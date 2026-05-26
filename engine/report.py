"""
report.py
=========
Experiment report generator for AB Audit.

Produces a professional PDF report formatted like a real company
experiment readout — the kind a data team would actually send to
a PM or engineering lead after running an A/B test.

Sections
--------
1. Header         Experiment name, date, metadata
2. Executive Summary  Plain-English verdict in 3 sentences
3. Experiment Setup   Config: alpha, power, metric type, peeking
4. Results Table      Raw numbers: n, rates, lift, CI, p-value
5. Validity Checks    All 8 checks with Pass/Warn/Fail
6. Bayesian Analysis  P(treatment > control) + credible interval
7. CUPED Summary      Variance reduction stats if available
8. Recommendations    Actionable next steps based on audit result
9. Footer         References + methodology note

Public API
----------
generate_report(audit, config, ...) -> bytes   PDF as bytes
save_report(audit, config, path)               Save PDF to disk
generate_markdown(audit, config, ...) -> str   Markdown string
"""

import io
import os
from datetime import datetime
from typing import Optional

from engine import AuditResult, ExperimentConfig, Severity


# ════════════════════════════════════════════════════════════
# COLOUR PALETTE  (matches dashboard tokens)
# ════════════════════════════════════════════════════════════

C_BG      = (28,  28,  30)
C_CARD    = (42,  42,  45)
C_CARD2   = (51,  51,  54)
C_TXT     = (255, 255, 255)
C_TXT2    = (160, 160, 168)
C_TXT3    = (90,  90,  98)
C_MINT    = (78,  205, 196)
C_CORAL   = (251, 113, 133)
C_YELLOW  = (252, 211, 77)
C_PURPLE  = (167, 139, 250)
C_BLUE    = (96,  165, 250)
C_BORDER  = (58,  58,  62)

SEV_COLOR = {
    Severity.PASS: C_MINT,
    Severity.WARN: C_YELLOW,
    Severity.FAIL: C_CORAL,
}
SEV_LABEL = {
    Severity.PASS: "PASS",
    Severity.WARN: "WARN",
    Severity.FAIL: "FAIL",
}


# ════════════════════════════════════════════════════════════
# PDF GENERATOR  (uses reportlab)
# ════════════════════════════════════════════════════════════

def generate_report(
    audit:          AuditResult,
    config:         ExperimentConfig,
    bayesian_result=None,
    cuped_result=None,
    title:          str = "AB Audit — Experiment Report",
) -> bytes:
    """
    Generate a PDF experiment report.

    Parameters
    ----------
    audit : AuditResult
        Output of run_full_audit().
    config : ExperimentConfig
        Experiment configuration used for the audit.
    bayesian_result : BayesianResult, optional
        Output of run_bayesian_ab() or bayesian_from_dataframe().
    cuped_result : CUPEDResult, optional
        Output of run_cuped_analysis().
    title : str
        Report title shown in the header.

    Returns
    -------
    bytes
        PDF file as bytes — pass directly to st.download_button().
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table,
            TableStyle, HRFlowable, KeepTogether,
        )
        from reportlab.lib import colors
        from reportlab.lib.colors import HexColor, Color

        def rgb(t): return Color(t[0]/255, t[1]/255, t[2]/255)

        buf  = io.BytesIO()
        W, H = A4
        M    = 20 * mm

        doc = SimpleDocTemplate(
            buf,
            pagesize=A4,
            leftMargin=M, rightMargin=M,
            topMargin=M, bottomMargin=M,
            title=title,
        )

        # ── Style helpers ──────────────────────────────────────
        def sty(name, **kw):
            base = {
                "fontName":  "Helvetica",
                "fontSize":  10,
                "textColor": rgb(C_TXT2),
                "leading":   14,
            }
            base.update(kw)
            return ParagraphStyle(name, **base)

        S_H1   = sty("h1",  fontSize=20, textColor=rgb(C_TXT),
                     fontName="Helvetica-Bold", leading=26)
        S_H2   = sty("h2",  fontSize=13, textColor=rgb(C_TXT),
                     fontName="Helvetica-Bold", leading=18, spaceBefore=12)
        S_H3   = sty("h3",  fontSize=10, textColor=rgb(C_MINT),
                     fontName="Helvetica-Bold", leading=14,
                     textTransform="uppercase", spaceBefore=6)
        S_BODY = sty("body", fontSize=9, leading=14)
        S_MONO = sty("mono", fontSize=8, fontName="Courier",
                     textColor=rgb(C_TXT2), leading=12)
        S_SM   = sty("sm",  fontSize=8, textColor=rgb(C_TXT3), leading=11)

        def hr(color=C_BORDER, thickness=0.5):
            return HRFlowable(
                width="100%",
                thickness=thickness,
                color=rgb(color),
                spaceAfter=6,
                spaceBefore=6,
            )

        def sp(h=4): return Spacer(1, h)

        def p(text, style=None):
            return Paragraph(text, style or S_BODY)

        # ── Overall severity ───────────────────────────────────
        sev       = audit.overall_severity
        sev_color = SEV_COLOR[sev]
        sev_label = SEV_LABEL[sev]
        meta      = audit.experiment_meta

        # ════════════════════════════════════════════════════════
        story = []

        # ── HEADER ────────────────────────────────────────────
        story += [
            p("AB AUDIT", sty("eyebrow",
                fontSize=9, textColor=rgb(C_MINT),
                fontName="Helvetica-Bold",
                letterSpacing=2)),
            sp(4),
            p(config.name or "Experiment Report", S_H1),
            sp(4),
            p(f"Generated {datetime.now().strftime('%d %B %Y, %H:%M')}  |  "
              f"Metric: {config.metric_type.value.title()}  |  "
              f"&alpha; = {config.alpha}  |  "
              f"Target power = {config.target_power:.0%}",
              S_SM),
            sp(8),
            hr(C_MINT, 1),
            sp(4),
        ]

        # ── EXECUTIVE SUMMARY ─────────────────────────────────
        story.append(p("Executive Summary", S_H2))
        story.append(sp(4))

        lift_pp = meta.get("lift_pp", 0)
        p_ctrl  = meta.get("p_control", 0)
        p_trt   = meta.get("p_treatment", 0)
        n_c     = meta.get("n_control", 0)
        n_t     = meta.get("n_treatment", 0)

        if sev == Severity.PASS:
            summary = (
                f"This experiment is <b>statistically valid</b>. "
                f"The treatment arm showed a <b>{lift_pp:+.2f} pp</b> lift over control "
                f"({p_ctrl*100:.2f}% vs {p_trt*100:.2f}%) across "
                f"{n_c+n_t:,} users. "
                f"All 8 validity checks passed. "
                f"This result can be used to make a shipping decision with confidence."
            )
        elif sev == Severity.WARN:
            summary = (
                f"This experiment produced a result of "
                f"<b>{lift_pp:+.2f} pp</b> lift "
                f"({p_ctrl*100:.2f}% vs {p_trt*100:.2f}%) "
                f"but <b>{audit.n_warned} validity warning(s)</b> were detected. "
                f"Proceed with caution and review the flagged checks before shipping."
            )
        else:
            summary = (
                f"This experiment has <b>{audit.n_failed} critical failure(s)</b> "
                f"that undermine the validity of the result. "
                f"The observed lift of <b>{lift_pp:+.2f} pp</b> cannot be trusted. "
                f"Do not make a shipping decision based on this data."
            )

        story.append(p(summary, sty("summary",
            fontSize=10, textColor=rgb(C_TXT), leading=16,
            backColor=rgb(sev_color + (0.08,)) if False else None,
        )))
        story.append(sp(8))

        # Overall verdict badge as a table cell
        verdict_tbl = Table(
            [[p(f"  {sev_label}  —  {audit.score_summary}",
               sty("badge", fontSize=10, fontName="Helvetica-Bold",
                   textColor=rgb(sev_color)))]],
            colWidths=[W - 2*M],
        )
        verdict_tbl.setStyle(TableStyle([
            ("BACKGROUND",  (0,0), (-1,-1), rgb(sev_color) + (0.08,) if False else colors.Color(*[x/255 for x in sev_color], alpha=0.08)),
            ("ROUNDEDCORNERS", [8]),
            ("TOPPADDING",    (0,0), (-1,-1), 10),
            ("BOTTOMPADDING", (0,0), (-1,-1), 10),
            ("LEFTPADDING",   (0,0), (-1,-1), 14),
        ]))
        story += [verdict_tbl, sp(12), hr()]

        # ── EXPERIMENT RESULTS ────────────────────────────────
        story += [p("Results", S_H2), sp(4)]

        results_data = [
            ["Metric",        "Control",                    "Treatment",                  "Difference"],
            ["Users",         f"{n_c:,}",                   f"{n_t:,}",                   f"{n_c+n_t:,} total"],
            ["Rate / Mean",   f"{p_ctrl*100:.3f}%",         f"{p_trt*100:.3f}%",          f"{lift_pp:+.3f} pp"],
        ]

        rtbl = Table(results_data, colWidths=[(W - 2*M)/4]*4)
        rtbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,0),  colors.Color(*[x/255 for x in C_CARD])),
            ("TEXTCOLOR",     (0,0), (-1,0),  colors.Color(*[x/255 for x in C_TXT])),
            ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
            ("FONTSIZE",      (0,0), (-1,-1), 9),
            ("TEXTCOLOR",     (0,1), (-1,-1), colors.Color(*[x/255 for x in C_TXT2])),
            ("BACKGROUND",    (0,1), (-1,-1), colors.Color(*[x/255 for x in C_CARD])),
            ("ROWBACKGROUNDS",(0,1), (-1,-1), [
                colors.Color(*[x/255 for x in C_CARD]),
                colors.Color(*[x/255 for x in C_CARD2]),
            ]),
            ("GRID",          (0,0), (-1,-1), 0.3, colors.Color(*[x/255 for x in C_BORDER])),
            ("TOPPADDING",    (0,0), (-1,-1), 7),
            ("BOTTOMPADDING", (0,0), (-1,-1), 7),
            ("LEFTPADDING",   (0,0), (-1,-1), 10),
            ("ALIGN",         (1,0), (-1,-1), "CENTER"),
        ]))
        story += [rtbl, sp(12), hr()]

        # ── VALIDITY CHECKS ───────────────────────────────────
        story += [p("Validity Checks", S_H2), sp(4)]

        check_icons = ["01","02","03","04","05","06","07","08"]
        for i, check in enumerate(audit.checks):
            sc       = SEV_COLOR[check.severity]
            sl       = SEV_LABEL[check.severity]
            row_data = [[
                p(f"<b>{check_icons[i]}  {check.name}</b>",
                  sty(f"cn{i}", fontSize=9, fontName="Helvetica-Bold",
                      textColor=colors.Color(*[x/255 for x in C_TXT]))),
                p(sl, sty(f"cs{i}", fontSize=8, fontName="Helvetica-Bold",
                          textColor=colors.Color(*[x/255 for x in sc]),
                          alignment=TA_CENTER)),
                p(check.verdict,
                  sty(f"cv{i}", fontSize=8,
                      textColor=colors.Color(*[x/255 for x in C_TXT2]))),
            ]]
            ctbl = Table(row_data, colWidths=[
                (W-2*M)*0.28,
                (W-2*M)*0.10,
                (W-2*M)*0.62,
            ])
            ctbl.setStyle(TableStyle([
                ("BACKGROUND",    (0,0),(-1,-1), colors.Color(*[x/255 for x in C_CARD])),
                ("LEFTBORDER",    (0,0),(0,-1),  2, colors.Color(*[x/255 for x in sc])),
                ("TOPPADDING",    (0,0),(-1,-1), 8),
                ("BOTTOMPADDING", (0,0),(-1,-1), 8),
                ("LEFTPADDING",   (0,0),(0,-1),  12),
                ("LEFTPADDING",   (1,0),(-1,-1), 8),
                ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
                ("LINEBELOW",     (0,0),(-1,-1), 0.3,
                 colors.Color(*[x/255 for x in C_BORDER])),
            ]))
            story += [KeepTogether([ctbl, sp(3)])]

        story += [sp(8), hr()]

        # ── BAYESIAN ANALYSIS ─────────────────────────────────
        if bayesian_result is not None:
            bay = bayesian_result
            story += [p("Bayesian Analysis", S_H2), sp(4)]
            story += [
                p(f"<b>P(treatment &gt; control):</b> {bay.prob_treatment_better:.1%}  |  "
                  f"<b>Expected lift:</b> {bay.expected_lift*100:+.3f} pp  |  "
                  f"<b>95% Credible interval:</b> "
                  f"[{bay.credible_interval[0]*100:+.3f}, "
                  f"{bay.credible_interval[1]*100:+.3f}] pp",
                  sty("bay", fontSize=9, textColor=rgb(C_TXT), leading=14)),
                sp(4),
                p(bay.decision, sty("baydec", fontSize=9,
                  textColor=rgb(C_MINT), leading=14)),
                sp(8),
                hr(),
            ]

        # ── CUPED SUMMARY ─────────────────────────────────────
        if cuped_result is not None:
            cr = cuped_result
            story += [p("CUPED Variance Reduction", S_H2), sp(4)]
            story += [
                p(f"Pre/post correlation &rho; = <b>{cr.rho:+.3f}</b>  |  "
                  f"Variance reduction: <b>{cr.variance_reduction_pct:.1f}%</b>  |  "
                  f"CI width reduction: <b>{cr.ci_width_reduction_pct:.1f}%</b>  |  "
                  f"Sample size equivalent: <b>{cr.sample_size_equivalent:,.0f} users/arm</b>",
                  sty("cuped", fontSize=9, textColor=rgb(C_TXT), leading=14)),
                sp(8),
                hr(),
            ]

        # ── RECOMMENDATIONS ───────────────────────────────────
        story += [p("Recommendations", S_H2), sp(4)]

        recs = []
        if sev == Severity.PASS:
            recs += [
                "Ship the treatment variant.",
                "Monitor post-launch metrics for at least 7 days to confirm the lift holds.",
                "Consider CUPED in future experiments on the same metric to improve sensitivity.",
            ]
        elif sev == Severity.WARN:
            for check in audit.checks:
                if check.severity == Severity.WARN and check.recommendation != "N/A":
                    recs.append(check.recommendation)
            recs.append("Re-evaluate after addressing the flagged warnings.")
        else:
            for check in audit.checks:
                if check.severity == Severity.FAIL and check.recommendation != "N/A":
                    recs.append(check.recommendation)
            recs.append("Do not ship. Fix the identified issues and re-run the experiment.")

        for rec in recs:
            story.append(p(f"&bull;  {rec}",
                sty("rec", fontSize=9, textColor=rgb(C_TXT2),
                    leading=14, leftIndent=10)))
            story.append(sp(3))

        story += [sp(8), hr()]

        # ── METHODOLOGY NOTE ──────────────────────────────────
        story += [
            p("Methodology", S_H3),
            sp(3),
            p("Validity checks implemented from first principles in NumPy. "
              "Cross-validated against SciPy. "
              "References: Kohavi et al. (2020) Trustworthy Online Controlled Experiments; "
              "Deng et al. (2013) CUPED, Microsoft Research KDD; "
              "Johari et al. (2015) Always Valid Inference; "
              "Benjamini &amp; Hochberg (1995) Controlling the False Discovery Rate.",
              sty("meth", fontSize=7.5, textColor=rgb(C_TXT3), leading=12)),
            sp(6),
            p(f"AB Audit v1.0  |  Report generated {datetime.now().strftime('%Y-%m-%d %H:%M')}",
              sty("footer", fontSize=7, textColor=rgb(C_TXT3), alignment=TA_CENTER)),
        ]

        # ── BUILD ─────────────────────────────────────────────
        doc.build(story)
        return buf.getvalue()

    except ImportError:
        # reportlab not installed — fall back to markdown
        md = generate_markdown(audit, config, bayesian_result, cuped_result)
        return md.encode("utf-8")


# ════════════════════════════════════════════════════════════
# MARKDOWN GENERATOR  (fallback + GitHub README export)
# ════════════════════════════════════════════════════════════

def generate_markdown(
    audit:          AuditResult,
    config:         ExperimentConfig,
    bayesian_result=None,
    cuped_result=None,
) -> str:
    """
    Generate a Markdown experiment report.

    Works without reportlab. Useful for GitHub, Notion, Confluence.
    """
    sev   = audit.overall_severity
    label = SEV_LABEL[sev]
    meta  = audit.experiment_meta
    now   = datetime.now().strftime("%d %B %Y, %H:%M")

    lines = [
        f"# AB Audit — {config.name}",
        f"",
        f"**Generated:** {now}  ",
        f"**Metric type:** {config.metric_type.value.title()}  ",
        f"**Alpha:** {config.alpha}  |  **Target power:** {config.target_power:.0%}",
        f"",
        f"---",
        f"",
        f"## Executive Summary",
        f"",
        f"**Overall result: {label}** — {audit.score_summary}",
        f"",
        audit.overall_verdict,
        f"",
        f"---",
        f"",
        f"## Results",
        f"",
        f"| Metric | Control | Treatment | Difference |",
        f"|--------|---------|-----------|------------|",
        f"| Users | {meta.get('n_control',0):,} | {meta.get('n_treatment',0):,} | {meta.get('n_control',0)+meta.get('n_treatment',0):,} total |",
        f"| Rate | {meta.get('p_control',0)*100:.3f}% | {meta.get('p_treatment',0)*100:.3f}% | {meta.get('lift_pp',0):+.3f} pp |",
        f"",
        f"---",
        f"",
        f"## Validity Checks",
        f"",
    ]

    for i, check in enumerate(audit.checks):
        label_c = SEV_LABEL[check.severity]
        icon    = {"PASS": "[+]", "WARN": "[!]", "FAIL": "[x]"}[label_c]
        lines += [
            f"### {icon} {i+1:02d}. {check.name} — {label_c}",
            f"",
            f"{check.verdict}",
            f"",
        ]
        if check.severity != Severity.PASS:
            if check.cost_of_violation and check.cost_of_violation != "N/A":
                lines.append(f"**Cost:** {check.cost_of_violation}")
                lines.append(f"")
            if check.recommendation and check.recommendation != "N/A":
                lines.append(f"**Action:** {check.recommendation}")
                lines.append(f"")

    lines += ["---", "", "## Bayesian Analysis", ""]
    if bayesian_result is not None:
        bay = bayesian_result
        lines += [
            f"- **P(treatment > control):** {bay.prob_treatment_better:.1%}",
            f"- **Expected lift:** {bay.expected_lift*100:+.3f} pp",
            f"- **95% Credible interval:** [{bay.credible_interval[0]*100:+.3f}, {bay.credible_interval[1]*100:+.3f}] pp",
            f"- **Decision:** {bay.decision}",
            f"",
        ]
    else:
        lines += ["*Bayesian analysis not run.*", ""]

    lines += ["---", "", "## CUPED Variance Reduction", ""]
    if cuped_result is not None:
        cr = cuped_result
        lines += [
            f"- **Correlation rho:** {cr.rho:+.3f}",
            f"- **Variance reduction:** {cr.variance_reduction_pct:.1f}%",
            f"- **CI width reduction:** {cr.ci_width_reduction_pct:.1f}%",
            f"- **Sample size equivalent:** {cr.sample_size_equivalent:,.0f} users/arm",
            f"",
        ]
    else:
        lines += ["*CUPED analysis not run.*", ""]

    lines += [
        "---",
        "",
        "## Recommendations",
        "",
    ]
    for check in audit.checks:
        if check.severity != Severity.PASS and check.recommendation not in ("N/A", None, ""):
            lines.append(f"- {check.recommendation}")
    lines += [
        "",
        "---",
        "",
        "*AB Audit v1.0 — github.com/yourusername/ab_audit*",
    ]

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════
# SAVE UTILITY
# ════════════════════════════════════════════════════════════

def save_report(
    audit:          AuditResult,
    config:         ExperimentConfig,
    path:           str,
    bayesian_result=None,
    cuped_result=None,
) -> str:
    """
    Save the report to disk.

    If path ends in .pdf, generates PDF.
    If path ends in .md, generates Markdown.

    Returns the path it was saved to.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".md":
        content = generate_markdown(audit, config, bayesian_result, cuped_result)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    else:
        content = generate_report(audit, config, bayesian_result, cuped_result)
        with open(path, "wb") as f:
            f.write(content)

    return path


__all__ = [
    "generate_report",
    "generate_markdown",
    "save_report",
]
