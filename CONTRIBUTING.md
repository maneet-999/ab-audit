# Contributing to AB Audit

Thanks for your interest. Here's how to contribute without breaking things.

---

## Ground Rules

- All new statistical logic needs unit tests before a PR will be merged.
- Keep the engine (`engine/`) and the UI (`app/`) cleanly separated. No Streamlit imports in the engine.
- One feature or fix per PR. Keep them small and reviewable.

---

## Setup

```bash
git clone https://github.com/yourusername/ab-audit.git
cd ab-audit
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pytest tests/ -v   # all 53 should pass before you start
```

---

## Workflow

```bash
# 1. Fork and clone your fork
# 2. Create a branch
git checkout -b feature/your-feature-name

# 3. Make changes, write tests
# 4. Verify tests pass
pytest tests/ -v

# 5. Commit with a clear message
git commit -m "add: sequential testing via mSPRT"

# 6. Push and open a PR against main
git push origin feature/your-feature-name
```

---

## Commit Message Format

```
add: short description       # new feature
fix: short description       # bug fix
refactor: short description  # no behaviour change
docs: short description      # documentation only
test: short description      # tests only
```

---

## What's Worth Contributing

- Additional validity checks (e.g. interference testing, variance ratio test)
- Performance improvements to the Monte Carlo engine
- More demo scenarios / data generators
- Deployment configs (Docker, Streamlit Cloud)
- Documentation improvements

---

## What to Avoid

- Adding heavy dependencies without discussion
- Changing the public API of `engine/__init__.py` without a version bump
- Committing data files or generated PDFs
- Reformatting the entire codebase in one PR

---

## Questions

Open an issue or email directly. See the README for contact details.
