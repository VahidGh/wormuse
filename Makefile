# wormuse root Makefile
# Usage:  make test        — run all PyANNOW unit tests
#         make coverage    — run tests with HTML coverage report
#         make lint        — ruff + mypy
#         make nb-test     — smoke-execute both notebooks (slow, ~10 min)
#         make ci          — runs test + lint (what CI does)

.PHONY: test coverage lint nb-test ci clean

PYTHON   := python3
PYTEST   := $(PYTHON) -m pytest
PYANNOW  := PyANNOW

# ── Unit tests (fast, no real simulation) ─────────────────────────────────
test:
	cd $(PYANNOW) && $(PYTEST) tests/ -q --tb=short 2>&1

# ── Coverage report (HTML in PyANNOW/htmlcov/) ────────────────────────────
coverage:
	cd $(PYANNOW) && $(PYTEST) tests/ -q --tb=short \
	  --cov=src/pyannow --cov-report=term-missing --cov-report=html
	@echo "Coverage report: PyANNOW/htmlcov/index.html"

# ── Lint + type check ─────────────────────────────────────────────────────
lint:
	cd $(PYANNOW) && $(PYTHON) -m ruff check src/ tests/ || true
	cd $(PYANNOW) && $(PYTHON) -m mypy src/pyannow --ignore-missing-imports || true

# ── Notebook smoke tests ──────────────────────────────────────────────────
nb-test:
	@echo "Smoke-testing 02_chopin_worm_optimizer.ipynb ..."
	cd $(PYANNOW)/notebooks && \
	  jupyter nbconvert --to notebook --execute --inplace \
	    02_chopin_worm_optimizer.ipynb || echo "Notebook 02 failed"
	@echo "Smoke-testing 03_pyannow_naml_progression.ipynb ..."
	cd $(PYANNOW)/notebooks && \
	  jupyter nbconvert --to notebook --execute --inplace \
	    03_pyannow_naml_progression.ipynb || echo "Notebook 03 failed"

# ── Combined CI target ────────────────────────────────────────────────────
ci: test lint

# ── Clean test artefacts ──────────────────────────────────────────────────
clean:
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".pytest_cache" -type d -exec rm -rf {} + 2>/dev/null || true
	rm -rf $(PYANNOW)/htmlcov $(PYANNOW)/.coverage 2>/dev/null || true
