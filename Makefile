PY ?= .venv/bin/python

.PHONY: setup prep train curves eval-dev dev jev-dry

setup:
	python3 -m venv .venv && $(PY) -m pip install -r requirements.txt

prep:
	$(PY) -m src.prep

train:
	$(PY) -m src.models.train

curves:
	$(PY) -m src.models.learning_curve

eval-dev:
	$(PY) -m src.eval.run --split dev

jev-dry:
	$(PY) -m src.jev.run --dry-run

# Full dev pipeline from scratch (no 2025 data is touched)
dev: prep train curves eval-dev

.PHONY: test jev-pilot-preview
test:
	$(PY) -m unittest discover -s tests -v

jev-pilot-preview:
	$(PY) -m src.jev.pilot --plays 20
