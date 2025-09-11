PY=python
PIP=pip

.PHONY: setup data train eval app lint fmt test precommit

setup:
	$(PIP) install -U pip
	$(PIP) install -e .[dev]
	pre-commit install

data:
	$(PY) -m credsafe.data.ingest

train:
	$(PY) -m credsafe.models.train_pd

eval:
	$(PY) -m credsafe.models.evaluate

app:
	streamlit run app/Home.py

lint:
	ruff check .

fmt:
	black . && isort .

test:
	pytest -q

precommit:
	pre-commit run --all-files
