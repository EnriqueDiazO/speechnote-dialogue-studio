.PHONY: install run test lint doctor

install:
	python3 -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -e '.[dev]'

run:
	.venv/bin/python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8510

test:
	.venv/bin/python -m pytest -q

lint:
	.venv/bin/ruff check .

doctor:
	.venv/bin/python -m dialogue_studio.doctor

