.PHONY: format lint check

format:
	ruff format .
	ruff check --fix .

lint:
	ruff check .

check: lint
	pytest tests/
