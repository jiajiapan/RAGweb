.PHONY: format lint check eval

format:
	ruff format .
	ruff check --fix .

lint:
	ruff check .

check: lint
	pytest tests/

eval:
	python -m backend.evaluation --samples 100 --top-k 5
