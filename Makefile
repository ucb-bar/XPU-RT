.PHONY: test test-fast lint format typecheck clean clean-cache clean-generated all

test:
	uv run pytest tests/ -v

test-fast:
	uv run pytest tests/ -v -x --timeout=120

lint:
	uv run ruff check python/ benchmarks/ tests/

format:
	uv run ruff format python/ benchmarks/ tests/

typecheck:
	uv run mypy python/compgen/

clean: clean-generated

clean-cache:
	rm -rf .compgen_cache/ recipe_library/ __pycache__/

clean-generated:
	uv run python scripts/clean_generated.py

all: lint format typecheck test
