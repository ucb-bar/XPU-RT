.PHONY: test test-fast lint lint-check format format-check typecheck \
	clean clean-cache clean-generated lockfile-check docs docs-strict \
	smoke ci all

# Fast developer loop.
test:
	uv run pytest tests/ -v

test-fast:
	uv run pytest tests/ -v -x --timeout=120

# Lint + format (auto-fix).
lint:
	uv run ruff check python/ benchmarks/ tests/

format:
	uv run ruff format python/ benchmarks/ tests/

# CI-equivalent lint + format check (no mutations).
lint-check:
	uv run ruff check python/ benchmarks/ tests/

format-check:
	uv run ruff format --check python/ benchmarks/ tests/

typecheck:
	uv run mypy python/compgen/

lockfile-check:
	uv lock --check

# Docs.
docs:
	uv run mkdocs serve

docs-strict:
	uv run mkdocs build --strict

# Smoke: exercise the scaffold-pack + load_pack flow (mirrors pr.yml smoke-cli).
smoke:
	uv run compgen --version
	rm -rf /tmp/compgen-smoke
	uv run compgen scaffold-pack --kind quantization --name smoke_pack --out /tmp/compgen-smoke
	PYTHONPATH=/tmp/compgen-smoke/smoke_pack/src uv run python -c \
	  "from compgen.packs import load_pack; l=load_pack('smoke_pack'); print(l.manifest.name, l.manifest.kinds)"

# Mirrors the pr.yml CI gate. Run before pushing.
ci: lint-check format-check typecheck lockfile-check test

# Legacy: full dev pass incl. auto-format.
all: lint format typecheck test

# Housekeeping.
clean: clean-generated

clean-cache:
	rm -rf .compgen_cache/ recipe_library/ __pycache__/

clean-generated:
	uv run python scripts/clean_generated.py
