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

# --- Native runtime build targets -------------------------------------------
# build-cpu-rt:  Builds the CPU-only libcompgen_rt and stages the .so into
#                python/compgen/runtime/native/prebuilt/ so the wheel
#                ships it via package_data.
# build-cuda-rt: Same but with -DCG_RT_WITH_CUDA=ON. Needs CUDA toolkit
#                (>= 12.6) on PATH. Compiles for SM_90 (Hopper) +
#                SM_100 (datacenter Blackwell, GB100/B200) +
#                SM_120 (workstation Blackwell, GB202).
# clean-rt:      Wipes the prebuilt dir.

PREBUILT_DIR := python/compgen/runtime/native/prebuilt

build-cpu-rt:
	cmake -S runtime/native/libcompgen_rt -B build/rt-cpu \
	      -DCG_RT_WITH_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
	cmake --build build/rt-cpu --parallel
	mkdir -p $(PREBUILT_DIR)
	cp build/rt-cpu/libcompgen_rt.so $(PREBUILT_DIR)/libcompgen_rt-cpu.so

build-cuda-rt:
	cmake -S runtime/native/libcompgen_rt -B build/rt-cuda \
	      -DCG_RT_WITH_CUDA=ON \
	      -DCMAKE_CUDA_ARCHITECTURES="90;100;120" \
	      -DCMAKE_BUILD_TYPE=Release
	cmake --build build/rt-cuda --parallel
	mkdir -p $(PREBUILT_DIR)
	cp build/rt-cuda/libcompgen_rt.so $(PREBUILT_DIR)/libcompgen_rt-cuda.so

clean-rt:
	rm -rf build/rt-cpu build/rt-cuda
	rm -f $(PREBUILT_DIR)/libcompgen_rt-*.so

# Build the wheel. Run `make build-cpu-rt build-cuda-rt` first to bundle
# native libraries; otherwise the wheel ships pure-Python (still works
# for CPU paths).
wheel:
	uv build --wheel

# --- Bridge convenience targets (Blackwell remote agent communication) ------
# bridge-push:  Garden -> Blackwell. Ships tmp/blackwell_bridge/ and any
#               wheels under dist/. Requires SSH alias `bwell` on Garden.
# bridge-pull:  Blackwell -> Garden. Pulls remote-agent updates back.
# bridge-check: Validate thread.md ↔ thread.jsonl agreement locally.

bridge-push:
	@SOCK=$$(tmp/blackwell_bridge/bin/find_agent.sh) || { \
		echo "ERROR: no live agent socket with BWRC-authorized key" >&2; \
		echo "Open a fresh VSCode terminal to refresh the forwarded agent." >&2; \
		exit 1; \
	}; \
	export SSH_AUTH_SOCK="$$SOCK"; \
	rsync -av --update -e "ssh -A -o BatchMode=yes" tmp/blackwell_bridge/ bwell:~/compgen/bridge/; \
	if [ -d dist ] && [ -n "$$(ls -A dist 2>/dev/null)" ]; then \
		rsync -av --update -e "ssh -A -o BatchMode=yes" dist/ bwell:~/compgen/wheels/; \
	fi

bridge-pull:
	@SOCK=$$(tmp/blackwell_bridge/bin/find_agent.sh) || { \
		echo "ERROR: no live agent socket with BWRC-authorized key" >&2; \
		echo "Open a fresh VSCode terminal to refresh the forwarded agent." >&2; \
		exit 1; \
	}; \
	export SSH_AUTH_SOCK="$$SOCK"; \
	rsync -av --update -e "ssh -A -o BatchMode=yes" bwell:~/compgen/bridge/ tmp/blackwell_bridge/

bridge-check:
	uv run python tmp/blackwell_bridge/bin/append.py check
