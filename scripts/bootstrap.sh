#!/usr/bin/env bash
# CompGen bootstrap script
#
# Sets up the development environment from a fresh clone.
# Usage: ./scripts/bootstrap.sh
#
# Prerequisites: Python 3.11+, uv, git

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# --- Colors (if terminal supports them) ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# --- Check prerequisites ---
info "Checking prerequisites..."

# Python 3.11+
if ! command -v python3 &> /dev/null; then
    error "Python 3 not found. Please install Python 3.11+."
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [[ "$PYTHON_MAJOR" -lt 3 ]] || { [[ "$PYTHON_MAJOR" -eq 3 ]] && [[ "$PYTHON_MINOR" -lt 11 ]]; }; then
    error "Python 3.11+ required, found Python ${PYTHON_VERSION}"
    exit 1
fi
info "Python ${PYTHON_VERSION} OK"

# uv
if ! command -v uv &> /dev/null; then
    error "uv not found. Install it: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi
info "uv $(uv --version | head -1) OK"

# git
if ! command -v git &> /dev/null; then
    error "git not found."
    exit 1
fi
info "git OK"

# --- Initialize submodules ---
info "Initializing git submodules..."
cd "$REPO_ROOT"
git submodule update --init --recursive

if [[ ! -f "third_party/autocomp/pyproject.toml" ]]; then
    error "autocomp submodule not initialized. Check .gitmodules."
    exit 1
fi
info "Submodules initialized"

# --- Create virtual environment and install dependencies ---
info "Setting up virtual environment..."

if [[ ! -d ".venv" ]]; then
    uv venv
fi

info "Installing CompGen and dependencies..."
uv sync

info "Installing autocomp (editable)..."
uv pip install -e third_party/autocomp

# --- Smoke tests ---
info "Running smoke tests..."

# Test compgen import
if uv run python -c "import compgen; print(f'compgen {compgen.__version__}')"; then
    info "compgen import OK"
else
    error "Failed to import compgen"
    exit 1
fi

# Test autocomp import
if uv run python -c "from autocomp.hw_config import HardwareConfig; print('autocomp OK')"; then
    info "autocomp import OK"
else
    warn "autocomp import failed -- kernel search features may not work"
fi

# Test CLI
if uv run python -m compgen.cli --help > /dev/null 2>&1; then
    info "CLI help OK"
else
    warn "CLI help failed -- click may need installation"
fi

# --- Done ---
echo ""
info "Bootstrap complete!"
echo ""
echo "  Next steps:"
echo "    uv run python -m compgen.cli --help                    # See CLI commands"
echo "    uv run python scripts/e2e_demo.py                      # Run the current demo path"
echo "    uv run pytest tests/test_version.py                    # Run smoke test"
echo "    uv run ruff check python/                              # Lint check"
echo ""
