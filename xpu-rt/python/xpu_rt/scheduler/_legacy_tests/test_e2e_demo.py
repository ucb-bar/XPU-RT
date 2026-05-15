"""Test that the E2E demo script runs without error."""

from __future__ import annotations

import sys
from pathlib import Path


def test_e2e_demo_runs() -> None:
    """The E2E demo script must execute without raising."""
    # Add scripts/ to path so we can import
    scripts_dir = Path(__file__).parent.parent / "scripts"
    sys.path.insert(0, str(scripts_dir))

    from e2e_demo import main

    main()  # Should not raise
