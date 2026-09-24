"""Every script in examples/ must keep running.

They are the first code a new user copies and runs unmodified, so a break
there is worse than a break in a test: it fails in someone else's terminal.
Each one runs offline against MockProvider (no API key needed), so we run
each as a subprocess with the provider API keys stripped from the
environment and assert it exits cleanly and prints something.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
SRC = Path(__file__).resolve().parent.parent / "src"

EXAMPLE_SCRIPTS = sorted(EXAMPLES_DIR.glob("*.py"))


@pytest.mark.parametrize("script", EXAMPLE_SCRIPTS, ids=lambda p: p.name)
def test_example_runs(script: Path) -> None:
    env = {
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": str(SRC),
    }
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    assert result.returncode == 0, (
        f"{script.name} exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert result.stdout.strip(), f"{script.name} produced no output"


def test_examples_dir_is_not_empty() -> None:
    # Guards against a glob typo silently turning this file into a no-op.
    assert EXAMPLE_SCRIPTS, "no examples/*.py scripts found to test"
