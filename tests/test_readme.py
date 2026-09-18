"""The README's quickstart must actually run.

A quickstart that no longer works is worse than none: it is the first thing a
visitor copies, and it fails in their terminal rather than in our CI. So the
block is extracted from README.md and executed as part of the suite.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

README = Path(__file__).resolve().parent.parent / "README.md"
SRC = Path(__file__).resolve().parent.parent / "src"


def _quickstart_source() -> str:
    text = README.read_text()
    match = re.search(r"## Quickstart.*?```python\n(.*?)```", text, re.S)
    assert match, "README.md no longer has a ```python block under ## Quickstart"
    return match.group(1)


def test_readme_quickstart_runs():
    # A subprocess, not exec(): the snippet calls asyncio.run(), which cannot
    # start a second event loop inside the one pytest-asyncio is already running.
    result = subprocess.run(
        [sys.executable, "-c", _quickstart_source()],
        capture_output=True,
        text=True,
        timeout=120,
        env={"PYTHONPATH": str(SRC), "PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0, f"README quickstart failed:\n{result.stderr}"
    assert "500 responses" in result.stdout
    assert "batch size distribution" in result.stdout


def test_readme_quickstart_imports_exist():
    """Catch a renamed export before it reaches the front page."""
    source = _quickstart_source()
    imported = re.search(r"from llm_gateway import (.+)", source)
    assert imported, "quickstart no longer imports from llm_gateway"

    import llm_gateway

    for name in (n.strip() for n in imported.group(1).split(",")):
        assert hasattr(llm_gateway, name), f"README imports missing export: {name}"
