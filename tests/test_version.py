"""The version is written in two places; a release with them out of sync
would publish one number and report another."""

import tomllib
from pathlib import Path

import llm_gateway


def test_package_version_matches_pyproject() -> None:
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    declared = tomllib.loads(pyproject.read_text())["project"]["version"]
    assert llm_gateway.__version__ == declared


def test_changelog_has_a_section_for_this_version() -> None:
    changelog = Path(__file__).resolve().parent.parent / "CHANGELOG.md"
    assert f"## [{llm_gateway.__version__}]" in changelog.read_text()
