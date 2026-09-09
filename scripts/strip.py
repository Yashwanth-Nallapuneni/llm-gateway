#!/usr/bin/env python3
"""Blank the four core functions so they can be rebuilt from scratch.

    python scripts/strip.py            # blank them
    python scripts/strip.py --restore  # put the originals back
    python scripts/strip.py --check    # report current state, change nothing

Originals are copied to `.reference/` before anything is modified, so the
rebuild loop ends with a diff against the reference implementation.

Signatures, docstrings, tests and every other module are left untouched.
"""

from __future__ import annotations

import argparse
import ast
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "llm_gateway"
REFERENCE = ROOT / ".reference"

# module filename -> qualified names to blank
TARGETS: dict[str, tuple[str, ...]] = {
    "rate_limit.py": ("TokenBucket.acquire", "TokenBucket.try_acquire"),
    "retry.py": ("RetryPolicy.delay_for",),
    "batching.py": ("Batcher.collect",),
    "routing.py": ("ProviderRouter.select",),
}

MARKER = "raise NotImplementedError"


def _find(tree: ast.Module, qualname: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    cls_name, func_name = qualname.split(".")
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls_name:
            for item in node.body:
                if (
                    isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and item.name == func_name
                ):
                    return item
            raise SystemExit(f"error: {cls_name} has no method {func_name}")
    raise SystemExit(f"error: no class {cls_name}")


def _blank_one(lines: list[str], func) -> list[str]:
    """Return `lines` with this function's body replaced.

    Keeps everything from the `def` line through the docstring (if any) and
    replaces the rest of the body with a single raise.
    """
    first_stmt = func.body[0]
    keep_through = func.lineno - 1  # 0-indexed, exclusive end of "signature"

    # Walk forward to the end of the parameter list: that is the line before
    # the first body statement starts.
    body_start = first_stmt.lineno - 1
    keep_through = body_start

    has_docstring = (
        isinstance(first_stmt, ast.Expr)
        and isinstance(first_stmt.value, ast.Constant)
        and isinstance(first_stmt.value.value, str)
    )
    if has_docstring:
        keep_through = first_stmt.end_lineno  # 0-indexed exclusive

    indent = " " * first_stmt.col_offset
    replacement = [
        f"{indent}{MARKER}(\n",
        f'{indent}    "rebuild me: see the test suite, then diff against '
        f'.reference/"\n',
        f"{indent})\n",
    ]
    return lines[:keep_through] + replacement + lines[func.end_lineno :]


def _is_stripped(path: Path, qualnames: tuple[str, ...]) -> bool:
    tree = ast.parse(path.read_text())
    for q in qualnames:
        func = _find(tree, q)
        body = [s for s in func.body if not _is_docstring(s)]
        if not (len(body) == 1 and isinstance(body[0], ast.Raise)):
            return False
    return True


def _is_docstring(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def strip() -> int:
    REFERENCE.mkdir(exist_ok=True)
    for filename, qualnames in TARGETS.items():
        path = SRC / filename
        ref = REFERENCE / filename

        if _is_stripped(path, qualnames):
            print(f"  {filename}: already stripped, skipping")
            continue

        # Snapshot BEFORE touching anything. Never overwrite an existing
        # reference with an already-stripped file -- that would destroy the
        # only copy of the implementation.
        if not ref.exists():
            shutil.copy2(path, ref)

        lines = path.read_text().splitlines(keepends=True)
        # Blank in reverse source order so earlier line numbers stay valid.
        tree = ast.parse("".join(lines))
        funcs = sorted(
            (_find(tree, q) for q in qualnames), key=lambda f: f.lineno, reverse=True
        )
        for func in funcs:
            lines = _blank_one(lines, func)

        new_source = "".join(lines)
        ast.parse(new_source)  # refuse to write anything that will not parse
        path.write_text(new_source)
        print(f"  {filename}: blanked {', '.join(qualnames)}")

    print("\nreference copies in .reference/")
    print("next: pytest  ->  expect failures in exactly those four areas")
    return 0


def restore() -> int:
    if not REFERENCE.exists():
        print("nothing to restore: .reference/ does not exist", file=sys.stderr)
        return 1
    for filename in TARGETS:
        ref = REFERENCE / filename
        if not ref.exists():
            print(f"  {filename}: no reference copy, skipping")
            continue
        shutil.copy2(ref, SRC / filename)
        print(f"  {filename}: restored")
    return 0


def check() -> int:
    for filename, qualnames in TARGETS.items():
        state = "stripped" if _is_stripped(SRC / filename, qualnames) else "implemented"
        ref = "reference saved" if (REFERENCE / filename).exists() else "no reference"
        print(f"  {filename:<16} {state:<12} ({ref})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--restore", action="store_true", help="restore originals")
    group.add_argument("--check", action="store_true", help="report state only")
    args = parser.parse_args()

    if args.restore:
        return restore()
    if args.check:
        return check()
    return strip()


if __name__ == "__main__":
    raise SystemExit(main())
