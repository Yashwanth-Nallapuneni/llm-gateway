"""Generate docs/failover-demo.svg from the REAL output of examples/failover_demo.py.

This is the source for the animated terminal recording at the top of the
README. It is not a video: it is a self-contained SVG that types out a
command and then reveals the program's actual captured stdout line by line
using CSS/SMIL animation, so it renders (and animates) directly on GitHub
with no external player.

Run:  python3 docs/make_demo_svg.py
Writes: docs/failover-demo.svg

No tool like vhs/asciinema was usable in the environment this was built in
(vhs needed a Go toolchain build blocked by outdated Command Line Tools,
which would have required sudo to fix). This script is the reproducible
substitute: it always re-runs the real demo and renders its real output,
nothing here is hand-typed.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "examples" / "failover_demo.py"
OUT = ROOT / "docs" / "failover-demo.svg"

PROMPT = "$ python examples/failover_demo.py"

FONT_SIZE = 14
LINE_HEIGHT = 20
PAD_X = 20
PAD_TOP = 44  # room for the title bar
CHROME_HEIGHT = 34
CHAR_WIDTH = FONT_SIZE * 0.6  # monospace approx, for sizing the prompt typing

# Timing
TYPE_DURATION = 1.2  # seconds to "type" the prompt
PRE_OUTPUT_PAUSE = 0.35
LINE_STAGGER = 0.16  # seconds between successive output lines appearing
HOLD_AT_END = 3.0
FADE_IN = 0.25


def capture_real_output() -> list[str]:
    result = subprocess.run(
        [sys.executable, str(DEMO)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    text = result.stdout.rstrip("\n")
    return text.split("\n")


def build_svg(lines: list[str]) -> str:
    n = len(lines)
    width = 860
    content_height = PAD_TOP + n * LINE_HEIGHT + 24
    height = max(content_height, 200)

    # Colors (dark terminal theme)
    bg = "#1e1e2e"
    chrome_bg = "#181825"
    fg = "#cdd6f4"
    prompt_color = "#a6e3a1"
    dim = "#6c7086"
    accent = "#89b4fa"

    total_type_end = TYPE_DURATION
    first_line_start = total_type_end + PRE_OUTPUT_PAUSE

    svg_parts: list[str] = []
    svg_parts.append(
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'xmlns="http://www.w3.org/2000/svg" font-family="ui-monospace, SFMono-Regular, '
        f'\'SF Mono\', Menlo, Consolas, monospace" role="img" '
        f'aria-label="Terminal recording of examples/failover_demo.py">'
    )

    # Background + rounded terminal window
    svg_parts.append(f'<rect width="{width}" height="{height}" rx="10" fill="{bg}"/>')
    svg_parts.append(
        f'<rect width="{width}" height="{CHROME_HEIGHT}" rx="10" fill="{chrome_bg}"/>'
    )
    svg_parts.append(
        f'<rect y="{CHROME_HEIGHT - 10}" width="{width}" height="10" fill="{chrome_bg}"/>'
    )
    for i, color in enumerate(["#ff5f56", "#ffbd2e", "#27c93f"]):
        cx = 20 + i * 20
        svg_parts.append(f'<circle cx="{cx}" cy="{CHROME_HEIGHT / 2}" r="6" fill="{color}"/>')
    svg_parts.append(
        f'<text x="{width / 2}" y="{CHROME_HEIGHT / 2 + 4}" fill="{dim}" '
        f'font-size="12" text-anchor="middle">failover_demo.py</text>'
    )

    # Typed prompt line, revealed via a growing clip rect (typing effect)
    prompt_y = PAD_TOP
    prompt_full = PROMPT
    clip_id = "typeclip"
    full_text_width = len(prompt_full) * CHAR_WIDTH + 4
    svg_parts.append(
        f'<clipPath id="{clip_id}"><rect x="0" y="0" height="{LINE_HEIGHT}" width="0">'
        f'<animate attributeName="width" from="0" to="{full_text_width:.1f}" '
        f'begin="0.3s" dur="{TYPE_DURATION}s" fill="freeze" calcMode="linear"/>'
        f'</rect></clipPath>'
    )
    svg_parts.append(
        f'<g clip-path="url(#{clip_id})" transform="translate({PAD_X}, {prompt_y - LINE_HEIGHT + 5})">'
        f'<text x="0" y="{LINE_HEIGHT - 5}" fill="{prompt_color}" font-size="{FONT_SIZE}" '
        f'xml:space="preserve">{escape(prompt_full)}</text></g>'
    )
    # blinking cursor block that stops blinking once typing completes, then fades
    cursor_x = PAD_X
    svg_parts.append(
        f'<rect x="{cursor_x}" y="{prompt_y - LINE_HEIGHT + 6}" width="{CHAR_WIDTH:.1f}" '
        f'height="{FONT_SIZE + 2}" fill="{prompt_color}">'
        f'<animate attributeName="x" from="{cursor_x}" to="{cursor_x + full_text_width - CHAR_WIDTH:.1f}" '
        f'begin="0.3s" dur="{TYPE_DURATION}s" fill="freeze" calcMode="linear"/>'
        f'<animate attributeName="opacity" values="1;0;1" dur="0.9s" '
        f'begin="0s" repeatCount="indefinite"/>'
        f'<animate attributeName="opacity" to="0" begin="{first_line_start}s" dur="0.01s" fill="freeze"/>'
        f'</rect>'
    )

    # Output lines fade in one by one, in real captured order
    for idx, line in enumerate(lines):
        y = PAD_TOP + (idx + 1) * LINE_HEIGHT
        begin = first_line_start + idx * LINE_STAGGER
        color = fg
        stripped = line.strip()
        if stripped.startswith("=="):
            color = dim
        elif stripped.startswith("phase"):
            color = accent
        elif "breaker:" in line:
            color = prompt_color
        safe = escape(line) if line.strip() else " "
        svg_parts.append(
            f'<text x="{PAD_X}" y="{y}" fill="{color}" font-size="{FONT_SIZE}" '
            f'xml:space="preserve" opacity="0">{safe}'
            f'<animate attributeName="opacity" from="0" to="1" begin="{begin:.2f}s" '
            f'dur="{FADE_IN}s" fill="freeze"/></text>'
        )

    svg_parts.append("</svg>")
    return "".join(svg_parts)


def main() -> None:
    lines = capture_real_output()
    svg = build_svg(lines)
    OUT.write_text(svg)
    print(f"wrote {OUT} ({len(svg)} bytes, {len(lines)} captured lines)")


if __name__ == "__main__":
    main()
