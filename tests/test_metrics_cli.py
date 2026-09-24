"""CLI test for --metrics-json. Runs offline against the mock provider."""

from __future__ import annotations

import json

from llm_gateway.cli import main


def test_metrics_json_written(tmp_path):
    input_path = tmp_path / "prompts.jsonl"
    input_path.write_text(
        "\n".join(json.dumps({"id": f"p{i}", "prompt": f"hi {i}"}) for i in range(4))
    )
    metrics_path = tmp_path / "metrics.json"

    code = main(
        [
            "run",
            str(input_path),
            "--provider",
            "mock",
            "--output",
            str(tmp_path / "out.jsonl"),
            "--metrics-json",
            str(metrics_path),
        ]
    )

    assert code == 0
    assert metrics_path.exists()
    data = json.loads(metrics_path.read_text())
    assert data["completed"] == 4
    assert data["providers"]["mock"]["succeeded"] == 4
