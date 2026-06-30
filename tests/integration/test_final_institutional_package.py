from __future__ import annotations

import json

from quant_platform import cli


def test_build_final_institutional_package_orchestrates_without_network(
    monkeypatch, capsys, tmp_path
) -> None:  # noqa: ANN001
    monkeypatch.setattr(cli, "load_quant_terminal_config", lambda path: {"config": path})
    monkeypatch.setattr(
        cli,
        "build_quant_terminal_report",
        lambda **kwargs: {
            "report_type": "professional_quant_terminal",
            "data": {"mode": "provider_yfinance"},
            "warnings": [],
        },
    )

    def fake_write_report(report, output_dir):  # noqa: ANN001
        path = output_dir / "terminal_report.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        return path

    def fake_frontier(report, output_dir):  # noqa: ANN001
        path = output_dir / "frontier.csv"
        path.write_text("x,y\n", encoding="utf-8")
        return path

    def fake_institutional(report, output_dir, **kwargs):  # noqa: ANN001
        output_dir.mkdir(parents=True, exist_ok=True)
        study = {
            "report_type": "institutional_state_of_art_quant_study",
            "research_only": True,
            "tables": {},
            "source_terminal_report": {"data_mode": "provider_yfinance", "warnings": []},
        }
        (output_dir / "institutional_quant_study.json").write_text(
            json.dumps(study),
            encoding="utf-8",
        )
        return {"outputs": {"json": str(output_dir / "institutional_quant_study.json")}}

    def fake_final_paper(institutional_study_path, output_dir, **kwargs):  # noqa: ANN001
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "tables").mkdir(exist_ok=True)
        (output_dir / "figures").mkdir(exist_ok=True)
        manifest = output_dir / "reproducibility_manifest.json"
        manifest.write_text("{}", encoding="utf-8")
        return {
            "outputs": {
                "markdown": str(output_dir / "paper.md"),
                "html": str(output_dir / "paper.html"),
            },
            "reproducibility_manifest": str(manifest),
        }

    monkeypatch.setattr(cli, "write_quant_terminal_report", fake_write_report)
    monkeypatch.setattr(cli, "write_frontier_csv", fake_frontier)
    monkeypatch.setattr(cli, "write_institutional_study", fake_institutional)
    monkeypatch.setattr(cli, "generate_final_academic_paper", fake_final_paper)

    exit_code = cli.main(
        [
            "build-final-institutional-package",
            "--config",
            "configs/quant_terminal_10_stocks.yaml",
            "--output-dir",
            str(tmp_path / "package"),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["research_only"] is True
    assert payload["validations"]["no_broker_order_generated"] is True
    assert (tmp_path / "package" / "metadata.json").exists()
    assert (tmp_path / "package" / "reproducibility_manifest.json").exists()
