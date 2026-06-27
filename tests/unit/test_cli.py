from __future__ import annotations

import json

from quant_platform import cli


def test_cli_show_settings_outputs_safe_defaults(capsys) -> None:  # noqa: ANN001
    exit_code = cli.main(["show-settings"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["safety"]["allow_live_trading"] is False


def test_cli_download_real_data_dry_run(capsys) -> None:  # noqa: ANN001
    exit_code = cli.main(
        [
            "download-real-data",
            "--config",
            "configs/universe_etfs_crypto_daily.yaml",
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-31",
            "--limit-equity",
            "1",
            "--limit-crypto",
            "1",
            "--dry-run",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["dry_run"] is True
    assert payload["limit_equity"] == 1


def test_cli_run_backtest_demo_uses_pipeline(monkeypatch, capsys) -> None:  # noqa: ANN001
    def fake_pipeline(**kwargs):  # noqa: ANN001
        return {"dataset_id": kwargs["dataset_id"], "final_equity": 10001.0}

    monkeypatch.setattr(cli, "run_real_data_backtest_pipeline", fake_pipeline)

    exit_code = cli.main(
        [
            "run-backtest-demo",
            "--dataset-id",
            "demo",
            "--version",
            "v1",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["dataset_id"] == "demo"
    assert payload["final_equity"] == 10001.0


def test_cli_compare_profiles_dry_run_uses_pipeline(monkeypatch, capsys) -> None:  # noqa: ANN001
    def fake_pipeline(**kwargs):  # noqa: ANN001
        return {"dry_run": kwargs["dry_run"], "dataset_id": kwargs["dataset_id"]}

    monkeypatch.setattr(cli, "run_profile_comparison_pipeline", fake_pipeline)

    exit_code = cli.main(
        [
            "compare-profiles",
            "--dataset-id",
            "demo",
            "--version",
            "v1",
            "--config",
            "configs/universe_etfs_crypto_daily.yaml",
            "--risk-config",
            "configs/risk_profiles.yaml",
            "--dry-run",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["dry_run"] is True
    assert payload["dataset_id"] == "demo"


def test_cli_compare_profiles_missing_dataset_fails_without_network(tmp_path) -> None:  # noqa: ANN001
    try:
        cli.main(
            [
                "compare-profiles",
                "--dataset-id",
                "missing",
                "--version",
                "v1",
                "--config",
                "configs/universe_etfs_crypto_daily.yaml",
                "--risk-config",
                "configs/risk_profiles.yaml",
                "--registry-dir",
                str(tmp_path / "registry"),
            ]
        )
    except ValueError as exc:
        assert "Dataset manifest not found" in str(exc)
    else:  # pragma: no cover - explicit failure path for clarity.
        raise AssertionError("missing dataset should fail before any network call")
