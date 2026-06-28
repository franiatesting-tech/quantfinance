from __future__ import annotations

import json

from quant_platform import cli


def test_cli_show_settings_outputs_safe_defaults(capsys) -> None:  # noqa: ANN001
    exit_code = cli.main(["show-settings"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["safety"]["allow_live_trading"] is False
    assert "alpha_vantage_api_key" not in str(payload)


def test_cli_list_providers_outputs_sanitized_json(capsys) -> None:  # noqa: ANN001
    exit_code = cli.main(["list-providers"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert "providers" in payload
    assert "secret" not in json.dumps(payload).lower()


def test_cli_validate_providers_without_network(capsys) -> None:  # noqa: ANN001
    exit_code = cli.main(["validate-providers"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["network_smoke"] is False
    assert "smoke_results" not in payload


def test_cli_ui_status_outputs_read_only_summary(capsys, tmp_path) -> None:  # noqa: ANN001
    exit_code = cli.main(
        [
            "ui-status",
            "--registry-dir",
            str(tmp_path / "registry"),
            "--report-dir",
            str(tmp_path / "reports"),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert "ui_available" in payload
    assert "streamlit_installed" in payload
    assert "plotly_installed" in payload
    assert payload["app_path"].endswith("app.py")
    assert payload["read_only"] is True
    assert payload["network_auto_run"] is False
    assert payload["no_secrets_exposed"] is True
    assert payload["dataset_count"] == 0
    assert "alpha_vantage_api_key" not in json.dumps(payload).lower()


def test_cli_launch_ui_dry_run_does_not_start_streamlit(capsys) -> None:  # noqa: ANN001
    exit_code = cli.main(["launch-ui", "--dry-run", "--no-browser", "--port", "8509"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["command"][1:4] == ["-m", "streamlit", "run"]
    assert "8509" in payload["command"]


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
    assert payload["fallback_enabled"] is True


def test_cli_download_real_data_dry_run_accepts_provider_args(capsys) -> None:  # noqa: ANN001
    exit_code = cli.main(
        [
            "download-real-data",
            "--config",
            "configs/universe_etfs_crypto_daily.yaml",
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-31",
            "--equity-provider",
            "polygon",
            "--crypto-provider",
            "cryptocompare",
            "--fallback-provider",
            "yfinance",
            "--no-fallback",
            "--dry-run",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["equity_provider"] == "polygon"
    assert payload["crypto_provider"] == "cryptocompare"
    assert payload["fallback_enabled"] is False


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


def test_cli_build_quant_terminal_report_dry_run(capsys) -> None:  # noqa: ANN001
    exit_code = cli.main(
        [
            "build-quant-terminal-report",
            "--config",
            "configs/quant_terminal_3_stocks.yaml",
            "--dry-run",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["dry_run"] is True
    assert payload["network_auto_run"] is False
    assert payload["research_only"] is True


def test_cli_build_quant_terminal_report_uses_pipeline(monkeypatch, capsys, tmp_path) -> None:  # noqa: ANN001
    def fake_builder(**kwargs):  # noqa: ANN001
        return {
            "report_path": str(tmp_path / "report.json"),
            "frontier_csv_path": str(tmp_path / "frontier.csv"),
            "data_mode": "offline_synthetic"
            if kwargs["offline_synthetic"]
            else "provider_yfinance",
            "symbols": ["AAPL", "MSFT", "NVDA"],
            "warnings": [],
            "research_only": True,
        }

    monkeypatch.setattr(cli, "build_and_write_quant_terminal_report", fake_builder)

    exit_code = cli.main(
        [
            "build-quant-terminal-report",
            "--config",
            "configs/quant_terminal_3_stocks.yaml",
            "--offline-synthetic",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["data_mode"] == "offline_synthetic"
    assert payload["research_only"] is True


def test_cli_generate_stock_academic_report_from_local_json(capsys, tmp_path) -> None:  # noqa: ANN001
    terminal_report = tmp_path / "terminal.json"
    terminal_report.write_text(json.dumps(_sample_academic_terminal_report()), encoding="utf-8")

    exit_code = cli.main(
        [
            "generate-stock-academic-report",
            "--asset",
            "AAPL",
            "--terminal-report",
            str(terminal_report),
            "--output-dir",
            str(tmp_path / "academic"),
            "--format",
            "md",
            "--format",
            "html",
            "--format",
            "pdf",
            "--overwrite",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["asset_id"] == "AAPL"
    assert payload["research_only"] is True
    assert "pdf" in payload["formats"]
    assert payload["pdf_export"] is not None
    assert (tmp_path / "academic" / "AAPL" / "AAPL_academic_report.md").exists()
    assert (tmp_path / "academic" / "AAPL" / "AAPL_academic_report.html").exists()


def test_cli_generate_all_stock_academic_reports_from_local_json(capsys, tmp_path) -> None:  # noqa: ANN001
    terminal_report = tmp_path / "terminal.json"
    terminal_report.write_text(json.dumps(_sample_academic_terminal_report()), encoding="utf-8")

    exit_code = cli.main(
        [
            "generate-all-stock-academic-reports",
            "--terminal-report",
            str(terminal_report),
            "--output-dir",
            str(tmp_path / "academic"),
            "--format",
            "html",
            "--overwrite",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["report_count"] == 1
    assert payload["research_only"] is True
    assert (tmp_path / "academic" / "AAPL" / "AAPL_academic_report.html").exists()


def test_cli_generate_stock_academic_report_missing_base_report_fails(capsys, tmp_path) -> None:  # noqa: ANN001
    exit_code = cli.main(
        [
            "generate-stock-academic-report",
            "--asset",
            "AAPL",
            "--terminal-report",
            str(tmp_path / "missing.json"),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["status"] == "failed"
    assert "Terminal report not found" in payload["error"]


def test_cli_validate_providers_network_smoke_uses_mock(monkeypatch, capsys) -> None:  # noqa: ANN001
    class FakeProvider:
        def download_ohlcv(self, request):  # noqa: ANN001
            return type(
                "Response",
                (),
                {"successful_symbols": (request.symbols[0],), "failed_symbols": {}},
            )()

    monkeypatch.setattr(
        cli,
        "list_available_providers",
        lambda settings: [
            {
                "name": "polygon",
                "enabled": True,
                "configured": True,
                "requires_api_key": True,
                "market_types": ["equity"],
                "status": "available",
            }
        ],
    )
    monkeypatch.setattr(cli, "create_market_data_provider", lambda name, settings: FakeProvider())

    exit_code = cli.main(["validate-providers", "--network-smoke"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["smoke_results"][0]["status"] == "success"


def test_cli_validate_providers_network_smoke_skips_disabled(monkeypatch, capsys) -> None:  # noqa: ANN001
    def fail_create_provider(name, settings):  # noqa: ANN001
        raise AssertionError("disabled provider should not be instantiated")

    monkeypatch.setattr(
        cli,
        "list_available_providers",
        lambda settings: [
            {
                "name": "polygon",
                "enabled": False,
                "configured": True,
                "requires_api_key": True,
                "market_types": ["equity"],
                "status": "configured_but_disabled",
            }
        ],
    )
    monkeypatch.setattr(cli, "create_market_data_provider", fail_create_provider)

    exit_code = cli.main(["validate-providers", "--network-smoke"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["smoke_results"] == [{"provider": "polygon", "status": "skipped_disabled"}]


def _sample_academic_terminal_report() -> dict[str, object]:
    stock = {
        "asset_id": "AAPL",
        "data_used": {
            "provider": "synthetic",
            "data_mode": "offline_synthetic",
            "frequency": "1d",
            "currency": "USD",
            "benchmark": "SPY",
            "risk_free_proxy": "^IRX",
            "risk_free_rate_annual": 0.0,
            "start_timestamp": "2024-01-01",
            "end_timestamp": "2024-01-02",
            "observations": 2,
        },
        "data_quality": {},
        "metrics": {"annualized_return": 0.1, "annualized_volatility": 0.2},
        "var": {},
        "monte_carlo": {},
        "backtesting_results": {},
        "options_theoretical_analytics": {},
    }
    return {
        "metadata": {"report_version": "test"},
        "data_provenance": {},
        "universe": {"selected_stocks": ["AAPL"]},
        "stocks": {"AAPL": stock},
        "bibliography": [],
    }
