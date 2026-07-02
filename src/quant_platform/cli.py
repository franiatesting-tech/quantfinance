"""Local command line interface for safe research operations."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from quant_platform.config.settings import load_settings_from_env, public_settings_dict
from quant_platform.data.ingestion import (
    download_combined_daily_universe,
    load_universe_config,
    register_real_dataset,
)
from quant_platform.data.providers.base import OHLCVRequest
from quant_platform.data.providers.factory import (
    create_market_data_provider,
    list_available_providers,
)
from quant_platform.data.schemas import MarketType
from quant_platform.pipelines.profile_comparison import run_profile_comparison_pipeline
from quant_platform.pipelines.real_data_backtest import run_real_data_backtest_pipeline
from quant_platform.reporting.academic_stock_report import (
    build_stock_academic_report_model,
    write_all_stock_academic_reports,
    write_stock_academic_report,
)
from quant_platform.reporting.latex_report import (
    write_all_stock_latex_reports,
    write_stock_latex_report,
)
from quant_platform.research.asset_dataset import load_quant_terminal_config
from quant_platform.research.final_academic_paper import generate_final_academic_paper
from quant_platform.research.institutional_study import write_institutional_study
from quant_platform.research.report import (
    build_and_write_quant_terminal_report,
    build_quant_terminal_report,
    write_frontier_csv,
    write_quant_terminal_report,
)
from quant_platform.research.seven_algorithms_study import write_seven_algorithm_study
from quant_platform.ui.actions import build_ui_status, launch_ui_command


def _settings_json() -> str:
    settings = load_settings_from_env()
    return json.dumps(public_settings_dict(settings), indent=2, sort_keys=True)


def build_parser() -> argparse.ArgumentParser:
    """Build the safe local CLI parser."""

    parser = argparse.ArgumentParser(prog="quant-platform")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("show-settings")
    subparsers.add_parser("list-providers")

    ui_status = subparsers.add_parser("ui-status")
    ui_status.add_argument("--registry-dir", default="data/registry")
    ui_status.add_argument("--report-dir", default="reports/generated")
    ui_status.add_argument("--universe-config", default="configs/universe_etfs_crypto_daily.yaml")
    ui_status.add_argument("--risk-profiles", default="configs/risk_profiles.yaml")

    launch_ui = subparsers.add_parser("launch-ui")
    launch_ui.add_argument("--host", default="localhost")
    launch_ui.add_argument("--port", type=int, default=8501)
    launch_ui.add_argument("--no-browser", action="store_true")
    launch_ui.add_argument("--dry-run", action="store_true")

    validate = subparsers.add_parser("validate-providers")
    validate.add_argument("--network-smoke", action="store_true")

    download = subparsers.add_parser("download-real-data")
    download.add_argument("--config", required=True)
    download.add_argument("--start", required=True)
    download.add_argument("--end", required=True)
    download.add_argument("--registry-dir", default="data/registry")
    download.add_argument("--dataset-id", default="real_daily_demo")
    download.add_argument("--version", default="v1")
    download.add_argument("--limit-equity", type=int, default=None)
    download.add_argument("--limit-crypto", type=int, default=None)
    download.add_argument("--equity-provider", default=None)
    download.add_argument("--crypto-provider", default=None)
    download.add_argument("--fallback-provider", default=None)
    download.add_argument("--no-fallback", action="store_true")
    download.add_argument("--nasdaq-dataset-code", default=None)
    download.add_argument("--dry-run", action="store_true")

    backtest = subparsers.add_parser("run-backtest-demo")
    backtest.add_argument("--dataset-id", required=True)
    backtest.add_argument("--version", required=True)
    backtest.add_argument("--profile", default="conservative")
    backtest.add_argument("--registry-dir", default="data/registry")
    backtest.add_argument("--risk-profiles", default="configs/risk_profiles.yaml")
    backtest.add_argument("--report-dir", default="reports/generated")
    backtest.add_argument("--trial-registry", default="reports/generated/strategy_trials.jsonl")

    compare = subparsers.add_parser("compare-profiles")
    compare.add_argument("--dataset-id", required=True)
    compare.add_argument("--version", required=True)
    compare.add_argument("--config", required=True)
    compare.add_argument("--risk-config", default="configs/risk_profiles.yaml")
    compare.add_argument("--registry-dir", default="data/registry")
    compare.add_argument("--report-dir", default="reports/generated")
    compare.add_argument("--trial-registry", default="reports/generated/strategy_trials.jsonl")
    compare.add_argument("--dry-run", action="store_true")

    quant_terminal = subparsers.add_parser("build-quant-terminal-report")
    quant_terminal.add_argument("--config", required=True)
    quant_terminal.add_argument("--report-dir", default="reports/generated/quant_terminal")
    quant_terminal.add_argument("--export-dir", default="reports/generated/portfolio_optimization")
    quant_terminal.add_argument("--provider", default="yfinance")
    quant_terminal.add_argument("--offline-synthetic", action="store_true")
    quant_terminal.add_argument("--dry-run", action="store_true")

    institutional = subparsers.add_parser("generate-institutional-study")
    institutional.add_argument(
        "--terminal-report",
        default="reports/generated/quant_terminal/10stocks_10y_report.json",
    )
    institutional.add_argument("--output-dir", default="reports/generated/institutional_study")
    institutional.add_argument("--max-table-rows", type=int, default=20)
    institutional.add_argument("--allow-synthetic", action="store_true")

    build_institutional = subparsers.add_parser("build-institutional-study")
    build_institutional.add_argument("--config", required=True)
    build_institutional.add_argument("--report-dir", default="reports/generated/quant_terminal")
    build_institutional.add_argument(
        "--export-dir", default="reports/generated/portfolio_optimization"
    )
    build_institutional.add_argument(
        "--output-dir", default="reports/generated/institutional_study"
    )
    build_institutional.add_argument("--provider", default="yfinance")
    build_institutional.add_argument("--max-table-rows", type=int, default=20)
    build_institutional.add_argument("--offline-synthetic", action="store_true")
    build_institutional.add_argument("--allow-synthetic", action="store_true")
    build_institutional.add_argument("--dry-run", action="store_true")

    final_paper = subparsers.add_parser("generate-final-academic-paper")
    final_paper.add_argument(
        "--institutional-study",
        default="reports/generated/institutional_study/institutional_quant_study.json",
    )
    final_paper.add_argument("--output-dir", default="reports/generated/final_paper")
    final_paper.add_argument("--include-figures", action="store_true")
    final_paper.add_argument("--overwrite", action="store_true")

    final_package = subparsers.add_parser("build-final-institutional-package")
    final_package.add_argument("--config", required=True)
    final_package.add_argument("--provider", default="yfinance")
    final_package.add_argument("--output-dir", default="reports/generated/final_package")
    final_package.add_argument("--max-table-rows", type=int, default=20)
    final_package.add_argument("--offline-synthetic", action="store_true")
    final_package.add_argument("--allow-synthetic", action="store_true")
    final_package.add_argument("--dry-run", action="store_true")

    seven = subparsers.add_parser("generate-seven-algorithm-study")
    seven.add_argument(
        "--terminal-report",
        default="reports/generated/final_package/terminal_report/10stocks_10y_report.json",
    )
    seven.add_argument("--output-dir", default="reports/generated/seven_algorithms_study")
    seven.add_argument(
        "--bibliography-pack",
        default="../quant_finance_papers_pack",
    )
    seven.add_argument("--max-table-rows", type=int, default=30)
    seven.add_argument("--allow-synthetic", action="store_true")
    seven.add_argument("--overwrite", action="store_true")

    build_seven = subparsers.add_parser("build-seven-algorithm-study")
    build_seven.add_argument("--config", required=True)
    build_seven.add_argument("--provider", default="yfinance")
    build_seven.add_argument("--output-dir", default="reports/generated/seven_algorithms_study")
    build_seven.add_argument(
        "--bibliography-pack",
        default="../quant_finance_papers_pack",
    )
    build_seven.add_argument("--max-table-rows", type=int, default=30)
    build_seven.add_argument("--offline-synthetic", action="store_true")
    build_seven.add_argument("--allow-synthetic", action="store_true")
    build_seven.add_argument("--dry-run", action="store_true")

    stock_report = subparsers.add_parser("generate-stock-academic-report")
    stock_report.add_argument("--asset", required=True)
    stock_report.add_argument(
        "--terminal-report",
        default="reports/generated/quant_terminal/10stocks_10y_report.json",
    )
    stock_report.add_argument("--output-dir", default="reports/generated/academic_stock_reports")
    stock_report.add_argument(
        "--format",
        action="append",
        dest="formats",
        choices=("md", "html", "pdf"),
    )
    stock_report.add_argument("--include-figures", action="store_true")
    stock_report.add_argument("--overwrite", action="store_true")

    all_stock_reports = subparsers.add_parser("generate-all-stock-academic-reports")
    all_stock_reports.add_argument(
        "--terminal-report",
        default="reports/generated/quant_terminal/10stocks_10y_report.json",
    )
    all_stock_reports.add_argument(
        "--output-dir", default="reports/generated/academic_stock_reports"
    )
    all_stock_reports.add_argument(
        "--format",
        action="append",
        dest="formats",
        choices=("md", "html", "pdf"),
    )
    all_stock_reports.add_argument("--include-figures", action="store_true")
    all_stock_reports.add_argument("--overwrite", action="store_true")

    latex_report = subparsers.add_parser("generate-stock-latex-report")
    latex_report.add_argument("--asset", required=True)
    latex_report.add_argument(
        "--terminal-report",
        default="reports/generated/quant_terminal/10stocks_10y_report.json",
    )
    latex_report.add_argument("--output-dir", default="reports/generated/academic_stock_reports")
    latex_report.add_argument("--no-compile", action="store_true")
    latex_report.add_argument("--no-figures", action="store_true")
    latex_report.add_argument("--overwrite", action="store_true")

    latex_all = subparsers.add_parser("generate-all-stock-latex-reports")
    latex_all.add_argument(
        "--terminal-report",
        default="reports/generated/quant_terminal/10stocks_10y_report.json",
    )
    latex_all.add_argument("--output-dir", default="reports/generated/academic_stock_reports")
    latex_all.add_argument("--no-compile", action="store_true")
    latex_all.add_argument("--no-figures", action="store_true")
    latex_all.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run a safe local CLI command."""

    load_dotenv()
    settings = load_settings_from_env()
    if settings.safety.allow_live_trading:
        raise SystemExit("Live trading is out of scope and must remain disabled.")

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "show-settings":
        print(_settings_json())
        return 0
    if args.command == "list-providers":
        print(
            json.dumps(
                {"providers": list_available_providers(settings)},
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "ui-status":
        print(
            json.dumps(
                build_ui_status(
                    settings=settings,
                    registry_dir=args.registry_dir,
                    report_dir=args.report_dir,
                    universe_config_path=args.universe_config,
                    risk_profiles_path=args.risk_profiles,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "launch-ui":
        command = launch_ui_command(
            host=args.host,
            port=args.port,
            show_browser=not args.no_browser,
        )
        if args.dry_run:
            print(json.dumps({"command": command}, indent=2, sort_keys=True))
            return 0
        return subprocess.call(command)
    if args.command == "validate-providers":
        result = _validate_providers(settings, network_smoke=args.network_smoke)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "download-real-data":
        universe = load_universe_config(args.config)
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "dry_run": True,
                        "config": args.config,
                        "start": args.start,
                        "end": args.end,
                        "equity_provider": args.equity_provider,
                        "crypto_provider": args.crypto_provider,
                        "fallback_provider": args.fallback_provider,
                        "fallback_enabled": not args.no_fallback,
                        "limit_equity": args.limit_equity,
                        "limit_crypto": args.limit_crypto,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        try:
            response = download_combined_daily_universe(
                universe,
                start=args.start,
                end=args.end,
                equity_provider_name=args.equity_provider,
                crypto_provider_name=args.crypto_provider,
                fallback_provider_name=args.fallback_provider,
                allow_fallback=not args.no_fallback,
                settings=settings,
                nasdaq_dataset_code=args.nasdaq_dataset_code,
                limit_equity=args.limit_equity,
                limit_crypto=args.limit_crypto,
            )
            registered = register_real_dataset(
                response,
                args.registry_dir,
                args.dataset_id,
                args.version,
            )
        except Exception as exc:  # noqa: BLE001 - CLI reports sanitized provider errors.
            print(
                json.dumps(
                    {"status": "failed", "error": _sanitize_error(exc)},
                    indent=2,
                    sort_keys=True,
                )
            )
            return 1
        print(json.dumps({"registered": str(registered.manifest_path)}, indent=2))
        return 0
    if args.command == "run-backtest-demo":
        summary = run_real_data_backtest_pipeline(
            dataset_id=args.dataset_id,
            version=args.version,
            registry_dir=args.registry_dir,
            risk_profiles_path=args.risk_profiles,
            profile_name=args.profile,
            report_dir=args.report_dir,
            trial_registry_path=args.trial_registry,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "compare-profiles":
        summary = run_profile_comparison_pipeline(
            dataset_id=args.dataset_id,
            version=args.version,
            universe_config=args.config,
            risk_profiles_config=args.risk_config,
            registry_dir=args.registry_dir,
            output_dir=args.report_dir,
            trial_registry_path=args.trial_registry,
            dry_run=args.dry_run,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "build-quant-terminal-report":
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "dry_run": True,
                        "config": args.config,
                        "report_dir": args.report_dir,
                        "export_dir": args.export_dir,
                        "provider": args.provider,
                        "offline_synthetic": args.offline_synthetic,
                        "network_auto_run": False,
                        "research_only": True,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        try:
            summary = build_and_write_quant_terminal_report(
                config_path=args.config,
                settings=settings,
                output_dir=args.report_dir,
                export_dir=args.export_dir,
                provider_name=args.provider,
                offline_synthetic=args.offline_synthetic,
            )
        except Exception as exc:  # noqa: BLE001 - CLI reports sanitized local failures.
            print(
                json.dumps(
                    {"status": "failed", "error": _sanitize_error(exc)},
                    indent=2,
                    sort_keys=True,
                )
            )
            return 1
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "generate-institutional-study":
        try:
            terminal_report = _load_terminal_report(args.terminal_report)
            summary = write_institutional_study(
                terminal_report,
                output_dir=args.output_dir,
                max_table_rows=args.max_table_rows,
                strict_real_data=not args.allow_synthetic,
            )
        except Exception as exc:  # noqa: BLE001 - CLI reports sanitized local failures.
            print(
                json.dumps(
                    {"status": "failed", "error": _sanitize_error(exc)},
                    indent=2,
                    sort_keys=True,
                )
            )
            return 1
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "build-institutional-study":
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "dry_run": True,
                        "config": args.config,
                        "provider": args.provider,
                        "offline_synthetic": args.offline_synthetic,
                        "allow_synthetic": args.allow_synthetic,
                        "strict_real_data": not args.allow_synthetic,
                        "network_auto_run": False,
                        "research_only": True,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        try:
            config = load_quant_terminal_config(args.config)
            terminal_report = build_quant_terminal_report(
                config=config,
                settings=settings,
                provider_name=args.provider,
                offline_synthetic=args.offline_synthetic,
            )
            summary = write_institutional_study(
                terminal_report,
                output_dir=args.output_dir,
                max_table_rows=args.max_table_rows,
                strict_real_data=not args.allow_synthetic,
            )
            report_path = write_quant_terminal_report(terminal_report, args.report_dir)
            frontier_path = write_frontier_csv(terminal_report, args.export_dir)
            summary["terminal_report_path"] = str(report_path)
            summary["frontier_csv_path"] = str(frontier_path) if frontier_path is not None else None
        except Exception as exc:  # noqa: BLE001 - CLI reports sanitized local failures.
            print(
                json.dumps(
                    {"status": "failed", "error": _sanitize_error(exc)},
                    indent=2,
                    sort_keys=True,
                )
            )
            return 1
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "generate-final-academic-paper":
        try:
            summary = generate_final_academic_paper(
                institutional_study_path=args.institutional_study,
                output_dir=args.output_dir,
                include_figures=args.include_figures,
                overwrite=args.overwrite,
            )
        except Exception as exc:  # noqa: BLE001 - CLI reports sanitized local failures.
            print(
                json.dumps(
                    {"status": "failed", "error": _sanitize_error(exc)},
                    indent=2,
                    sort_keys=True,
                )
            )
            return 1
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "build-final-institutional-package":
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "dry_run": True,
                        "config": args.config,
                        "provider": args.provider,
                        "output_dir": args.output_dir,
                        "offline_synthetic": args.offline_synthetic,
                        "strict_real_data": not args.allow_synthetic,
                        "network_auto_run": False,
                        "research_only": True,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        try:
            summary = _build_final_institutional_package(args, settings)
        except Exception as exc:  # noqa: BLE001 - CLI reports sanitized local failures.
            print(
                json.dumps(
                    {"status": "failed", "error": _sanitize_error(exc)},
                    indent=2,
                    sort_keys=True,
                )
            )
            return 1
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "generate-seven-algorithm-study":
        try:
            terminal_report = _load_terminal_report(args.terminal_report)
            summary = write_seven_algorithm_study(
                terminal_report,
                output_dir=args.output_dir,
                bibliography_pack_dir=args.bibliography_pack,
                strict_real_data=not args.allow_synthetic,
                max_table_rows=args.max_table_rows,
                overwrite=args.overwrite,
            )
        except Exception as exc:  # noqa: BLE001 - CLI reports sanitized local failures.
            print(
                json.dumps(
                    {"status": "failed", "error": _sanitize_error(exc)},
                    indent=2,
                    sort_keys=True,
                )
            )
            return 1
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "build-seven-algorithm-study":
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "dry_run": True,
                        "config": args.config,
                        "provider": args.provider,
                        "output_dir": args.output_dir,
                        "bibliography_pack": args.bibliography_pack,
                        "offline_synthetic": args.offline_synthetic,
                        "strict_real_data": not args.allow_synthetic,
                        "network_auto_run": False,
                        "research_only": True,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        try:
            config = load_quant_terminal_config(args.config)
            terminal_report = build_quant_terminal_report(
                config=config,
                settings=settings,
                provider_name=args.provider,
                offline_synthetic=args.offline_synthetic,
            )
            data_mode = str(terminal_report.get("data", {}).get("mode", "UNKNOWN"))
            if not args.allow_synthetic and not data_mode.startswith("provider_"):
                raise ValueError(
                    f"Seven-algorithm study requires provider data, got {data_mode!r}."
                )
            summary = write_seven_algorithm_study(
                terminal_report,
                output_dir=args.output_dir,
                bibliography_pack_dir=args.bibliography_pack,
                strict_real_data=not args.allow_synthetic,
                max_table_rows=args.max_table_rows,
                overwrite=True,
            )
            terminal_dir = Path(args.output_dir) / "terminal_report"
            terminal_dir.mkdir(parents=True, exist_ok=True)
            terminal_report_path = write_quant_terminal_report(terminal_report, terminal_dir)
            summary["terminal_report_path"] = str(terminal_report_path)
        except Exception as exc:  # noqa: BLE001 - CLI reports sanitized local failures.
            print(
                json.dumps(
                    {"status": "failed", "error": _sanitize_error(exc)},
                    indent=2,
                    sort_keys=True,
                )
            )
            return 1
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "generate-stock-academic-report":
        try:
            terminal_report = _load_terminal_report(args.terminal_report)
            model = build_stock_academic_report_model(terminal_report, args.asset)
            summary = write_stock_academic_report(
                model,
                output_dir=args.output_dir,
                formats=_normalized_formats(args.formats),
                include_figures=args.include_figures,
                overwrite=args.overwrite,
            )
        except Exception as exc:  # noqa: BLE001 - CLI reports sanitized local failures.
            print(
                json.dumps(
                    {"status": "failed", "error": _sanitize_error(exc)},
                    indent=2,
                    sort_keys=True,
                )
            )
            return 1
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "generate-all-stock-academic-reports":
        try:
            terminal_report = _load_terminal_report(args.terminal_report)
            summaries = write_all_stock_academic_reports(
                terminal_report,
                output_dir=args.output_dir,
                formats=_normalized_formats(args.formats),
                include_figures=args.include_figures,
                overwrite=args.overwrite,
            )
        except Exception as exc:  # noqa: BLE001 - CLI reports sanitized local failures.
            print(
                json.dumps(
                    {"status": "failed", "error": _sanitize_error(exc)},
                    indent=2,
                    sort_keys=True,
                )
            )
            return 1
        print(
            json.dumps(
                {
                    "report_count": len(summaries),
                    "reports": summaries,
                    "research_only": True,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "generate-stock-latex-report":
        try:
            terminal_report = _load_terminal_report(args.terminal_report)
            model = build_stock_academic_report_model(terminal_report, args.asset)
            summary = write_stock_latex_report(
                model,
                output_dir=args.output_dir,
                overwrite=args.overwrite,
                compile_pdf=not args.no_compile,
                include_figures=not args.no_figures,
            )
        except Exception as exc:
            print(
                json.dumps(
                    {"status": "failed", "error": _sanitize_error(exc)},
                    indent=2,
                    sort_keys=True,
                )
            )
            return 1
        print(json.dumps(summary, indent=2, sort_keys=True, default=str))
        return 0

    if args.command == "generate-all-stock-latex-reports":
        try:
            terminal_report = _load_terminal_report(args.terminal_report)
            summaries = write_all_stock_latex_reports(
                terminal_report,
                output_dir=args.output_dir,
                overwrite=args.overwrite,
                compile_pdf=not args.no_compile,
                include_figures=not args.no_figures,
            )
        except Exception as exc:
            print(
                json.dumps(
                    {"status": "failed", "error": _sanitize_error(exc)},
                    indent=2,
                    sort_keys=True,
                )
            )
            return 1
        print(
            json.dumps(
                {
                    "report_count": len(summaries),
                    "reports": summaries,
                    "research_only": True,
                },
                indent=2,
                sort_keys=True,
                default=str,
            )
        )
        return 0

    parser.error("Unknown command")
    return 2


def _validate_providers(settings, network_smoke: bool) -> dict[str, object]:  # noqa: ANN001
    providers = list_available_providers(settings)
    result: dict[str, object] = {
        "network_smoke": network_smoke,
        "providers": providers,
    }
    if not network_smoke:
        return result

    smoke_results = []
    for provider in providers:
        name = str(provider["name"])
        if not bool(provider["enabled"]):
            smoke_results.append({"provider": name, "status": "skipped_disabled"})
            continue
        if bool(provider["requires_api_key"]) and not bool(provider["configured"]):
            smoke_results.append({"provider": name, "status": "skipped_missing_api_key"})
            continue
        if name == "nasdaq_data_link":
            smoke_results.append({"provider": name, "status": "skipped_dataset_code_required"})
            continue
        if name not in {"alpha_vantage", "polygon", "cryptocompare"}:
            continue
        try:
            market_type = MarketType.CRYPTO if name == "cryptocompare" else MarketType.EQUITY
            symbol = "BTC/USD" if market_type == MarketType.CRYPTO else "SPY"
            provider_instance = create_market_data_provider(name, settings)
            response = provider_instance.download_ohlcv(
                OHLCVRequest(
                    symbols=(symbol,),
                    start="2024-01-01",
                    end="2024-01-05",
                    frequency="1d",
                    market_type=market_type,
                    source=name,
                    currency="USD",
                )
            )
            status = "success" if response.successful_symbols else "provider_error"
            smoke_results.append(
                {
                    "provider": name,
                    "status": status,
                    "successful_symbols": list(response.successful_symbols),
                    "failed_symbols": sorted(response.failed_symbols),
                }
            )
        except Exception as exc:  # noqa: BLE001 - CLI must report sanitized provider failures.
            smoke_results.append(
                {"provider": name, "status": "provider_error", "error": _sanitize_error(exc)}
            )
    result["smoke_results"] = smoke_results
    return result


def _build_final_institutional_package(args, settings) -> dict[str, object]:  # noqa: ANN001
    output_dir = Path(args.output_dir)
    terminal_dir = output_dir / "terminal_report"
    institutional_dir = output_dir / "institutional_study"
    final_paper_dir = output_dir / "final_paper"
    root_tables_dir = output_dir / "tables"
    root_figures_dir = output_dir / "figures"
    for directory in (
        terminal_dir,
        institutional_dir,
        final_paper_dir,
        root_tables_dir,
        root_figures_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    config = load_quant_terminal_config(args.config)
    terminal_report = build_quant_terminal_report(
        config=config,
        settings=settings,
        provider_name=args.provider,
        offline_synthetic=args.offline_synthetic,
    )
    data_mode = str(terminal_report.get("data", {}).get("mode", "UNKNOWN"))
    if not args.allow_synthetic and not data_mode.startswith("provider_"):
        raise ValueError(f"Final package requires provider data, but mode is {data_mode!r}.")

    terminal_report_path = write_quant_terminal_report(terminal_report, terminal_dir)
    frontier_path = write_frontier_csv(terminal_report, root_tables_dir)
    institutional_metadata = write_institutional_study(
        terminal_report,
        output_dir=institutional_dir,
        max_table_rows=args.max_table_rows,
        strict_real_data=not args.allow_synthetic,
    )
    final_paper_metadata = generate_final_academic_paper(
        institutional_study_path=institutional_dir / "institutional_quant_study.json",
        output_dir=final_paper_dir,
        include_figures=True,
        overwrite=True,
    )
    _copy_directory_files(final_paper_dir / "tables", root_tables_dir)
    _copy_directory_files(final_paper_dir / "figures", root_figures_dir)
    final_manifest = Path(str(final_paper_metadata.get("reproducibility_manifest")))
    root_manifest = output_dir / "reproducibility_manifest.json"
    if final_manifest.exists():
        shutil.copy2(final_manifest, root_manifest)

    metadata = {
        "report_type": "final_institutional_quant_package_metadata",
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "research_only": True,
        "not_investment_advice": True,
        "data_mode": data_mode,
        "warnings": terminal_report.get("warnings", []),
        "outputs": {
            "terminal_report": str(terminal_report_path),
            "frontier_csv": str(frontier_path) if frontier_path is not None else None,
            "institutional_study": institutional_metadata.get("outputs", {}),
            "final_paper": final_paper_metadata.get("outputs", {}),
            "root_tables": str(root_tables_dir),
            "root_figures": str(root_figures_dir),
            "reproducibility_manifest": str(root_manifest),
        },
        "validations": {
            "synthetic_rejected_by_default": not args.allow_synthetic,
            "no_broker_order_generated": True,
            "no_secrets_required": True,
        },
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    metadata["metadata_path"] = str(metadata_path)
    return metadata


def _copy_directory_files(source: Path, target: Path) -> None:
    if not source.exists():
        return
    for path in source.iterdir():
        if path.is_file():
            shutil.copy2(path, target / path.name)


def _sanitize_error(exc: Exception) -> str:
    message = str(exc)
    for marker in ("apikey", "api_key", "apiKey", "token", "key"):
        message = message.replace(marker, "credential_param")
    return message


def _load_terminal_report(path: str | Path) -> dict[str, object]:
    report_path = Path(path)
    if not report_path.exists():
        raise FileNotFoundError(f"Terminal report not found: {report_path}")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Terminal report must be a JSON object.")
    return payload


def _normalized_formats(formats: list[str] | None) -> tuple[str, ...]:
    if not formats:
        return ("md", "html")
    return tuple(dict.fromkeys(formats))


if __name__ == "__main__":
    raise SystemExit(main())
