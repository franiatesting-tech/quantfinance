"""Local command line interface for safe research operations."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from dotenv import load_dotenv

from quant_platform.config.settings import load_settings_from_env
from quant_platform.data.ingestion import (
    download_combined_daily_universe,
    load_universe_config,
    register_real_dataset,
)
from quant_platform.pipelines.real_data_backtest import run_real_data_backtest_pipeline


def _settings_json() -> str:
    settings = load_settings_from_env()
    return json.dumps(asdict(settings), indent=2, sort_keys=True)


def build_parser() -> argparse.ArgumentParser:
    """Build the safe local CLI parser."""

    parser = argparse.ArgumentParser(prog="quant-platform")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("show-settings")

    download = subparsers.add_parser("download-real-data")
    download.add_argument("--config", required=True)
    download.add_argument("--start", required=True)
    download.add_argument("--end", required=True)
    download.add_argument("--registry-dir", default="data/registry")
    download.add_argument("--dataset-id", default="real_daily_demo")
    download.add_argument("--version", default="v1")
    download.add_argument("--limit-equity", type=int, default=None)
    download.add_argument("--limit-crypto", type=int, default=None)
    download.add_argument("--dry-run", action="store_true")

    backtest = subparsers.add_parser("run-backtest-demo")
    backtest.add_argument("--dataset-id", required=True)
    backtest.add_argument("--version", required=True)
    backtest.add_argument("--profile", default="conservative")
    backtest.add_argument("--registry-dir", default="data/registry")
    backtest.add_argument("--risk-profiles", default="configs/risk_profiles.yaml")
    backtest.add_argument("--report-dir", default="reports/generated")
    backtest.add_argument("--trial-registry", default="reports/generated/strategy_trials.jsonl")
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
                        "limit_equity": args.limit_equity,
                        "limit_crypto": args.limit_crypto,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        response = download_combined_daily_universe(
            universe,
            start=args.start,
            end=args.end,
            limit_equity=args.limit_equity,
            limit_crypto=args.limit_crypto,
        )
        registered = register_real_dataset(
            response,
            args.registry_dir,
            args.dataset_id,
            args.version,
        )
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
    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
