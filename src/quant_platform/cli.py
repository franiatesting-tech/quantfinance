"""Local command line interface for safe research operations."""

from __future__ import annotations

import argparse
import json

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


def _settings_json() -> str:
    settings = load_settings_from_env()
    return json.dumps(public_settings_dict(settings), indent=2, sort_keys=True)


def build_parser() -> argparse.ArgumentParser:
    """Build the safe local CLI parser."""

    parser = argparse.ArgumentParser(prog="quant-platform")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("show-settings")
    subparsers.add_parser("list-providers")

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
        if bool(provider["requires_api_key"]) and not bool(provider["configured"]):
            smoke_results.append({"provider": name, "status": "skipped_missing_api_key"})
            continue
        if name == "nasdaq_data_link":
            smoke_results.append({"provider": name, "status": "skipped_dataset_code_required"})
            continue
        if name not in {"alpha_vantage", "polygon", "cryptocompare"}:
            continue
        try:
            market_type = (
                MarketType.CRYPTO if name == "cryptocompare" else MarketType.EQUITY
            )
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


def _sanitize_error(exc: Exception) -> str:
    message = str(exc)
    for marker in ("apikey", "api_key", "apiKey", "token", "key"):
        message = message.replace(marker, "credential_param")
    return message


if __name__ == "__main__":
    raise SystemExit(main())
