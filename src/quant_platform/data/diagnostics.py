"""Diagnostics for read-only OHLCV coverage and data quality reporting."""

from __future__ import annotations

from typing import Any

import pandas as pd

from quant_platform.data.schemas import MarketBarSchema, MarketType


def _first_non_null(values: pd.Series) -> object:
    clean = values.dropna().astype(str).unique().tolist()
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    return "MIXED"


def _required_missing_count(frame: pd.DataFrame) -> int:
    missing_columns = [column for column in MarketBarSchema.required_columns if column not in frame]
    missing_values = 0
    for column in MarketBarSchema.required_columns:
        if column in frame:
            missing_values += int(frame[column].isna().sum())
    return missing_values + len(missing_columns) * len(frame)


def _duplicate_mask(frame: pd.DataFrame) -> pd.Series:
    key_columns = [column for column in MarketBarSchema.duplicate_key_columns(tuple(frame.columns))]
    key_columns = [column for column in key_columns if column in frame.columns]
    if not key_columns:
        return pd.Series(False, index=frame.index)
    return frame.duplicated(subset=key_columns, keep=False)


def summarize_ohlcv_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize OHLCV coverage and basic quality counters by asset.

    This diagnostic is intentionally tolerant: it reports violations instead of
    raising so partially bad provider payloads can be audited safely.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    columns = [
        "asset_id",
        "market_type",
        "source",
        "venue",
        "currency",
        "start_timestamp",
        "end_timestamp",
        "row_count",
        "missing_required_values",
        "duplicate_rows",
        "min_close",
        "max_close",
        "zero_volume_rows",
        "available_at_violations",
    ]
    if df.empty or "asset_id" not in df.columns:
        return pd.DataFrame(columns=columns)

    frame = df.copy()
    frame["timestamp"] = pd.to_datetime(
        frame.get("timestamp"),
        utc=True,
        errors="coerce",
        format="mixed",
    )
    frame["available_at"] = pd.to_datetime(
        frame.get("available_at"),
        utc=True,
        errors="coerce",
        format="mixed",
    )
    close = pd.to_numeric(frame.get("close"), errors="coerce")
    volume = pd.to_numeric(frame.get("volume"), errors="coerce")
    duplicate_mask = _duplicate_mask(frame)

    rows: list[dict[str, Any]] = []
    for asset_id, asset in frame.groupby("asset_id", dropna=False, sort=True):
        asset_index = asset.index
        timestamp = frame.loc[asset_index, "timestamp"]
        available_at = frame.loc[asset_index, "available_at"]
        available_violations = available_at.isna() | timestamp.isna() | (available_at < timestamp)
        rows.append(
            {
                "asset_id": str(asset_id),
                "market_type": _first_non_null(asset.get("market_type", pd.Series(dtype=object))),
                "source": _first_non_null(asset.get("source", pd.Series(dtype=object))),
                "venue": _first_non_null(asset.get("venue", pd.Series(dtype=object))),
                "currency": _first_non_null(asset.get("currency", pd.Series(dtype=object))),
                "start_timestamp": timestamp.min(),
                "end_timestamp": timestamp.max(),
                "row_count": int(len(asset)),
                "missing_required_values": _required_missing_count(asset),
                "duplicate_rows": int(duplicate_mask.loc[asset_index].sum()),
                "min_close": _finite_or_none(close.loc[asset_index].min()),
                "max_close": _finite_or_none(close.loc[asset_index].max()),
                "zero_volume_rows": int((volume.loc[asset_index] == 0).sum()),
                "available_at_violations": int(available_violations.sum()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _finite_or_none(value: object) -> float | None:
    try:
        clean = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(clean):
        return None
    return clean


def detect_date_gaps(df: pd.DataFrame, expected_frequency: str = "1d") -> pd.DataFrame:
    """Detect per-asset date gaps for daily read-only research datasets.

    Crypto spot trades 24/7, so daily gaps are failed checks. Equity gaps are
    warnings because exchange holidays and provider-specific calendars are not
    fully modeled in this iteration.
    """

    if expected_frequency != "1d":
        raise ValueError("Only expected_frequency='1d' is supported.")
    columns = [
        "asset_id",
        "market_type",
        "source",
        "venue",
        "gap_start",
        "gap_end",
        "missing_periods",
        "severity",
        "message",
    ]
    if df.empty or "asset_id" not in df.columns or "timestamp" not in df.columns:
        return pd.DataFrame(columns=columns)

    frame = df.copy()
    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"],
        utc=True,
        errors="coerce",
        format="mixed",
    )
    rows: list[dict[str, Any]] = []
    for asset_id, asset in frame.dropna(subset=["timestamp"]).groupby("asset_id", sort=True):
        dates = pd.Series(asset["timestamp"].dt.normalize().drop_duplicates().sort_values())
        if len(dates) < 2:
            continue
        market_type = str(_first_non_null(asset.get("market_type", pd.Series(dtype=object))))
        severity = "failed_check" if market_type == MarketType.CRYPTO.value else "warning"
        for previous, current in zip(dates.iloc[:-1], dates.iloc[1:], strict=False):
            gap_days = int((current - previous).days)
            if gap_days <= 1:
                continue
            missing_periods = gap_days - 1
            rows.append(
                {
                    "asset_id": str(asset_id),
                    "market_type": market_type,
                    "source": _first_non_null(asset.get("source", pd.Series(dtype=object))),
                    "venue": _first_non_null(asset.get("venue", pd.Series(dtype=object))),
                    "gap_start": previous,
                    "gap_end": current,
                    "missing_periods": int(missing_periods),
                    "severity": severity,
                    "message": (
                        "Crypto daily data is expected to be 24/7."
                        if severity == "failed_check"
                        else "Equity gap requires calendar-aware review."
                    ),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def build_data_quality_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Build a serializable quality summary for registry and reports."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    coverage = summarize_ohlcv_coverage(df)
    gaps = detect_date_gaps(df)
    failed_checks: list[str] = []
    warnings: list[str] = []
    if df.empty:
        failed_checks.append("empty_dataset")
    if not coverage.empty:
        if int(coverage["duplicate_rows"].sum()) > 0:
            failed_checks.append("duplicate_rows")
        if int(coverage["missing_required_values"].sum()) > 0:
            failed_checks.append("missing_required_values")
        if int(coverage["available_at_violations"].sum()) > 0:
            failed_checks.append("available_at_violations")
        if int(coverage["zero_volume_rows"].sum()) > 0:
            warnings.append("zero_volume_rows")
    if not gaps.empty:
        if (gaps["severity"] == "failed_check").any():
            failed_checks.append("crypto_date_gaps")
        if (gaps["severity"] == "warning").any():
            warnings.append("equity_date_gaps_require_calendar_review")

    return {
        "total_rows": int(len(df)),
        "assets": _unique_strings(df, "asset_id"),
        "sources": _unique_strings(df, "source"),
        "venues": _unique_strings(df, "venue"),
        "failed_checks": sorted(set(failed_checks)),
        "warnings": sorted(set(warnings)),
        "coverage_table": _serializable_records(coverage),
        "date_gap_table": _serializable_records(gaps),
    }


def _unique_strings(df: pd.DataFrame, column: str) -> list[str]:
    if column not in df.columns:
        return []
    values = df[column].dropna().astype(str).unique().tolist()
    return sorted(values)


def _serializable_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        records.append({key: _json_value(value) for key, value in row.items()})
    return records


def _json_value(value: object) -> object:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value
