"""Local CSV dataset registry for auditable research inputs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from quant_platform.data.quality import run_market_data_quality_checks
from quant_platform.data.schemas import DatasetMetadata, Frequency, MarketType


class DatasetRegistryError(ValueError):
    """Raised when a dataset cannot be registered or loaded safely."""


@dataclass(frozen=True)
class RegisteredDataset:
    """Manifest entry for one registered dataset version."""

    metadata: DatasetMetadata
    data_path: Path
    manifest_path: Path
    row_count: int
    columns: tuple[str, ...]
    data_format: str = "csv"


def _safe_path_component(value: str, name: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise DatasetRegistryError(f"{name} must be a safe path component.")
    return value


def _dataset_dir(registry_dir: str | Path, dataset_id: str, version: str) -> Path:
    safe_dataset_id = _safe_path_component(dataset_id, "dataset_id")
    safe_version = _safe_path_component(version, "version")
    return Path(registry_dir) / safe_dataset_id / safe_version


def _metadata_to_dict(metadata: DatasetMetadata) -> dict[str, str]:
    return {
        "dataset_id": metadata.dataset_id,
        "version": metadata.version,
        "source": metadata.source,
        "market_type": MarketType(metadata.market_type).value,
        "frequency": Frequency(metadata.frequency).value,
        "created_at": metadata.created_at.isoformat(),
        "schema_version": metadata.schema_version,
        "description": metadata.description or "",
    }


def _metadata_from_dict(values: dict[str, str]) -> DatasetMetadata:
    created_at = pd.Timestamp(values["created_at"])
    if created_at.tzinfo is None:
        created_at = created_at.tz_localize("UTC")
    return DatasetMetadata(
        dataset_id=values["dataset_id"],
        version=values["version"],
        source=values["source"],
        market_type=MarketType(values["market_type"]),
        frequency=Frequency(values["frequency"]),
        created_at=created_at.to_pydatetime(),
        schema_version=values.get("schema_version", "market_bars_v1"),
        description=values.get("description") or None,
    )


def _registered_from_manifest(
    manifest_path: Path,
    manifest: dict[str, object],
) -> RegisteredDataset:
    metadata = _metadata_from_dict(manifest["metadata"])  # type: ignore[arg-type]
    data_path = manifest_path.parent / str(manifest["data_file"])
    return RegisteredDataset(
        metadata=metadata,
        data_path=data_path,
        manifest_path=manifest_path,
        row_count=int(manifest["row_count"]),
        columns=tuple(str(column) for column in manifest["columns"]),  # type: ignore[union-attr]
        data_format=str(manifest.get("data_format", "csv")),
    )


def register_dataset(
    df: pd.DataFrame,
    metadata: DatasetMetadata,
    registry_dir: str | Path,
    overwrite: bool = False,
) -> RegisteredDataset:
    """Validate and register a market-bar dataset as CSV plus JSON manifest.

    Existing dataset versions are immutable by default. Set `overwrite=True` only
    for disposable local fixtures, never for auditable research runs.
    """

    run_market_data_quality_checks(df)
    dataset_dir = _dataset_dir(registry_dir, metadata.dataset_id, metadata.version)
    if dataset_dir.exists() and not overwrite:
        raise DatasetRegistryError(
            f"Dataset version already exists: {metadata.dataset_id}/{metadata.version}"
        )
    dataset_dir.mkdir(parents=True, exist_ok=True)

    data_path = dataset_dir / "data.csv"
    manifest_path = dataset_dir / "manifest.json"
    df.to_csv(data_path, index=False)
    data_sha256 = _file_sha256(data_path)

    manifest: dict[str, object] = {
        "manifest_schema_version": "dataset_manifest_v2",
        "metadata": _metadata_to_dict(metadata),
        "data_file": data_path.name,
        "data_format": "csv",
        "data_sha256": data_sha256,
        "hash_algorithm": "sha256",
        "row_count": int(len(df)),
        "columns": [str(column) for column in df.columns],
        "registered_at": datetime.now().astimezone().isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return _registered_from_manifest(manifest_path, manifest)


def read_dataset_manifest(
    registry_dir: str | Path,
    dataset_id: str,
    version: str,
) -> RegisteredDataset:
    """Read a registered dataset manifest without loading the data file."""

    manifest_path = _dataset_dir(registry_dir, dataset_id, version) / "manifest.json"
    if not manifest_path.exists():
        raise DatasetRegistryError(f"Dataset manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return _registered_from_manifest(manifest_path, manifest)


def load_dataset(
    registry_dir: str | Path,
    dataset_id: str,
    version: str,
) -> tuple[pd.DataFrame, RegisteredDataset]:
    """Load a registered market-bar dataset and re-run quality checks."""

    registered = read_dataset_manifest(registry_dir, dataset_id, version)
    if registered.data_format != "csv":
        raise DatasetRegistryError(f"Unsupported data format: {registered.data_format}")
    if not registered.data_path.exists():
        raise DatasetRegistryError(f"Registered data file not found: {registered.data_path}")
    _verify_registered_hash(registered.manifest_path, registered.data_path)
    df = pd.read_csv(registered.data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="mixed")
    df["available_at"] = pd.to_datetime(df["available_at"], utc=True, format="mixed")
    run_market_data_quality_checks(df)
    return df, registered


def list_dataset_versions(registry_dir: str | Path, dataset_id: str) -> list[str]:
    """List local versions for a registered dataset id."""

    dataset_root = Path(registry_dir) / _safe_path_component(dataset_id, "dataset_id")
    if not dataset_root.exists():
        return []
    return sorted(path.name for path in dataset_root.iterdir() if path.is_dir())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_registered_hash(manifest_path: Path, data_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get("data_sha256")
    if expected is None:
        return
    observed = _file_sha256(data_path)
    if observed != expected:
        raise DatasetRegistryError(
            "Registered data file hash mismatch: data.csv no longer matches manifest.json."
        )
