from __future__ import annotations

import json
import time
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .analytics import analyze_history, compute_good_snapshots
from .config import SimulationConfig
from .dto import MarketSnapshot
from .engine import SimulationEngine
from .metrics import MetricsSnapshot

_CHECKPOINT_STEM = "checkpoint_latest"
_CHECKPOINT_VERSION = 1


def save_checkpoint(engine: SimulationEngine, destination_dir: str | Path) -> Path:
    destination = Path(destination_dir)
    destination.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    scalars: dict[str, bool | int | float] = {}
    _flatten_dataclass("state", engine.state, engine.backend, arrays, scalars)

    metadata = {
        "version": _CHECKPOINT_VERSION,
        "backend_name": engine.backend.metadata.name,
        "cycle": engine.cycle,
        "config": asdict(engine.config),
        "history": [asdict(item) for item in engine.history],
        "scalars": scalars,
    }

    metadata_path = destination / f"{_CHECKPOINT_STEM}.json"
    arrays_path = destination / f"{_CHECKPOINT_STEM}.npz"
    _atomic_write_npz(arrays_path, arrays)
    _atomic_write_json(metadata_path, metadata)
    return metadata_path


def load_checkpoint(source: str | Path) -> SimulationEngine:
    metadata_path, arrays_path = _resolve_checkpoint_files(source)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if int(metadata.get("version", 0)) != _CHECKPOINT_VERSION:
        raise ValueError("Unsupported checkpoint version")

    config = SimulationConfig(**metadata["config"])
    backend_name = str(metadata["backend_name"])
    engine = SimulationEngine.create(config=config, backend_name=backend_name)

    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    _restore_dataclass("state", engine.state, engine.backend, arrays, metadata["scalars"])

    engine.cycle = int(metadata["cycle"])
    engine.history = [MetricsSnapshot(**item) for item in metadata.get("history", [])]
    return engine


def run_long_simulation(
    *,
    cycles: int,
    checkpoint_dir: str | Path,
    config: SimulationConfig | None = None,
    backend_name: str = "numpy",
    checkpoint_every: int = 50,
    sample_every: int = 10,
    resume_from: str | Path | None = None,
    top_goods: int = 8,
) -> dict[str, Any]:
    if cycles <= 0:
        raise ValueError("cycles must be positive")
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")
    if sample_every <= 0:
        raise ValueError("sample_every must be positive")

    artifact_dir = Path(checkpoint_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = artifact_dir / "metrics.jsonl"
    summary_path = artifact_dir / "summary.json"

    if resume_from is not None:
        engine = load_checkpoint(resume_from)
    else:
        engine = SimulationEngine.create(config=config, backend_name=backend_name)

    start_cycle = engine.cycle
    metrics_mode = "a" if start_cycle > 0 and metrics_path.exists() else "w"
    started_at = time.perf_counter()

    with metrics_path.open(metrics_mode, encoding="utf-8") as metrics_file:
        for offset in range(1, cycles + 1):
            metrics = engine.step()
            if offset % sample_every == 0 or offset == cycles:
                metrics_file.write(json.dumps(asdict(metrics), sort_keys=True) + "\n")
                metrics_file.flush()
            if offset % checkpoint_every == 0 or offset == cycles:
                save_checkpoint(engine, artifact_dir)

    runtime_seconds = time.perf_counter() - started_at
    latest_metrics = engine.history[-1] if engine.history else engine.snapshot_metrics()
    latest_market = MarketSnapshot.from_metrics(latest_metrics)
    goods = compute_good_snapshots(state=engine.state, backend=engine.backend, limit=top_goods)
    phenomena = analyze_history(engine.history, goods)

    summary = {
        "start_cycle": start_cycle,
        "cycles_executed": cycles,
        "end_cycle": engine.cycle,
        "runtime_seconds": runtime_seconds,
        "backend_name": engine.backend.metadata.name,
        "device": engine.backend.metadata.device,
        "config": asdict(engine.config),
        "latest_market": asdict(latest_market),
        "phenomena": asdict(phenomena),
        "top_goods": [asdict(item) for item in goods],
        "artifacts": {
            "checkpoint_json": str((artifact_dir / f"{_CHECKPOINT_STEM}.json").resolve()),
            "checkpoint_npz": str((artifact_dir / f"{_CHECKPOINT_STEM}.npz").resolve()),
            "metrics_jsonl": str(metrics_path.resolve()),
            "summary_json": str(summary_path.resolve()),
        },
    }
    _atomic_write_json(summary_path, summary)
    return summary


def _flatten_dataclass(
    prefix: str,
    value: Any,
    backend,
    arrays: dict[str, np.ndarray],
    scalars: dict[str, bool | int | float],
) -> None:
    if is_dataclass(value):
        for field in fields(value):
            _flatten_dataclass(f"{prefix}__{field.name}", getattr(value, field.name), backend, arrays, scalars)
        return

    if isinstance(value, (bool, int, float, np.bool_, np.integer, np.floating)):
        scalar = value.item() if hasattr(value, "item") else value
        scalars[prefix] = scalar
        return

    arrays[prefix] = np.asarray(backend.to_numpy(value))


def _restore_dataclass(prefix: str, target: Any, backend, arrays: dict[str, np.ndarray], scalars: dict[str, Any]) -> None:
    for field in fields(target):
        field_prefix = f"{prefix}__{field.name}"
        current_value = getattr(target, field.name)
        if is_dataclass(current_value):
            _restore_dataclass(field_prefix, current_value, backend, arrays, scalars)
            continue
        if field_prefix in scalars:
            setattr(target, field.name, scalars[field_prefix])
            continue
        restored = arrays[field_prefix]
        setattr(target, field.name, backend.asarray(restored, dtype=restored.dtype))


def _resolve_checkpoint_files(source: str | Path) -> tuple[Path, Path]:
    source_path = Path(source)
    if source_path.is_dir() or source_path.suffix == "":
        base = source_path / _CHECKPOINT_STEM
    elif source_path.suffix in {".json", ".npz"}:
        base = source_path.with_suffix("")
    else:
        raise ValueError("resume_from must point to a checkpoint directory, .json, or .npz file")

    metadata_path = base.with_suffix(".json")
    arrays_path = base.with_suffix(".npz")
    if not metadata_path.exists() or not arrays_path.exists():
        raise FileNotFoundError(f"Checkpoint files not found for base path {base}")
    return metadata_path, arrays_path


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temp_path.replace(path)


def _atomic_write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temp_path.replace(path)
