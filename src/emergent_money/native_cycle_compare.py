from __future__ import annotations

import json
import time
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import legacy_cycle_native
from .config import SimulationConfig
from .engine import SimulationEngine


def run_native_cycle_comparison(
    *,
    cycles: int,
    seeds: list[int],
    config: SimulationConfig,
    backend_name: str = 'numpy',
    output_path: str | Path | None = None,
    mismatch_example_limit: int = 10,
) -> dict[str, Any]:
    if cycles <= 0:
        raise ValueError('cycles must be positive')
    if not seeds:
        raise ValueError('seeds must not be empty')
    if mismatch_example_limit <= 0:
        raise ValueError('mismatch_example_limit must be positive')
    if not _native_cycle_entrypoint_available():
        raise RuntimeError(
            'native exact-cycle entrypoint is not available; build the optional Rust module '
            'before running compare-native-cycle'
        )

    started_at = time.perf_counter()
    total_python_seconds = 0.0
    total_native_seconds = 0.0
    per_seed: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []

    for seed in seeds:
        python_engine = SimulationEngine.create(
            config=_reference_compare_config(config, seed=seed),
            backend_name=backend_name,
        )
        native_engine = SimulationEngine.create(
            config=_target_compare_config(config, seed=seed),
            backend_name=backend_name,
        )
        if config.experimental_native_exchange_stage:
            setattr(native_engine, '_allow_rejected_native_exchange_stage', True)

        first_mismatch: dict[str, Any] | None = None
        cycles_completed = 0
        for cycle_index in range(1, cycles + 1):
            python_started = time.perf_counter()
            python_snapshot = _step_python_cycle(python_engine)
            total_python_seconds += time.perf_counter() - python_started

            native_started = time.perf_counter()
            native_snapshot = _step_native_cycle(native_engine)
            total_native_seconds += time.perf_counter() - native_started

            mismatch = _compare_cycle_state(
                seed=seed,
                cycle=cycle_index,
                python_engine=python_engine,
                native_engine=native_engine,
                python_snapshot=python_snapshot,
                native_snapshot=native_snapshot,
                mismatch_example_limit=mismatch_example_limit,
            )
            cycles_completed = cycle_index
            if mismatch is not None:
                mismatches.append(mismatch)
                first_mismatch = mismatch
                break

        per_seed.append(
            {
                'seed': seed,
                'cycles_requested': cycles,
                'cycles_completed': cycles_completed,
                'matched_all_cycles': first_mismatch is None,
                'first_mismatch': first_mismatch,
            }
        )

    speedup = 0.0
    if total_native_seconds > 0.0:
        speedup = total_python_seconds / total_native_seconds

    summary = {
        'cycles': cycles,
        'seeds': seeds,
        'config': asdict(config),
        'mismatch_count': len(mismatches),
        'matched_seed_count': sum(1 for item in per_seed if item['matched_all_cycles']),
        'per_seed': per_seed,
        'mismatch_examples': mismatches[:mismatch_example_limit],
        'benchmark': {
            'python_seconds': total_python_seconds,
            'native_seconds': total_native_seconds,
            'speedup_vs_python': speedup,
        },
        'comparison_runtime_seconds': time.perf_counter() - started_at,
    }
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding='utf-8')
    return summary


def _native_cycle_entrypoint_available() -> bool:
    return legacy_cycle_native.native_exact_cycle_available()


def _step_python_cycle(engine: SimulationEngine):
    return engine.step()


def _step_native_cycle(engine: SimulationEngine):
    if not legacy_cycle_native.run_native_legacy_cycle(engine):
        raise RuntimeError('native exact-cycle entrypoint was not available at execution time')
    engine.backend.synchronize()
    engine.cycle += 1
    snapshot = engine.snapshot_metrics()
    if engine.config.use_exact_legacy_mechanics:
        engine.state.market.total_stock_previous = snapshot.stock_total
    engine.history.append(snapshot)
    return snapshot


def _compare_cycle_state(
    *,
    seed: int,
    cycle: int,
    python_engine: SimulationEngine,
    native_engine: SimulationEngine,
    python_snapshot,
    native_snapshot,
    mismatch_example_limit: int,
) -> dict[str, Any] | None:
    snapshot_deltas = _snapshot_deltas(python_snapshot, native_snapshot)
    engine_mismatches = _compare_value_maps(
        _flatten_engine_state(python_engine),
        _flatten_engine_state(native_engine),
        mismatch_example_limit=mismatch_example_limit,
    )
    if not engine_mismatches and all(delta == 0.0 for delta in snapshot_deltas.values()):
        return None
    return {
        'seed': seed,
        'cycle': cycle,
        'snapshot_deltas': snapshot_deltas,
        'field_mismatch_count': len(engine_mismatches),
        'field_mismatch_examples': engine_mismatches[:mismatch_example_limit],
    }


def _flatten_engine_state(engine: SimulationEngine) -> dict[str, Any]:
    values: dict[str, Any] = {
        'engine.cycle': int(engine.cycle),
        'engine._cycle_need_total': float(engine._cycle_need_total),
        'engine._proposed_trade_count': int(engine._proposed_trade_count),
        'engine._accepted_trade_count': int(engine._accepted_trade_count),
        'engine._accepted_trade_volume': float(engine._accepted_trade_volume),
        'engine._production_total': float(engine._production_total),
        'engine._surplus_output_total': float(engine._surplus_output_total),
        'engine._stock_consumption_total': float(engine._stock_consumption_total),
        'engine._leisure_extra_need_total': float(engine._leisure_extra_need_total),
        'engine._inventory_trade_volume': float(engine._inventory_trade_volume),
        'engine.exact_cycle_diagnostics': engine.exact_cycle_diagnostics,
    }
    _flatten_dataclass('state', engine.state, values)
    return values


def _flatten_dataclass(prefix: str, value: Any, output: dict[str, Any]) -> None:
    if not is_dataclass(value):
        output[prefix] = value
        return
    for field in fields(value):
        child = getattr(value, field.name)
        child_name = f'{prefix}.{field.name}'
        if is_dataclass(child):
            _flatten_dataclass(child_name, child, output)
        elif isinstance(child, np.ndarray):
            output[child_name] = np.array(child, copy=True, order='C')
        elif isinstance(child, np.generic):
            output[child_name] = child.item()
        else:
            output[child_name] = child


def _compare_value_maps(
    python_values: dict[str, Any],
    native_values: dict[str, Any],
    *,
    mismatch_example_limit: int,
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    for name in sorted(python_values):
        python_value = python_values[name]
        native_value = native_values[name]
        if isinstance(python_value, np.ndarray):
            if np.array_equal(python_value, native_value):
                continue
            mismatches.append(_array_mismatch_payload(name, python_value, native_value))
        elif python_value != native_value:
            mismatches.append(
                {
                    'field': name,
                    'python': python_value,
                    'native': native_value,
                }
            )
        if len(mismatches) >= mismatch_example_limit:
            break
    return mismatches


def _array_mismatch_payload(name: str, python_value: np.ndarray, native_value: np.ndarray) -> dict[str, Any]:
    payload: dict[str, Any] = {
        'field': name,
        'shape': list(python_value.shape),
        'dtype': str(python_value.dtype),
        'mismatch_entries': int(np.count_nonzero(python_value != native_value)),
    }
    if np.issubdtype(python_value.dtype, np.number):
        delta = np.abs(python_value.astype(np.float64) - native_value.astype(np.float64))
        max_index = int(np.argmax(delta))
        payload['max_abs_diff'] = float(delta.reshape(-1)[max_index])
        payload['max_abs_diff_index'] = [int(item) for item in np.unravel_index(max_index, python_value.shape)]
        python_flat = python_value.reshape(-1)
        native_flat = native_value.reshape(-1)
        payload['python_at_max_abs_diff'] = float(python_flat[max_index])
        payload['native_at_max_abs_diff'] = float(native_flat[max_index])
    return payload


def _snapshot_deltas(python_snapshot, native_snapshot) -> dict[str, float]:
    return {
        'fulfilled_share': float(python_snapshot.fulfilled_share) - float(native_snapshot.fulfilled_share),
        'utility_proxy_total': float(python_snapshot.utility_proxy_total) - float(native_snapshot.utility_proxy_total),
        'production_total': float(python_snapshot.production_total) - float(native_snapshot.production_total),
        'stock_total': float(python_snapshot.stock_total) - float(native_snapshot.stock_total),
        'proposed_trade_count': float(python_snapshot.proposed_trade_count) - float(native_snapshot.proposed_trade_count),
        'accepted_trade_count': float(python_snapshot.accepted_trade_count) - float(native_snapshot.accepted_trade_count),
        'accepted_trade_volume': float(python_snapshot.accepted_trade_volume) - float(native_snapshot.accepted_trade_volume),
        'rare_goods_monetary_share': float(python_snapshot.rare_goods_monetary_share) - float(native_snapshot.rare_goods_monetary_share),
    }


def _reference_compare_config(config: SimulationConfig, *, seed: int) -> SimulationConfig:
    payload = asdict(config)
    payload['seed'] = seed
    payload['experimental_hybrid_batches'] = 0
    payload['experimental_hybrid_frontier_size'] = 0
    payload['experimental_hybrid_consumption_stage'] = False
    payload['experimental_hybrid_surplus_stage'] = False
    payload['experimental_hybrid_block_frontier_partners'] = True
    payload['experimental_hybrid_preserve_proposer_order'] = False
    payload['experimental_hybrid_rolling_frontier'] = False
    payload['experimental_native_stage_math'] = False
    payload['experimental_native_exchange_stage'] = False
    return SimulationConfig(**payload)


def _target_compare_config(config: SimulationConfig, *, seed: int) -> SimulationConfig:
    payload = asdict(config)
    payload['seed'] = seed
    payload['experimental_hybrid_batches'] = 0
    payload['experimental_hybrid_frontier_size'] = 0
    payload['experimental_hybrid_consumption_stage'] = False
    payload['experimental_hybrid_surplus_stage'] = False
    return SimulationConfig(**payload)
