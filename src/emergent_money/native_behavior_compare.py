from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import SimulationConfig
from .engine import SimulationEngine


_BEHAVIOR_METRICS = (
    'fulfilled_share',
    'utility_proxy_total',
    'production_total',
    'stock_total',
    'proposed_trade_count',
    'accepted_trade_count',
    'accepted_trade_volume',
    'rare_goods_monetary_share',
)

_DEFAULT_TOLERANCES: dict[str, float] = {
    'fulfilled_share': 0.0,
    'utility_proxy_total': 1.0e-6,
    'production_total': 1.0e-3,
    'stock_total': 1.0e-3,
    'proposed_trade_count': 0.0,
    'accepted_trade_count': 0.0,
    'accepted_trade_volume': 1.0e-4,
    'rare_goods_monetary_share': 1.0e-6,
}


def run_native_behavior_comparison(
    *,
    cycles: int,
    seeds: list[int],
    config: SimulationConfig,
    backend_name: str = 'numpy',
    output_path: str | Path | None = None,
    tolerances: dict[str, float] | None = None,
) -> dict[str, Any]:
    if cycles <= 0:
        raise ValueError('cycles must be positive')
    if not seeds:
        raise ValueError('seeds must not be empty')

    resolved_tolerances = dict(_DEFAULT_TOLERANCES)
    if tolerances is not None:
        for key, value in tolerances.items():
            if key not in resolved_tolerances:
                raise ValueError(f'unknown behavior tolerance metric: {key}')
            if value < 0.0:
                raise ValueError(f'tolerance for {key} must be non-negative')
            resolved_tolerances[key] = float(value)

    started_at = time.perf_counter()
    total_reference_seconds = 0.0
    total_target_seconds = 0.0
    per_seed: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    final_delta_rows: list[dict[str, float]] = []
    peak_abs_delta = {
        metric: {'seed': None, 'cycle': None, 'delta': 0.0}
        for metric in _BEHAVIOR_METRICS
    }

    for seed in seeds:
        reference_engine = SimulationEngine.create(
            config=_reference_compare_config(config, seed=seed),
            backend_name=backend_name,
        )
        target_engine = SimulationEngine.create(
            config=_target_compare_config(config, seed=seed),
            backend_name=backend_name,
        )
        if config.experimental_native_exchange_stage:
            setattr(target_engine, '_allow_rejected_native_exchange_stage', True)

        first_behavioral_mismatch: dict[str, Any] | None = None
        final_deltas = {metric: 0.0 for metric in _BEHAVIOR_METRICS}
        for cycle_index in range(1, cycles + 1):
            reference_started = time.perf_counter()
            reference_snapshot = reference_engine.step()
            total_reference_seconds += time.perf_counter() - reference_started

            target_started = time.perf_counter()
            target_snapshot = target_engine.step()
            total_target_seconds += time.perf_counter() - target_started

            deltas = _snapshot_deltas(reference_snapshot, target_snapshot)
            final_deltas = deltas
            _update_peak_abs_delta(peak_abs_delta, deltas, seed=seed, cycle=cycle_index)
            if first_behavioral_mismatch is None:
                exceeded = {
                    metric: delta
                    for metric, delta in deltas.items()
                    if abs(delta) > resolved_tolerances[metric]
                }
                if exceeded:
                    first_behavioral_mismatch = {
                        'cycle': cycle_index,
                        'deltas': exceeded,
                    }
                    mismatches.append(
                        {
                            'seed': seed,
                            'cycle': cycle_index,
                            'deltas': exceeded,
                        }
                    )

        final_delta_rows.append(final_deltas)
        per_seed.append(
            {
                'seed': seed,
                'cycles_requested': cycles,
                'cycles_completed': cycles,
                'first_behavioral_mismatch': first_behavioral_mismatch,
                'final_deltas': final_deltas,
            }
        )

    speedup = 0.0
    if total_target_seconds > 0.0:
        speedup = total_reference_seconds / total_target_seconds

    summary = {
        'cycles': cycles,
        'seeds': seeds,
        'config': asdict(config),
        'behavior_tolerances': resolved_tolerances,
        'mismatch_count': len(mismatches),
        'matched_seed_count': sum(1 for item in per_seed if item['first_behavioral_mismatch'] is None),
        'per_seed': per_seed,
        'mismatch_examples': mismatches[:10],
        'mean_final_delta': _mean_delta_rows(final_delta_rows),
        'peak_abs_delta': peak_abs_delta,
        'benchmark': {
            'reference_seconds': total_reference_seconds,
            'target_seconds': total_target_seconds,
            'speedup_vs_reference': speedup,
        },
        'comparison_runtime_seconds': time.perf_counter() - started_at,
    }
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding='utf-8')
    return summary


def _snapshot_deltas(reference_snapshot, target_snapshot) -> dict[str, float]:
    return {
        'fulfilled_share': float(reference_snapshot.fulfilled_share) - float(target_snapshot.fulfilled_share),
        'utility_proxy_total': float(reference_snapshot.utility_proxy_total) - float(target_snapshot.utility_proxy_total),
        'production_total': float(reference_snapshot.production_total) - float(target_snapshot.production_total),
        'stock_total': float(reference_snapshot.stock_total) - float(target_snapshot.stock_total),
        'proposed_trade_count': float(reference_snapshot.proposed_trade_count) - float(target_snapshot.proposed_trade_count),
        'accepted_trade_count': float(reference_snapshot.accepted_trade_count) - float(target_snapshot.accepted_trade_count),
        'accepted_trade_volume': float(reference_snapshot.accepted_trade_volume) - float(target_snapshot.accepted_trade_volume),
        'rare_goods_monetary_share': float(reference_snapshot.rare_goods_monetary_share) - float(target_snapshot.rare_goods_monetary_share),
    }


def _mean_delta_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {metric: 0.0 for metric in _BEHAVIOR_METRICS}
    totals = {metric: 0.0 for metric in _BEHAVIOR_METRICS}
    for row in rows:
        for metric in _BEHAVIOR_METRICS:
            totals[metric] += float(row[metric])
    count = float(len(rows))
    return {metric: totals[metric] / count for metric in _BEHAVIOR_METRICS}


def _update_peak_abs_delta(
    peak_abs_delta: dict[str, dict[str, Any]],
    deltas: dict[str, float],
    *,
    seed: int,
    cycle: int,
) -> None:
    for metric, delta in deltas.items():
        current = peak_abs_delta[metric]
        if abs(delta) > abs(float(current['delta'])):
            current['seed'] = seed
            current['cycle'] = cycle
            current['delta'] = float(delta)


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
    return SimulationConfig(**payload)
