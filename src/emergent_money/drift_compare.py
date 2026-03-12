from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import SimulationConfig
from .engine import SimulationEngine

_COMPARE_METRICS = (
    'accepted_trade_count',
    'accepted_trade_volume',
    'production_total',
    'utility_proxy_total',
    'rare_goods_monetary_share',
)


def run_hybrid_consumption_comparison(
    *,
    cycles: int,
    seeds: list[int],
    config: SimulationConfig,
    backend_name: str = 'numpy',
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    if cycles <= 0:
        raise ValueError('cycles must be positive')
    if not seeds:
        raise ValueError('seeds must not be empty')
    if config.experimental_hybrid_batches <= 0 or not (
        config.experimental_hybrid_consumption_stage or config.experimental_hybrid_surplus_stage
    ):
        raise ValueError('config must enable an experimental hybrid exchange stage for comparison')

    sequential_rows: list[dict[str, Any]] = []
    hybrid_rows: list[dict[str, Any]] = []
    started_at = time.perf_counter()

    for seed in seeds:
        sequential_config = _comparison_config(config, seed=seed, enable_hybrid=False)
        hybrid_config = _comparison_config(config, seed=seed, enable_hybrid=True)
        sequential_rows.append(_run_variant(cycles=cycles, config=sequential_config, backend_name=backend_name))
        hybrid_rows.append(_run_variant(cycles=cycles, config=hybrid_config, backend_name=backend_name))

    deltas = [_metric_delta(seq_row, hyb_row) for seq_row, hyb_row in zip(sequential_rows, hybrid_rows, strict=True)]
    mean_delta = {
        metric: sum(row[metric] for row in deltas) / len(deltas)
        for metric in _COMPARE_METRICS
    }
    per_cycle_mean_delta = _summarize_per_cycle_deltas(sequential_rows, hybrid_rows, cycles)

    summary = {
        'cycles': cycles,
        'seeds': seeds,
        'runtime_seconds': time.perf_counter() - started_at,
        'config': asdict(config),
        'sequential': sequential_rows,
        'hybrid': hybrid_rows,
        'deltas': deltas,
        'mean_delta': mean_delta,
        'per_cycle_mean_delta': per_cycle_mean_delta,
        'peak_delta_cycles': _peak_delta_cycles(per_cycle_mean_delta),
        'hybrid_wave_diagnostics': _summarize_hybrid_wave_diagnostics(hybrid_rows),
    }
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding='utf-8')
    return summary



def run_hybrid_consumption_frontier_sweep(
    *,
    cycles: int,
    seeds: list[int],
    config: SimulationConfig,
    frontier_sizes: list[int],
    backend_name: str = 'numpy',
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    if not frontier_sizes:
        raise ValueError('frontier_sizes must not be empty')
    normalized_frontiers = tuple(sorted({int(size) for size in frontier_sizes}))
    if normalized_frontiers[0] <= 0:
        raise ValueError('frontier_sizes must all be positive')

    started_at = time.perf_counter()
    comparisons: list[dict[str, Any]] = []
    for frontier_size in normalized_frontiers:
        frontier_config = _frontier_config(config, frontier_size=frontier_size)
        comparison = run_hybrid_consumption_comparison(
            cycles=cycles,
            seeds=seeds,
            config=frontier_config,
            backend_name=backend_name,
        )
        comparison['frontier_size'] = frontier_size
        comparisons.append(comparison)

    summary = {
        'cycles': cycles,
        'seeds': seeds,
        'frontier_sizes': list(normalized_frontiers),
        'runtime_seconds': time.perf_counter() - started_at,
        'config': asdict(config),
        'comparisons': comparisons,
        'best_by_abs_mean_delta': _best_frontiers_by_abs_mean_delta(comparisons),
        'best_by_abs_peak_delta': _best_frontiers_by_abs_peak_delta(comparisons),
        'recommended_frontier': _recommended_frontier(comparisons),
        'recommended_nontrivial_frontier': _recommended_nontrivial_frontier(comparisons),
    }
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding='utf-8')
    return summary


def _comparison_config(config: SimulationConfig, *, seed: int, enable_hybrid: bool) -> SimulationConfig:
    payload = asdict(config)
    payload['seed'] = seed
    if not enable_hybrid:
        payload['experimental_hybrid_batches'] = 0
        payload['experimental_hybrid_frontier_size'] = 0
        payload['experimental_hybrid_consumption_stage'] = False
        payload['experimental_hybrid_surplus_stage'] = False
    return SimulationConfig(**payload)


def _frontier_config(config: SimulationConfig, *, frontier_size: int) -> SimulationConfig:
    payload = asdict(config)
    payload['experimental_hybrid_frontier_size'] = frontier_size
    return SimulationConfig(**payload)


def _run_variant(*, cycles: int, config: SimulationConfig, backend_name: str) -> dict[str, Any]:
    engine = SimulationEngine.create(config=config, backend_name=backend_name)
    latest = None
    cycle_diagnostics: list[dict[str, Any]] = []
    cycle_metrics: list[dict[str, Any]] = []
    for _ in range(cycles):
        latest = engine.step()
        cycle_metrics.append(_snapshot_row(latest))
        if engine.exact_cycle_diagnostics is not None:
            cycle_diagnostics.append(engine.exact_cycle_diagnostics)
    if latest is None:
        raise RuntimeError('comparison variant produced no metrics')
    row = {
        'seed': config.seed,
        'experimental_hybrid_batches': config.experimental_hybrid_batches,
        'experimental_hybrid_frontier_size': config.experimental_hybrid_frontier_size,
        'experimental_hybrid_consumption_stage': config.experimental_hybrid_consumption_stage,
        'experimental_hybrid_surplus_stage': config.experimental_hybrid_surplus_stage,
        'cycle_diagnostics': cycle_diagnostics,
        'cycle_metrics': cycle_metrics,
    }
    for metric in _COMPARE_METRICS:
        row[metric] = getattr(latest, metric)
    return row


def _snapshot_row(snapshot) -> dict[str, Any]:
    row = {'cycle': snapshot.cycle}
    for metric in _COMPARE_METRICS:
        row[metric] = getattr(snapshot, metric)
    return row


def _metric_delta(sequential_row: dict[str, Any], hybrid_row: dict[str, Any]) -> dict[str, Any]:
    delta = {'seed': sequential_row['seed']}
    for metric in _COMPARE_METRICS:
        delta[metric] = hybrid_row[metric] - sequential_row[metric]
    return delta


def _summarize_per_cycle_deltas(
    sequential_rows: list[dict[str, Any]],
    hybrid_rows: list[dict[str, Any]],
    cycles: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cycle_index in range(cycles):
        cycle_row: dict[str, Any] = {'cycle': cycle_index + 1}
        for metric in _COMPARE_METRICS:
            deltas = [
                hybrid_row['cycle_metrics'][cycle_index][metric] - sequential_row['cycle_metrics'][cycle_index][metric]
                for sequential_row, hybrid_row in zip(sequential_rows, hybrid_rows, strict=True)
            ]
            cycle_row[metric] = sum(deltas) / len(deltas)
        rows.append(cycle_row)
    return rows


def _peak_delta_cycles(per_cycle_mean_delta: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    peaks: dict[str, dict[str, float | int]] = {}
    for metric in _COMPARE_METRICS:
        best_row = max(per_cycle_mean_delta, key=lambda row: abs(float(row[metric])))
        peaks[metric] = {
            'cycle': int(best_row['cycle']),
            'delta': float(best_row[metric]),
            'abs_delta': abs(float(best_row[metric])),
        }
    return peaks


def _summarize_hybrid_wave_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cycle_diagnostics = [
        cycle_row
        for row in rows
        for cycle_row in row.get('cycle_diagnostics', ())
    ]
    stage_activation = _summarize_stage_activation(rows)
    wave_count_total = sum(int(item['wave_count']) for item in cycle_diagnostics)
    frontier_count_total = sum(int(item['frontier_count']) for item in cycle_diagnostics)
    candidate_agents_total = sum(int(item['candidate_agents_total']) for item in cycle_diagnostics)
    no_candidate_agents_total = sum(int(item['no_candidate_agents_total']) for item in cycle_diagnostics)
    scheduled_exchanges_total = sum(int(item['scheduled_exchanges_total']) for item in cycle_diagnostics)
    scheduled_quantity_total = sum(float(item['scheduled_quantity_total']) for item in cycle_diagnostics)
    dropped_exchanges_total = sum(int(item['dropped_exchanges_total']) for item in cycle_diagnostics)
    executed_exchanges_total = sum(int(item['executed_exchanges_total']) for item in cycle_diagnostics)
    executed_quantity_total = sum(float(item['executed_quantity_total']) for item in cycle_diagnostics)
    execution_failures_total = sum(int(item['execution_failures_total']) for item in cycle_diagnostics)
    retry_exhausted_agents_total = sum(int(item['retry_exhausted_agents_total']) for item in cycle_diagnostics)
    stalled_waves_total = sum(int(item['stalled_waves_total']) for item in cycle_diagnostics)
    no_candidate_reasons_total: dict[str, int] = {}
    execution_failure_reasons_total: dict[str, int] = {}
    for item in cycle_diagnostics:
        _merge_reason_counts(no_candidate_reasons_total, item.get('no_candidate_reasons_total', {}))
        _merge_reason_counts(execution_failure_reasons_total, item.get('execution_failure_reasons_total', {}))
    return {
        'seeds_with_diagnostics': sum(1 for row in rows if row.get('cycle_diagnostics')),
        'cycles_with_diagnostics': len(cycle_diagnostics),
        'frontier_count_total': frontier_count_total,
        'wave_count_total': wave_count_total,
        'candidate_agents_total': candidate_agents_total,
        'no_candidate_agents_total': no_candidate_agents_total,
        'no_candidate_reasons_total': no_candidate_reasons_total,
        'scheduled_exchanges_total': scheduled_exchanges_total,
        'scheduled_quantity_total': scheduled_quantity_total,
        'dropped_exchanges_total': dropped_exchanges_total,
        'scheduler_conflict_exchanges_total': dropped_exchanges_total,
        'executed_exchanges_total': executed_exchanges_total,
        'executed_quantity_total': executed_quantity_total,
        'execution_failures_total': execution_failures_total,
        'execution_failure_reasons_total': execution_failure_reasons_total,
        'retry_exhausted_agents_total': retry_exhausted_agents_total,
        'stalled_waves_total': stalled_waves_total,
        'stage_activation': stage_activation,
        'mean_cycle': {
            'frontiers': _safe_average(frontier_count_total, len(cycle_diagnostics)),
            'waves': _safe_average(wave_count_total, len(cycle_diagnostics)),
            'candidates': _safe_average(candidate_agents_total, len(cycle_diagnostics)),
            'no_candidates': _safe_average(no_candidate_agents_total, len(cycle_diagnostics)),
            'scheduled': _safe_average(scheduled_exchanges_total, len(cycle_diagnostics)),
            'scheduled_quantity': _safe_average(scheduled_quantity_total, len(cycle_diagnostics)),
            'executed': _safe_average(executed_exchanges_total, len(cycle_diagnostics)),
            'executed_quantity': _safe_average(executed_quantity_total, len(cycle_diagnostics)),
            'mean_scheduled_quantity_per_exchange': _safe_average(scheduled_quantity_total, scheduled_exchanges_total),
            'mean_executed_quantity_per_exchange': _safe_average(executed_quantity_total, executed_exchanges_total),
            'execution_failures': _safe_average(execution_failures_total, len(cycle_diagnostics)),
            'retry_exhausted_agents': _safe_average(retry_exhausted_agents_total, len(cycle_diagnostics)),
        },
        'mean_wave': {
            'candidates': _safe_average(candidate_agents_total, wave_count_total),
            'no_candidates': _safe_average(no_candidate_agents_total, wave_count_total),
            'scheduled': _safe_average(scheduled_exchanges_total, wave_count_total),
            'scheduled_quantity': _safe_average(scheduled_quantity_total, wave_count_total),
            'executed': _safe_average(executed_exchanges_total, wave_count_total),
            'executed_quantity': _safe_average(executed_quantity_total, wave_count_total),
            'mean_scheduled_quantity_per_exchange': _safe_average(scheduled_quantity_total, scheduled_exchanges_total),
            'mean_executed_quantity_per_exchange': _safe_average(executed_quantity_total, executed_exchanges_total),
            'dropped': _safe_average(dropped_exchanges_total, wave_count_total),
            'execution_failures': _safe_average(execution_failures_total, wave_count_total),
            'retry_exhausted_agents': _safe_average(retry_exhausted_agents_total, wave_count_total),
        },
    }


def _merge_reason_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for reason, count in source.items():
        target[reason] = target.get(reason, 0) + int(count)


def _safe_average(total: float, count: int) -> float:
    if count <= 0:
        return 0.0
    return float(total) / float(count)


def _summarize_stage_activation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stage_cycle_counts: dict[str, int] = {}
    stage_wave_counts: dict[str, int] = {}
    stage_seed_counts: dict[str, int] = {}
    first_cycle_by_seed: dict[str, dict[str, int]] = {}

    for row in rows:
        seed_key = str(row['seed'])
        seed_firsts: dict[str, int] = {}
        seed_stages: set[str] = set()
        for cycle_index, cycle_row in enumerate(row.get('cycle_diagnostics', ()), start=1):
            for stage, count in cycle_row.get('stage_wave_counts', {}).items():
                wave_count = int(count)
                if wave_count <= 0:
                    continue
                stage_wave_counts[stage] = stage_wave_counts.get(stage, 0) + wave_count
                stage_cycle_counts[stage] = stage_cycle_counts.get(stage, 0) + 1
                seed_stages.add(stage)
                if stage not in seed_firsts:
                    seed_firsts[stage] = cycle_index
        if seed_firsts:
            first_cycle_by_seed[seed_key] = seed_firsts
        for stage in seed_stages:
            stage_seed_counts[stage] = stage_seed_counts.get(stage, 0) + 1

    first_cycle_overall = {
        stage: min(seed_cycles[stage] for seed_cycles in first_cycle_by_seed.values() if stage in seed_cycles)
        for stage in stage_cycle_counts
    }
    return {
        'stage_cycle_counts': stage_cycle_counts,
        'stage_wave_counts': stage_wave_counts,
        'stage_seed_counts': stage_seed_counts,
        'first_cycle_by_seed': first_cycle_by_seed,
        'first_cycle_overall': first_cycle_overall,
        'surplus_activated': stage_wave_counts.get('surplus', 0) > 0,
    }



def _best_frontiers_by_abs_mean_delta(comparisons: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    best: dict[str, dict[str, float | int]] = {}
    for metric in _COMPARE_METRICS:
        winner = min(comparisons, key=lambda item: (abs(float(item['mean_delta'][metric])), int(item['frontier_size'])))
        best[metric] = {
            'frontier_size': int(winner['frontier_size']),
            'delta': float(winner['mean_delta'][metric]),
            'abs_delta': abs(float(winner['mean_delta'][metric])),
        }
    return best


def _best_frontiers_by_abs_peak_delta(comparisons: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    best: dict[str, dict[str, float | int]] = {}
    for metric in _COMPARE_METRICS:
        winner = min(
            comparisons,
            key=lambda item: (
                abs(float(item['peak_delta_cycles'][metric]['delta'])),
                abs(float(item['mean_delta'][metric])),
                int(item['frontier_size']),
            ),
        )
        best[metric] = {
            'frontier_size': int(winner['frontier_size']),
            'cycle': int(winner['peak_delta_cycles'][metric]['cycle']),
            'delta': float(winner['peak_delta_cycles'][metric]['delta']),
            'abs_delta': abs(float(winner['peak_delta_cycles'][metric]['delta'])),
        }
    return best


def _recommended_frontier(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    winner = min(
        comparisons,
        key=lambda item: (
            abs(float(item['mean_delta']['accepted_trade_count'])),
            abs(float(item['mean_delta']['accepted_trade_volume'])),
            abs(float(item['peak_delta_cycles']['accepted_trade_volume']['delta'])),
            abs(float(item['mean_delta']['production_total'])),
            abs(float(item['peak_delta_cycles']['production_total']['delta'])),
            abs(float(item['mean_delta']['utility_proxy_total'])),
            abs(float(item['mean_delta']['rare_goods_monetary_share'])),
            int(item['frontier_size']),
        ),
    )
    return {
        'frontier_size': int(winner['frontier_size']),
        'mean_delta': dict(winner['mean_delta']),
        'peak_delta_cycles': dict(winner['peak_delta_cycles']),
        'selection_rule': 'minimize trade-count drift, then mean/peak trade-volume drift, then production and utility drift',
    }



def _recommended_nontrivial_frontier(comparisons: list[dict[str, Any]]) -> dict[str, Any] | None:
    nontrivial = [item for item in comparisons if int(item['frontier_size']) > 1]
    if not nontrivial:
        return None
    winner = min(
        nontrivial,
        key=lambda item: (
            abs(float(item['mean_delta']['accepted_trade_volume'])),
            abs(float(item['peak_delta_cycles']['accepted_trade_volume']['delta'])),
            abs(float(item['mean_delta']['accepted_trade_count'])),
            abs(float(item['mean_delta']['production_total'])),
            abs(float(item['peak_delta_cycles']['production_total']['delta'])),
            abs(float(item['mean_delta']['utility_proxy_total'])),
            abs(float(item['mean_delta']['rare_goods_monetary_share'])),
            int(item['frontier_size']),
        ),
    )
    return {
        'frontier_size': int(winner['frontier_size']),
        'mean_delta': dict(winner['mean_delta']),
        'peak_delta_cycles': dict(winner['peak_delta_cycles']),
        'selection_rule': 'exclude exact reference frontier=1 and prioritize mean/peak trade-volume drift before trade-count drift',
    }
