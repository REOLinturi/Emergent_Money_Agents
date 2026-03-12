from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .config import SimulationConfig
from .engine import SimulationEngine
from .legacy_cycle import LegacyCycleRunner


_ENGINE_FLOAT_FIELDS = (
    'engine._cycle_need_total',
    'engine._production_total',
    'engine._surplus_output_total',
    'engine._stock_consumption_total',
    'engine._leisure_extra_need_total',
    'engine._accepted_trade_volume',
    'engine._inventory_trade_volume',
)

_ENGINE_INT_FIELDS = (
    'engine._proposed_trade_count',
    'engine._accepted_trade_count',
)

_STATE_SCALAR_FIELDS = (
    'state.time_remaining',
    'state.period_failure',
    'state.period_time_debt',
    'state.needs_level',
    'state.recent_needs_increment',
    'state.timeout',
    'state.periodic_spoilage',
)

_STATE_ROW_FIELDS = (
    'state.need',
    'state.stock',
    'state.recent_production',
    'state.produced_this_period',
    'state.friend_id',
    'state.friend_activity',
    'state.sales_price',
    'state.purchase_price',
    'state.stock_limit',
    'state.previous_stock_limit',
    'state.efficiency',
    'state.learned_efficiency',
    'state.recent_sales',
    'state.recent_purchases',
    'state.recent_inventory_inflow',
    'state.sold_this_period',
    'state.purchased_this_period',
    'state.sold_last_period',
    'state.purchased_last_period',
    'state.purchase_times',
    'state.sales_times',
    'state.sum_period_purchase_value',
    'state.sum_period_sales_value',
    'state.spoilage',
    'state.role',
)

_STAGE_SEQUENCE = (
    'prepare_consumption',
    'consumption_exchange',
    'basic_produce_need',
    'post_basic_transition',
    'surplus_production',
    'leisure_round',
    'surplus_exchange',
    'post_period',
)


def run_native_stage_math_trace_comparison(
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
    if not config.experimental_native_stage_math:
        raise ValueError('compare-native-stage-math-trace requires experimental_native_stage_math=True')

    started_at = time.perf_counter()
    total_reference_seconds = 0.0
    total_target_seconds = 0.0
    per_seed: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []

    for seed in seeds:
        reference_engine = SimulationEngine.create(
            config=_reference_compare_config(config, seed=seed),
            backend_name=backend_name,
        )
        target_engine = SimulationEngine.create(
            config=_target_compare_config(config, seed=seed),
            backend_name=backend_name,
        )
        reference_runner = LegacyCycleRunner(reference_engine)
        target_runner = LegacyCycleRunner(target_engine)
        if not _target_runner_has_stage_math(target_runner):
            raise RuntimeError(
                'native stage-math helpers are not available; build the optional Rust module '
                'before running compare-native-stage-math-trace'
            )

        first_mismatch: dict[str, Any] | None = None
        cycles_completed = 0
        for cycle_index in range(1, cycles + 1):
            reference_started = time.perf_counter()
            reference_runner._reset_cycle_state()
            total_reference_seconds += time.perf_counter() - reference_started

            target_started = time.perf_counter()
            target_runner._reset_cycle_state()
            total_target_seconds += time.perf_counter() - target_started

            mismatch = _run_cycle_trace(
                seed=seed,
                cycle=cycle_index,
                reference_engine=reference_engine,
                target_engine=target_engine,
                reference_runner=reference_runner,
                target_runner=target_runner,
                mismatch_example_limit=mismatch_example_limit,
            )
            if mismatch is not None:
                mismatches.append(mismatch)
                first_mismatch = mismatch
                break

            reference_started = time.perf_counter()
            reference_runner._finalize_cycle_after_agent_loop()
            reference_engine.backend.synchronize()
            reference_engine.cycle += 1
            reference_snapshot = reference_engine.snapshot_metrics()
            reference_engine.state.market.total_stock_previous = reference_snapshot.stock_total
            reference_engine.history.append(reference_snapshot)
            total_reference_seconds += time.perf_counter() - reference_started

            target_started = time.perf_counter()
            target_runner._finalize_cycle_after_agent_loop()
            target_engine.backend.synchronize()
            target_engine.cycle += 1
            target_snapshot = target_engine.snapshot_metrics()
            target_engine.state.market.total_stock_previous = target_snapshot.stock_total
            target_engine.history.append(target_snapshot)
            total_target_seconds += time.perf_counter() - target_started

            cycles_completed = cycle_index

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
    if total_target_seconds > 0.0:
        speedup = total_reference_seconds / total_target_seconds

    summary = {
        'cycles': cycles,
        'seeds': seeds,
        'config': asdict(config),
        'stages': list(_STAGE_SEQUENCE),
        'mismatch_count': len(mismatches),
        'matched_seed_count': sum(1 for item in per_seed if item['matched_all_cycles']),
        'per_seed': per_seed,
        'mismatch_examples': mismatches[:mismatch_example_limit],
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


def _target_runner_has_stage_math(runner: LegacyCycleRunner) -> bool:
    backend = runner._native_cycle
    return bool(
        backend is not None
        and backend.supports_prepare_agent_for_consumption
        and backend.supports_produce_need
        and backend.supports_prepare_leisure_round
    )


def _run_cycle_trace(
    *,
    seed: int,
    cycle: int,
    reference_engine: SimulationEngine,
    target_engine: SimulationEngine,
    reference_runner: LegacyCycleRunner,
    target_runner: LegacyCycleRunner,
    mismatch_example_limit: int,
) -> dict[str, Any] | None:
    for agent_id in range(reference_runner.config.population):
        for stage_name, reference_action, target_action in _agent_stage_actions(reference_runner, target_runner, agent_id):
            reference_started = time.perf_counter()
            reference_action()
            reference_engine.backend.synchronize()
            reference_seconds = time.perf_counter() - reference_started

            target_started = time.perf_counter()
            target_action()
            target_engine.backend.synchronize()
            target_seconds = time.perf_counter() - target_started

            mismatch = _compare_stage_state(
                seed=seed,
                cycle=cycle,
                agent_id=agent_id,
                stage=stage_name,
                reference_runner=reference_runner,
                target_runner=target_runner,
                mismatch_example_limit=mismatch_example_limit,
                reference_seconds=reference_seconds,
                target_seconds=target_seconds,
            )
            if mismatch is not None:
                return mismatch
    return None


def _agent_stage_actions(
    reference_runner: LegacyCycleRunner,
    target_runner: LegacyCycleRunner,
    agent_id: int,
):
    return (
        (
            'prepare_consumption',
            lambda: reference_runner._prepare_agent_for_consumption(agent_id),
            lambda: target_runner._prepare_agent_for_consumption(agent_id),
        ),
        (
            'consumption_exchange',
            lambda: reference_runner._satisfy_needs_by_exchange(agent_id),
            lambda: target_runner._satisfy_needs_by_exchange(agent_id),
        ),
        (
            'basic_produce_need',
            lambda: reference_runner._produce_need(agent_id),
            lambda: target_runner._produce_need(agent_id),
        ),
        (
            'post_basic_transition',
            lambda: _post_basic_transition(reference_runner, agent_id),
            lambda: _post_basic_transition(target_runner, agent_id),
        ),
        (
            'surplus_production',
            lambda: reference_runner._surplus_production(agent_id),
            lambda: target_runner._surplus_production(agent_id),
        ),
        (
            'leisure_round',
            lambda: reference_runner._run_leisure_round(agent_id),
            lambda: target_runner._run_leisure_round(agent_id),
        ),
        (
            'surplus_exchange',
            lambda: reference_runner._make_surplus_deals(agent_id),
            lambda: target_runner._make_surplus_deals(agent_id),
        ),
        (
            'post_period',
            lambda: reference_runner._complete_agent_period_after_surplus(agent_id),
            lambda: target_runner._complete_agent_period_after_surplus(agent_id),
        ),
    )


def _post_basic_transition(runner: LegacyCycleRunner, agent_id: int) -> None:
    runner.state.period_failure[agent_id] = runner.state.time_remaining[agent_id] < 0.0
    runner._add_random_friend(agent_id)
    if runner.state.period_time_debt[agent_id] < 0.0:
        half_debt = runner.state.period_time_debt[agent_id] / 2.0
        runner.state.time_remaining[agent_id] += half_debt
        runner.state.period_time_debt[agent_id] = half_debt


def _compare_stage_state(
    *,
    seed: int,
    cycle: int,
    agent_id: int,
    stage: str,
    reference_runner: LegacyCycleRunner,
    target_runner: LegacyCycleRunner,
    mismatch_example_limit: int,
    reference_seconds: float,
    target_seconds: float,
) -> dict[str, Any] | None:
    mismatches = _collect_stage_mismatches(
        reference_runner=reference_runner,
        target_runner=target_runner,
        agent_id=agent_id,
        mismatch_example_limit=mismatch_example_limit,
    )
    if not mismatches:
        return None
    return {
        'seed': seed,
        'cycle': cycle,
        'agent_id': agent_id,
        'stage': stage,
        'field_mismatch_count': len(mismatches),
        'field_mismatch_examples': mismatches[:mismatch_example_limit],
        'benchmark': {
            'reference_seconds': reference_seconds,
            'target_seconds': target_seconds,
        },
    }


def _collect_stage_mismatches(
    *,
    reference_runner: LegacyCycleRunner,
    target_runner: LegacyCycleRunner,
    agent_id: int,
    mismatch_example_limit: int,
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    reference_engine = reference_runner.engine
    target_engine = target_runner.engine

    for field_name in _ENGINE_INT_FIELDS:
        reference_value = _resolve_attr_path(reference_engine, field_name.removeprefix('engine.'))
        target_value = _resolve_attr_path(target_engine, field_name.removeprefix('engine.'))
        if reference_value != target_value:
            mismatches.append({'field': field_name, 'reference': int(reference_value), 'target': int(target_value)})
            if len(mismatches) >= mismatch_example_limit:
                return mismatches

    for field_name in _ENGINE_FLOAT_FIELDS:
        reference_value = float(_resolve_attr_path(reference_engine, field_name.removeprefix('engine.')))
        target_value = float(_resolve_attr_path(target_engine, field_name.removeprefix('engine.')))
        if reference_value != target_value:
            mismatches.append(
                {
                    'field': field_name,
                    'reference': reference_value,
                    'target': target_value,
                    'delta': reference_value - target_value,
                }
            )
            if len(mismatches) >= mismatch_example_limit:
                return mismatches

    for field_name in _STATE_SCALAR_FIELDS:
        reference_value = _resolve_state_scalar(reference_runner, field_name, agent_id)
        target_value = _resolve_state_scalar(target_runner, field_name, agent_id)
        if reference_value != target_value:
            payload = {'field': f'{field_name}[{agent_id}]', 'reference': reference_value, 'target': target_value}
            if isinstance(reference_value, float) or isinstance(target_value, float):
                payload['delta'] = float(reference_value) - float(target_value)
            mismatches.append(payload)
            if len(mismatches) >= mismatch_example_limit:
                return mismatches

    for field_name in _STATE_ROW_FIELDS:
        reference_row = _resolve_state_row(reference_runner, field_name, agent_id)
        target_row = _resolve_state_row(target_runner, field_name, agent_id)
        if np.array_equal(reference_row, target_row):
            continue
        mismatches.append(_array_mismatch_payload(f'{field_name}[{agent_id}]', reference_row, target_row))
        if len(mismatches) >= mismatch_example_limit:
            return mismatches

    return mismatches


def _resolve_attr_path(root: object, path: str) -> Any:
    current = root
    for part in path.split('.'):
        current = getattr(current, part)
    return current


def _resolve_state_scalar(runner: LegacyCycleRunner, path: str, agent_id: int) -> Any:
    array = _resolve_attr_path(runner, path)
    value = array[agent_id]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _resolve_state_row(runner: LegacyCycleRunner, path: str, agent_id: int) -> np.ndarray:
    array = _resolve_attr_path(runner, path)
    return np.array(array[agent_id], copy=True, order='C')


def _array_mismatch_payload(name: str, reference_value: np.ndarray, target_value: np.ndarray) -> dict[str, Any]:
    payload: dict[str, Any] = {
        'field': name,
        'shape': list(reference_value.shape),
        'dtype': str(reference_value.dtype),
        'mismatch_entries': int(np.count_nonzero(reference_value != target_value)),
    }
    if np.issubdtype(reference_value.dtype, np.number):
        delta = np.abs(reference_value.astype(np.float64) - target_value.astype(np.float64))
        max_index = int(np.argmax(delta))
        payload['max_abs_diff'] = float(delta.reshape(-1)[max_index])
        payload['max_abs_diff_index'] = [int(item) for item in np.unravel_index(max_index, reference_value.shape)]
        reference_flat = reference_value.reshape(-1)
        target_flat = target_value.reshape(-1)
        payload['reference_at_max_abs_diff'] = float(reference_flat[max_index])
        payload['target_at_max_abs_diff'] = float(target_flat[max_index])
    return payload


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
    payload['experimental_hybrid_block_frontier_partners'] = True
    payload['experimental_hybrid_preserve_proposer_order'] = False
    payload['experimental_hybrid_rolling_frontier'] = False
    payload['experimental_native_exchange_stage'] = False
    return SimulationConfig(**payload)
