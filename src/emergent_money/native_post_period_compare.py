from __future__ import annotations

import copy
import json
import time
from contextlib import contextmanager
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import SimulationConfig
from .engine import SimulationEngine
from .legacy_cycle import LegacyCycleRunner
from . import legacy_cycle_native
from .native_cycle_compare import _compare_value_maps


def run_native_post_period_comparison(
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
    if not _native_post_period_available(config, backend_name=backend_name):
        raise RuntimeError(
            'native post-period helpers are not available; build the optional Rust module '
            'before running compare-native-post-period'
        )

    started_at = time.perf_counter()
    total_reference_seconds = 0.0
    total_target_seconds = 0.0
    per_seed: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []

    with _force_python_legacy_orchestration(), _capture_native_post_period_comparison():
        for seed in seeds:
            engine = SimulationEngine.create(
                config=_target_compare_config(config, seed=seed),
                backend_name=backend_name,
            )
            compare_log: list[dict[str, Any]] = []
            benchmark = {'reference_seconds': 0.0, 'target_seconds': 0.0}
            setattr(engine, '_native_post_period_compare_log', compare_log)
            setattr(engine, '_native_post_period_compare_benchmark', benchmark)
            setattr(engine, '_native_post_period_compare_limit', mismatch_example_limit)

            first_mismatch: dict[str, Any] | None = None
            cycles_completed = 0
            cursor = 0
            for cycle_index in range(1, cycles + 1):
                engine.step()
                cycles_completed = cycle_index
                total_reference_seconds += float(benchmark['reference_seconds'])
                total_target_seconds += float(benchmark['target_seconds'])
                benchmark['reference_seconds'] = 0.0
                benchmark['target_seconds'] = 0.0

                new_entries = compare_log[cursor:]
                cursor = len(compare_log)
                mismatch_entry = next((entry for entry in new_entries if entry['field_mismatch_count'] > 0), None)
                if mismatch_entry is not None:
                    mismatch_with_seed = dict(mismatch_entry)
                    mismatch_with_seed['seed'] = seed
                    mismatches.append(mismatch_with_seed)
                    first_mismatch = mismatch_with_seed
                    break

            delattr(engine, '_native_post_period_compare_log')
            delattr(engine, '_native_post_period_compare_benchmark')
            delattr(engine, '_native_post_period_compare_limit')

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


def _native_post_period_available(config: SimulationConfig, *, backend_name: str) -> bool:
    engine = SimulationEngine.create(config=config, backend_name=backend_name)
    runner = LegacyCycleRunner(engine)
    backend = runner._native_cycle
    return bool(
        backend is not None
        and backend.supports_leisure_production
        and backend.supports_end_agent_period
    )


@contextmanager
def _force_python_legacy_orchestration():
    original = legacy_cycle_native.can_use_native_legacy_cycle
    legacy_cycle_native.can_use_native_legacy_cycle = lambda engine: False
    try:
        yield
    finally:
        legacy_cycle_native.can_use_native_legacy_cycle = original


@contextmanager
def _capture_native_post_period_comparison():
    original = LegacyCycleRunner._complete_agent_period_after_surplus

    def wrapped(self: LegacyCycleRunner, agent_id: int) -> None:
        compare_log = getattr(self.engine, '_native_post_period_compare_log', None)
        benchmark = getattr(self.engine, '_native_post_period_compare_benchmark', None)
        mismatch_example_limit = int(getattr(self.engine, '_native_post_period_compare_limit', 10))
        if compare_log is None or benchmark is None:
            original(self, agent_id)
            return

        candidate_engine = _clone_engine(self.engine)
        candidate_runner = LegacyCycleRunner(candidate_engine)

        target_started = time.perf_counter()
        _run_native_post_period_candidate(candidate_runner, agent_id)
        benchmark['target_seconds'] += time.perf_counter() - target_started

        reference_started = time.perf_counter()
        original(self, agent_id)
        benchmark['reference_seconds'] += time.perf_counter() - reference_started

        mismatches = _compare_value_maps(
            _flatten_post_period_state(self.engine),
            _flatten_post_period_state(candidate_engine),
            mismatch_example_limit=mismatch_example_limit,
        )
        compare_log.append(
            {
                'cycle': int(self.engine.cycle) + 1,
                'agent_id': int(agent_id),
                'field_mismatch_count': len(mismatches),
                'field_mismatch_examples': mismatches[:mismatch_example_limit],
            }
        )

    LegacyCycleRunner._complete_agent_period_after_surplus = wrapped
    try:
        yield
    finally:
        LegacyCycleRunner._complete_agent_period_after_surplus = original


def _clone_engine(engine: SimulationEngine) -> SimulationEngine:
    clone = SimulationEngine.create(config=engine.config, backend_name=engine.backend.metadata.name)
    clone.cycle = int(engine.cycle)
    clone._cycle_need_total = float(engine._cycle_need_total)
    clone._proposed_trade_count = int(engine._proposed_trade_count)
    clone._accepted_trade_count = int(engine._accepted_trade_count)
    clone._accepted_trade_volume = float(engine._accepted_trade_volume)
    clone._production_total = float(engine._production_total)
    clone._surplus_output_total = float(engine._surplus_output_total)
    clone._stock_consumption_total = float(engine._stock_consumption_total)
    clone._leisure_extra_need_total = float(engine._leisure_extra_need_total)
    clone._inventory_trade_volume = float(engine._inventory_trade_volume)
    clone.exact_cycle_diagnostics = copy.deepcopy(engine.exact_cycle_diagnostics)
    _copy_dataclass(engine.state, clone.state)
    return clone


def _copy_dataclass(source: Any, target: Any) -> None:
    if not is_dataclass(source) or not is_dataclass(target):
        raise TypeError('expected dataclass instances when cloning engine state')
    for field in fields(source):
        source_value = getattr(source, field.name)
        target_value = getattr(target, field.name)
        if is_dataclass(source_value):
            _copy_dataclass(source_value, target_value)
            continue
        if isinstance(source_value, np.ndarray):
            target_value[...] = source_value
            continue
        setattr(target, field.name, copy.deepcopy(source_value))


def _run_native_post_period_candidate(runner: LegacyCycleRunner, agent_id: int) -> None:
    backend = runner._native_cycle
    if backend is None or not (backend.supports_leisure_production and backend.supports_end_agent_period):
        raise RuntimeError('native post-period helpers are not available in this build')

    runner.state.period_time_debt[agent_id] += runner.state.time_remaining[agent_id]
    if runner.state.period_time_debt[agent_id] > 0.0:
        runner.state.period_time_debt[agent_id] = 0.0
    produced_total = backend.leisure_production(agent_id=agent_id)
    runner.engine._production_total += produced_total
    runner.engine._surplus_output_total += produced_total
    backend.end_agent_period(cycle=runner.engine.cycle, agent_id=agent_id)


def _flatten_post_period_state(engine: SimulationEngine) -> dict[str, Any]:
    state = engine.state
    market = state.market
    return {
        'engine._production_total': float(engine._production_total),
        'engine._surplus_output_total': float(engine._surplus_output_total),
        'state.time_remaining': np.array(state.time_remaining, copy=True, order='C'),
        'state.period_time_debt': np.array(state.period_time_debt, copy=True, order='C'),
        'state.stock': np.array(state.stock, copy=True, order='C'),
        'state.stock_limit': np.array(state.stock_limit, copy=True, order='C'),
        'state.previous_stock_limit': np.array(state.previous_stock_limit, copy=True, order='C'),
        'state.efficiency': np.array(state.efficiency, copy=True, order='C'),
        'state.learned_efficiency': np.array(state.learned_efficiency, copy=True, order='C'),
        'state.recent_production': np.array(state.recent_production, copy=True, order='C'),
        'state.recent_sales': np.array(state.recent_sales, copy=True, order='C'),
        'state.recent_purchases': np.array(state.recent_purchases, copy=True, order='C'),
        'state.recent_inventory_inflow': np.array(state.recent_inventory_inflow, copy=True, order='C'),
        'state.produced_this_period': np.array(state.produced_this_period, copy=True, order='C'),
        'state.sold_this_period': np.array(state.sold_this_period, copy=True, order='C'),
        'state.purchased_this_period': np.array(state.purchased_this_period, copy=True, order='C'),
        'state.produced_last_period': np.array(state.produced_last_period, copy=True, order='C'),
        'state.sold_last_period': np.array(state.sold_last_period, copy=True, order='C'),
        'state.purchased_last_period': np.array(state.purchased_last_period, copy=True, order='C'),
        'state.purchase_times': np.array(state.purchase_times, copy=True, order='C'),
        'state.sales_times': np.array(state.sales_times, copy=True, order='C'),
        'state.sum_period_purchase_value': np.array(state.sum_period_purchase_value, copy=True, order='C'),
        'state.sum_period_sales_value': np.array(state.sum_period_sales_value, copy=True, order='C'),
        'state.spoilage': np.array(state.spoilage, copy=True, order='C'),
        'state.periodic_spoilage': np.array(state.periodic_spoilage, copy=True, order='C'),
        'state.role': np.array(state.role, copy=True, order='C'),
        'state.purchase_price': np.array(state.purchase_price, copy=True, order='C'),
        'state.sales_price': np.array(state.sales_price, copy=True, order='C'),
        'state.friend_activity': np.array(state.friend_activity, copy=True, order='C'),
        'state.transparency': np.array(state.transparency, copy=True, order='C'),
        'state.market.periodic_spoilage': np.array(market.periodic_spoilage, copy=True, order='C'),
    }


def _target_compare_config(config: SimulationConfig, *, seed: int) -> SimulationConfig:
    payload = asdict(config)
    payload['seed'] = seed
    return SimulationConfig(**payload)
