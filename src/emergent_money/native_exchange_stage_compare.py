from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from . import legacy_cycle_native
from .config import SimulationConfig
from .engine import SimulationEngine
from .legacy_cycle import LegacyCycleRunner, _CONSUMPTION_DEAL, _SURPLUS_DEAL
from .native_cycle_compare import _compare_value_maps
from .native_post_period_compare import _clone_engine, _force_python_legacy_orchestration


def run_native_exchange_stage_comparison(
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
    if not _native_exchange_stage_available(config, backend_name=backend_name):
        raise RuntimeError(
            'native exchange-stage helper is not available; build the optional Rust module '
            'before running compare-native-exchange-stage'
        )

    started_at = time.perf_counter()
    total_reference_seconds = 0.0
    total_target_seconds = 0.0
    per_seed: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []

    with _force_python_legacy_orchestration(), _capture_native_exchange_stage_comparison():
        for seed in seeds:
            engine = SimulationEngine.create(
                config=_target_compare_config(config, seed=seed),
                backend_name=backend_name,
            )
            compare_log: list[dict[str, Any]] = []
            benchmark = {'reference_seconds': 0.0, 'target_seconds': 0.0}
            setattr(engine, '_native_exchange_stage_compare_log', compare_log)
            setattr(engine, '_native_exchange_stage_compare_benchmark', benchmark)
            setattr(engine, '_native_exchange_stage_compare_limit', mismatch_example_limit)

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

            delattr(engine, '_native_exchange_stage_compare_log')
            delattr(engine, '_native_exchange_stage_compare_benchmark')
            delattr(engine, '_native_exchange_stage_compare_limit')

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


def _native_exchange_stage_available(config: SimulationConfig, *, backend_name: str) -> bool:
    engine = SimulationEngine.create(config=config, backend_name=backend_name)
    runner = LegacyCycleRunner(engine)
    backend = runner._native_cycle
    return bool(backend is not None and backend.supports_run_exchange_stage)


@contextmanager
def _capture_native_exchange_stage_comparison():
    original_consumption = LegacyCycleRunner._satisfy_needs_by_exchange
    original_surplus = LegacyCycleRunner._make_surplus_deals

    def wrap(stage_name: str, deal_type: int, original):
        def wrapped(self: LegacyCycleRunner, agent_id: int) -> None:
            compare_log = getattr(self.engine, '_native_exchange_stage_compare_log', None)
            benchmark = getattr(self.engine, '_native_exchange_stage_compare_benchmark', None)
            mismatch_example_limit = int(getattr(self.engine, '_native_exchange_stage_compare_limit', 10))
            if compare_log is None or benchmark is None:
                original(self, agent_id)
                return

            candidate_engine = _clone_engine(self.engine)
            candidate_runner = LegacyCycleRunner(candidate_engine)

            target_started = time.perf_counter()
            _run_native_exchange_stage_candidate(candidate_runner, agent_id, deal_type=deal_type)
            benchmark['target_seconds'] += time.perf_counter() - target_started

            reference_started = time.perf_counter()
            original(self, agent_id)
            benchmark['reference_seconds'] += time.perf_counter() - reference_started

            mismatches = _compare_value_maps(
                _flatten_exchange_stage_state(self.engine),
                _flatten_exchange_stage_state(candidate_engine),
                mismatch_example_limit=mismatch_example_limit,
            )
            compare_log.append(
                {
                    'cycle': int(self.engine.cycle) + 1,
                    'agent_id': int(agent_id),
                    'stage': stage_name,
                    'field_mismatch_count': len(mismatches),
                    'field_mismatch_examples': mismatches[:mismatch_example_limit],
                }
            )

        return wrapped

    LegacyCycleRunner._satisfy_needs_by_exchange = wrap('consumption', _CONSUMPTION_DEAL, original_consumption)
    LegacyCycleRunner._make_surplus_deals = wrap('surplus', _SURPLUS_DEAL, original_surplus)
    try:
        yield
    finally:
        LegacyCycleRunner._satisfy_needs_by_exchange = original_consumption
        LegacyCycleRunner._make_surplus_deals = original_surplus


def _run_native_exchange_stage_candidate(runner: LegacyCycleRunner, agent_id: int, *, deal_type: int) -> None:
    backend = runner._native_cycle
    if backend is None or not backend.supports_run_exchange_stage:
        raise RuntimeError('native exchange-stage helper is not available in this build')
    proposed_count, accepted_count, accepted_volume, inventory_trade_volume = backend.run_exchange_stage(
        agent_id=agent_id,
        deal_type=deal_type,
    )
    runner.engine._proposed_trade_count += proposed_count
    runner.engine._accepted_trade_count += accepted_count
    runner.engine._accepted_trade_volume += accepted_volume
    runner.engine._inventory_trade_volume += inventory_trade_volume


def _flatten_exchange_stage_state(engine: SimulationEngine) -> dict[str, Any]:
    state = engine.state
    market = state.market
    trade = state.trade
    return {
        'engine._proposed_trade_count': int(engine._proposed_trade_count),
        'engine._accepted_trade_count': int(engine._accepted_trade_count),
        'engine._accepted_trade_volume': float(engine._accepted_trade_volume),
        'engine._inventory_trade_volume': float(engine._inventory_trade_volume),
        'state.need': np.array(state.need, copy=True, order='C'),
        'state.stock': np.array(state.stock, copy=True, order='C'),
        'state.recent_sales': np.array(state.recent_sales, copy=True, order='C'),
        'state.recent_purchases': np.array(state.recent_purchases, copy=True, order='C'),
        'state.recent_inventory_inflow': np.array(state.recent_inventory_inflow, copy=True, order='C'),
        'state.sold_this_period': np.array(state.sold_this_period, copy=True, order='C'),
        'state.purchased_this_period': np.array(state.purchased_this_period, copy=True, order='C'),
        'state.purchase_times': np.array(state.purchase_times, copy=True, order='C'),
        'state.sales_times': np.array(state.sales_times, copy=True, order='C'),
        'state.sum_period_purchase_value': np.array(state.sum_period_purchase_value, copy=True, order='C'),
        'state.sum_period_sales_value': np.array(state.sum_period_sales_value, copy=True, order='C'),
        'state.friend_id': np.array(state.friend_id, copy=True, order='C'),
        'state.friend_activity': np.array(state.friend_activity, copy=True, order='C'),
        'state.friend_purchased': np.array(state.friend_purchased, copy=True, order='C'),
        'state.friend_sold': np.array(state.friend_sold, copy=True, order='C'),
        'state.transparency': np.array(state.transparency, copy=True, order='C'),
        'state.trade.proposal_friend_slot': np.array(trade.proposal_friend_slot, copy=True, order='C'),
        'state.trade.proposal_target_agent': np.array(trade.proposal_target_agent, copy=True, order='C'),
        'state.trade.proposal_need_good': np.array(trade.proposal_need_good, copy=True, order='C'),
        'state.trade.proposal_offer_good': np.array(trade.proposal_offer_good, copy=True, order='C'),
        'state.trade.proposal_quantity': np.array(trade.proposal_quantity, copy=True, order='C'),
        'state.trade.proposal_score': np.array(trade.proposal_score, copy=True, order='C'),
        'state.trade.accepted_mask': np.array(trade.accepted_mask, copy=True, order='C'),
        'state.trade.accepted_quantity': np.array(trade.accepted_quantity, copy=True, order='C'),
        'state.market.periodic_tce_cost': np.array(market.periodic_tce_cost, copy=True, order='C'),
        'state.market.total_cost_of_tce_in_time': float(market.total_cost_of_tce_in_time),
    }


def _target_compare_config(config: SimulationConfig, *, seed: int) -> SimulationConfig:
    payload = asdict(config)
    payload['seed'] = seed
    return SimulationConfig(**payload)
