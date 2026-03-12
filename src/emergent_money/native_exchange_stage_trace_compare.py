from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .config import SimulationConfig
from .engine import SimulationEngine
from .legacy_cycle import LegacyCycleRunner

_DEFAULT_FLOAT_TOLERANCES: dict[str, float] = {
    'accepted_volume_delta': 1.0e-5,
    'inventory_trade_volume_delta': 1.0e-5,
    'proposal_quantity': 1.0e-5,
    'proposal_score': 1.0e-6,
    'agent_need_total': 1.0e-5,
    'agent_stock_total': 1.0e-5,
    'agent_recent_sales_total': 1.0e-5,
    'agent_recent_purchases_total': 1.0e-5,
    'agent_purchase_value_total': 1.0e-5,
    'agent_sales_value_total': 1.0e-5,
    'agent_friend_activity_total': 1.0e-5,
    'market_tce_total': 1.0e-5,
    'partner_stock_total': 1.0e-5,
    'partner_recent_sales_total': 1.0e-5,
    'partner_recent_purchases_total': 1.0e-5,
}

_EXACT_FIELDS = (
    'stage',
    'agent_id',
    'proposed_delta',
    'accepted_delta',
    'proposal_friend_slot',
    'proposal_target_agent',
    'proposal_need_good',
    'proposal_offer_good',
    'accepted_mask',
)

_FLOAT_FIELDS = tuple(_DEFAULT_FLOAT_TOLERANCES)


def run_native_exchange_stage_trace_comparison(
    *,
    cycles: int,
    seeds: list[int],
    config: SimulationConfig,
    backend_name: str = 'numpy',
    output_path: str | Path | None = None,
    float_tolerances: dict[str, float] | None = None,
) -> dict[str, Any]:
    if cycles <= 0:
        raise ValueError('cycles must be positive')
    if not seeds:
        raise ValueError('seeds must not be empty')

    resolved_tolerances = dict(_DEFAULT_FLOAT_TOLERANCES)
    if float_tolerances is not None:
        for key, value in float_tolerances.items():
            if key not in resolved_tolerances:
                raise ValueError(f'unknown exchange-trace tolerance field: {key}')
            if value < 0.0:
                raise ValueError(f'tolerance for {key} must be non-negative')
            resolved_tolerances[key] = float(value)

    started_at = time.perf_counter()
    total_reference_seconds = 0.0
    total_target_seconds = 0.0
    mismatches: list[dict[str, Any]] = []
    per_seed: list[dict[str, Any]] = []

    with _capture_exchange_stage_traces():
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

            first_mismatch: dict[str, Any] | None = None
            cycles_completed = 0
            for cycle_index in range(1, cycles + 1):
                reference_trace: list[dict[str, Any]] = []
                target_trace: list[dict[str, Any]] = []
                setattr(reference_engine, '_exchange_stage_trace_log', reference_trace)
                setattr(target_engine, '_exchange_stage_trace_log', target_trace)

                reference_started = time.perf_counter()
                reference_engine.step()
                total_reference_seconds += time.perf_counter() - reference_started

                target_started = time.perf_counter()
                target_engine.step()
                total_target_seconds += time.perf_counter() - target_started

                delattr(reference_engine, '_exchange_stage_trace_log')
                delattr(target_engine, '_exchange_stage_trace_log')

                cycles_completed = cycle_index
                mismatch = _compare_trace_logs(
                    seed=seed,
                    cycle=cycle_index,
                    reference_events=reference_trace,
                    target_events=target_trace,
                    float_tolerances=resolved_tolerances,
                )
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
    if total_target_seconds > 0.0:
        speedup = total_reference_seconds / total_target_seconds

    summary = {
        'cycles': cycles,
        'seeds': seeds,
        'config': asdict(config),
        'float_tolerances': resolved_tolerances,
        'mismatch_count': len(mismatches),
        'matched_seed_count': sum(1 for item in per_seed if item['matched_all_cycles']),
        'per_seed': per_seed,
        'mismatch_examples': mismatches[:10],
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


@contextmanager
def _capture_exchange_stage_traces():
    original_consumption = LegacyCycleRunner._satisfy_needs_by_exchange
    original_surplus = LegacyCycleRunner._make_surplus_deals
    original_leisure_round = LegacyCycleRunner._run_leisure_round

    def _wrap(stage: str, original):
        def wrapped(self: LegacyCycleRunner, agent_id: int) -> None:
            before = _engine_counter_snapshot(self.engine)
            original(self, agent_id)
            trace_log = getattr(self.engine, '_exchange_stage_trace_log', None)
            if trace_log is not None:
                trace_log.append(_stage_event_payload(self, stage, agent_id, before))
        return wrapped

    def _wrap_leisure(original):
        def wrapped(self: LegacyCycleRunner, agent_id: int) -> None:
            previous = getattr(self, '_trace_exchange_context', 'basic')
            self._trace_exchange_context = 'leisure'
            try:
                original(self, agent_id)
            finally:
                self._trace_exchange_context = previous
        return wrapped

    LegacyCycleRunner._satisfy_needs_by_exchange = _wrap('consumption', original_consumption)
    LegacyCycleRunner._make_surplus_deals = _wrap('surplus', original_surplus)
    LegacyCycleRunner._run_leisure_round = _wrap_leisure(original_leisure_round)
    try:
        yield
    finally:
        LegacyCycleRunner._satisfy_needs_by_exchange = original_consumption
        LegacyCycleRunner._make_surplus_deals = original_surplus
        LegacyCycleRunner._run_leisure_round = original_leisure_round


def _engine_counter_snapshot(engine) -> dict[str, float]:
    return {
        'proposed_trade_count': float(engine._proposed_trade_count),
        'accepted_trade_count': float(engine._accepted_trade_count),
        'accepted_trade_volume': float(engine._accepted_trade_volume),
        'inventory_trade_volume': float(engine._inventory_trade_volume),
    }


def _stage_event_payload(
    runner: LegacyCycleRunner,
    stage: str,
    agent_id: int,
    before: dict[str, float],
) -> dict[str, Any]:
    engine = runner.engine
    state = runner.state
    market = runner.market
    trade = state.trade
    partner_id = int(trade.proposal_target_agent[agent_id])
    resolved_stage = stage
    if stage == 'consumption' and getattr(runner, '_trace_exchange_context', 'basic') == 'leisure':
        resolved_stage = 'leisure_consumption'

    payload = {
        'stage': resolved_stage,
        'agent_id': int(agent_id),
        'proposed_delta': int(engine._proposed_trade_count - before['proposed_trade_count']),
        'accepted_delta': int(engine._accepted_trade_count - before['accepted_trade_count']),
        'accepted_volume_delta': float(engine._accepted_trade_volume - before['accepted_trade_volume']),
        'inventory_trade_volume_delta': float(engine._inventory_trade_volume - before['inventory_trade_volume']),
        'proposal_friend_slot': int(trade.proposal_friend_slot[agent_id]),
        'proposal_target_agent': partner_id,
        'proposal_need_good': int(trade.proposal_need_good[agent_id]),
        'proposal_offer_good': int(trade.proposal_offer_good[agent_id]),
        'proposal_quantity': float(trade.proposal_quantity[agent_id]),
        'proposal_score': float(trade.proposal_score[agent_id]),
        'accepted_mask': bool(trade.accepted_mask[agent_id]),
        'agent_need_total': float(np.sum(state.need[agent_id], dtype=np.float64)),
        'agent_stock_total': float(np.sum(state.stock[agent_id], dtype=np.float64)),
        'agent_recent_sales_total': float(np.sum(state.recent_sales[agent_id], dtype=np.float64)),
        'agent_recent_purchases_total': float(np.sum(state.recent_purchases[agent_id], dtype=np.float64)),
        'agent_purchase_value_total': float(np.sum(state.sum_period_purchase_value[agent_id], dtype=np.float64)),
        'agent_sales_value_total': float(np.sum(state.sum_period_sales_value[agent_id], dtype=np.float64)),
        'agent_friend_activity_total': float(np.sum(state.friend_activity[agent_id], dtype=np.float64)),
        'market_tce_total': float(np.sum(market.periodic_tce_cost, dtype=np.float64)),
    }
    if partner_id >= 0:
        payload['partner_stock_total'] = float(np.sum(state.stock[partner_id], dtype=np.float64))
        payload['partner_recent_sales_total'] = float(np.sum(state.recent_sales[partner_id], dtype=np.float64))
        payload['partner_recent_purchases_total'] = float(np.sum(state.recent_purchases[partner_id], dtype=np.float64))
    else:
        payload['partner_stock_total'] = 0.0
        payload['partner_recent_sales_total'] = 0.0
        payload['partner_recent_purchases_total'] = 0.0
    return payload


def _compare_trace_logs(
    *,
    seed: int,
    cycle: int,
    reference_events: list[dict[str, Any]],
    target_events: list[dict[str, Any]],
    float_tolerances: dict[str, float],
) -> dict[str, Any] | None:
    common_length = min(len(reference_events), len(target_events))
    for event_index, (reference_event, target_event) in enumerate(zip(reference_events, target_events)):
        event_mismatch = _compare_trace_event(
            reference_event=reference_event,
            target_event=target_event,
            float_tolerances=float_tolerances,
        )
        if event_mismatch is not None:
            return {
                'seed': seed,
                'cycle': cycle,
                'reason': 'event_mismatch',
                'event_index': event_index,
                'stage': reference_event['stage'],
                'agent_id': reference_event['agent_id'],
                'field_mismatch_examples': event_mismatch,
                'reference_event': reference_event,
                'target_event': target_event,
            }

    if len(reference_events) != len(target_events):
        last_matching_index = common_length - 1
        last_matching_event = reference_events[last_matching_index] if last_matching_index >= 0 else None
        next_reference_event = reference_events[common_length] if len(reference_events) > common_length else None
        next_target_event = target_events[common_length] if len(target_events) > common_length else None
        return {
            'seed': seed,
            'cycle': cycle,
            'reason': 'event_count_mismatch',
            'reference_event_count': len(reference_events),
            'target_event_count': len(target_events),
            'last_matching_event_index': last_matching_index,
            'last_matching_event': last_matching_event,
            'next_reference_event': next_reference_event,
            'next_target_event': next_target_event,
        }

    return None


def _compare_trace_event(
    *,
    reference_event: dict[str, Any],
    target_event: dict[str, Any],
    float_tolerances: dict[str, float],
) -> list[dict[str, Any]] | None:
    mismatches: list[dict[str, Any]] = []
    for field in _EXACT_FIELDS:
        if field == 'stage' and _stages_equivalent(reference_event[field], target_event[field]):
            continue
        if reference_event[field] != target_event[field]:
            mismatches.append({
                'field': field,
                'reference': reference_event[field],
                'target': target_event[field],
            })
    for field in _FLOAT_FIELDS:
        reference_value = float(reference_event[field])
        target_value = float(target_event[field])
        delta = reference_value - target_value
        if abs(delta) > float_tolerances[field]:
            mismatches.append({
                'field': field,
                'reference': reference_value,
                'target': target_value,
                'delta': delta,
                'tolerance': float_tolerances[field],
            })
    if mismatches:
        return mismatches
    return None


def _stages_equivalent(reference_stage: str, target_stage: str) -> bool:
    if reference_stage == target_stage:
        return True
    consumption_family = {'consumption', 'leisure_consumption'}
    return reference_stage in consumption_family and target_stage in consumption_family


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
