from __future__ import annotations

import pytest

from emergent_money.config import SimulationConfig
from emergent_money.native_behavior_compare import run_native_behavior_comparison


def test_run_native_behavior_comparison_matches_reference_when_no_experimental_flags() -> None:
    config = SimulationConfig(
        population=8,
        goods=3,
        acquaintances=2,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
        seed=2009,
    )

    summary = run_native_behavior_comparison(
        cycles=2,
        seeds=[2009, 2011],
        config=config,
    )

    assert summary['mismatch_count'] == 0
    assert summary['matched_seed_count'] == 2
    assert summary['mean_final_delta']['production_total'] == 0.0
    assert summary['benchmark']['target_seconds'] > 0.0


def test_run_native_behavior_comparison_reports_behavioral_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    config = SimulationConfig(
        population=8,
        goods=3,
        acquaintances=2,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
        seed=2009,
    )

    calls = {'count': 0}

    def fake_snapshot_deltas(reference_snapshot, target_snapshot):
        calls['count'] += 1
        if calls['count'] == 1:
            return {
                'fulfilled_share': 0.0,
                'utility_proxy_total': 0.0,
                'production_total': 2.5,
                'stock_total': 0.0,
                'proposed_trade_count': 0.0,
                'accepted_trade_count': 0.0,
                'accepted_trade_volume': 0.0,
                'rare_goods_monetary_share': 0.0,
            }
        return {
            'fulfilled_share': 0.0,
            'utility_proxy_total': 0.0,
            'production_total': 0.0,
            'stock_total': 0.0,
            'proposed_trade_count': 0.0,
            'accepted_trade_count': 0.0,
            'accepted_trade_volume': 0.0,
            'rare_goods_monetary_share': 0.0,
        }

    monkeypatch.setattr('emergent_money.native_behavior_compare._snapshot_deltas', fake_snapshot_deltas)

    summary = run_native_behavior_comparison(
        cycles=2,
        seeds=[2009],
        config=config,
    )

    assert summary['mismatch_count'] == 1
    mismatch = summary['mismatch_examples'][0]
    assert mismatch['seed'] == 2009
    assert mismatch['cycle'] == 1
    assert mismatch['deltas']['production_total'] == 2.5


def test_run_native_behavior_comparison_rejects_unknown_tolerance() -> None:
    config = SimulationConfig(population=4, goods=2, acquaintances=1, active_acquaintances=1, demand_candidates=1, supply_candidates=1)

    try:
        run_native_behavior_comparison(cycles=1, seeds=[2009], config=config, tolerances={'unknown_metric': 1.0})
    except ValueError as exc:
        assert 'unknown behavior tolerance metric' in str(exc)
    else:
        raise AssertionError('expected ValueError for unknown tolerance metric')
