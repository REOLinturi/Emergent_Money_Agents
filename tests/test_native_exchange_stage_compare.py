from __future__ import annotations

import pytest

from emergent_money.config import SimulationConfig
from emergent_money.legacy_cycle import LegacyCycleRunner, _CONSUMPTION_DEAL
from emergent_money.native_exchange_stage_compare import run_native_exchange_stage_comparison


def test_run_native_exchange_stage_comparison_requires_native_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    config = SimulationConfig(
        population=8,
        goods=3,
        acquaintances=2,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
        seed=2009,
    )
    monkeypatch.setattr(
        'emergent_money.native_exchange_stage_compare._native_exchange_stage_available',
        lambda config, backend_name: False,
    )

    with pytest.raises(RuntimeError, match='native exchange-stage helper is not available'):
        run_native_exchange_stage_comparison(cycles=1, seeds=[2009], config=config)


def test_run_native_exchange_stage_comparison_matches_reference_when_candidate_uses_python(monkeypatch: pytest.MonkeyPatch) -> None:
    config = SimulationConfig(
        population=8,
        goods=3,
        acquaintances=2,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
        seed=2009,
    )
    monkeypatch.setattr(
        'emergent_money.native_exchange_stage_compare._native_exchange_stage_available',
        lambda config, backend_name: True,
    )
    monkeypatch.setattr(
        'emergent_money.native_exchange_stage_compare._run_native_exchange_stage_candidate',
        lambda runner, agent_id, deal_type: LegacyCycleRunner._satisfy_needs_by_exchange(runner, agent_id)
        if deal_type == _CONSUMPTION_DEAL
        else LegacyCycleRunner._make_surplus_deals(runner, agent_id),
    )

    summary = run_native_exchange_stage_comparison(cycles=2, seeds=[2009], config=config)

    assert summary['mismatch_count'] == 0
    assert summary['matched_seed_count'] == 1


def test_run_native_exchange_stage_comparison_reports_first_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    config = SimulationConfig(
        population=8,
        goods=3,
        acquaintances=2,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
        seed=2009,
    )
    monkeypatch.setattr(
        'emergent_money.native_exchange_stage_compare._native_exchange_stage_available',
        lambda config, backend_name: True,
    )

    def mismatching_candidate(runner, agent_id, deal_type):
        if deal_type == _CONSUMPTION_DEAL:
            LegacyCycleRunner._satisfy_needs_by_exchange(runner, agent_id)
        else:
            LegacyCycleRunner._make_surplus_deals(runner, agent_id)
        runner.state.stock[agent_id, 0] += 1.0

    monkeypatch.setattr(
        'emergent_money.native_exchange_stage_compare._run_native_exchange_stage_candidate',
        mismatching_candidate,
    )

    summary = run_native_exchange_stage_comparison(cycles=1, seeds=[2009], config=config)

    assert summary['mismatch_count'] == 1
    mismatch = summary['mismatch_examples'][0]
    assert mismatch['seed'] == 2009
    assert mismatch['field_mismatch_examples'][0]['field'] == 'state.stock'
