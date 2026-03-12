from __future__ import annotations

import pytest

from emergent_money.config import SimulationConfig
from emergent_money.legacy_cycle import LegacyCycleRunner
from emergent_money.native_post_period_compare import run_native_post_period_comparison


def test_run_native_post_period_comparison_matches_reference_when_candidate_uses_python(monkeypatch: pytest.MonkeyPatch) -> None:
    config = SimulationConfig(
        population=8,
        goods=3,
        acquaintances=2,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
        seed=2009,
    )

    monkeypatch.setattr('emergent_money.native_post_period_compare._native_post_period_available', lambda config, backend_name: True)
    monkeypatch.setattr(
        'emergent_money.native_post_period_compare._run_native_post_period_candidate',
        lambda runner, agent_id: LegacyCycleRunner._complete_agent_period_after_surplus(runner, agent_id),
    )

    summary = run_native_post_period_comparison(
        cycles=2,
        seeds=[2009, 2011],
        config=config,
    )

    assert summary['mismatch_count'] == 0
    assert summary['matched_seed_count'] == 2
    assert summary['benchmark']['target_seconds'] > 0.0


def test_run_native_post_period_comparison_reports_first_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    config = SimulationConfig(
        population=8,
        goods=3,
        acquaintances=2,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
        seed=2009,
    )

    monkeypatch.setattr('emergent_money.native_post_period_compare._native_post_period_available', lambda config, backend_name: True)

    def mismatching_candidate(runner, agent_id):
        LegacyCycleRunner._complete_agent_period_after_surplus(runner, agent_id)
        runner.state.stock[agent_id, 0] += 1.0

    monkeypatch.setattr(
        'emergent_money.native_post_period_compare._run_native_post_period_candidate',
        mismatching_candidate,
    )

    summary = run_native_post_period_comparison(
        cycles=1,
        seeds=[2009],
        config=config,
    )

    assert summary['mismatch_count'] == 1
    mismatch = summary['mismatch_examples'][0]
    assert mismatch['seed'] == 2009
    assert mismatch['cycle'] == 1
    assert mismatch['field_mismatch_examples'][0]['field'] == 'state.stock'


def test_run_native_post_period_comparison_requires_native_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    config = SimulationConfig(population=4, goods=2, acquaintances=1, active_acquaintances=1, demand_candidates=1, supply_candidates=1)
    monkeypatch.setattr('emergent_money.native_post_period_compare._native_post_period_available', lambda config, backend_name: False)

    with pytest.raises(RuntimeError, match='native post-period helpers are not available'):
        run_native_post_period_comparison(cycles=1, seeds=[2009], config=config)
