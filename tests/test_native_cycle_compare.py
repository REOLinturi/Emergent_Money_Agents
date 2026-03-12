from __future__ import annotations

import pytest

from emergent_money.config import SimulationConfig
from emergent_money.native_cycle_compare import run_native_cycle_comparison


def test_run_native_cycle_comparison_matches_python_when_native_step_is_stubbed(monkeypatch: pytest.MonkeyPatch) -> None:
    config = SimulationConfig(
        population=8,
        goods=3,
        acquaintances=2,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
        seed=2009,
    )

    monkeypatch.setattr('emergent_money.native_cycle_compare._native_cycle_entrypoint_available', lambda: True)
    monkeypatch.setattr(
        'emergent_money.native_cycle_compare._step_native_cycle',
        lambda engine: engine.step(),
    )

    summary = run_native_cycle_comparison(
        cycles=3,
        seeds=[2009, 2011],
        config=config,
    )

    assert summary['mismatch_count'] == 0
    assert summary['matched_seed_count'] == 2
    assert len(summary['per_seed']) == 2
    assert summary['benchmark']['native_seconds'] > 0.0


def test_run_native_cycle_comparison_reports_first_cycle_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    config = SimulationConfig(
        population=6,
        goods=3,
        acquaintances=2,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
        seed=2009,
    )

    monkeypatch.setattr('emergent_money.native_cycle_compare._native_cycle_entrypoint_available', lambda: True)

    def mismatching_step(engine):
        snapshot = engine.step()
        engine.state.stock[0, 0] += 1.0
        return snapshot

    monkeypatch.setattr('emergent_money.native_cycle_compare._step_native_cycle', mismatching_step)

    summary = run_native_cycle_comparison(
        cycles=2,
        seeds=[2009],
        config=config,
    )

    assert summary['mismatch_count'] == 1
    mismatch = summary['mismatch_examples'][0]
    assert mismatch['seed'] == 2009
    assert mismatch['cycle'] == 1
    assert mismatch['field_mismatch_examples'][0]['field'] == 'state.stock'


def test_run_native_cycle_comparison_requires_native_entrypoint(monkeypatch: pytest.MonkeyPatch) -> None:
    config = SimulationConfig(population=4, goods=2, acquaintances=1, active_acquaintances=1, demand_candidates=1, supply_candidates=1)
    monkeypatch.setattr('emergent_money.native_cycle_compare._native_cycle_entrypoint_available', lambda: False)

    with pytest.raises(RuntimeError, match='native exact-cycle entrypoint is not available'):
        run_native_cycle_comparison(cycles=1, seeds=[2009], config=config)
