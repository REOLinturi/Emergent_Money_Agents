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
    assert summary['material_mismatch_count'] == 0
    assert summary['matched_seed_count'] == 2
    assert summary['matched_seed_count_tolerant'] == 2
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
    assert summary['material_mismatch_count'] == 1
    assert summary['matched_seed_count_tolerant'] == 0
    mismatch = summary['mismatch_examples'][0]
    assert mismatch['seed'] == 2009
    assert mismatch['cycle'] == 1
    assert mismatch['field_mismatch_examples'][0]['field'] == 'state.stock'


def test_run_native_cycle_comparison_classifies_inventory_volume_roundoff_as_tolerated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    def roundoff_step(engine):
        snapshot = engine.step()
        engine._inventory_trade_volume += 1.0e-12
        return snapshot

    monkeypatch.setattr('emergent_money.native_cycle_compare._step_native_cycle', roundoff_step)

    summary = run_native_cycle_comparison(
        cycles=2,
        seeds=[2009],
        config=config,
    )

    assert summary['mismatch_count'] == 1
    assert summary['material_mismatch_count'] == 0
    assert summary['matched_seed_count'] == 0
    assert summary['matched_seed_count_tolerant'] == 1
    assert summary['tolerated_mismatch_count'] >= 1
    mismatch = summary['mismatch_examples'][0]
    assert mismatch['tolerated_boundary_mismatch'] is True
    assert mismatch['field_mismatch_examples'][0]['field'] == 'engine._inventory_trade_volume'


def test_run_native_cycle_comparison_tolerates_report_scale_inventory_roundoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    def report_scale_roundoff_step(engine):
        snapshot = engine.step()
        engine._inventory_trade_volume += 5.0e-7
        return snapshot

    monkeypatch.setattr('emergent_money.native_cycle_compare._step_native_cycle', report_scale_roundoff_step)

    summary = run_native_cycle_comparison(
        cycles=1,
        seeds=[2009],
        config=config,
    )

    assert summary['material_mismatch_count'] == 0
    assert summary['matched_seed_count_tolerant'] == 1


def test_run_native_cycle_comparison_tolerates_relative_inventory_roundoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    def large_accumulator_roundoff_step(engine):
        snapshot = engine.step()
        engine._inventory_trade_volume += 100_000_000.0
        engine._inventory_trade_volume += 5.0e-5
        return snapshot

    def large_accumulator_reference_step(engine):
        snapshot = engine.step()
        engine._inventory_trade_volume += 100_000_000.0
        return snapshot

    monkeypatch.setattr('emergent_money.native_cycle_compare._step_native_cycle', large_accumulator_roundoff_step)
    monkeypatch.setattr('emergent_money.native_cycle_compare._step_python_cycle', large_accumulator_reference_step)

    summary = run_native_cycle_comparison(
        cycles=1,
        seeds=[2009],
        config=config,
    )

    assert summary['material_mismatch_count'] == 0
    assert summary['matched_seed_count_tolerant'] == 1


def test_run_native_cycle_comparison_requires_native_entrypoint(monkeypatch: pytest.MonkeyPatch) -> None:
    config = SimulationConfig(population=4, goods=2, acquaintances=1, active_acquaintances=1, demand_candidates=1, supply_candidates=1)
    monkeypatch.setattr('emergent_money.native_cycle_compare._native_cycle_entrypoint_available', lambda: False)

    with pytest.raises(RuntimeError, match='native exact-cycle entrypoint is not available'):
        run_native_cycle_comparison(cycles=1, seeds=[2009], config=config)
