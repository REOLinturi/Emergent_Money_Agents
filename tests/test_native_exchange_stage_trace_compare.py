from __future__ import annotations

import pytest

from emergent_money.config import SimulationConfig
from emergent_money.native_exchange_stage_trace_compare import run_native_exchange_stage_trace_comparison


def test_run_native_exchange_stage_trace_comparison_matches_when_target_equals_reference() -> None:
    config = SimulationConfig(
        population=8,
        goods=3,
        acquaintances=2,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
        seed=2009,
    )

    summary = run_native_exchange_stage_trace_comparison(
        cycles=2,
        seeds=[2009, 2011],
        config=config,
    )

    assert summary['mismatch_count'] == 0
    assert summary['matched_seed_count'] == 2
    assert summary['benchmark']['target_seconds'] > 0.0


def test_run_native_exchange_stage_trace_comparison_reports_first_event_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    config = SimulationConfig(
        population=8,
        goods=3,
        acquaintances=2,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
        seed=2009,
        experimental_native_exchange_stage=True,
    )

    original_payload = __import__('emergent_money.native_exchange_stage_trace_compare', fromlist=['_stage_event_payload'])._stage_event_payload

    def fake_payload(runner, stage, agent_id, before):
        payload = original_payload(runner, stage, agent_id, before)
        if runner.config.experimental_native_exchange_stage and stage == 'consumption' and agent_id == 0:
            payload['proposal_target_agent'] = int(payload['proposal_target_agent']) + 1
        return payload

    monkeypatch.setattr('emergent_money.native_exchange_stage_trace_compare._stage_event_payload', fake_payload)

    summary = run_native_exchange_stage_trace_comparison(
        cycles=1,
        seeds=[2009],
        config=config,
    )

    assert summary['mismatch_count'] == 1
    mismatch = summary['mismatch_examples'][0]
    assert mismatch['seed'] == 2009
    assert mismatch['cycle'] == 1
    assert mismatch['reason'] == 'event_mismatch'
    assert mismatch['field_mismatch_examples'][0]['field'] == 'proposal_target_agent'


def test_run_native_exchange_stage_trace_comparison_rejects_unknown_tolerance_field() -> None:
    config = SimulationConfig(population=4, goods=2, acquaintances=1, active_acquaintances=1, demand_candidates=1, supply_candidates=1)

    with pytest.raises(ValueError, match='unknown exchange-trace tolerance field'):
        run_native_exchange_stage_trace_comparison(
            cycles=1,
            seeds=[2009],
            config=config,
            float_tolerances={'unknown_field': 1.0},
        )
