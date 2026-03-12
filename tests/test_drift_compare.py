from __future__ import annotations

import json

from emergent_money.config import SimulationConfig
from emergent_money.drift_compare import (
    _summarize_hybrid_wave_diagnostics,
    run_hybrid_consumption_comparison,
    run_hybrid_consumption_frontier_sweep,
)


def test_hybrid_consumption_comparison_writes_summary_and_mean_deltas(tmp_path) -> None:
    config = SimulationConfig(
        population=8,
        goods=3,
        acquaintances=3,
        active_acquaintances=2,
        demand_candidates=2,
        supply_candidates=2,
        experimental_hybrid_batches=1,
        experimental_hybrid_frontier_size=1,
        experimental_hybrid_consumption_stage=True,
        experimental_hybrid_rolling_frontier=True,
    )

    output_path = tmp_path / 'comparison.json'
    summary = run_hybrid_consumption_comparison(
        cycles=2,
        seeds=[2009, 2011],
        config=config,
        output_path=output_path,
    )

    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding='utf-8'))
    assert payload['cycles'] == 2
    assert payload['seeds'] == [2009, 2011]
    assert len(summary['sequential']) == 2
    assert len(summary['hybrid']) == 2
    assert len(summary['deltas']) == 2
    assert len(summary['per_cycle_mean_delta']) == 2
    assert summary['per_cycle_mean_delta'][0]['cycle'] == 1
    assert 'accepted_trade_volume' in summary['peak_delta_cycles']
    assert set(summary['mean_delta']) == {
        'accepted_trade_count',
        'accepted_trade_volume',
        'production_total',
        'utility_proxy_total',
        'rare_goods_monetary_share',
    }
    assert summary['hybrid_wave_diagnostics']['cycles_with_diagnostics'] == 4
    assert summary['hybrid_wave_diagnostics']['seeds_with_diagnostics'] == 2
    assert summary['hybrid_wave_diagnostics']['wave_count_total'] >= 2
    assert 'no_candidate_reasons_total' in summary['hybrid_wave_diagnostics']
    assert 'execution_failure_reasons_total' in summary['hybrid_wave_diagnostics']
    assert 'scheduled_quantity_total' in summary['hybrid_wave_diagnostics']
    assert 'executed_quantity_total' in summary['hybrid_wave_diagnostics']
    assert 'mean_executed_quantity_per_exchange' in summary['hybrid_wave_diagnostics']['mean_wave']
    assert summary['hybrid_wave_diagnostics']['stage_activation']['surplus_activated'] is False
    assert summary['hybrid_wave_diagnostics']['stage_activation']['stage_cycle_counts']['consumption'] == 4
    assert summary['hybrid_wave_diagnostics']['stage_activation']['first_cycle_overall']['consumption'] == 1
    assert len(summary['hybrid'][0]['cycle_diagnostics']) == 2
    assert len(summary['hybrid'][0]['cycle_metrics']) == 2
    assert payload['hybrid_wave_diagnostics']['cycles_with_diagnostics'] == 4


def test_hybrid_consumption_comparison_requires_enabled_hybrid_config(tmp_path) -> None:
    config = SimulationConfig(
        population=8,
        goods=3,
        acquaintances=3,
        active_acquaintances=2,
        demand_candidates=2,
        supply_candidates=2,
    )

    try:
        run_hybrid_consumption_comparison(
            cycles=1,
            seeds=[2009],
            config=config,
            output_path=tmp_path / 'comparison.json',
        )
    except ValueError as exc:
        assert 'experimental hybrid exchange stage' in str(exc)
    else:
        raise AssertionError('Expected ValueError for disabled hybrid comparison config')



def test_hybrid_wave_diagnostics_summarize_stage_activation() -> None:
    def cycle(stage_wave_counts: dict[str, int]) -> dict[str, object]:
        return {
            'wave_count': sum(stage_wave_counts.values()),
            'frontier_count': 1,
            'candidate_agents_total': 1,
            'no_candidate_agents_total': 0,
            'scheduled_exchanges_total': 1,
            'scheduled_quantity_total': 1.0,
            'dropped_exchanges_total': 0,
            'executed_exchanges_total': 1,
            'executed_quantity_total': 1.0,
            'execution_failures_total': 0,
            'retry_exhausted_agents_total': 0,
            'stalled_waves_total': 0,
            'no_candidate_reasons_total': {},
            'execution_failure_reasons_total': {},
            'stage_wave_counts': stage_wave_counts,
        }

    summary = _summarize_hybrid_wave_diagnostics(
        [
            {'seed': 2009, 'cycle_diagnostics': [cycle({'consumption': 2}), cycle({'consumption': 1, 'surplus': 2})]},
            {'seed': 2011, 'cycle_diagnostics': [cycle({'consumption': 1}), cycle({'consumption': 1, 'surplus': 1})]},
        ]
    )

    stage_activation = summary['stage_activation']
    assert stage_activation['surplus_activated'] is True
    assert stage_activation['stage_cycle_counts'] == {'consumption': 4, 'surplus': 2}
    assert stage_activation['stage_wave_counts'] == {'consumption': 5, 'surplus': 3}
    assert stage_activation['stage_seed_counts'] == {'consumption': 2, 'surplus': 2}
    assert stage_activation['first_cycle_overall'] == {'consumption': 1, 'surplus': 2}
    assert stage_activation['first_cycle_by_seed']['2009']['surplus'] == 2
    assert stage_activation['first_cycle_by_seed']['2011']['consumption'] == 1


def test_hybrid_consumption_frontier_sweep_summarizes_recommendation(tmp_path) -> None:
    config = SimulationConfig(
        population=8,
        goods=3,
        acquaintances=3,
        active_acquaintances=2,
        demand_candidates=2,
        supply_candidates=2,
        experimental_hybrid_batches=1,
        experimental_hybrid_frontier_size=2,
        experimental_hybrid_consumption_stage=True,
    )

    output_path = tmp_path / 'frontier-sweep.json'
    summary = run_hybrid_consumption_frontier_sweep(
        cycles=2,
        seeds=[2009, 2011],
        config=config,
        frontier_sizes=[2, 1, 2],
        output_path=output_path,
    )

    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding='utf-8'))
    assert summary['frontier_sizes'] == [1, 2]
    assert payload['frontier_sizes'] == [1, 2]
    assert len(summary['comparisons']) == 2
    assert summary['recommended_frontier']['frontier_size'] == 1
    assert summary['recommended_nontrivial_frontier']['frontier_size'] == 2
    assert summary['best_by_abs_mean_delta']['accepted_trade_count']['frontier_size'] == 1
    assert summary['best_by_abs_peak_delta']['accepted_trade_volume']['frontier_size'] == 1


def test_hybrid_consumption_frontier_sweep_rejects_empty_frontier_sizes(tmp_path) -> None:
    config = SimulationConfig(
        population=8,
        goods=3,
        acquaintances=3,
        active_acquaintances=2,
        demand_candidates=2,
        supply_candidates=2,
        experimental_hybrid_batches=1,
        experimental_hybrid_frontier_size=1,
        experimental_hybrid_consumption_stage=True,
    )

    try:
        run_hybrid_consumption_frontier_sweep(
            cycles=1,
            seeds=[2009],
            config=config,
            frontier_sizes=[],
            output_path=tmp_path / 'frontier-sweep.json',
        )
    except ValueError as exc:
        assert 'frontier_sizes' in str(exc)
    else:
        raise AssertionError('Expected ValueError for empty frontier_sizes')



def test_hybrid_consumption_frontier_sweep_nontrivial_recommendation_is_none_without_hybrid_frontiers(tmp_path) -> None:
    config = SimulationConfig(
        population=8,
        goods=3,
        acquaintances=3,
        active_acquaintances=2,
        demand_candidates=2,
        supply_candidates=2,
        experimental_hybrid_batches=1,
        experimental_hybrid_frontier_size=1,
        experimental_hybrid_consumption_stage=True,
    )

    summary = run_hybrid_consumption_frontier_sweep(
        cycles=1,
        seeds=[2009],
        config=config,
        frontier_sizes=[1],
        output_path=tmp_path / 'frontier-sweep-single.json',
    )

    assert summary['recommended_frontier']['frontier_size'] == 1
    assert summary['recommended_nontrivial_frontier'] is None



def test_hybrid_consumption_frontier_sweep_nontrivial_recommendation_prioritizes_volume_stability(tmp_path) -> None:
    config = SimulationConfig(
        population=32,
        goods=6,
        acquaintances=6,
        active_acquaintances=6,
        demand_candidates=4,
        supply_candidates=4,
        experimental_hybrid_batches=2,
        experimental_hybrid_frontier_size=2,
        experimental_hybrid_consumption_stage=True,
    )

    summary = run_hybrid_consumption_frontier_sweep(
        cycles=12,
        seeds=[2009, 2011, 2013, 2015, 2017],
        config=config,
        frontier_sizes=[2, 3, 4],
        output_path=tmp_path / 'frontier-sweep-volume-priority.json',
    )

    assert summary['recommended_nontrivial_frontier']['frontier_size'] == 2
