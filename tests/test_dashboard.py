from __future__ import annotations

from pathlib import Path

import pytest

from emergent_money.config import SimulationConfig
from emergent_money.dashboard import ArtifactDashboardController
from emergent_money.long_run import run_long_simulation


def test_artifact_dashboard_controller_reads_long_run_artifacts(tmp_path) -> None:
    artifact_dir = tmp_path / 'artifact_run'
    config = SimulationConfig(
        population=24,
        goods=8,
        acquaintances=5,
        active_acquaintances=5,
        demand_candidates=3,
        supply_candidates=3,
        seed=2009,
        experimental_native_stage_math=True,
    )
    run_long_simulation(
        cycles=3,
        checkpoint_dir=artifact_dir,
        config=config,
        backend_name='numpy',
        checkpoint_every=1,
        sample_every=1,
    )
    (artifact_dir / 'runner.log').write_text(
        '\n'.join(
            [
                '[2026-04-14 12:00:00] continuous exact runner starting target_cycle=3 chunk_cycles=3',
                '[2026-04-14 12:00:01] starting chunk from_cycle=0 chunk_cycles=3',
                'long_run start=0 end=3 seconds=30.0 utility=1.0 production=2.0 rare_money=0.1',
                '[2026-04-14 12:00:12] chunk completed end_cycle=3',
            ]
        ),
        encoding='utf-8',
    )

    controller = ArtifactDashboardController(artifact_dir)
    status = controller.get_status_payload()
    market = controller.get_market_payload()
    history = controller.get_history_payload(limit=10)
    goods = controller.get_goods_payload(limit=5)
    phenomena = controller.get_phenomena_payload(top_goods=5)
    recent_trends = controller.get_recent_trends_payload()
    role_mix = controller.get_role_mix_payload(limit=5)
    inequality = controller.get_inequality_payload()
    progress = controller.get_progress_payload()

    assert status['read_only'] is True
    assert status['completed'] is True
    assert status['cycle'] == 3
    assert status['config']['population'] == 24
    assert market['cycle'] == 3
    assert 0.0 <= market['value_weighted_rare_goods_monetary_share'] <= 1.0
    assert 0.0 <= market['rare_goods_exchange_media_share'] <= 1.0
    assert 0.0 <= market['exchange_media_concentration'] <= 1.0
    assert market['living_standard_gini'] >= 0.0
    assert market['living_standard_p10'] >= 0.0
    assert market['living_standard_p90'] >= 0.0
    assert 0.0 <= market['aspiration_shortfall_share'] <= 1.0
    assert market['smith_cost_p90'] >= 0.0
    assert market['friction_share_of_time_budget'] >= 0.0
    assert len(history) == 3
    assert history[-1]['living_standard_gini'] >= 0.0
    assert history[-1]['smith_cost_median'] >= 0.0
    assert len(goods) == 5
    assert 'value_weighted_monetary_score' in goods[0]
    assert 'exchange_media_score' in goods[0]
    assert 'relative_tce_loss' in goods[0]
    assert 'excess_stock_breadth' in goods[0]
    assert 'round_trip_breadth' in goods[0]
    assert 'consumer_flow_share' in goods[0]
    assert phenomena['cycles_observed'] == 3
    assert 0.0 <= phenomena['value_weighted_rare_goods_monetary_share'] <= 1.0
    assert 0.0 <= phenomena['rare_goods_exchange_media_share'] <= 1.0
    assert recent_trends
    assert recent_trends[0]['to_cycle'] == 3
    assert len(role_mix) == 5
    assert {'producer_count', 'retailer_count', 'consumer_count', 'retailer_inventory_inflow_total'} <= set(role_mix[0])
    assert 0.0 <= inequality['stock_value_gini'] <= 1.0
    assert 0.0 <= inequality['stock_value_top_decile_share'] <= 1.0
    assert 0.0 <= inequality['living_standard_gini'] <= 1.0
    assert 0.0 <= inequality['living_standard_top_decile_share'] <= 1.0
    assert inequality['living_standard_p10'] >= 0.0
    assert 0.0 <= inequality['aspiration_shortfall_share'] <= 1.0
    assert inequality['aspiration_shortfall_mean'] >= 0.0
    assert inequality['smith_cost_median'] >= 0.0
    assert 0.0 <= inequality['friction_share_of_output_value']
    assert progress['current_cycle'] == 3
    assert progress['target_cycle'] == 3
    assert progress['progress_share'] == pytest.approx(1.0)
    assert progress['latest_chunk_from_cycle'] == 0
    assert progress['latest_chunk_target_cycle'] == 3
    assert progress['recent_seconds_per_cycle'] == pytest.approx(10.0)

    with pytest.raises(RuntimeError):
        controller.start()
    with pytest.raises(RuntimeError):
        controller.reset({})


def test_dashboard_page_marks_exact_legacy_candidate_controls_as_scaffold_only() -> None:
    html = (Path(__file__).resolve().parents[1] / 'src' / 'emergent_money' / 'dashboard_page.html').read_text(encoding='utf-8')

    assert "Exact legacy mode uses each agent's full acquaintance list with exhaustive barter scoring." in html
    assert 'class="exact-scaffold-field"' in html
    assert 'id="recentTrendCards"' in html
    assert 'id="progressCards"' in html
    assert 'id="roleTable"' in html
    assert 'Stock value Gini' in html
    assert 'Living standard Gini' in html
    assert 'Welfare path' in html
    assert 'Value rare-money' in html
    assert 'Exchange-media rare' in html
    assert 'Exchange-media candidates' in html
    assert 'sort_by=exchange_media_score' in html
    assert 'Excess holders' in html
    assert 'Round trips' in html
    assert 'Cons flow' in html
    assert 'Expectation shortfall' in html
    assert 'Aspirational balance' in html
    assert 'Smith need cost' in html
    assert 'Friction / output' in html
    assert 'id="helpOverlay"' in html
    assert 'data-help-key="chart_macro"' in html
    assert 'data-help-key="chart_welfare"' in html
    assert 'living_mean' in html
    assert 'Indexed utility proxy and production.' in html
    assert 'Top output focus' in html
    assert 'Top prod focus' in html
