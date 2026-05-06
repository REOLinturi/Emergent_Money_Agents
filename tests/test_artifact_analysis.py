from __future__ import annotations

import json

from emergent_money.artifact_analysis import summarize_run_artifact


def test_summarize_run_artifact_reports_phenomenon_flags(tmp_path) -> None:
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    rows = [
        {
            'cycle': 5,
            'production_total': 10.0,
            'utility_proxy_total': 1.0,
            'accepted_trade_volume': 2.0,
            'stock_total': 4.0,
            'rare_goods_monetary_share': 0.0,
            'living_standard_mean': 1.0,
            'living_standard_gini': 0.1,
            'friction_share_of_time_budget': 0.01,
            'tce_share_of_time_budget': 0.01,
            'spoilage_share_of_time_budget': 0.0,
        },
        {
            'cycle': 10,
            'production_total': 40.0,
            'utility_proxy_total': 5.0,
            'accepted_trade_volume': 8.0,
            'stock_total': 10.0,
            'rare_goods_monetary_share': 0.7,
            'living_standard_mean': 3.0,
            'living_standard_gini': 0.45,
            'friction_share_of_time_budget': 0.08,
            'tce_share_of_time_budget': 0.07,
            'spoilage_share_of_time_budget': 0.01,
        },
        {
            'cycle': 15,
            'production_total': 80.0,
            'utility_proxy_total': 4.0,
            'accepted_trade_volume': 16.0,
            'stock_total': 30.0,
            'rare_goods_monetary_share': 0.9,
            'living_standard_mean': 2.5,
            'living_standard_gini': 0.5,
            'friction_share_of_time_budget': 0.12,
            'tce_share_of_time_budget': 0.1,
            'spoilage_share_of_time_budget': 0.02,
        },
    ]
    (run_dir / 'metrics.jsonl').write_text(
        ''.join(json.dumps(row) + '\n' for row in rows),
        encoding='utf-8',
    )

    summary = summarize_run_artifact(run_dir)

    assert summary['sample_count'] == 3
    assert summary['first_cycle'] == 5
    assert summary['last_cycle'] == 15
    assert summary['fields']['production_total']['max_cycle'] == 15
    assert summary['fields']['utility_proxy_total']['max_cycle'] == 10
    assert summary['phenomenon_flags']['production_grew'] is True
    assert summary['phenomenon_flags']['rare_money_emerged'] is True
    assert summary['phenomenon_flags']['utility_peaked_before_end'] is True
    assert summary['phenomenon_flags']['friction_rose'] is True
    assert summary['phenomenon_flags']['living_standard_inequality_high'] is True


def test_summarize_run_artifact_uses_latest_duplicate_cycle_sample(tmp_path) -> None:
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    rows = [
        {
            'cycle': 5,
            'production_total': 10.0,
            'utility_proxy_total': 1.0,
            'rare_goods_monetary_share': 0.0,
            'living_standard_gini': 0.1,
            'friction_share_of_time_budget': 0.01,
        },
        {
            'cycle': 10,
            'production_total': 20.0,
            'utility_proxy_total': 2.0,
            'rare_goods_monetary_share': 0.1,
            'living_standard_gini': 0.2,
            'friction_share_of_time_budget': 0.02,
        },
        {
            'cycle': 10,
            'production_total': 40.0,
            'utility_proxy_total': 4.0,
            'rare_goods_monetary_share': 0.7,
            'living_standard_gini': 0.45,
            'friction_share_of_time_budget': 0.08,
        },
    ]
    (run_dir / 'metrics.jsonl').write_text(
        ''.join(json.dumps(row) + '\n' for row in rows),
        encoding='utf-8',
    )

    summary = summarize_run_artifact(run_dir)

    assert summary['sample_count'] == 2
    assert summary['fields']['production_total']['last'] == 40.0
    assert summary['fields']['rare_goods_monetary_share']['last'] == 0.7
