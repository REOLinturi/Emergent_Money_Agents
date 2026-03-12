from __future__ import annotations

from emergent_money.config import SimulationConfig
from emergent_money.service import SimulationService


def test_service_exposes_snapshots_without_raw_backend_arrays() -> None:
    config = SimulationConfig(
        population=16,
        goods=6,
        acquaintances=4,
        active_acquaintances=2,
        demand_candidates=2,
        supply_candidates=2,
    )
    service = SimulationService.create(config=config, backend_name="numpy")

    status = service.get_status()
    assert status.backend_name == "numpy"
    assert status.cycle == 0

    snapshots = service.step(1)
    assert len(snapshots) == 1
    assert snapshots[0].cycle == 1
    assert snapshots[0].accepted_trade_count >= 0
    assert snapshots[0].production_total > 0.0
    assert 0.0 <= snapshots[0].rare_goods_monetary_share <= 1.0

    agent = service.get_agent_snapshot(0)
    assert len(agent.need) == config.goods
    assert len(agent.active_friends) == config.active_acquaintances
    assert len(agent.recent_inventory_inflow) == config.goods

    network = service.get_network_slice(0, limit=2)
    assert network.root_agent_id == 0
    assert len(network.friend_ids) == 2


def test_service_exposes_history_goods_and_phenomena_snapshots() -> None:
    config = SimulationConfig(
        population=24,
        goods=8,
        acquaintances=5,
        active_acquaintances=2,
        demand_candidates=2,
        supply_candidates=2,
    )
    service = SimulationService.create(config=config, backend_name="numpy")

    service.step(4)

    history = service.get_history(limit=3)
    goods = service.get_goods_snapshot(limit=4)
    phenomena = service.get_phenomena_snapshot(top_goods=4)
    report = service.run_experiment(0, top_goods=4)

    assert len(history) == 3
    assert len(goods) == 4
    assert all(goods[idx].monetary_score >= goods[idx + 1].monetary_score for idx in range(len(goods) - 1))
    assert phenomena.cycles_observed == 4
    assert 0.0 <= phenomena.rare_goods_monetary_share <= 1.0
    assert report.latest_market.cycle == 4
    assert len(report.goods) == 4
