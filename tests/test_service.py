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

    agent = service.get_agent_snapshot(0)
    assert len(agent.need) == config.goods
    assert len(agent.active_friends) == config.active_acquaintances

    network = service.get_network_slice(0, limit=2)
    assert network.root_agent_id == 0
    assert len(network.friend_ids) == 2
