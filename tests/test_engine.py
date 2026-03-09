from __future__ import annotations

import numpy as np

from emergent_money.config import SimulationConfig
from emergent_money.engine import SimulationEngine


def test_single_step_keeps_state_non_negative_and_engages_market() -> None:
    config = SimulationConfig(
        population=32,
        goods=8,
        acquaintances=6,
        active_acquaintances=3,
        demand_candidates=2,
        supply_candidates=2,
    )
    engine = SimulationEngine.create(config=config, backend_name="numpy")
    snapshot = engine.step()

    xp = engine.backend.xp
    assert snapshot.cycle == 1
    assert snapshot.fulfilled_share > 0.95
    assert snapshot.proposed_trade_count >= snapshot.accepted_trade_count >= 0
    assert snapshot.proposed_trade_count > 0
    assert float(engine.backend.to_scalar(xp.min(engine.state.need))) >= 0.0
    assert float(engine.backend.to_scalar(xp.min(engine.state.stock))) >= 0.0
    assert float(engine.backend.to_scalar(xp.min(engine.state.time_remaining))) >= 0.0
    assert engine.state.trade.active_friend_id.shape == config.active_friend_shape
    assert engine.state.trade.active_friend_slot.shape == config.active_friend_shape


def test_learning_can_surpass_innate_advantage() -> None:
    config = SimulationConfig(
        population=2,
        goods=2,
        acquaintances=1,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
        talent_probability=0.0,
        activity_discount=0.8,
    )
    engine = SimulationEngine.create(config=config, backend_name="numpy")
    window = config.learning_window
    target_production = engine.backend.xp.asarray([[4.0 * window, 0.0], [0.0, 0.0]], dtype=engine.backend.xp.float32)

    engine.state.innate_efficiency[...] = 1.2
    engine.state.recent_production[...] = target_production
    engine._update_efficiency_from_learning()

    learned = float(engine.backend.to_scalar(engine.state.learned_efficiency[0, 0]))
    effective = float(engine.backend.to_scalar(engine.state.efficiency[0, 0]))
    assert learned >= 2.0
    assert effective == learned


def test_prepare_trade_frontier_prefers_recently_active_friends() -> None:
    config = SimulationConfig(
        population=3,
        goods=2,
        acquaintances=4,
        active_acquaintances=2,
        demand_candidates=1,
        supply_candidates=1,
    )
    engine = SimulationEngine.create(config=config, backend_name="numpy")
    engine.state.friend_id[...] = np.array(
        [
            [1, 2, 1, 2],
            [0, 2, 0, 2],
            [0, 1, 0, 1],
        ],
        dtype=np.int32,
    )
    engine.state.friend_activity[...] = np.array(
        [
            [0.1, 5.0, 3.0, 0.2],
            [2.0, 0.5, 4.0, 0.1],
            [0.0, 1.0, 2.0, 3.0],
        ],
        dtype=np.float32,
    )

    engine._prepare_trade_frontier()

    active_slots = engine.backend.to_numpy(engine.state.trade.active_friend_slot)
    active_ids = engine.backend.to_numpy(engine.state.trade.active_friend_id)
    assert np.array_equal(active_slots[0], np.array([1, 2], dtype=np.int32))
    assert np.array_equal(active_ids[0], np.array([2, 1], dtype=np.int32))


def test_random_contact_introduction_fills_sparse_network_one_slot_per_cycle() -> None:
    config = SimulationConfig(
        population=8,
        goods=3,
        acquaintances=4,
        active_acquaintances=2,
        demand_candidates=1,
        supply_candidates=1,
    )
    engine = SimulationEngine.create(config=config, backend_name="numpy")

    engine._introduce_random_contacts()

    friend_id = engine.backend.to_numpy(engine.state.friend_id)
    known_counts = np.sum(friend_id >= 0, axis=1)
    self_ids = np.arange(config.population, dtype=np.int32)[:, None]
    assert np.all(known_counts == 1)
    assert not np.any(friend_id == self_ids)


def test_apply_leisure_demand_weights_cheaper_goods_more() -> None:
    config = SimulationConfig(
        population=2,
        goods=2,
        acquaintances=1,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
        max_leisure_extra_multiplier=1.0,
    )
    engine = SimulationEngine.create(config=config, backend_name="numpy")
    engine.state.need[...] = 0.0
    engine.state.time_remaining[...] = engine.state.cycle_time_budget * 0.5
    engine.state.efficiency[...] = np.array(
        [
            [2.0, 1.0],
            [2.0, 1.0],
        ],
        dtype=np.float32,
    )

    applied = engine._apply_leisure_demand()
    need = engine.backend.to_numpy(engine.state.need)
    base_need = engine.backend.to_numpy(engine.state.base_need)
    normalized_extra = need / base_need

    assert applied is True
    assert normalized_extra[0, 0] > normalized_extra[0, 1]