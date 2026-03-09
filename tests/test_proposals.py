from __future__ import annotations

import numpy as np

from emergent_money.config import SimulationConfig
from emergent_money.engine import SimulationEngine


def test_proposal_scoring_finds_profitable_barter_pair() -> None:
    config = SimulationConfig(
        population=2,
        goods=2,
        acquaintances=1,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
    )
    engine = SimulationEngine.create(config=config, backend_name="numpy")
    state = engine.state

    state.friend_id[...] = np.array([[1], [0]], dtype=np.int32)
    state.transparency[...] = 1.0

    state.need[...] = np.array(
        [
            [0.0, 5.0],
            [4.0, 0.0],
        ],
        dtype=np.float32,
    )
    state.stock[...] = np.array(
        [
            [4.0, 0.0],
            [0.0, 5.0],
        ],
        dtype=np.float32,
    )
    state.stock_limit[...] = 10.0
    state.purchase_price[...] = np.array(
        [
            [1.0, 3.0],
            [3.0, 1.0],
        ],
        dtype=np.float32,
    )
    state.sales_price[...] = 1.0

    engine._prepare_trade_frontier()
    engine._select_trade_candidates(allow_stock_trade=False)
    engine._score_trade_proposals(allow_stock_trade=False)

    assert float(engine.backend.to_scalar(state.trade.proposal_score[0])) > 0.0
    assert int(engine.backend.to_scalar(state.trade.proposal_target_agent[0])) == 1
    assert int(engine.backend.to_scalar(state.trade.proposal_need_good[0])) == 1
    assert int(engine.backend.to_scalar(state.trade.proposal_offer_good[0])) == 0
    assert float(engine.backend.to_scalar(state.trade.proposal_quantity[0])) == 4.0

    assert float(engine.backend.to_scalar(state.trade.proposal_score[1])) > 0.0
    assert int(engine.backend.to_scalar(state.trade.proposal_target_agent[1])) == 0


def test_proposal_scoring_ignores_snapshot_candidate_buffers() -> None:
    config = SimulationConfig(
        population=2,
        goods=3,
        acquaintances=1,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
    )
    engine = SimulationEngine.create(config=config, backend_name="numpy")
    state = engine.state

    state.friend_id[...] = np.array([[1], [0]], dtype=np.int32)
    state.transparency[...] = 1.0
    state.need[...] = np.array(
        [
            [0.0, 0.0, 5.0],
            [0.0, 4.0, 0.0],
        ],
        dtype=np.float32,
    )
    state.stock[...] = np.array(
        [
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 5.0],
        ],
        dtype=np.float32,
    )
    state.stock_limit[...] = 10.0
    state.purchase_price[...] = np.array(
        [
            [1.0, 1.0, 3.0],
            [1.0, 3.0, 1.0],
        ],
        dtype=np.float32,
    )
    state.sales_price[...] = 1.0

    engine._prepare_trade_frontier()
    state.trade.candidate_need_good[...] = 0
    state.trade.candidate_offer_good[...] = 0
    engine._score_trade_proposals(allow_stock_trade=False)

    assert int(engine.backend.to_scalar(state.trade.proposal_need_good[0])) == 2
    assert int(engine.backend.to_scalar(state.trade.proposal_offer_good[0])) == 1
    assert float(engine.backend.to_scalar(state.trade.proposal_score[0])) > 0.0


def test_commit_executes_single_barter_without_double_counting() -> None:
    config = SimulationConfig(
        population=2,
        goods=2,
        acquaintances=1,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
    )
    engine = SimulationEngine.create(config=config, backend_name="numpy")
    state = engine.state

    state.friend_id[...] = np.array([[1], [0]], dtype=np.int32)
    state.transparency[...] = 1.0
    state.need[...] = np.array(
        [
            [0.0, 5.0],
            [4.0, 0.0],
        ],
        dtype=np.float32,
    )
    state.stock[...] = np.array(
        [
            [4.0, 0.0],
            [0.0, 5.0],
        ],
        dtype=np.float32,
    )
    state.stock_limit[...] = 10.0
    state.purchase_price[...] = np.array(
        [
            [1.0, 3.0],
            [3.0, 1.0],
        ],
        dtype=np.float32,
    )
    state.sales_price[...] = 1.0

    engine._prepare_trade_frontier()
    engine._select_trade_candidates(allow_stock_trade=False)
    engine._score_trade_proposals(allow_stock_trade=False)
    accepted_count, accepted_volume = engine._commit_trades()

    assert accepted_count == 1
    assert accepted_volume == 4.0
    assert bool(engine.backend.to_scalar(state.trade.accepted_mask[0])) is True
    assert bool(engine.backend.to_scalar(state.trade.accepted_mask[1])) is False
    assert float(engine.backend.to_scalar(state.trade.accepted_quantity[0])) == 4.0
    assert float(engine.backend.to_scalar(state.need[0, 1])) == 1.0
    assert float(engine.backend.to_scalar(state.need[1, 0])) == 0.0
    assert float(engine.backend.to_scalar(state.stock[0, 0])) == 0.0
    assert float(engine.backend.to_scalar(state.stock[1, 1])) == 1.0
    assert float(engine.backend.to_scalar(state.recent_sales[0, 0])) == 4.0
    assert float(engine.backend.to_scalar(state.recent_purchases[0, 1])) == 4.0


def test_stock_trade_round_can_exchange_for_inventory_room() -> None:
    config = SimulationConfig(
        population=2,
        goods=2,
        acquaintances=1,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
    )
    engine = SimulationEngine.create(config=config, backend_name="numpy")
    state = engine.state

    state.friend_id[...] = np.array([[1], [0]], dtype=np.int32)
    state.transparency[...] = 1.0
    state.need[...] = 0.0
    state.stock[...] = np.array(
        [
            [4.0, 0.0],
            [0.0, 4.0],
        ],
        dtype=np.float32,
    )
    state.stock_limit[...] = np.array(
        [
            [4.0, 8.0],
            [8.0, 4.0],
        ],
        dtype=np.float32,
    )
    state.purchase_price[...] = np.array(
        [
            [1.0, 3.0],
            [3.0, 1.0],
        ],
        dtype=np.float32,
    )
    state.sales_price[...] = 1.0

    engine._run_market_round(allow_stock_trade=True)

    assert engine._accepted_trade_count == 1
    assert float(engine.backend.to_scalar(state.trade.accepted_quantity[0])) > 0.0