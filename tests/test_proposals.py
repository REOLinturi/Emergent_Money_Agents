from __future__ import annotations

import numpy as np
import pytest

from emergent_money.backend import create_backend
from emergent_money.config import SimulationConfig
from emergent_money.engine import SimulationEngine


def _cuda_available() -> bool:
    try:
        create_backend("cuda")
    except Exception:
        return False
    return True

def _seed_proposal_fixture(engine: SimulationEngine) -> None:
    state = engine.state
    backend = engine.backend
    state.friend_id[...] = backend.asarray(
        np.array(
            [
                [1, 2],
                [0, -1],
                [0, 1],
            ],
            dtype=np.int32,
        ),
        dtype=np.int32,
    )
    state.transparency[...] = backend.asarray(
        np.array(
            [
                [[0.9, 1.0, 0.7, 0.8], [0.8, 0.6, 1.0, 0.9]],
                [[1.0, 0.9, 0.8, 0.7], [0.0, 0.0, 0.0, 0.0]],
                [[0.7, 0.8, 0.9, 1.0], [0.9, 0.7, 0.8, 1.0]],
            ],
            dtype=np.float32,
        ),
        dtype=np.float32,
    )
    state.need[...] = backend.asarray(
        np.array(
            [
                [0.0, 5.0, 1.0, 0.0],
                [3.0, 0.0, 0.0, 2.0],
                [0.0, 2.0, 0.0, 4.0],
            ],
            dtype=np.float32,
        ),
        dtype=np.float32,
    )
    state.stock[...] = backend.asarray(
        np.array(
            [
                [4.0, 0.0, 3.0, 0.0],
                [0.0, 6.0, 1.0, 3.0],
                [2.0, 1.0, 5.0, 0.0],
            ],
            dtype=np.float32,
        ),
        dtype=np.float32,
    )
    state.stock_limit[...] = 8.0
    state.purchase_price[...] = backend.asarray(
        np.array(
            [
                [1.0, 3.0, 1.5, 2.0],
                [3.0, 1.0, 2.0, 1.2],
                [1.2, 2.4, 1.0, 3.0],
            ],
            dtype=np.float32,
        ),
        dtype=np.float32,
    )
    state.sales_price[...] = backend.asarray(
        np.array(
            [
                [1.0, 1.2, 1.1, 1.5],
                [1.0, 1.1, 1.4, 1.0],
                [1.3, 1.0, 1.1, 1.0],
            ],
            dtype=np.float32,
        ),
        dtype=np.float32,
    )


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


def test_blocked_proposal_scoring_matches_reference_loop() -> None:
    config = SimulationConfig(
        population=3,
        goods=4,
        acquaintances=2,
        active_acquaintances=2,
        demand_candidates=2,
        supply_candidates=2,
        cuda_friend_block=1,
        cuda_goods_block=2,
    )
    engine = SimulationEngine.create(config=config, backend_name="numpy")
    _seed_proposal_fixture(engine)
    state = engine.state

    engine._score_trade_proposals(allow_stock_trade=True)
    reference_friend_slot = state.trade.proposal_friend_slot.copy()
    reference_target_agent = state.trade.proposal_target_agent.copy()
    reference_need_good = state.trade.proposal_need_good.copy()
    reference_offer_good = state.trade.proposal_offer_good.copy()
    reference_quantity = state.trade.proposal_quantity.copy()
    reference_score = state.trade.proposal_score.copy()

    state.trade.proposal_friend_slot[...] = -1
    state.trade.proposal_target_agent[...] = -1
    state.trade.proposal_need_good[...] = -1
    state.trade.proposal_offer_good[...] = -1
    state.trade.proposal_quantity[...] = 0.0
    state.trade.proposal_score[...] = 0.0

    engine._score_trade_proposals_blocked(allow_stock_trade=True)

    np.testing.assert_array_equal(state.trade.proposal_friend_slot, reference_friend_slot)
    np.testing.assert_array_equal(state.trade.proposal_target_agent, reference_target_agent)
    np.testing.assert_array_equal(state.trade.proposal_need_good, reference_need_good)
    np.testing.assert_array_equal(state.trade.proposal_offer_good, reference_offer_good)
    np.testing.assert_allclose(state.trade.proposal_quantity, reference_quantity)
    np.testing.assert_allclose(state.trade.proposal_score, reference_score)


@pytest.mark.skipif(not _cuda_available(), reason="CUDA backend unavailable")
def test_cuda_proposal_scoring_matches_numpy_reference() -> None:
    config = SimulationConfig(
        population=3,
        goods=4,
        acquaintances=2,
        active_acquaintances=2,
        demand_candidates=2,
        supply_candidates=2,
        cuda_friend_block=1,
        cuda_goods_block=2,
    )
    numpy_engine = SimulationEngine.create(config=config, backend_name="numpy")
    cuda_engine = SimulationEngine.create(config=config, backend_name="cuda")
    _seed_proposal_fixture(numpy_engine)
    _seed_proposal_fixture(cuda_engine)

    numpy_engine._score_trade_proposals(allow_stock_trade=True)
    cuda_engine._score_trade_proposals(allow_stock_trade=True)
    cuda_engine.backend.synchronize()

    np.testing.assert_array_equal(
        numpy_engine.state.trade.proposal_friend_slot,
        cuda_engine.backend.to_numpy(cuda_engine.state.trade.proposal_friend_slot),
    )
    np.testing.assert_array_equal(
        numpy_engine.state.trade.proposal_target_agent,
        cuda_engine.backend.to_numpy(cuda_engine.state.trade.proposal_target_agent),
    )
    np.testing.assert_array_equal(
        numpy_engine.state.trade.proposal_need_good,
        cuda_engine.backend.to_numpy(cuda_engine.state.trade.proposal_need_good),
    )
    np.testing.assert_array_equal(
        numpy_engine.state.trade.proposal_offer_good,
        cuda_engine.backend.to_numpy(cuda_engine.state.trade.proposal_offer_good),
    )
    np.testing.assert_allclose(
        numpy_engine.state.trade.proposal_quantity,
        cuda_engine.backend.to_numpy(cuda_engine.state.trade.proposal_quantity),
    )
    np.testing.assert_allclose(
        numpy_engine.state.trade.proposal_score,
        cuda_engine.backend.to_numpy(cuda_engine.state.trade.proposal_score),
    )


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