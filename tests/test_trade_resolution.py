from __future__ import annotations

import numpy as np

from emergent_money.backend import create_backend
from emergent_money.trade_resolution import commit_resolved_trades, resolve_trade_proposals


def _sample_trade_inputs() -> dict[str, np.ndarray]:
    return {
        "stock": np.array(
            [
                [3.0, 0.0, 0.0],
                [0.0, 0.0, 3.0],
                [0.0, 3.0, 0.0],
            ],
            dtype=np.float32,
        ),
        "need": np.array(
            [
                [0.0, 3.0, 0.0],
                [0.0, 3.0, 0.0],
                [3.0, 0.0, 3.0],
            ],
            dtype=np.float32,
        ),
        "stock_limit": np.full((3, 3), 10.0, dtype=np.float32),
        "target_agent": np.array([2, 2, -1], dtype=np.int32),
        "need_good": np.array([1, 1, -1], dtype=np.int32),
        "offer_good": np.array([0, 2, -1], dtype=np.int32),
        "quantity": np.array([3.0, 3.0, 0.0], dtype=np.float32),
        "score": np.array([12.0, 9.0, 0.0], dtype=np.float32),
    }


def _sample_commit_inputs() -> dict[str, np.ndarray | float]:
    return {
        "stock": np.array(
            [
                [4.0, 0.0],
                [0.0, 5.0],
            ],
            dtype=np.float32,
        ),
        "need": np.array(
            [
                [0.0, 5.0],
                [4.0, 0.0],
            ],
            dtype=np.float32,
        ),
        "recent_sales": np.zeros((2, 2), dtype=np.float32),
        "recent_purchases": np.zeros((2, 2), dtype=np.float32),
        "friend_id": np.array(
            [
                [1],
                [-1],
            ],
            dtype=np.int32,
        ),
        "friend_activity": np.zeros((2, 1), dtype=np.float32),
        "transparency": np.full((2, 1, 2), 0.7, dtype=np.float32),
        "proposal_friend_slot": np.array([0, -1], dtype=np.int32),
        "proposal_target_agent": np.array([1, -1], dtype=np.int32),
        "proposal_need_good": np.array([1, -1], dtype=np.int32),
        "proposal_offer_good": np.array([0, -1], dtype=np.int32),
        "accepted_mask": np.array([True, False], dtype=np.bool_),
        "accepted_quantity": np.array([4.0, 0.0], dtype=np.float32),
        "initial_transparency": 0.7,
    }


def test_resolve_trade_proposals_prefers_higher_score_under_shared_supply() -> None:
    resolved = resolve_trade_proposals(**_sample_trade_inputs())

    assert resolved.accepted_mask.tolist() == [True, False, False]
    assert resolved.accepted_quantity.tolist() == [3.0, 0.0, 0.0]


def test_commit_resolved_trades_updates_goods_and_social_state() -> None:
    committed = commit_resolved_trades(**_sample_commit_inputs())
    transparency_gain = min(0.05, 0.01 * np.log1p(4.0))

    assert committed.stock.tolist() == [[0.0, 0.0], [0.0, 1.0]]
    assert committed.need.tolist() == [[0.0, 1.0], [0.0, 0.0]]
    assert committed.recent_sales.tolist() == [[4.0, 0.0], [0.0, 4.0]]
    assert committed.recent_purchases.tolist() == [[0.0, 4.0], [4.0, 0.0]]
    assert committed.friend_id.tolist() == [[1], [0]]
    assert committed.friend_activity.tolist() == [[4.0], [10.0]]
    assert np.isclose(committed.transparency[0, 0, 1], 0.7 + transparency_gain)
    assert np.isclose(committed.transparency[1, 0, 0], 0.7 + transparency_gain)


def test_numpy_backend_resolution_contract_matches_reference_semantics() -> None:
    inputs = _sample_trade_inputs()
    backend = create_backend("numpy")

    resolved = backend.resolve_trade_proposals(**inputs)

    assert backend.to_numpy(resolved.accepted_mask).tolist() == [True, False, False]
    assert backend.to_numpy(resolved.accepted_quantity).tolist() == [3.0, 0.0, 0.0]


def test_numpy_backend_commit_contract_matches_reference_semantics() -> None:
    inputs = _sample_commit_inputs()
    expected = commit_resolved_trades(**inputs)
    backend = create_backend("numpy")

    committed = backend.commit_resolved_trades(**inputs)

    assert backend.to_numpy(committed.stock).tolist() == expected.stock.tolist()
    assert backend.to_numpy(committed.need).tolist() == expected.need.tolist()
    assert backend.to_numpy(committed.recent_sales).tolist() == expected.recent_sales.tolist()
    assert backend.to_numpy(committed.recent_purchases).tolist() == expected.recent_purchases.tolist()
    assert backend.to_numpy(committed.friend_id).tolist() == expected.friend_id.tolist()
    assert backend.to_numpy(committed.friend_activity).tolist() == expected.friend_activity.tolist()
    assert np.allclose(backend.to_numpy(committed.transparency), expected.transparency)