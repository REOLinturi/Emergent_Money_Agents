from __future__ import annotations

import numpy as np
import pytest

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
                [0.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        "need": np.array(
            [
                [0.0, 1.0],
                [0.0, 0.0],
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


def _cuda_available() -> bool:
    try:
        create_backend("cuda")
    except Exception:
        return False
    return True


def test_resolve_trade_proposals_prefers_higher_score_under_shared_supply() -> None:
    resolved = resolve_trade_proposals(**_sample_trade_inputs())

    assert resolved.accepted_mask.tolist() == [True, False, False]
    assert resolved.accepted_quantity.tolist() == [3.0, 0.0, 0.0]
    assert resolved.stock.tolist() == [[0.0, 0.0, 0.0], [0.0, 0.0, 3.0], [0.0, 0.0, 0.0]]
    assert resolved.need.tolist() == [[0.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]


def test_commit_resolved_trades_updates_social_state_without_changing_resolved_goods() -> None:
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


def test_commit_preserves_resolve_output_goods_state() -> None:
    resolve_inputs = _sample_trade_inputs()
    resolved = resolve_trade_proposals(**resolve_inputs)
    committed = commit_resolved_trades(
        stock=resolved.stock.copy(),
        need=resolved.need.copy(),
        recent_sales=np.zeros_like(resolve_inputs["stock"]),
        recent_purchases=np.zeros_like(resolve_inputs["stock"]),
        friend_id=np.array([[2, -1], [2, -1], [-1, -1]], dtype=np.int32),
        friend_activity=np.zeros((3, 2), dtype=np.float32),
        transparency=np.full((3, 2, 3), 0.7, dtype=np.float32),
        proposal_friend_slot=np.array([0, 0, -1], dtype=np.int32),
        proposal_target_agent=resolve_inputs["target_agent"],
        proposal_need_good=resolve_inputs["need_good"],
        proposal_offer_good=resolve_inputs["offer_good"],
        accepted_mask=resolved.accepted_mask,
        accepted_quantity=resolved.accepted_quantity,
        initial_transparency=0.7,
    )

    assert np.allclose(committed.stock, resolved.stock)
    assert np.allclose(committed.need, resolved.need)


def test_numpy_backend_resolution_contract_matches_reference_semantics() -> None:
    inputs = _sample_trade_inputs()
    expected = resolve_trade_proposals(**inputs)
    backend = create_backend("numpy")

    resolved = backend.resolve_trade_proposals(**inputs)

    assert backend.to_numpy(resolved.accepted_mask).tolist() == expected.accepted_mask.tolist()
    assert backend.to_numpy(resolved.accepted_quantity).tolist() == expected.accepted_quantity.tolist()
    assert backend.to_numpy(resolved.stock).tolist() == expected.stock.tolist()
    assert backend.to_numpy(resolved.need).tolist() == expected.need.tolist()


def test_numpy_backend_commit_contract_matches_reference_semantics() -> None:
    expected = commit_resolved_trades(**_sample_commit_inputs())
    inputs = _sample_commit_inputs()
    backend = create_backend("numpy")

    committed = backend.commit_resolved_trades(**inputs)

    assert backend.to_numpy(committed.stock).tolist() == expected.stock.tolist()
    assert backend.to_numpy(committed.need).tolist() == expected.need.tolist()
    assert backend.to_numpy(committed.recent_sales).tolist() == expected.recent_sales.tolist()
    assert backend.to_numpy(committed.recent_purchases).tolist() == expected.recent_purchases.tolist()
    assert backend.to_numpy(committed.friend_id).tolist() == expected.friend_id.tolist()
    assert backend.to_numpy(committed.friend_activity).tolist() == expected.friend_activity.tolist()
    assert np.allclose(backend.to_numpy(committed.transparency), expected.transparency)


@pytest.mark.skipif(not _cuda_available(), reason="CUDA backend unavailable")
def test_cuda_backend_resolution_contract_matches_reference_semantics() -> None:
    inputs = _sample_trade_inputs()
    expected = resolve_trade_proposals(**inputs)
    backend = create_backend("cuda")
    device_inputs = {key: backend.asarray(value) for key, value in inputs.items()}

    resolved = backend.resolve_trade_proposals(**device_inputs)

    assert backend.to_numpy(resolved.accepted_mask).tolist() == expected.accepted_mask.tolist()
    assert backend.to_numpy(resolved.accepted_quantity).tolist() == expected.accepted_quantity.tolist()
    assert backend.to_numpy(resolved.stock).tolist() == expected.stock.tolist()
    assert backend.to_numpy(resolved.need).tolist() == expected.need.tolist()


@pytest.mark.skipif(not _cuda_available(), reason="CUDA backend unavailable")
def test_cuda_backend_commit_contract_matches_reference_semantics() -> None:
    expected = commit_resolved_trades(**_sample_commit_inputs())
    inputs = _sample_commit_inputs()
    backend = create_backend("cuda")
    device_inputs = {
        key: backend.asarray(value) if isinstance(value, np.ndarray) else value
        for key, value in inputs.items()
    }

    committed = backend.commit_resolved_trades(**device_inputs)

    assert backend.to_numpy(committed.stock).tolist() == expected.stock.tolist()
    assert backend.to_numpy(committed.need).tolist() == expected.need.tolist()
    assert backend.to_numpy(committed.recent_sales).tolist() == expected.recent_sales.tolist()
    assert backend.to_numpy(committed.recent_purchases).tolist() == expected.recent_purchases.tolist()
    assert backend.to_numpy(committed.friend_id).tolist() == expected.friend_id.tolist()
    assert backend.to_numpy(committed.friend_activity).tolist() == expected.friend_activity.tolist()
    assert np.allclose(backend.to_numpy(committed.transparency), expected.transparency)
