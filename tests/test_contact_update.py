from __future__ import annotations

import numpy as np
import pytest

from emergent_money.backend import create_backend
from emergent_money.contact_update import apply_contact_candidates_in_place, plan_contact_candidates


def _sample_contact_inputs() -> dict[str, np.ndarray | float]:
    return {
        "friend_id": np.array(
            [
                [2, -1, -1],
                [2, 3, 4],
                [1, 2, -1],
            ],
            dtype=np.int32,
        ),
        "friend_activity": np.array(
            [
                [0.5, 0.0, 0.0],
                [5.0, 1.0, 2.0],
                [3.0, 4.0, 0.0],
            ],
            dtype=np.float32,
        ),
        "transparency": np.array(
            [
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
                [[0.9, 0.8, 0.7], [0.6, 0.5, 0.4], [0.3, 0.2, 0.1]],
                [[0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.9, 1.0]],
            ],
            dtype=np.float32,
        ),
        "candidate_ids": np.array([2, 5, 4], dtype=np.int32),
        "initial_activity": 2.0,
        "initial_transparency": 0.7,
    }


def _sample_planning_inputs() -> dict[str, np.ndarray | int]:
    return {
        "friend_id": np.array(
            [
                [2, -1, -1],
                [0, 2, 3],
                [1, -1, -1],
                [1, 0, -1],
            ],
            dtype=np.int32,
        ),
        "seed": 17,
        "cycle": 3,
    }


def _cuda_available() -> bool:
    try:
        create_backend("cuda")
    except Exception:
        return False
    return True


def test_reference_contact_planning_avoids_self_and_known_contacts() -> None:
    inputs = _sample_planning_inputs()
    candidate_ids = plan_contact_candidates(**inputs)

    assert candidate_ids.tolist()[1] == -1
    for agent_id, candidate_id in enumerate(candidate_ids.tolist()):
        if candidate_id < 0:
            continue
        assert candidate_id != agent_id
        assert candidate_id not in inputs["friend_id"][agent_id].tolist()


def test_numpy_backend_contact_planning_matches_reference() -> None:
    inputs = _sample_planning_inputs()
    expected = plan_contact_candidates(**inputs)
    backend = create_backend("numpy")

    candidate_ids = backend.plan_contact_candidates(**inputs)

    assert np.array_equal(candidate_ids, expected)


@pytest.mark.skipif(not _cuda_available(), reason="CUDA backend unavailable")
def test_cuda_backend_contact_planning_matches_reference() -> None:
    inputs = _sample_planning_inputs()
    expected = plan_contact_candidates(**inputs)
    backend = create_backend("cuda")

    candidate_ids = backend.plan_contact_candidates(
        friend_id=backend.asarray(inputs["friend_id"]),
        seed=int(inputs["seed"]),
        cycle=int(inputs["cycle"]),
    )
    backend.synchronize()

    assert np.array_equal(backend.to_numpy(candidate_ids), expected)


def test_reference_contact_update_mutates_expected_slots() -> None:
    inputs = _sample_contact_inputs()
    friend_id = inputs["friend_id"].copy()
    friend_activity = inputs["friend_activity"].copy()
    transparency = inputs["transparency"].copy()

    apply_contact_candidates_in_place(
        friend_id=friend_id,
        friend_activity=friend_activity,
        transparency=transparency,
        candidate_ids=inputs["candidate_ids"],
        initial_activity=float(inputs["initial_activity"]),
        initial_transparency=float(inputs["initial_transparency"]),
    )

    assert friend_id.tolist() == [[2, -1, -1], [2, 5, 4], [1, 2, 4]]
    assert friend_activity.tolist() == [[2.0, 0.0, 0.0], [5.0, 2.0, 2.0], [3.0, 4.0, 2.0]]
    assert np.allclose(transparency[0, 0], np.array([0.1, 0.2, 0.3], dtype=np.float32))
    assert np.allclose(transparency[1, 1], np.array([0.7, 0.7, 0.7], dtype=np.float32))
    assert np.allclose(transparency[2, 2], np.array([0.7, 0.7, 0.7], dtype=np.float32))


def test_numpy_backend_contact_update_matches_reference() -> None:
    inputs = _sample_contact_inputs()
    expected_friend_id = inputs["friend_id"].copy()
    expected_friend_activity = inputs["friend_activity"].copy()
    expected_transparency = inputs["transparency"].copy()
    apply_contact_candidates_in_place(
        friend_id=expected_friend_id,
        friend_activity=expected_friend_activity,
        transparency=expected_transparency,
        candidate_ids=inputs["candidate_ids"],
        initial_activity=float(inputs["initial_activity"]),
        initial_transparency=float(inputs["initial_transparency"]),
    )

    backend = create_backend("numpy")
    friend_id = inputs["friend_id"].copy()
    friend_activity = inputs["friend_activity"].copy()
    transparency = inputs["transparency"].copy()
    backend.apply_contact_candidates(
        friend_id=friend_id,
        friend_activity=friend_activity,
        transparency=transparency,
        candidate_ids=inputs["candidate_ids"],
        initial_activity=float(inputs["initial_activity"]),
        initial_transparency=float(inputs["initial_transparency"]),
    )

    assert np.array_equal(friend_id, expected_friend_id)
    assert np.allclose(friend_activity, expected_friend_activity)
    assert np.allclose(transparency, expected_transparency)


@pytest.mark.skipif(not _cuda_available(), reason="CUDA backend unavailable")
def test_cuda_backend_contact_update_matches_reference() -> None:
    inputs = _sample_contact_inputs()
    expected_friend_id = inputs["friend_id"].copy()
    expected_friend_activity = inputs["friend_activity"].copy()
    expected_transparency = inputs["transparency"].copy()
    apply_contact_candidates_in_place(
        friend_id=expected_friend_id,
        friend_activity=expected_friend_activity,
        transparency=expected_transparency,
        candidate_ids=inputs["candidate_ids"],
        initial_activity=float(inputs["initial_activity"]),
        initial_transparency=float(inputs["initial_transparency"]),
    )

    backend = create_backend("cuda")
    friend_id = backend.asarray(inputs["friend_id"])
    friend_activity = backend.asarray(inputs["friend_activity"])
    transparency = backend.asarray(inputs["transparency"])
    candidate_ids = backend.asarray(inputs["candidate_ids"])
    backend.apply_contact_candidates(
        friend_id=friend_id,
        friend_activity=friend_activity,
        transparency=transparency,
        candidate_ids=candidate_ids,
        initial_activity=float(inputs["initial_activity"]),
        initial_transparency=float(inputs["initial_transparency"]),
    )
    backend.synchronize()

    assert np.array_equal(backend.to_numpy(friend_id), expected_friend_id)
    assert np.allclose(backend.to_numpy(friend_activity), expected_friend_activity)
    assert np.allclose(backend.to_numpy(transparency), expected_transparency)