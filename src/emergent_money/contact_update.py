from __future__ import annotations

import numpy as np


def plan_contact_candidates(*, friend_id: np.ndarray, seed: int, cycle: int) -> np.ndarray:
    friend_id = friend_id.astype(np.int32, copy=False)
    population, acquaintances = friend_id.shape
    candidate_ids = np.full((population,), -1, dtype=np.int32)
    if population <= 1:
        return candidate_ids

    max_attempts = max(2 * acquaintances, 8)
    for agent_id in range(population):
        friend_row = friend_id[agent_id]
        for attempt in range(max_attempts):
            raw_candidate = _sample_contact_candidate(
                seed=seed,
                cycle=cycle,
                agent_id=agent_id,
                attempt=attempt,
                population=population,
            )
            if not np.any(friend_row == raw_candidate):
                candidate_ids[agent_id] = raw_candidate
                break

        if candidate_ids[agent_id] >= 0:
            continue

        for candidate in range(population):
            if candidate == agent_id:
                continue
            if not np.any(friend_row == candidate):
                candidate_ids[agent_id] = candidate
                break

    return candidate_ids


def apply_contact_candidates_in_place(
    *,
    friend_id: np.ndarray,
    friend_activity: np.ndarray,
    transparency: np.ndarray,
    candidate_ids: np.ndarray,
    initial_activity: float,
    initial_transparency: float,
) -> None:
    candidate_ids = candidate_ids.astype(np.int32, copy=False)

    for agent_id, candidate_id in enumerate(candidate_ids):
        if candidate_id < 0:
            continue

        existing_slot = _find_friend_slot(friend_id[agent_id], int(candidate_id))
        if existing_slot >= 0:
            friend_activity[agent_id, existing_slot] = max(friend_activity[agent_id, existing_slot], initial_activity)
            continue

        target_slot = _select_friend_slot(friend_id[agent_id], friend_activity[agent_id])
        if target_slot < 0:
            continue

        friend_id[agent_id, target_slot] = candidate_id
        friend_activity[agent_id, target_slot] = initial_activity
        transparency[agent_id, target_slot, :] = initial_transparency


def _sample_contact_candidate(*, seed: int, cycle: int, agent_id: int, attempt: int, population: int) -> int:
    if population <= 1:
        return -1

    mask = 0xFFFFFFFF
    mixed = seed & mask
    mixed ^= ((cycle + 1) * 0x9E3779B9) & mask
    mixed ^= ((agent_id + 1) * 0x85EBCA6B) & mask
    mixed ^= ((attempt + 1) * 0xC2B2AE35) & mask
    mixed = _mix_u32(mixed)
    candidate = mixed % (population - 1)
    if candidate >= agent_id:
        candidate += 1
    return int(candidate)


def _mix_u32(value: int) -> int:
    mask = 0xFFFFFFFF
    x = value & mask
    x ^= x >> 16
    x = (x * 0x7FEB352D) & mask
    x ^= x >> 15
    x = (x * 0x846CA68B) & mask
    x ^= x >> 16
    return x & mask


def _find_friend_slot(friend_row: np.ndarray, agent_id: int) -> int:
    matches = np.nonzero(friend_row == agent_id)[0]
    if matches.size == 0:
        return -1
    return int(matches[0])


def _select_friend_slot(friend_row: np.ndarray, activity_row: np.ndarray) -> int:
    empty_slots = np.flatnonzero(friend_row < 0)
    if empty_slots.size > 0:
        return int(empty_slots[0])
    return int(np.argmin(activity_row))