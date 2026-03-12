from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable

import numpy as np


@dataclass(slots=True, frozen=True)
class NegotiationCandidate:
    proposer_id: int
    partner_id: int
    priority: float

    def __post_init__(self) -> None:
        if self.proposer_id < 0:
            raise ValueError("proposer_id must be non-negative")
        if self.partner_id < 0:
            raise ValueError("partner_id must be non-negative")
        if self.proposer_id == self.partner_id:
            raise ValueError("partner_id must differ from proposer_id")
        if not isfinite(self.priority):
            raise ValueError("priority must be finite")

    def participants(self) -> tuple[int, int]:
        return (self.proposer_id, self.partner_id)


@dataclass(slots=True, frozen=True)
class ConflictFreeBatchPlan:
    seed: int
    batches: tuple[tuple[NegotiationCandidate, ...], ...]
    dropped: tuple[NegotiationCandidate, ...]

    @property
    def scheduled_count(self) -> int:
        return sum(len(batch) for batch in self.batches)


def batch_is_conflict_free(batch: Iterable[NegotiationCandidate]) -> bool:
    seen_agents: set[int] = set()
    for candidate in batch:
        proposer_id, partner_id = candidate.participants()
        if proposer_id in seen_agents or partner_id in seen_agents:
            return False
        seen_agents.add(proposer_id)
        seen_agents.add(partner_id)
    return True


def schedule_conflict_free_batches(
    candidates: Iterable[NegotiationCandidate],
    *,
    batch_count: int,
    seed: int,
    preserve_input_order: bool = False,
) -> ConflictFreeBatchPlan:
    if batch_count <= 0:
        raise ValueError("batch_count must be positive")

    frozen_candidates = tuple(candidates)
    if not frozen_candidates:
        return ConflictFreeBatchPlan(seed=seed, batches=(), dropped=())

    rng = np.random.default_rng(seed)
    if preserve_input_order:
        ranked_candidate_items = tuple(candidate for candidate in frozen_candidates)
    else:
        mutable_ranked_candidates: list[tuple[float, float, int, NegotiationCandidate]] = []
        for index, candidate in enumerate(frozen_candidates):
            mutable_ranked_candidates.append((-candidate.priority, float(rng.random()), index, candidate))
        mutable_ranked_candidates.sort()
        ranked_candidate_items = tuple(candidate for _, _, _, candidate in mutable_ranked_candidates)

    batch_agents = [set() for _ in range(batch_count)]
    globally_reserved_agents: set[int] = set()
    mutable_batches: list[list[NegotiationCandidate]] = [[] for _ in range(batch_count)]
    dropped: list[NegotiationCandidate] = []

    for candidate in ranked_candidate_items:
        proposer_id, partner_id = candidate.participants()
        if proposer_id in globally_reserved_agents or partner_id in globally_reserved_agents:
            dropped.append(candidate)
            continue

        batch_order = rng.permutation(batch_count)
        for batch_id_raw in batch_order:
            batch_id = int(batch_id_raw)
            participants = batch_agents[batch_id]
            if proposer_id in participants or partner_id in participants:
                continue
            participants.add(proposer_id)
            participants.add(partner_id)
            globally_reserved_agents.add(proposer_id)
            globally_reserved_agents.add(partner_id)
            mutable_batches[batch_id].append(candidate)
            break
        else:
            dropped.append(candidate)

    non_empty_batches = tuple(tuple(batch) for batch in mutable_batches if batch)
    return ConflictFreeBatchPlan(
        seed=seed,
        batches=non_empty_batches,
        dropped=tuple(dropped),
    )
