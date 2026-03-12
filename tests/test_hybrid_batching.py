from __future__ import annotations

import pytest

from emergent_money.hybrid_batching import (
    NegotiationCandidate,
    batch_is_conflict_free,
    schedule_conflict_free_batches,
)


def _signature(plan) -> tuple[tuple[tuple[tuple[int, int], ...], ...], tuple[tuple[int, int], ...]]:
    return (
        tuple(
            tuple((candidate.proposer_id, candidate.partner_id) for candidate in batch)
            for batch in plan.batches
        ),
        tuple((candidate.proposer_id, candidate.partner_id) for candidate in plan.dropped),
    )


def test_schedule_conflict_free_batches_rejects_non_positive_batch_count() -> None:
    with pytest.raises(ValueError):
        schedule_conflict_free_batches([], batch_count=0, seed=1)


def test_schedule_conflict_free_batches_returns_only_conflict_free_batches() -> None:
    candidates = [
        NegotiationCandidate(0, 1, 10.0),
        NegotiationCandidate(1, 2, 9.0),
        NegotiationCandidate(3, 4, 8.0),
        NegotiationCandidate(4, 5, 7.0),
        NegotiationCandidate(6, 7, 6.0),
    ]

    plan = schedule_conflict_free_batches(candidates, batch_count=2, seed=17)

    assert 1 <= len(plan.batches) <= 2
    assert plan.scheduled_count + len(plan.dropped) == len(candidates)
    assert all(batch_is_conflict_free(batch) for batch in plan.batches)


def test_schedule_conflict_free_batches_is_seed_deterministic() -> None:
    candidates = [
        NegotiationCandidate(0, 1, 10.0),
        NegotiationCandidate(1, 2, 9.0),
        NegotiationCandidate(3, 4, 8.0),
        NegotiationCandidate(4, 5, 7.0),
        NegotiationCandidate(6, 7, 6.0),
    ]

    plan_a = schedule_conflict_free_batches(candidates, batch_count=2, seed=17)
    plan_b = schedule_conflict_free_batches(candidates, batch_count=2, seed=17)

    assert _signature(plan_a) == _signature(plan_b)


def test_schedule_conflict_free_batches_respects_priority_before_tie_breaking() -> None:
    candidates = [
        NegotiationCandidate(0, 1, 10.0),
        NegotiationCandidate(0, 2, 5.0),
        NegotiationCandidate(3, 4, 1.0),
    ]

    plan = schedule_conflict_free_batches(candidates, batch_count=1, seed=3)
    scheduled = {pair for batch in _signature(plan)[0] for pair in batch}
    dropped = set(_signature(plan)[1])

    assert (0, 1) in scheduled
    assert (0, 2) in dropped
    assert (3, 4) in scheduled


def test_schedule_conflict_free_batches_reseeds_tie_breaks_across_cycles() -> None:
    candidates = [
        NegotiationCandidate(0, 1, 1.0),
        NegotiationCandidate(0, 2, 1.0),
        NegotiationCandidate(0, 3, 1.0),
        NegotiationCandidate(0, 4, 1.0),
        NegotiationCandidate(0, 5, 1.0),
        NegotiationCandidate(0, 6, 1.0),
    ]

    plan_a = schedule_conflict_free_batches(candidates, batch_count=1, seed=11)
    plan_b = schedule_conflict_free_batches(candidates, batch_count=1, seed=29)

    assert _signature(plan_a) != _signature(plan_b)


def test_schedule_conflict_free_batches_never_reuses_agents_across_batches() -> None:
    candidates = [
        NegotiationCandidate(0, 1, 10.0),
        NegotiationCandidate(2, 3, 9.0),
        NegotiationCandidate(0, 4, 8.0),
        NegotiationCandidate(5, 6, 7.0),
    ]

    plan = schedule_conflict_free_batches(candidates, batch_count=2, seed=17)

    seen_agents: set[int] = set()
    for batch in plan.batches:
        for candidate in batch:
            proposer_id, partner_id = candidate.participants()
            assert proposer_id not in seen_agents
            assert partner_id not in seen_agents
            seen_agents.add(proposer_id)
            seen_agents.add(partner_id)



def test_schedule_conflict_free_batches_can_preserve_input_order_over_priority() -> None:
    candidates = [
        NegotiationCandidate(0, 2, 1.0),
        NegotiationCandidate(0, 1, 10.0),
        NegotiationCandidate(3, 4, 5.0),
    ]

    plan = schedule_conflict_free_batches(candidates, batch_count=1, seed=17, preserve_input_order=True)
    scheduled = {pair for batch in _signature(plan)[0] for pair in batch}
    dropped = set(_signature(plan)[1])

    assert (0, 2) in scheduled
    assert (0, 1) in dropped
    assert (3, 4) in scheduled
