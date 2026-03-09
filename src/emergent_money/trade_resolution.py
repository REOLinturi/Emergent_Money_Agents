from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True, frozen=True)
class ResolvedTrades:
    accepted_mask: Any
    accepted_quantity: Any


@dataclass(slots=True, frozen=True)
class CommittedTradeState:
    stock: Any
    need: Any
    recent_sales: Any
    recent_purchases: Any
    friend_id: Any
    friend_activity: Any
    transparency: Any


def resolve_trade_proposals(
    *,
    stock: np.ndarray,
    need: np.ndarray,
    stock_limit: np.ndarray,
    target_agent: np.ndarray,
    need_good: np.ndarray,
    offer_good: np.ndarray,
    quantity: np.ndarray,
    score: np.ndarray,
) -> ResolvedTrades:
    proposal_count = score.shape[0]
    accepted_mask = np.zeros((proposal_count,), dtype=np.bool_)
    accepted_quantity = np.zeros((proposal_count,), dtype=np.float32)

    available_stock = stock.astype(np.float32, copy=True)
    remaining_need = need.astype(np.float32, copy=True)

    for proposer in np.argsort(-score, kind="stable"):
        current_score = float(score[proposer])
        if current_score <= 0.0:
            break

        target = int(target_agent[proposer])
        wanted_good = int(need_good[proposer])
        offered_good = int(offer_good[proposer])
        proposed_quantity = float(quantity[proposer])

        if target < 0 or wanted_good < 0 or offered_good < 0 or proposer == target:
            continue
        if proposed_quantity <= 0.0 or wanted_good == offered_good:
            continue

        proposer_supply = float(available_stock[proposer, offered_good])
        target_supply = float(available_stock[target, wanted_good])
        proposer_need = float(remaining_need[proposer, wanted_good])
        proposer_stock_room = max(float(stock_limit[proposer, wanted_good] - available_stock[proposer, wanted_good]), 0.0)
        proposer_interest = proposer_need + proposer_stock_room
        target_need = float(remaining_need[target, offered_good])
        target_stock_room = max(float(stock_limit[target, offered_good] - available_stock[target, offered_good]), 0.0)
        target_interest = target_need + target_stock_room

        executable_quantity = min(
            proposed_quantity,
            proposer_supply,
            target_supply,
            proposer_interest,
            target_interest,
        )
        if executable_quantity <= 0.0:
            continue

        accepted_mask[proposer] = True
        accepted_quantity[proposer] = executable_quantity

        available_stock[proposer, offered_good] -= executable_quantity
        available_stock[target, wanted_good] -= executable_quantity

        proposer_consumed = min(float(remaining_need[proposer, wanted_good]), executable_quantity)
        remaining_need[proposer, wanted_good] -= proposer_consumed
        proposer_leftover = executable_quantity - proposer_consumed
        if proposer_leftover > 0.0:
            available_stock[proposer, wanted_good] += proposer_leftover

        target_consumed = min(float(remaining_need[target, offered_good]), executable_quantity)
        remaining_need[target, offered_good] -= target_consumed
        target_leftover = executable_quantity - target_consumed
        if target_leftover > 0.0:
            available_stock[target, offered_good] += target_leftover

    return ResolvedTrades(
        accepted_mask=accepted_mask,
        accepted_quantity=accepted_quantity,
    )


def commit_resolved_trades(
    *,
    stock: np.ndarray,
    need: np.ndarray,
    recent_sales: np.ndarray,
    recent_purchases: np.ndarray,
    friend_id: np.ndarray,
    friend_activity: np.ndarray,
    transparency: np.ndarray,
    proposal_friend_slot: np.ndarray,
    proposal_target_agent: np.ndarray,
    proposal_need_good: np.ndarray,
    proposal_offer_good: np.ndarray,
    accepted_mask: np.ndarray,
    accepted_quantity: np.ndarray,
    initial_transparency: float,
) -> CommittedTradeState:
    updated_stock = stock.astype(np.float32, copy=True)
    updated_need = need.astype(np.float32, copy=True)
    updated_recent_sales = recent_sales.astype(np.float32, copy=True)
    updated_recent_purchases = recent_purchases.astype(np.float32, copy=True)
    updated_friend_id = friend_id.astype(np.int32, copy=True)
    updated_friend_activity = friend_activity.astype(np.float32, copy=True)
    updated_transparency = transparency.astype(np.float32, copy=True)

    accepted_mask = accepted_mask.astype(np.bool_, copy=False)
    accepted_quantity = accepted_quantity.astype(np.float32, copy=False)
    proposal_friend_slot = proposal_friend_slot.astype(np.int32, copy=False)
    proposal_target_agent = proposal_target_agent.astype(np.int32, copy=False)
    proposal_need_good = proposal_need_good.astype(np.int32, copy=False)
    proposal_offer_good = proposal_offer_good.astype(np.int32, copy=False)

    for proposer in np.flatnonzero(accepted_mask):
        target = int(proposal_target_agent[proposer])
        friend_slot = int(proposal_friend_slot[proposer])
        need_good = int(proposal_need_good[proposer])
        offer_good = int(proposal_offer_good[proposer])
        quantity = float(accepted_quantity[proposer])

        if quantity <= 0.0:
            continue

        updated_stock[proposer, offer_good] -= quantity
        updated_stock[target, need_good] -= quantity

        proposer_consumed = min(float(updated_need[proposer, need_good]), quantity)
        updated_need[proposer, need_good] -= proposer_consumed
        updated_stock[proposer, need_good] += quantity - proposer_consumed

        target_consumed = min(float(updated_need[target, offer_good]), quantity)
        updated_need[target, offer_good] -= target_consumed
        updated_stock[target, offer_good] += quantity - target_consumed

        updated_recent_sales[proposer, offer_good] += quantity
        updated_recent_purchases[proposer, need_good] += quantity
        updated_recent_sales[target, need_good] += quantity
        updated_recent_purchases[target, offer_good] += quantity

        updated_friend_activity[proposer, friend_slot] += quantity
        transparency_gain = min(0.05, 0.01 * np.log1p(quantity))
        updated_transparency[proposer, friend_slot, need_good] = min(
            1.0,
            updated_transparency[proposer, friend_slot, need_good] + transparency_gain,
        )

        reciprocal_slot = _find_friend_slot(updated_friend_id[target], proposer)
        if reciprocal_slot < 0:
            reciprocal_slot = _select_friend_slot(updated_friend_id[target], updated_friend_activity[target])
            updated_friend_id[target, reciprocal_slot] = proposer
            updated_friend_activity[target, reciprocal_slot] = 2.0 + quantity
            updated_transparency[target, reciprocal_slot, :] = initial_transparency

        updated_friend_activity[target, reciprocal_slot] += quantity
        updated_transparency[target, reciprocal_slot, offer_good] = min(
            1.0,
            updated_transparency[target, reciprocal_slot, offer_good] + transparency_gain,
        )

    updated_stock = np.maximum(updated_stock, 0.0)
    updated_need = np.maximum(updated_need, 0.0)

    return CommittedTradeState(
        stock=updated_stock,
        need=updated_need,
        recent_sales=updated_recent_sales,
        recent_purchases=updated_recent_purchases,
        friend_id=updated_friend_id,
        friend_activity=updated_friend_activity,
        transparency=updated_transparency,
    )


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