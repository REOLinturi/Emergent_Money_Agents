from __future__ import annotations

from dataclasses import dataclass
from typing import Any

ArrayLike = Any


@dataclass(slots=True)
class TradeBuffers:
    active_friend_slot: ArrayLike
    active_friend_id: ArrayLike
    candidate_need_good: ArrayLike
    candidate_offer_good: ArrayLike
    proposal_friend_slot: ArrayLike
    proposal_target_agent: ArrayLike
    proposal_need_good: ArrayLike
    proposal_offer_good: ArrayLike
    proposal_quantity: ArrayLike
    proposal_score: ArrayLike
    accepted_mask: ArrayLike
    accepted_quantity: ArrayLike


@dataclass(slots=True)
class SimulationState:
    base_need: ArrayLike
    need: ArrayLike
    stock: ArrayLike
    stock_limit: ArrayLike
    innate_efficiency: ArrayLike
    learned_efficiency: ArrayLike
    efficiency: ArrayLike
    purchase_price: ArrayLike
    sales_price: ArrayLike
    recent_production: ArrayLike
    recent_sales: ArrayLike
    recent_purchases: ArrayLike
    talent_mask: ArrayLike
    friend_id: ArrayLike
    friend_activity: ArrayLike
    transparency: ArrayLike
    cycle_time_budget: ArrayLike
    time_remaining: ArrayLike
    trade: TradeBuffers
