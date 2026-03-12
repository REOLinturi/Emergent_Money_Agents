from __future__ import annotations

from dataclasses import dataclass
from typing import Any

ArrayLike = Any

ROLE_CONSUMER = 10
ROLE_RETAILER = 11
ROLE_PRODUCER = 12


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
class MarketState:
    elastic_need: ArrayLike
    previous_elastic_need: ArrayLike
    average_price: ArrayLike
    recent_production: ArrayLike
    produced_this_period: ArrayLike
    periodic_tce_cost: ArrayLike
    periodic_spoilage: ArrayLike
    cost_of_tce_in_time: ArrayLike
    cost_of_spoilage_in_time: ArrayLike
    consumer_count: ArrayLike
    retailer_count: ArrayLike
    producer_count: ArrayLike
    price_average: float
    total_cost_of_tce_in_time: float
    total_cost_of_spoilage_in_time: float
    total_stock_previous: float
    losers: int


@dataclass(slots=True)
class SimulationState:
    base_need: ArrayLike
    need: ArrayLike
    stock: ArrayLike
    stock_limit: ArrayLike
    previous_stock_limit: ArrayLike
    innate_efficiency: ArrayLike
    learned_efficiency: ArrayLike
    efficiency: ArrayLike
    purchase_price: ArrayLike
    sales_price: ArrayLike
    purchase_times: ArrayLike
    sales_times: ArrayLike
    sum_period_purchase_value: ArrayLike
    sum_period_sales_value: ArrayLike
    recent_production: ArrayLike
    produced_this_period: ArrayLike
    produced_last_period: ArrayLike
    recent_sales: ArrayLike
    sold_this_period: ArrayLike
    sold_last_period: ArrayLike
    recent_purchases: ArrayLike
    purchased_this_period: ArrayLike
    purchased_last_period: ArrayLike
    recent_inventory_inflow: ArrayLike
    spoilage: ArrayLike
    periodic_spoilage: ArrayLike
    talent_mask: ArrayLike
    role: ArrayLike
    friend_id: ArrayLike
    friend_activity: ArrayLike
    friend_purchased: ArrayLike
    friend_sold: ArrayLike
    transparency: ArrayLike
    cycle_time_budget: ArrayLike
    time_remaining: ArrayLike
    period_time_debt: ArrayLike
    period_failure: ArrayLike
    timeout: ArrayLike
    needs_level: ArrayLike
    recent_needs_increment: ArrayLike
    trade: TradeBuffers
    market: MarketState
