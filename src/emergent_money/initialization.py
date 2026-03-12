from __future__ import annotations

import numpy as np

from .backend.base import BaseBackend
from .config import SimulationConfig
from .state import MarketState, ROLE_CONSUMER, SimulationState, TradeBuffers


def _gifted_mask(config: SimulationConfig, rng: np.random.Generator) -> np.ndarray:
    gifted = np.zeros(config.agent_good_shape, dtype=np.float32)
    gifted_count = min(config.population, max(0, config.gifted_count()))
    if gifted_count == 0:
        return gifted

    population_index = np.arange(config.population, dtype=np.int32)
    for good_id in range(config.goods):
        selected = rng.choice(population_index, size=gifted_count, replace=False)
        gifted[selected, good_id] = 1.0
    return gifted


def create_initial_state(config: SimulationConfig, backend: BaseBackend) -> SimulationState:
    rng = np.random.default_rng(config.seed)
    base_need_vector = config.base_need_vector()
    base_need = np.broadcast_to(base_need_vector, config.agent_good_shape).copy()

    talent_mask = _gifted_mask(config, rng)
    innate_efficiency = np.full(config.agent_good_shape, config.initial_efficiency, dtype=np.float32)
    innate_efficiency += talent_mask * config.gifted_efficiency_bonus
    learned_efficiency = np.full(config.agent_good_shape, config.initial_efficiency, dtype=np.float32)
    efficiency = np.maximum(innate_efficiency, learned_efficiency)

    stock_limit = (base_need * config.stock_limit_multiplier).astype(np.float32)
    previous_stock_limit = stock_limit.copy()
    stock = (base_need * config.initial_stock_fraction).astype(np.float32)

    friend_id = np.full(config.friend_shape, -1, dtype=np.int32)
    friend_activity = np.zeros(config.friend_shape, dtype=np.float32)
    friend_purchased = np.zeros(config.transparency_shape, dtype=np.float32)
    friend_sold = np.zeros(config.transparency_shape, dtype=np.float32)
    transparency = np.full(config.transparency_shape, config.initial_transparency, dtype=np.float32)

    cycle_time_budget = np.full((config.population,), config.cycle_time_budget, dtype=np.float32)
    time_remaining = cycle_time_budget.copy()

    initial_price = np.full(config.agent_good_shape, config.initial_price, dtype=np.float32)
    need = base_need.copy()
    recent_production = (base_need * config.history).astype(np.float32)
    produced_this_period = base_need.copy().astype(np.float32)
    produced_last_period = base_need.copy().astype(np.float32)
    recent_sales = np.zeros(config.agent_good_shape, dtype=np.float32)
    sold_this_period = np.zeros(config.agent_good_shape, dtype=np.float32)
    sold_last_period = np.zeros(config.agent_good_shape, dtype=np.float32)
    recent_purchases = np.zeros(config.agent_good_shape, dtype=np.float32)
    purchased_this_period = np.zeros(config.agent_good_shape, dtype=np.float32)
    purchased_last_period = np.zeros(config.agent_good_shape, dtype=np.float32)
    recent_inventory_inflow = np.zeros(config.agent_good_shape, dtype=np.float32)
    purchase_times = np.zeros(config.agent_good_shape, dtype=np.int32)
    sales_times = np.zeros(config.agent_good_shape, dtype=np.int32)
    sum_period_purchase_value = np.full(config.agent_good_shape, config.initial_price, dtype=np.float32)
    sum_period_sales_value = np.full(config.agent_good_shape, config.initial_price, dtype=np.float32)
    spoilage = np.zeros(config.agent_good_shape, dtype=np.float32)
    periodic_spoilage = np.zeros((config.population,), dtype=np.float32)
    role = np.full(config.agent_good_shape, ROLE_CONSUMER, dtype=np.int32)
    period_time_debt = np.zeros((config.population,), dtype=np.float32)
    period_failure = np.zeros((config.population,), dtype=np.bool_)
    timeout = np.zeros((config.population,), dtype=np.int32)
    needs_level = np.ones((config.population,), dtype=np.float32)
    recent_needs_increment = np.ones((config.population,), dtype=np.float32)

    market_recent_production = (base_need_vector * config.history * config.population).astype(np.float32)
    market = MarketState(
        elastic_need=backend.asarray(base_need_vector.copy(), dtype=np.float32),
        previous_elastic_need=backend.asarray(base_need_vector.copy(), dtype=np.float32),
        average_price=backend.asarray(np.full((config.goods,), config.initial_price, dtype=np.float32), dtype=np.float32),
        recent_production=backend.asarray(market_recent_production, dtype=np.float32),
        produced_this_period=backend.asarray(np.zeros((config.goods,), dtype=np.float32), dtype=np.float32),
        periodic_tce_cost=backend.asarray(np.zeros((config.goods,), dtype=np.float32), dtype=np.float32),
        periodic_spoilage=backend.asarray(np.zeros((config.goods,), dtype=np.float32), dtype=np.float32),
        cost_of_tce_in_time=backend.asarray(np.zeros((config.goods,), dtype=np.float32), dtype=np.float32),
        cost_of_spoilage_in_time=backend.asarray(np.zeros((config.goods,), dtype=np.float32), dtype=np.float32),
        consumer_count=backend.asarray(np.zeros((config.goods,), dtype=np.int32), dtype=np.int32),
        retailer_count=backend.asarray(np.zeros((config.goods,), dtype=np.int32), dtype=np.int32),
        producer_count=backend.asarray(np.zeros((config.goods,), dtype=np.int32), dtype=np.int32),
        price_average=float(config.initial_price),
        total_cost_of_tce_in_time=0.0,
        total_cost_of_spoilage_in_time=0.0,
        total_stock_previous=float(stock.sum()),
        losers=0,
    )

    trade = TradeBuffers(
        active_friend_slot=np.full(config.active_friend_shape, -1, dtype=np.int32),
        active_friend_id=np.full(config.active_friend_shape, -1, dtype=np.int32),
        candidate_need_good=np.zeros(config.demand_candidate_shape, dtype=np.int32),
        candidate_offer_good=np.zeros(config.supply_candidate_shape, dtype=np.int32),
        proposal_friend_slot=np.full((config.population,), -1, dtype=np.int32),
        proposal_target_agent=np.full((config.population,), -1, dtype=np.int32),
        proposal_need_good=np.full((config.population,), -1, dtype=np.int32),
        proposal_offer_good=np.full((config.population,), -1, dtype=np.int32),
        proposal_quantity=np.zeros((config.population,), dtype=np.float32),
        proposal_score=np.zeros((config.population,), dtype=np.float32),
        accepted_mask=np.zeros((config.population,), dtype=np.bool_),
        accepted_quantity=np.zeros((config.population,), dtype=np.float32),
    )

    return SimulationState(
        base_need=backend.asarray(base_need, dtype=np.float32),
        need=backend.asarray(need, dtype=np.float32),
        stock=backend.asarray(stock, dtype=np.float32),
        stock_limit=backend.asarray(stock_limit, dtype=np.float32),
        previous_stock_limit=backend.asarray(previous_stock_limit, dtype=np.float32),
        innate_efficiency=backend.asarray(innate_efficiency, dtype=np.float32),
        learned_efficiency=backend.asarray(learned_efficiency, dtype=np.float32),
        efficiency=backend.asarray(efficiency, dtype=np.float32),
        purchase_price=backend.asarray(initial_price.copy(), dtype=np.float32),
        sales_price=backend.asarray(initial_price.copy(), dtype=np.float32),
        purchase_times=backend.asarray(purchase_times, dtype=np.int32),
        sales_times=backend.asarray(sales_times, dtype=np.int32),
        sum_period_purchase_value=backend.asarray(sum_period_purchase_value, dtype=np.float32),
        sum_period_sales_value=backend.asarray(sum_period_sales_value, dtype=np.float32),
        recent_production=backend.asarray(recent_production, dtype=np.float32),
        produced_this_period=backend.asarray(produced_this_period, dtype=np.float32),
        produced_last_period=backend.asarray(produced_last_period, dtype=np.float32),
        recent_sales=backend.asarray(recent_sales, dtype=np.float32),
        sold_this_period=backend.asarray(sold_this_period, dtype=np.float32),
        sold_last_period=backend.asarray(sold_last_period, dtype=np.float32),
        recent_purchases=backend.asarray(recent_purchases, dtype=np.float32),
        purchased_this_period=backend.asarray(purchased_this_period, dtype=np.float32),
        purchased_last_period=backend.asarray(purchased_last_period, dtype=np.float32),
        recent_inventory_inflow=backend.asarray(recent_inventory_inflow, dtype=np.float32),
        spoilage=backend.asarray(spoilage, dtype=np.float32),
        periodic_spoilage=backend.asarray(periodic_spoilage, dtype=np.float32),
        talent_mask=backend.asarray(talent_mask, dtype=np.float32),
        role=backend.asarray(role, dtype=np.int32),
        friend_id=backend.asarray(friend_id, dtype=np.int32),
        friend_activity=backend.asarray(friend_activity, dtype=np.float32),
        friend_purchased=backend.asarray(friend_purchased, dtype=np.float32),
        friend_sold=backend.asarray(friend_sold, dtype=np.float32),
        transparency=backend.asarray(transparency, dtype=np.float32),
        cycle_time_budget=backend.asarray(cycle_time_budget, dtype=np.float32),
        time_remaining=backend.asarray(time_remaining, dtype=np.float32),
        period_time_debt=backend.asarray(period_time_debt, dtype=np.float32),
        period_failure=backend.asarray(period_failure, dtype=np.bool_),
        timeout=backend.asarray(timeout, dtype=np.int32),
        needs_level=backend.asarray(needs_level, dtype=np.float32),
        recent_needs_increment=backend.asarray(recent_needs_increment, dtype=np.float32),
        trade=TradeBuffers(
            active_friend_slot=backend.asarray(trade.active_friend_slot, dtype=np.int32),
            active_friend_id=backend.asarray(trade.active_friend_id, dtype=np.int32),
            candidate_need_good=backend.asarray(trade.candidate_need_good, dtype=np.int32),
            candidate_offer_good=backend.asarray(trade.candidate_offer_good, dtype=np.int32),
            proposal_friend_slot=backend.asarray(trade.proposal_friend_slot, dtype=np.int32),
            proposal_target_agent=backend.asarray(trade.proposal_target_agent, dtype=np.int32),
            proposal_need_good=backend.asarray(trade.proposal_need_good, dtype=np.int32),
            proposal_offer_good=backend.asarray(trade.proposal_offer_good, dtype=np.int32),
            proposal_quantity=backend.asarray(trade.proposal_quantity, dtype=np.float32),
            proposal_score=backend.asarray(trade.proposal_score, dtype=np.float32),
            accepted_mask=backend.asarray(trade.accepted_mask, dtype=np.bool_),
            accepted_quantity=backend.asarray(trade.accepted_quantity, dtype=np.float32),
        ),
        market=market,
    )
