from __future__ import annotations

import numpy as np

from .backend.base import BaseBackend
from .config import SimulationConfig
from .state import SimulationState, TradeBuffers


def create_initial_state(config: SimulationConfig, backend: BaseBackend) -> SimulationState:
    rng = np.random.default_rng(config.seed)
    base_need_vector = config.base_need_vector()
    base_need = np.broadcast_to(base_need_vector, config.agent_good_shape).copy()

    talent_mask_bool = rng.random(config.agent_good_shape) < config.talent_probability
    innate_bonus = talent_mask_bool.astype(np.float32) * config.gifted_efficiency_bonus

    innate_efficiency = np.full(config.agent_good_shape, config.initial_efficiency, dtype=np.float32) + innate_bonus
    learned_efficiency = np.full(config.agent_good_shape, config.initial_efficiency, dtype=np.float32)
    efficiency = np.maximum(innate_efficiency, learned_efficiency)
    production_cost = (1.0 / np.maximum(innate_efficiency, 1e-6)).astype(np.float32)

    stock_limit = base_need * config.stock_limit_multiplier
    # Paper-level starting point: everyone has a small stock of everything,
    # while talent gives some agents slightly more initial room for exchange.
    stock = (base_need * config.initial_stock_fraction).astype(np.float32)
    stock += np.maximum(
        (innate_efficiency - config.initial_efficiency) * base_need * config.initial_stock_fraction,
        0.0,
    ).astype(np.float32)

    friend_id = np.full(config.friend_shape, -1, dtype=np.int32)
    friend_activity = np.zeros(config.friend_shape, dtype=np.float32)
    transparency = np.full(config.transparency_shape, config.initial_transparency, dtype=np.float32)

    cycle_time_budget = np.full((config.population,), config.cycle_time_budget, dtype=np.float32)
    time_remaining = cycle_time_budget.copy()

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
        need=backend.asarray(base_need.copy(), dtype=np.float32),
        stock=backend.asarray(stock, dtype=np.float32),
        stock_limit=backend.asarray(stock_limit, dtype=np.float32),
        innate_efficiency=backend.asarray(innate_efficiency, dtype=np.float32),
        learned_efficiency=backend.asarray(learned_efficiency, dtype=np.float32),
        efficiency=backend.asarray(efficiency, dtype=np.float32),
        purchase_price=backend.asarray(production_cost.copy(), dtype=np.float32),
        sales_price=backend.asarray(production_cost.copy(), dtype=np.float32),
        recent_production=backend.zeros(config.agent_good_shape, dtype=np.float32),
        recent_sales=backend.zeros(config.agent_good_shape, dtype=np.float32),
        recent_purchases=backend.zeros(config.agent_good_shape, dtype=np.float32),
        talent_mask=backend.asarray(talent_mask_bool.astype(np.float32), dtype=np.float32),
        friend_id=backend.asarray(friend_id, dtype=np.int32),
        friend_activity=backend.asarray(friend_activity, dtype=np.float32),
        transparency=backend.asarray(transparency, dtype=np.float32),
        cycle_time_budget=backend.asarray(cycle_time_budget, dtype=np.float32),
        time_remaining=backend.asarray(time_remaining, dtype=np.float32),
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
    )
