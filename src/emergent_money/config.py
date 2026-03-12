from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class SimulationConfig:
    population: int = 10_000
    goods: int = 30
    acquaintances: int = 100
    active_acquaintances: int = 24
    demand_candidates: int = 4
    supply_candidates: int = 4
    seed: int = 2009
    initial_efficiency: float = 1.0
    gifted_efficiency_bonus: float = 1.0
    gifted_efficiency_floor: float = 1.5
    talent_probability: float = 0.20
    initial_stock_fraction: float = 1.0
    initial_price: float = 1.0
    initial_transparency: float = 0.70
    stock_limit_multiplier: float = 2.0
    activity_discount: float = 0.80
    spoilage_rate: float = 0.10
    max_leisure_extra_multiplier: float = 1.0
    leisure_stock_trade_bias: float = 0.35
    history: int = 4
    price_demand_elasticity: int = 2
    basic_round_elastic: bool = True
    stock_spoil_threshold: float = 2.0
    price_reduction: float = 0.95
    price_hike: float = 1.05
    price_leap: float = 1.30
    max_rise_in_elastic_need: float = 1.01
    max_drop_in_elastic_need: float = 0.98
    max_stocklimit_decrease: float = 0.95
    max_stocklimit_increase: float = 1.20
    max_efficiency_downgrade: float = 0.98
    max_efficiency_upgrade: float = 1.05
    max_needs_increase: float = 1.50
    max_needs_reduction: float = 0.70
    small_needs_increase: float = 1.05
    small_needs_reduction: float = 0.95
    switch_time: float = 1.0
    min_trade_quantity: float = 0.5
    trade_rounding_buffer: float = 1.0 / 3.0
    cuda_friend_block: int = 12
    cuda_goods_block: int = 25
    experimental_hybrid_batches: int = 0
    experimental_hybrid_frontier_size: int = 0
    experimental_hybrid_seed_stride: int = 9_973
    experimental_hybrid_consumption_stage: bool = False
    experimental_hybrid_surplus_stage: bool = False
    experimental_hybrid_block_frontier_partners: bool = True
    experimental_hybrid_preserve_proposer_order: bool = False
    experimental_hybrid_rolling_frontier: bool = False
    experimental_native_stage_math: bool = False
    experimental_native_exchange_stage: bool = False
    use_exact_legacy_mechanics: bool = True

    def __post_init__(self) -> None:
        if self.population <= 0:
            raise ValueError("population must be positive")
        if self.goods <= 0:
            raise ValueError("goods must be positive")
        if self.acquaintances <= 0:
            raise ValueError("acquaintances must be positive")
        if self.active_acquaintances <= 0 or self.active_acquaintances > self.acquaintances:
            raise ValueError("active_acquaintances must be between 1 and acquaintances")
        if self.demand_candidates <= 0 or self.demand_candidates > self.goods:
            raise ValueError("demand_candidates must be between 1 and goods")
        if self.supply_candidates <= 0 or self.supply_candidates > self.goods:
            raise ValueError("supply_candidates must be between 1 and goods")
        if not 0.0 <= self.talent_probability <= 1.0:
            raise ValueError("talent_probability must be between 0 and 1")
        if self.gifted_efficiency_bonus < 0.0:
            raise ValueError("gifted_efficiency_bonus must be non-negative")
        if self.gifted_efficiency_floor < self.initial_efficiency:
            raise ValueError("gifted_efficiency_floor must be at least initial_efficiency")
        if self.initial_stock_fraction < 0.0:
            raise ValueError("initial_stock_fraction must be non-negative")
        if self.initial_price <= 0.0:
            raise ValueError("initial_price must be positive")
        if not 0.0 <= self.initial_transparency <= 1.0:
            raise ValueError("initial_transparency must be between 0 and 1")
        if not 0.0 <= self.activity_discount <= 1.0:
            raise ValueError("activity_discount must be between 0 and 1")
        if self.initial_efficiency <= 0.0:
            raise ValueError("initial_efficiency must be positive")
        if self.stock_limit_multiplier <= 0.0:
            raise ValueError("stock_limit_multiplier must be positive")
        if not 0.0 <= self.spoilage_rate <= 1.0:
            raise ValueError("spoilage_rate must be between 0 and 1")
        if self.max_leisure_extra_multiplier < 0.0:
            raise ValueError("max_leisure_extra_multiplier must be non-negative")
        if self.leisure_stock_trade_bias < 0.0:
            raise ValueError("leisure_stock_trade_bias must be non-negative")
        if self.history <= 0:
            raise ValueError("history must be positive")
        if self.price_demand_elasticity not in {0, 1, 2}:
            raise ValueError("price_demand_elasticity must be 0, 1, or 2")
        if self.stock_spoil_threshold <= 0.0:
            raise ValueError("stock_spoil_threshold must be positive")
        if self.price_reduction <= 0.0 or self.price_reduction > 1.0:
            raise ValueError("price_reduction must be in (0, 1]")
        if self.price_hike < 1.0:
            raise ValueError("price_hike must be at least 1")
        if self.price_leap < 1.0:
            raise ValueError("price_leap must be at least 1")
        if self.max_rise_in_elastic_need < 1.0:
            raise ValueError("max_rise_in_elastic_need must be at least 1")
        if not 0.0 < self.max_drop_in_elastic_need <= 1.0:
            raise ValueError("max_drop_in_elastic_need must be in (0, 1]")
        if not 0.0 < self.max_stocklimit_decrease <= 1.0:
            raise ValueError("max_stocklimit_decrease must be in (0, 1]")
        if self.max_stocklimit_increase < 1.0:
            raise ValueError("max_stocklimit_increase must be at least 1")
        if not 0.0 < self.max_efficiency_downgrade <= 1.0:
            raise ValueError("max_efficiency_downgrade must be in (0, 1]")
        if self.max_efficiency_upgrade < 1.0:
            raise ValueError("max_efficiency_upgrade must be at least 1")
        if self.max_needs_increase < 1.0:
            raise ValueError("max_needs_increase must be at least 1")
        if not 0.0 < self.max_needs_reduction <= 1.0:
            raise ValueError("max_needs_reduction must be in (0, 1]")
        if self.small_needs_increase < 1.0:
            raise ValueError("small_needs_increase must be at least 1")
        if not 0.0 < self.small_needs_reduction <= 1.0:
            raise ValueError("small_needs_reduction must be in (0, 1]")
        if self.switch_time < 0.0:
            raise ValueError("switch_time must be non-negative")
        if self.min_trade_quantity <= 0.0:
            raise ValueError("min_trade_quantity must be positive")
        if self.trade_rounding_buffer < 0.0:
            raise ValueError("trade_rounding_buffer must be non-negative")
        if self.cuda_friend_block <= 0:
            raise ValueError("cuda_friend_block must be positive")
        if self.cuda_goods_block <= 0:
            raise ValueError("cuda_goods_block must be positive")
        if self.experimental_hybrid_batches < 0:
            raise ValueError("experimental_hybrid_batches must be non-negative")
        if self.experimental_hybrid_frontier_size < 0:
            raise ValueError("experimental_hybrid_frontier_size must be non-negative")
        if self.experimental_hybrid_seed_stride <= 0:
            raise ValueError("experimental_hybrid_seed_stride must be positive")

    @property
    def agent_good_shape(self) -> tuple[int, int]:
        return (self.population, self.goods)

    @property
    def friend_shape(self) -> tuple[int, int]:
        return (self.population, self.acquaintances)

    @property
    def active_friend_shape(self) -> tuple[int, int]:
        return (self.population, self.active_acquaintances)

    @property
    def transparency_shape(self) -> tuple[int, int, int]:
        return (self.population, self.acquaintances, self.goods)

    @property
    def demand_candidate_shape(self) -> tuple[int, int]:
        return (self.population, self.demand_candidates)

    @property
    def supply_candidate_shape(self) -> tuple[int, int]:
        return (self.population, self.supply_candidates)

    @property
    def cycle_time_budget(self) -> float:
        return float(self.base_need_vector().sum())

    @property
    def leisure_time(self) -> float:
        return self.cycle_time_budget / float(self.goods)

    @property
    def learning_window(self) -> float:
        if self.activity_discount >= 0.999:
            return 1.0
        return 1.0 / max(1.0 - self.activity_discount, 1e-6)

    def gifted_count(self) -> int:
        return int(round(self.population * self.talent_probability))

    def base_need_vector(self) -> np.ndarray:
        return np.arange(1, self.goods + 1, dtype=np.float32) ** 2
