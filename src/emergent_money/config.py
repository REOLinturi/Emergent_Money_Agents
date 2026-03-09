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
    gifted_efficiency_bonus: float = 0.5
    talent_probability: float = 0.50
    initial_stock_fraction: float = 0.10
    initial_price: float = 1.0
    initial_transparency: float = 0.70
    stock_limit_multiplier: float = 2.0
    activity_discount: float = 0.80
    spoilage_rate: float = 0.05
    max_leisure_extra_multiplier: float = 1.0
    leisure_stock_trade_bias: float = 0.35

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
        if self.initial_stock_fraction < 0.0:
            raise ValueError("initial_stock_fraction must be non-negative")
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
    def learning_window(self) -> float:
        if self.activity_discount >= 0.999:
            return 1.0
        return 1.0 / max(1.0 - self.activity_discount, 1e-6)

    def base_need_vector(self) -> np.ndarray:
        return np.arange(1, self.goods + 1, dtype=np.float32) ** 2
