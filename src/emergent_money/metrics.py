from __future__ import annotations

from dataclasses import dataclass

from .backend.base import BaseBackend
from .state import SimulationState


@dataclass(slots=True, frozen=True)
class MetricsSnapshot:
    cycle: int
    fulfilled_share: float
    unmet_need_total: float
    stock_total: float
    average_efficiency: float
    mean_time_remaining: float
    proposed_trade_count: int
    accepted_trade_count: int
    accepted_trade_volume: float


def compute_metrics(
    *,
    cycle: int,
    state: SimulationState,
    backend: BaseBackend,
    cycle_need_total: float,
    proposed_trade_count: int,
    accepted_trade_count: int,
    accepted_trade_volume: float,
) -> MetricsSnapshot:
    xp = backend.xp
    unmet_need_total = float(backend.to_scalar(xp.sum(state.need)))
    stock_total = float(backend.to_scalar(xp.sum(state.stock)))
    average_efficiency = float(backend.to_scalar(xp.mean(state.efficiency)))
    mean_time_remaining = float(backend.to_scalar(xp.mean(state.time_remaining)))
    fulfilled_share = 1.0
    if cycle_need_total > 0.0:
        fulfilled_share = max(0.0, 1.0 - (unmet_need_total / cycle_need_total))

    return MetricsSnapshot(
        cycle=cycle,
        fulfilled_share=fulfilled_share,
        unmet_need_total=unmet_need_total,
        stock_total=stock_total,
        average_efficiency=average_efficiency,
        mean_time_remaining=mean_time_remaining,
        proposed_trade_count=proposed_trade_count,
        accepted_trade_count=accepted_trade_count,
        accepted_trade_volume=accepted_trade_volume,
    )
