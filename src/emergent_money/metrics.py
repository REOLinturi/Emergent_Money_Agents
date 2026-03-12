from __future__ import annotations

from dataclasses import dataclass

from .backend.base import BaseBackend
from .state import SimulationState


@dataclass(slots=True, frozen=True)
class MetricsSnapshot:
    cycle: int
    fulfilled_share: float
    fulfilled_need_total: float
    utility_proxy_total: float
    unmet_need_total: float
    stock_total: float
    average_efficiency: float
    mean_time_remaining: float
    proposed_trade_count: int
    accepted_trade_count: int
    accepted_trade_volume: float
    production_total: float
    surplus_output_total: float
    stock_consumption_total: float
    leisure_extra_need_total: float
    inventory_trade_volume: float
    network_density: float
    monetary_concentration: float
    rare_goods_monetary_share: float
    average_needs_level: float
    periodic_tce_cost_total: float
    periodic_spoilage_total: float
    tce_cost_in_time_total: float
    spoilage_cost_in_time_total: float
    stored_delta_total: float
    loser_share: float
    price_average: float


def compute_metrics(
    *,
    cycle: int,
    state: SimulationState,
    backend: BaseBackend,
    cycle_need_total: float,
    proposed_trade_count: int,
    accepted_trade_count: int,
    accepted_trade_volume: float,
    production_total: float,
    surplus_output_total: float,
    stock_consumption_total: float,
    leisure_extra_need_total: float,
    inventory_trade_volume: float,
    network_density: float,
    monetary_concentration: float,
    rare_goods_monetary_share: float,
) -> MetricsSnapshot:
    xp = backend.xp
    unmet_need_total = float(backend.to_scalar(xp.sum(state.need)))
    stock_total = float(backend.to_scalar(xp.sum(state.stock)))
    average_efficiency = float(backend.to_scalar(xp.mean(state.efficiency)))
    mean_time_remaining = float(backend.to_scalar(xp.mean(state.time_remaining)))
    average_needs_level = float(backend.to_scalar(xp.mean(state.needs_level)))
    fulfilled_need_total = max(cycle_need_total - unmet_need_total, 0.0)
    fulfilled_share = 1.0
    if cycle_need_total > 0.0:
        fulfilled_share = max(0.0, 1.0 - (unmet_need_total / cycle_need_total))

    periodic_tce_cost_total = float(backend.to_scalar(xp.sum(state.market.periodic_tce_cost)))
    periodic_spoilage_total = float(backend.to_scalar(xp.sum(state.market.periodic_spoilage)))
    stored_delta_total = stock_total - float(state.market.total_stock_previous)
    loser_share = float(state.market.losers) / float(state.base_need.shape[0]) if state.base_need.shape[0] else 0.0

    return MetricsSnapshot(
        cycle=cycle,
        fulfilled_share=fulfilled_share,
        fulfilled_need_total=fulfilled_need_total,
        utility_proxy_total=average_needs_level,
        unmet_need_total=unmet_need_total,
        stock_total=stock_total,
        average_efficiency=average_efficiency,
        mean_time_remaining=mean_time_remaining,
        proposed_trade_count=proposed_trade_count,
        accepted_trade_count=accepted_trade_count,
        accepted_trade_volume=accepted_trade_volume,
        production_total=production_total,
        surplus_output_total=surplus_output_total,
        stock_consumption_total=stock_consumption_total,
        leisure_extra_need_total=leisure_extra_need_total,
        inventory_trade_volume=inventory_trade_volume,
        network_density=network_density,
        monetary_concentration=monetary_concentration,
        rare_goods_monetary_share=rare_goods_monetary_share,
        average_needs_level=average_needs_level,
        periodic_tce_cost_total=periodic_tce_cost_total,
        periodic_spoilage_total=periodic_spoilage_total,
        tce_cost_in_time_total=float(state.market.total_cost_of_tce_in_time),
        spoilage_cost_in_time_total=float(state.market.total_cost_of_spoilage_in_time),
        stored_delta_total=stored_delta_total,
        loser_share=loser_share,
        price_average=float(state.market.price_average),
    )
