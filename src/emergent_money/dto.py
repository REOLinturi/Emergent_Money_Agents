from __future__ import annotations

from dataclasses import dataclass

from .metrics import MetricsSnapshot


@dataclass(slots=True, frozen=True)
class RunStatus:
    cycle: int
    backend_name: str
    device: str
    history_length: int
    is_running: bool


@dataclass(slots=True, frozen=True)
class MarketSnapshot:
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

    @classmethod
    def from_metrics(cls, metrics: MetricsSnapshot) -> "MarketSnapshot":
        return cls(
            cycle=metrics.cycle,
            fulfilled_share=metrics.fulfilled_share,
            fulfilled_need_total=metrics.fulfilled_need_total,
            utility_proxy_total=metrics.utility_proxy_total,
            unmet_need_total=metrics.unmet_need_total,
            stock_total=metrics.stock_total,
            average_efficiency=metrics.average_efficiency,
            mean_time_remaining=metrics.mean_time_remaining,
            proposed_trade_count=metrics.proposed_trade_count,
            accepted_trade_count=metrics.accepted_trade_count,
            accepted_trade_volume=metrics.accepted_trade_volume,
            production_total=metrics.production_total,
            surplus_output_total=metrics.surplus_output_total,
            stock_consumption_total=metrics.stock_consumption_total,
            leisure_extra_need_total=metrics.leisure_extra_need_total,
            inventory_trade_volume=metrics.inventory_trade_volume,
            network_density=metrics.network_density,
            monetary_concentration=metrics.monetary_concentration,
            rare_goods_monetary_share=metrics.rare_goods_monetary_share,
            average_needs_level=metrics.average_needs_level,
            periodic_tce_cost_total=metrics.periodic_tce_cost_total,
            periodic_spoilage_total=metrics.periodic_spoilage_total,
            tce_cost_in_time_total=metrics.tce_cost_in_time_total,
            spoilage_cost_in_time_total=metrics.spoilage_cost_in_time_total,
            stored_delta_total=metrics.stored_delta_total,
            loser_share=metrics.loser_share,
            price_average=metrics.price_average,
        )


@dataclass(slots=True, frozen=True)
class TradeProposalView:
    proposer_id: int
    target_agent_id: int
    friend_slot: int
    need_good: int
    offer_good: int
    proposed_quantity: float
    accepted_quantity: float
    score: float
    accepted: bool


@dataclass(slots=True, frozen=True)
class AgentSnapshot:
    agent_id: int
    time_remaining: float
    need: list[float]
    stock: list[float]
    stock_limit: list[float]
    innate_efficiency: list[float]
    learned_efficiency: list[float]
    efficiency: list[float]
    purchase_price: list[float]
    sales_price: list[float]
    recent_inventory_inflow: list[float]
    active_friends: list[int]
    candidate_need_goods: list[int]
    candidate_offer_goods: list[int]
    proposal: TradeProposalView | None


@dataclass(slots=True, frozen=True)
class NetworkSlice:
    root_agent_id: int
    friend_ids: list[int]
    friend_activity: list[float]


@dataclass(slots=True, frozen=True)
class GoodSnapshot:
    good_id: int
    base_need: float
    demand_rank: int
    is_rare: bool
    stock_total: float
    average_efficiency: float
    recent_purchase_total: float
    recent_sales_total: float
    recent_inventory_inflow_total: float
    purchase_breadth: float
    inventory_acceptance_share: float
    turnover_balance: float
    monetary_score: float


@dataclass(slots=True, frozen=True)
class PhenomenaSnapshot:
    cycles_observed: int
    production_trend: float
    utility_trend: float
    stock_trend: float
    cycle_strength: float
    dominant_cycle_length: int | None
    rare_goods_monetary_share: float
    monetary_concentration: float
    economy_growing: bool
    utility_growing: bool
    cycles_detected: bool
    top_monetary_good_ids: list[int]


@dataclass(slots=True, frozen=True)
class ExperimentReport:
    status: RunStatus
    latest_market: MarketSnapshot
    history: list[MarketSnapshot]
    goods: list[GoodSnapshot]
    phenomena: PhenomenaSnapshot
