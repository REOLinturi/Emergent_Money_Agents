from __future__ import annotations

from dataclasses import dataclass, field

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
    value_weighted_monetary_concentration: float = 0.0
    value_weighted_rare_goods_monetary_share: float = 0.0
    exchange_media_concentration: float = 0.0
    rare_goods_exchange_media_share: float = 0.0
    stock_value_gini: float = 0.0
    stock_value_top_decile_share: float = 0.0
    stock_value_mean: float = 0.0
    stock_value_median: float = 0.0
    living_standard_gini: float = 0.0
    living_standard_top_decile_share: float = 0.0
    living_standard_mean: float = 0.0
    living_standard_median: float = 0.0
    living_standard_p10: float = 0.0
    living_standard_p25: float = 0.0
    living_standard_p75: float = 0.0
    living_standard_p90: float = 0.0
    living_standard_p99: float = 0.0
    aspiration_balance_mean: float = 0.0
    aspiration_balance_median: float = 0.0
    aspiration_balance_p10: float = 0.0
    aspiration_balance_p90: float = 0.0
    aspiration_shortfall_share: float = 0.0
    aspiration_shortfall_mean: float = 0.0
    aspiration_shortfall_p90: float = 0.0
    smith_cost_gini: float = 0.0
    smith_cost_top_decile_share: float = 0.0
    smith_cost_mean: float = 0.0
    smith_cost_median: float = 0.0
    smith_cost_p10: float = 0.0
    smith_cost_p25: float = 0.0
    smith_cost_p75: float = 0.0
    smith_cost_p90: float = 0.0
    smith_cost_p99: float = 0.0
    production_time_value: float = 0.0
    direct_production_time: float = 0.0
    production_time_share_of_budget: float = 0.0
    tce_share_of_output_value: float = 0.0
    spoilage_share_of_output_value: float = 0.0
    friction_share_of_output_value: float = 0.0
    tce_share_of_time_budget: float = 0.0
    spoilage_share_of_time_budget: float = 0.0
    friction_share_of_time_budget: float = 0.0

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
            value_weighted_monetary_concentration=metrics.value_weighted_monetary_concentration,
            value_weighted_rare_goods_monetary_share=metrics.value_weighted_rare_goods_monetary_share,
            exchange_media_concentration=metrics.exchange_media_concentration,
            rare_goods_exchange_media_share=metrics.rare_goods_exchange_media_share,
            stock_value_gini=metrics.stock_value_gini,
            stock_value_top_decile_share=metrics.stock_value_top_decile_share,
            stock_value_mean=metrics.stock_value_mean,
            stock_value_median=metrics.stock_value_median,
            living_standard_gini=metrics.living_standard_gini,
            living_standard_top_decile_share=metrics.living_standard_top_decile_share,
            living_standard_mean=metrics.living_standard_mean,
            living_standard_median=metrics.living_standard_median,
            living_standard_p10=metrics.living_standard_p10,
            living_standard_p25=metrics.living_standard_p25,
            living_standard_p75=metrics.living_standard_p75,
            living_standard_p90=metrics.living_standard_p90,
            living_standard_p99=metrics.living_standard_p99,
            aspiration_balance_mean=metrics.aspiration_balance_mean,
            aspiration_balance_median=metrics.aspiration_balance_median,
            aspiration_balance_p10=metrics.aspiration_balance_p10,
            aspiration_balance_p90=metrics.aspiration_balance_p90,
            aspiration_shortfall_share=metrics.aspiration_shortfall_share,
            aspiration_shortfall_mean=metrics.aspiration_shortfall_mean,
            aspiration_shortfall_p90=metrics.aspiration_shortfall_p90,
            smith_cost_gini=metrics.smith_cost_gini,
            smith_cost_top_decile_share=metrics.smith_cost_top_decile_share,
            smith_cost_mean=metrics.smith_cost_mean,
            smith_cost_median=metrics.smith_cost_median,
            smith_cost_p10=metrics.smith_cost_p10,
            smith_cost_p25=metrics.smith_cost_p25,
            smith_cost_p75=metrics.smith_cost_p75,
            smith_cost_p90=metrics.smith_cost_p90,
            smith_cost_p99=metrics.smith_cost_p99,
            production_time_value=metrics.production_time_value,
            direct_production_time=metrics.direct_production_time,
            production_time_share_of_budget=metrics.production_time_share_of_budget,
            tce_share_of_output_value=metrics.tce_share_of_output_value,
            spoilage_share_of_output_value=metrics.spoilage_share_of_output_value,
            friction_share_of_output_value=metrics.friction_share_of_output_value,
            tce_share_of_time_budget=metrics.tce_share_of_time_budget,
            spoilage_share_of_time_budget=metrics.spoilage_share_of_time_budget,
            friction_share_of_time_budget=metrics.friction_share_of_time_budget,
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
    recent_purchase_value_total: float = 0.0
    recent_sales_value_total: float = 0.0
    recent_inventory_inflow_value_total: float = 0.0
    value_weighted_monetary_score: float = 0.0
    report_good_id: int | None = None
    exchange_media_score: float = 0.0
    relative_tce_loss: float = 0.0
    relative_trade_flow: float = 0.0
    relative_stock: float = 0.0
    network_circulation_breadth: float = 0.0
    excess_stock_breadth: float = 0.0
    excess_stock_ratio: float = 0.0
    round_trip_breadth: float = 0.0
    round_trip_turnover_share: float = 0.0
    consumer_flow_share: float = 0.0
    retailer_stock_share: float = 0.0
    local_liquidity_score: float = 0.0
    local_liquidity_acceptance_breadth: float = 0.0
    local_liquidity_visible_acceptance: float = 0.0
    local_liquidity_target_increment: float = 0.0
    exchange_media_reserve_score: float = 0.0
    exchange_media_reserve_scale: float = 0.0
    exchange_media_reserve_gap: float = 0.0
    exchange_media_spread_ok_share: float = 0.0


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
    value_weighted_rare_goods_monetary_share: float = 0.0
    value_weighted_monetary_concentration: float = 0.0
    top_value_weighted_monetary_good_ids: list[int] = field(default_factory=list)
    rare_goods_exchange_media_share: float = 0.0
    exchange_media_concentration: float = 0.0
    top_exchange_media_good_ids: list[int] = field(default_factory=list)


@dataclass(slots=True, frozen=True)
class RecentTrendWindowSnapshot:
    window_cycles: int
    sample_count: int
    from_cycle: int
    to_cycle: int
    production_trend: float
    utility_trend: float
    trade_trend: float
    rare_goods_monetary_trend: float
    production_change: float | None
    utility_change: float | None
    trade_change: float | None
    rare_goods_monetary_change: float | None
    cycle_strength: float
    dominant_cycle_length: int | None
    economy_growing: bool
    utility_growing: bool
    value_weighted_rare_goods_monetary_trend: float = 0.0
    value_weighted_rare_goods_monetary_change: float | None = None
    rare_goods_exchange_media_trend: float = 0.0
    rare_goods_exchange_media_change: float | None = None


@dataclass(slots=True, frozen=True)
class GoodRoleSnapshot:
    good_id: int
    base_need: float
    demand_rank: int
    is_rare: bool
    consumer_count: int
    retailer_count: int
    producer_count: int
    retailer_purchase_total: float
    retailer_sales_total: float
    retailer_inventory_inflow_total: float
    retailer_stock_limit_ratio_mean: float
    top1_producer_output_share: float
    top_producer_output_focus_share: float
    top_producer_time_share: float
    report_good_id: int | None = None


@dataclass(slots=True, frozen=True)
class InequalitySnapshot:
    stock_value_gini: float
    stock_value_top_decile_share: float
    stock_value_mean: float
    stock_value_median: float
    living_standard_gini: float
    living_standard_top_decile_share: float
    living_standard_mean: float
    living_standard_median: float
    living_standard_p10: float
    living_standard_p25: float
    living_standard_p75: float
    living_standard_p90: float
    living_standard_p99: float
    aspiration_balance_mean: float
    aspiration_balance_median: float
    aspiration_balance_p10: float
    aspiration_balance_p90: float
    aspiration_shortfall_share: float
    aspiration_shortfall_mean: float
    aspiration_shortfall_p90: float
    smith_cost_gini: float
    smith_cost_top_decile_share: float
    smith_cost_mean: float
    smith_cost_median: float
    smith_cost_p10: float
    smith_cost_p25: float
    smith_cost_p75: float
    smith_cost_p90: float
    smith_cost_p99: float
    production_time_value: float
    direct_production_time: float
    production_time_share_of_budget: float
    tce_share_of_output_value: float
    spoilage_share_of_output_value: float
    friction_share_of_output_value: float
    tce_share_of_time_budget: float
    spoilage_share_of_time_budget: float
    friction_share_of_time_budget: float


@dataclass(slots=True, frozen=True)
class ProgressSnapshot:
    current_cycle: int
    target_cycle: int | None
    progress_share: float | None
    checkpoint_updated_at: str | None
    checkpoint_age_seconds: float | None
    runner_log_updated_at: str | None
    runner_log_age_seconds: float | None
    latest_chunk_from_cycle: int | None
    latest_chunk_target_cycle: int | None
    recent_seconds_per_cycle: float | None
    eta_seconds: float | None


@dataclass(slots=True, frozen=True)
class ExperimentReport:
    status: RunStatus
    latest_market: MarketSnapshot
    history: list[MarketSnapshot]
    goods: list[GoodSnapshot]
    phenomena: PhenomenaSnapshot
