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
    unmet_need_total: float
    stock_total: float
    average_efficiency: float
    mean_time_remaining: float
    proposed_trade_count: int
    accepted_trade_count: int
    accepted_trade_volume: float

    @classmethod
    def from_metrics(cls, metrics: MetricsSnapshot) -> "MarketSnapshot":
        return cls(
            cycle=metrics.cycle,
            fulfilled_share=metrics.fulfilled_share,
            unmet_need_total=metrics.unmet_need_total,
            stock_total=metrics.stock_total,
            average_efficiency=metrics.average_efficiency,
            mean_time_remaining=metrics.mean_time_remaining,
            proposed_trade_count=metrics.proposed_trade_count,
            accepted_trade_count=metrics.accepted_trade_count,
            accepted_trade_volume=metrics.accepted_trade_volume,
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
    active_friends: list[int]
    candidate_need_goods: list[int]
    candidate_offer_goods: list[int]
    proposal: TradeProposalView | None


@dataclass(slots=True, frozen=True)
class NetworkSlice:
    root_agent_id: int
    friend_ids: list[int]
    friend_activity: list[float]
