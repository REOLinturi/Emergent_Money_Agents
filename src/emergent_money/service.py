from __future__ import annotations

import numpy as np

from .analytics import analyze_history, compute_good_snapshots
from .dto import (
    AgentSnapshot,
    ExperimentReport,
    GoodSnapshot,
    MarketSnapshot,
    NetworkSlice,
    PhenomenaSnapshot,
    RunStatus,
    TradeProposalView,
)
from .engine import SimulationEngine


class SimulationService:
    def __init__(self, engine: SimulationEngine) -> None:
        self.engine = engine
        self._is_running = False

    @classmethod
    def create(cls, config=None, backend_name: str = "numpy") -> "SimulationService":
        return cls(SimulationEngine.create(config=config, backend_name=backend_name))

    def reset(self, config=None, backend_name: str | None = None) -> RunStatus:
        resolved_backend = backend_name or self.engine.backend.metadata.name
        self.engine = SimulationEngine.create(config=config or self.engine.config, backend_name=resolved_backend)
        self._is_running = False
        return self.get_status()

    def pause(self) -> RunStatus:
        self._is_running = False
        return self.get_status()

    def resume(self) -> RunStatus:
        self._is_running = True
        return self.get_status()

    def step(self, cycles: int = 1) -> list[MarketSnapshot]:
        if cycles <= 0:
            raise ValueError("cycles must be positive")

        snapshots: list[MarketSnapshot] = []
        for _ in range(cycles):
            metrics = self.engine.step()
            snapshots.append(MarketSnapshot.from_metrics(metrics))
        self._is_running = False
        return snapshots

    def run_experiment(self, cycles: int, top_goods: int = 10) -> ExperimentReport:
        if cycles < 0:
            raise ValueError("cycles must be non-negative")
        if cycles > 0:
            self.step(cycles)
        history = self.get_history()
        goods = self.get_goods_snapshot(limit=top_goods)
        return ExperimentReport(
            status=self.get_status(),
            latest_market=self.get_market_snapshot(),
            history=history,
            goods=goods,
            phenomena=analyze_history(self.engine.history, goods),
        )

    def get_status(self) -> RunStatus:
        metadata = self.engine.backend.metadata
        return RunStatus(
            cycle=self.engine.cycle,
            backend_name=metadata.name,
            device=metadata.device,
            history_length=len(self.engine.history),
            is_running=self._is_running,
        )

    def get_market_snapshot(self) -> MarketSnapshot:
        if self.engine.history:
            return MarketSnapshot.from_metrics(self.engine.history[-1])
        return MarketSnapshot.from_metrics(self.engine.snapshot_metrics())

    def get_history(self, limit: int | None = None) -> list[MarketSnapshot]:
        history = self.engine.history if limit is None else self.engine.history[-limit:]
        return [MarketSnapshot.from_metrics(item) for item in history]

    def get_goods_snapshot(self, limit: int = 12, sort_by: str = "monetary_score") -> list[GoodSnapshot]:
        return compute_good_snapshots(
            state=self.engine.state,
            backend=self.engine.backend,
            limit=limit,
            sort_by=sort_by,
        )

    def get_phenomena_snapshot(self, top_goods: int = 8) -> PhenomenaSnapshot:
        goods = self.get_goods_snapshot(limit=top_goods)
        return analyze_history(self.engine.history, goods)

    def get_agent_snapshot(self, agent_id: int) -> AgentSnapshot:
        self._validate_agent_id(agent_id)
        proposal = self._build_agent_proposal(agent_id)
        return AgentSnapshot(
            agent_id=agent_id,
            time_remaining=self._scalar(self.engine.state.time_remaining[agent_id]),
            need=self._vector(self.engine.state.need[agent_id]),
            stock=self._vector(self.engine.state.stock[agent_id]),
            stock_limit=self._vector(self.engine.state.stock_limit[agent_id]),
            innate_efficiency=self._vector(self.engine.state.innate_efficiency[agent_id]),
            learned_efficiency=self._vector(self.engine.state.learned_efficiency[agent_id]),
            efficiency=self._vector(self.engine.state.efficiency[agent_id]),
            purchase_price=self._vector(self.engine.state.purchase_price[agent_id]),
            sales_price=self._vector(self.engine.state.sales_price[agent_id]),
            recent_inventory_inflow=self._vector(self.engine.state.recent_inventory_inflow[agent_id]),
            active_friends=self._int_vector(self.engine.state.trade.active_friend_id[agent_id]),
            candidate_need_goods=self._int_vector(self.engine.state.trade.candidate_need_good[agent_id]),
            candidate_offer_goods=self._int_vector(self.engine.state.trade.candidate_offer_good[agent_id]),
            proposal=proposal,
        )

    def get_network_slice(self, agent_id: int, limit: int | None = None) -> NetworkSlice:
        self._validate_agent_id(agent_id)
        resolved_limit = limit or self.engine.config.acquaintances
        if resolved_limit <= 0:
            raise ValueError("limit must be positive")
        friend_ids = self._int_vector(self.engine.state.friend_id[agent_id])[:resolved_limit]
        friend_activity = self._vector(self.engine.state.friend_activity[agent_id])[:resolved_limit]
        return NetworkSlice(
            root_agent_id=agent_id,
            friend_ids=friend_ids,
            friend_activity=friend_activity,
        )

    def get_trade_sample(self, agent_ids: list[int] | None = None, limit: int = 32) -> list[TradeProposalView]:
        if limit <= 0:
            raise ValueError("limit must be positive")

        if agent_ids is None:
            candidate_ids = range(self.engine.config.population)
        else:
            for agent_id in agent_ids:
                self._validate_agent_id(agent_id)
            candidate_ids = agent_ids

        proposals: list[TradeProposalView] = []
        for agent_id in candidate_ids:
            proposal = self._build_agent_proposal(agent_id)
            if proposal is not None:
                proposals.append(proposal)
            if len(proposals) >= limit:
                break
        return proposals

    def _build_agent_proposal(self, agent_id: int) -> TradeProposalView | None:
        friend_slot = int(self._scalar(self.engine.state.trade.proposal_friend_slot[agent_id]))
        score = self._scalar(self.engine.state.trade.proposal_score[agent_id])
        if friend_slot < 0 or score <= 0.0:
            return None
        accepted_quantity = self._scalar(self.engine.state.trade.accepted_quantity[agent_id])
        return TradeProposalView(
            proposer_id=agent_id,
            target_agent_id=int(self._scalar(self.engine.state.trade.proposal_target_agent[agent_id])),
            friend_slot=friend_slot,
            need_good=int(self._scalar(self.engine.state.trade.proposal_need_good[agent_id])),
            offer_good=int(self._scalar(self.engine.state.trade.proposal_offer_good[agent_id])),
            proposed_quantity=self._scalar(self.engine.state.trade.proposal_quantity[agent_id]),
            accepted_quantity=accepted_quantity,
            score=score,
            accepted=accepted_quantity > 0.0,
        )

    def _validate_agent_id(self, agent_id: int) -> None:
        if agent_id < 0 or agent_id >= self.engine.config.population:
            raise IndexError("agent_id out of range")

    def _vector(self, value) -> list[float]:
        return self.engine.backend.to_numpy(value).astype(np.float64).tolist()

    def _int_vector(self, value) -> list[int]:
        return self.engine.backend.to_numpy(value).astype(np.int64).tolist()

    def _scalar(self, value) -> float:
        return float(self.engine.backend.to_scalar(value))
