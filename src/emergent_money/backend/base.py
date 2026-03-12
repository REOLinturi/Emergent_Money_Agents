from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..contact_update import apply_contact_candidates_in_place, plan_contact_candidates
from ..trade_resolution import CommittedTradeState, ResolvedTrades, commit_resolved_trades, resolve_trade_proposals


class BackendUnavailableError(RuntimeError):
    pass


@dataclass(slots=True, frozen=True)
class BackendMetadata:
    name: str
    device: str


class BaseBackend:
    def __init__(self, *, name: str, device: str, xp: Any) -> None:
        self._metadata = BackendMetadata(name=name, device=device)
        self.xp = xp

    @property
    def metadata(self) -> BackendMetadata:
        return self._metadata

    def asarray(self, data: Any, dtype: Any | None = None) -> Any:
        return self.xp.asarray(data, dtype=dtype)

    def zeros(self, shape: tuple[int, ...], dtype: Any = np.float32) -> Any:
        return self.xp.zeros(shape, dtype=dtype)

    def full(self, shape: tuple[int, ...], fill_value: float, dtype: Any = np.float32) -> Any:
        return self.xp.full(shape, fill_value, dtype=dtype)

    def topk_indices(self, values: Any, k: int, axis: int = -1) -> Any:
        axis_size = values.shape[axis]
        if k <= 0 or k > axis_size:
            raise ValueError("k must be between 1 and the selected axis size")

        split = axis_size - k
        partition = self.xp.argpartition(values, split, axis=axis)
        topk = self.xp.take(partition, indices=range(split, axis_size), axis=axis)
        topk_values = self.xp.take_along_axis(values, topk, axis=axis)
        order = self.xp.argsort(topk_values, axis=axis)
        order = self.xp.flip(order, axis=axis)
        return self.xp.take_along_axis(topk, order, axis=axis)

    def resolve_trade_proposals(
        self,
        *,
        stock: Any,
        need: Any,
        stock_limit: Any,
        target_agent: Any,
        need_good: Any,
        offer_good: Any,
        quantity: Any,
        score: Any,
    ) -> ResolvedTrades:
        resolved = resolve_trade_proposals(
            stock=self.to_numpy(stock),
            need=self.to_numpy(need),
            stock_limit=self.to_numpy(stock_limit),
            target_agent=self.to_numpy(target_agent),
            need_good=self.to_numpy(need_good),
            offer_good=self.to_numpy(offer_good),
            quantity=self.to_numpy(quantity),
            score=self.to_numpy(score),
        )
        return ResolvedTrades(
            accepted_mask=self.asarray(resolved.accepted_mask, dtype=np.bool_),
            accepted_quantity=self.asarray(resolved.accepted_quantity, dtype=np.float32),
            proposer_need_satisfied=self.asarray(resolved.proposer_need_satisfied, dtype=np.float32),
            proposer_stock_added=self.asarray(resolved.proposer_stock_added, dtype=np.float32),
            target_need_satisfied=self.asarray(resolved.target_need_satisfied, dtype=np.float32),
            target_stock_added=self.asarray(resolved.target_stock_added, dtype=np.float32),
            stock=self.asarray(resolved.stock, dtype=np.float32),
            need=self.asarray(resolved.need, dtype=np.float32),
        )

    def commit_resolved_trades(
        self,
        *,
        stock: Any,
        need: Any,
        recent_sales: Any,
        recent_purchases: Any,
        recent_inventory_inflow: Any,
        friend_id: Any,
        friend_activity: Any,
        transparency: Any,
        proposal_friend_slot: Any,
        proposal_target_agent: Any,
        proposal_need_good: Any,
        proposal_offer_good: Any,
        accepted_mask: Any,
        accepted_quantity: Any,
        proposer_stock_added: Any,
        target_stock_added: Any,
        initial_transparency: float,
    ) -> CommittedTradeState:
        committed = commit_resolved_trades(
            stock=self.to_numpy(stock),
            need=self.to_numpy(need),
            recent_sales=self.to_numpy(recent_sales),
            recent_purchases=self.to_numpy(recent_purchases),
            recent_inventory_inflow=self.to_numpy(recent_inventory_inflow),
            friend_id=self.to_numpy(friend_id),
            friend_activity=self.to_numpy(friend_activity),
            transparency=self.to_numpy(transparency),
            proposal_friend_slot=self.to_numpy(proposal_friend_slot),
            proposal_target_agent=self.to_numpy(proposal_target_agent),
            proposal_need_good=self.to_numpy(proposal_need_good),
            proposal_offer_good=self.to_numpy(proposal_offer_good),
            accepted_mask=self.to_numpy(accepted_mask),
            accepted_quantity=self.to_numpy(accepted_quantity),
            proposer_stock_added=self.to_numpy(proposer_stock_added),
            target_stock_added=self.to_numpy(target_stock_added),
            initial_transparency=initial_transparency,
        )
        return CommittedTradeState(
            stock=self.asarray(committed.stock, dtype=np.float32),
            need=self.asarray(committed.need, dtype=np.float32),
            recent_sales=self.asarray(committed.recent_sales, dtype=np.float32),
            recent_purchases=self.asarray(committed.recent_purchases, dtype=np.float32),
            recent_inventory_inflow=self.asarray(committed.recent_inventory_inflow, dtype=np.float32),
            friend_id=self.asarray(committed.friend_id, dtype=np.int32),
            friend_activity=self.asarray(committed.friend_activity, dtype=np.float32),
            transparency=self.asarray(committed.transparency, dtype=np.float32),
        )

    def plan_contact_candidates(self, *, friend_id: Any, seed: int, cycle: int) -> Any:
        planned = plan_contact_candidates(
            friend_id=self.to_numpy(friend_id).astype(np.int32, copy=False),
            seed=seed,
            cycle=cycle,
        )
        return self.asarray(planned, dtype=np.int32)

    def apply_contact_candidates(
        self,
        *,
        friend_id: Any,
        friend_activity: Any,
        transparency: Any,
        candidate_ids: Any,
        initial_activity: float,
        initial_transparency: float,
    ) -> None:
        updated_friend_id = self.to_numpy(friend_id).astype(np.int32, copy=True)
        updated_friend_activity = self.to_numpy(friend_activity).astype(np.float32, copy=True)
        updated_transparency = self.to_numpy(transparency).astype(np.float32, copy=True)
        candidate_ids_np = self.to_numpy(candidate_ids).astype(np.int32, copy=False)

        apply_contact_candidates_in_place(
            friend_id=updated_friend_id,
            friend_activity=updated_friend_activity,
            transparency=updated_transparency,
            candidate_ids=candidate_ids_np,
            initial_activity=initial_activity,
            initial_transparency=initial_transparency,
        )

        friend_id[...] = self.asarray(updated_friend_id, dtype=np.int32)
        friend_activity[...] = self.asarray(updated_friend_activity, dtype=np.float32)
        transparency[...] = self.asarray(updated_transparency, dtype=np.float32)

    def to_numpy(self, value: Any) -> np.ndarray:
        return np.asarray(value)

    def to_scalar(self, value: Any) -> Any:
        if hasattr(value, "item"):
            return value.item()
        return value

    def synchronize(self) -> None:
        return None
