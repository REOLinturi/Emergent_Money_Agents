from __future__ import annotations

import numpy as np

from ..contact_update import apply_contact_candidates_in_place
from ..trade_resolution import CommittedTradeState, ResolvedTrades, commit_resolved_trades, resolve_trade_proposals
from .base import BaseBackend


class NumPyBackend(BaseBackend):
    def __init__(self) -> None:
        super().__init__(name="numpy", device="cpu", xp=np)

    def resolve_trade_proposals(
        self,
        *,
        stock,
        need,
        stock_limit,
        target_agent,
        need_good,
        offer_good,
        quantity,
        score,
    ) -> ResolvedTrades:
        return resolve_trade_proposals(
            stock=stock,
            need=need,
            stock_limit=stock_limit,
            target_agent=target_agent,
            need_good=need_good,
            offer_good=offer_good,
            quantity=quantity,
            score=score,
        )

    def commit_resolved_trades(
        self,
        *,
        stock,
        need,
        recent_sales,
        recent_purchases,
        friend_id,
        friend_activity,
        transparency,
        proposal_friend_slot,
        proposal_target_agent,
        proposal_need_good,
        proposal_offer_good,
        accepted_mask,
        accepted_quantity,
        initial_transparency: float,
    ) -> CommittedTradeState:
        return commit_resolved_trades(
            stock=stock,
            need=need,
            recent_sales=recent_sales,
            recent_purchases=recent_purchases,
            friend_id=friend_id,
            friend_activity=friend_activity,
            transparency=transparency,
            proposal_friend_slot=proposal_friend_slot,
            proposal_target_agent=proposal_target_agent,
            proposal_need_good=proposal_need_good,
            proposal_offer_good=proposal_offer_good,
            accepted_mask=accepted_mask,
            accepted_quantity=accepted_quantity,
            initial_transparency=initial_transparency,
        )

    def apply_contact_candidates(
        self,
        *,
        friend_id,
        friend_activity,
        transparency,
        candidate_ids,
        initial_activity: float,
        initial_transparency: float,
    ) -> None:
        apply_contact_candidates_in_place(
            friend_id=friend_id,
            friend_activity=friend_activity,
            transparency=transparency,
            candidate_ids=candidate_ids,
            initial_activity=initial_activity,
            initial_transparency=initial_transparency,
        )
