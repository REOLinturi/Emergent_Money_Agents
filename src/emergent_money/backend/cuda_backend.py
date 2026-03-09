from __future__ import annotations

from ..trade_resolution import CommittedTradeState, ResolvedTrades
from .base import BackendUnavailableError, BaseBackend

try:
    import cupy as cp
except ImportError:  # pragma: no cover - depends on the target machine
    cp = None


class CudaBackend(BaseBackend):
    @staticmethod
    def available() -> bool:
        return cp is not None

    def __init__(self) -> None:
        if cp is None:
            raise BackendUnavailableError(
                "CuPy is not installed. Install the CuPy build that matches the target CUDA stack."
            )
        super().__init__(name="cuda", device="cuda", xp=cp)

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
    ) -> ResolvedTrades:  # pragma: no cover - depends on CUDA runtime
        # CUDA keeps the same backend contract now; a device-native resolver can replace
        # this host fallback later without changing engine semantics.
        return super().resolve_trade_proposals(
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
    ) -> CommittedTradeState:  # pragma: no cover - depends on CUDA runtime
        # CUDA keeps the same backend contract now; a device-native commit kernel can replace
        # this host fallback later without changing engine semantics.
        return super().commit_resolved_trades(
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

    def to_numpy(self, value):  # pragma: no cover - depends on CUDA runtime
        return self.xp.asnumpy(value)

    def synchronize(self) -> None:  # pragma: no cover - depends on CUDA runtime
        self.xp.cuda.Stream.null.synchronize()