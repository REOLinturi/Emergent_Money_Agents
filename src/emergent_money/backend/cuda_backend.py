from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ..trade_resolution import CommittedTradeState, ResolvedTrades
from .base import BackendUnavailableError, BaseBackend

try:
    import cupy as cp
except ImportError:  # pragma: no cover - depends on the target machine
    cp = None


class CudaBackend(BaseBackend):
    _kernel_source: str | None = None

    @staticmethod
    def available() -> bool:
        return cp is not None

    def __init__(self) -> None:
        if cp is None:
            raise BackendUnavailableError(
                "CuPy is not installed. Install the CuPy build that matches the target CUDA stack."
            )
        self._ensure_cache_dir()
        super().__init__(name="cuda", device="cuda", xp=cp)
        kernel_source = self._load_kernel_source()
        self._resolve_kernel = cp.RawKernel(kernel_source, "resolve_trade_proposals_kernel")
        self._commit_kernel = cp.RawKernel(kernel_source, "commit_resolved_trades_kernel")
        self._plan_contact_kernel = cp.RawKernel(kernel_source, "plan_contact_candidates_kernel")
        self._contact_kernel = cp.RawKernel(kernel_source, "apply_contact_candidates_kernel")
        self._proposal_block_kernel = cp.RawKernel(kernel_source, "score_trade_block_kernel")

    @staticmethod
    def _ensure_cache_dir() -> None:
        if os.environ.get("CUPY_CACHE_DIR"):
            return
        workspace_root = Path(__file__).resolve().parents[3]
        cache_dir = workspace_root / ".cupy_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["CUPY_CACHE_DIR"] = str(cache_dir)

    @classmethod
    def _load_kernel_source(cls) -> str:
        if cls._kernel_source is None:
            kernel_path = Path(__file__).with_name("trade_kernels.cu")
            cls._kernel_source = kernel_path.read_text(encoding="utf-8")
        return cls._kernel_source

    def _sorted_proposers(self, score) -> object:
        proposal_count = int(score.shape[0])
        keys = self.xp.stack(
            (
                self.xp.arange(proposal_count, dtype=self.xp.float32),
                -score.astype(self.xp.float32, copy=False),
            )
        )
        try:
            return self.xp.lexsort(keys).astype(self.xp.int32, copy=False)
        except RuntimeError:
            order = np.argsort(-self.to_numpy(score), kind="stable")
            return self.asarray(order, dtype=np.int32)

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
        proposal_count = int(score.shape[0])
        sorted_proposers = self._sorted_proposers(score)
        accepted_mask = self.xp.zeros((proposal_count,), dtype=self.xp.uint8)
        accepted_quantity = self.xp.zeros((proposal_count,), dtype=self.xp.float32)
        working_stock = self.xp.array(stock, copy=True)
        working_need = self.xp.array(need, copy=True)

        self._resolve_kernel(
            (1,),
            (1,),
            (
                stock,
                need,
                stock_limit,
                sorted_proposers,
                target_agent,
                need_good,
                offer_good,
                quantity,
                score,
                np.int32(proposal_count),
                np.int32(stock.shape[1]),
                accepted_mask,
                accepted_quantity,
                working_stock,
                working_need,
            ),
        )
        return ResolvedTrades(
            accepted_mask=accepted_mask.astype(self.xp.bool_),
            accepted_quantity=accepted_quantity,
            stock=working_stock,
            need=working_need,
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
        accepted_mask_u8 = accepted_mask.astype(self.xp.uint8, copy=False)

        self._commit_kernel(
            (1,),
            (1,),
            (
                np.int32(friend_id.shape[0]),
                np.int32(stock.shape[1]),
                np.int32(friend_id.shape[1]),
                proposal_friend_slot,
                proposal_target_agent,
                proposal_need_good,
                proposal_offer_good,
                accepted_mask_u8,
                accepted_quantity,
                np.float32(initial_transparency),
                recent_sales,
                recent_purchases,
                friend_id,
                friend_activity,
                transparency,
            ),
        )
        return CommittedTradeState(
            stock=stock,
            need=need,
            recent_sales=recent_sales,
            recent_purchases=recent_purchases,
            friend_id=friend_id,
            friend_activity=friend_activity,
            transparency=transparency,
        )

    def plan_contact_candidates(self, *, friend_id, seed: int, cycle: int):  # pragma: no cover - depends on CUDA runtime
        candidate_ids = self.xp.full((friend_id.shape[0],), -1, dtype=self.xp.int32)
        threads = 128
        blocks = (int((friend_id.shape[0] + threads - 1) // threads),)
        self._plan_contact_kernel(
            blocks,
            (threads,),
            (
                np.int32(friend_id.shape[0]),
                np.int32(friend_id.shape[1]),
                np.uint32(seed),
                np.uint32(cycle),
                friend_id,
                candidate_ids,
            ),
        )
        return candidate_ids

    def apply_contact_candidates(
        self,
        *,
        friend_id,
        friend_activity,
        transparency,
        candidate_ids,
        initial_activity: float,
        initial_transparency: float,
    ) -> None:  # pragma: no cover - depends on CUDA runtime
        threads = 128
        blocks = (int((candidate_ids.shape[0] + threads - 1) // threads),)
        self._contact_kernel(
            blocks,
            (threads,),
            (
                np.int32(friend_id.shape[0]),
                np.int32(friend_id.shape[1]),
                np.int32(transparency.shape[2]),
                candidate_ids,
                np.float32(initial_activity),
                np.float32(initial_transparency),
                friend_id,
                friend_activity,
                transparency,
            ),
        )

    def score_trade_block(
        self,
        *,
        friend_start: int,
        need_start: int,
        offer_start: int,
        friend_index_block,
        self_interest_need,
        self_stock_offer,
        self_purchase_need,
        self_sales_offer,
        friend_stock_need,
        friend_interest_offer,
        friend_purchase_offer,
        friend_sales_need,
        transparency_need,
        best_score,
        best_friend_slot,
        best_target_agent,
        best_need_good,
        best_offer_good,
        best_quantity,
        initial_transparency: float,
    ) -> None:  # pragma: no cover - depends on CUDA runtime
        threads = 128
        population = int(best_score.shape[0])
        blocks = (int((population + threads - 1) // threads),)
        self._proposal_block_kernel(
            blocks,
            (threads,),
            (
                np.int32(population),
                np.int32(friend_start),
                np.int32(friend_index_block.shape[1]),
                np.int32(need_start),
                np.int32(self_interest_need.shape[1]),
                np.int32(offer_start),
                np.int32(self_stock_offer.shape[1]),
                np.float32(initial_transparency),
                friend_index_block,
                self_interest_need,
                self_stock_offer,
                self_purchase_need,
                self_sales_offer,
                friend_stock_need,
                friend_interest_offer,
                friend_purchase_offer,
                friend_sales_need,
                transparency_need,
                best_score,
                best_friend_slot,
                best_target_agent,
                best_need_good,
                best_offer_good,
                best_quantity,
            ),
        )

    def to_numpy(self, value):  # pragma: no cover - depends on CUDA runtime
        return self.xp.asnumpy(value)

    def synchronize(self) -> None:  # pragma: no cover - depends on CUDA runtime
        self.xp.cuda.Stream.null.synchronize()