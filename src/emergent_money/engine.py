from __future__ import annotations

import numpy as np

from .analytics import compute_monetary_aggregates
from .backend import create_backend
from .backend.base import BaseBackend
from .config import SimulationConfig
from .initialization import create_initial_state
from .legacy_cycle import run_legacy_cycle
from .metrics import MetricsSnapshot, compute_metrics
from .state import SimulationState


class SimulationEngine:
    def __init__(self, config: SimulationConfig, backend: BaseBackend, state: SimulationState | None = None) -> None:
        self.config = config
        self.backend = backend
        self.state = state or create_initial_state(config, backend)
        self.cycle = 0
        self.history: list[MetricsSnapshot] = []
        self.exact_cycle_diagnostics: dict[str, object] | None = None
        self._cycle_need_total = self._base_need_total()
        self._proposed_trade_count = 0
        self._accepted_trade_count = 0
        self._accepted_trade_volume = 0.0
        self._production_total = 0.0
        self._surplus_output_total = 0.0
        self._stock_consumption_total = 0.0
        self._leisure_extra_need_total = 0.0
        self._inventory_trade_volume = 0.0

    @classmethod
    def create(cls, config: SimulationConfig | None = None, backend_name: str = "numpy") -> "SimulationEngine":
        resolved_config = config or SimulationConfig()
        resolved_backend_name = backend_name
        if resolved_config.use_exact_legacy_mechanics:
            resolved_backend_name = "numpy"
        backend = create_backend(resolved_backend_name)
        state = create_initial_state(resolved_config, backend)
        return cls(config=resolved_config, backend=backend, state=state)

    def step(self) -> MetricsSnapshot:
        if self.config.use_exact_legacy_mechanics:
            run_legacy_cycle(self)
        else:
            self._start_cycle()
            self._run_basic_round()
            self._run_leisure_round()
            self._update_efficiency_from_learning()
            self._apply_spoilage()
            self._update_private_values()
        self.backend.synchronize()

        self.cycle += 1
        snapshot = self.snapshot_metrics()
        if self.config.use_exact_legacy_mechanics:
            self.state.market.total_stock_previous = snapshot.stock_total
        self.history.append(snapshot)
        return snapshot

    def snapshot_metrics(self) -> MetricsSnapshot:
        monetary_concentration, rare_goods_monetary_share = compute_monetary_aggregates(self.state, self.backend)
        return compute_metrics(
            cycle=self.cycle,
            state=self.state,
            backend=self.backend,
            cycle_need_total=self._cycle_need_total,
            proposed_trade_count=self._proposed_trade_count,
            accepted_trade_count=self._accepted_trade_count,
            accepted_trade_volume=self._accepted_trade_volume,
            production_total=self._production_total,
            surplus_output_total=self._surplus_output_total,
            stock_consumption_total=self._stock_consumption_total,
            leisure_extra_need_total=self._leisure_extra_need_total,
            inventory_trade_volume=self._inventory_trade_volume,
            network_density=self._network_density(),
            monetary_concentration=monetary_concentration,
            rare_goods_monetary_share=rare_goods_monetary_share,
        )

    def _base_need_total(self) -> float:
        return float(self.backend.to_scalar(self.backend.xp.sum(self.state.base_need)))

    def _network_density(self) -> float:
        xp = self.backend.xp
        known_share = xp.mean((self.state.friend_id >= 0).astype(xp.float32))
        return float(self.backend.to_scalar(known_share))

    def _start_cycle(self) -> None:
        self.state.need[...] = self.state.base_need
        self.state.time_remaining[...] = self.state.cycle_time_budget
        self.state.friend_activity *= self.config.activity_discount
        self.state.recent_production *= self.config.activity_discount
        self.state.recent_sales *= self.config.activity_discount
        self.state.recent_purchases *= self.config.activity_discount
        self.state.recent_inventory_inflow *= self.config.activity_discount
        self.state.trade.active_friend_slot[...] = -1
        self.state.trade.active_friend_id[...] = -1
        self.state.trade.proposal_friend_slot[...] = -1
        self.state.trade.proposal_target_agent[...] = -1
        self.state.trade.proposal_need_good[...] = -1
        self.state.trade.proposal_offer_good[...] = -1
        self.state.trade.proposal_quantity[...] = 0.0
        self.state.trade.proposal_score[...] = 0.0
        self.state.trade.accepted_mask[...] = False
        self.state.trade.accepted_quantity[...] = 0.0
        self._cycle_need_total = self._base_need_total()
        self._proposed_trade_count = 0
        self._accepted_trade_count = 0
        self._accepted_trade_volume = 0.0
        self._production_total = 0.0
        self._surplus_output_total = 0.0
        self._stock_consumption_total = 0.0
        self._leisure_extra_need_total = 0.0
        self._inventory_trade_volume = 0.0

    def _run_basic_round(self) -> None:
        self._run_market_round(allow_stock_trade=False)
        self._introduce_random_contacts()
        self._produce_surplus()

    def _run_leisure_round(self) -> None:
        if not self._apply_leisure_demand():
            return
        self._run_market_round(allow_stock_trade=True)
        self._produce_surplus()

    def _run_market_round(self, *, allow_stock_trade: bool) -> None:
        self._consume_from_stock()
        self._prepare_trade_frontier()
        self._select_trade_candidates(allow_stock_trade=allow_stock_trade)
        self._proposed_trade_count += self._score_trade_proposals(allow_stock_trade=allow_stock_trade)
        accepted_count, accepted_volume, inventory_trade_volume = self._commit_trades()
        self._accepted_trade_count += accepted_count
        self._accepted_trade_volume += accepted_volume
        self._inventory_trade_volume += inventory_trade_volume
        self._produce_for_needs()

    def _consume_from_stock(self) -> None:
        xp = self.backend.xp
        consumed = xp.minimum(self.state.need, self.state.stock)
        self._stock_consumption_total += float(self.backend.to_scalar(xp.sum(consumed)))
        self.state.need -= consumed
        self.state.stock -= consumed

    def _prepare_trade_frontier(self) -> None:
        xp = self.backend.xp
        slot_index = xp.arange(self.config.acquaintances, dtype=xp.float32)[None, :]
        invalid_score = xp.full(self.config.friend_shape, -1e9, dtype=xp.float32)
        activity_score = xp.where(
            self.state.friend_id >= 0,
            self.state.friend_activity - (slot_index * 1e-6),
            invalid_score,
        )
        active_slots = self.backend.topk_indices(activity_score, self.config.active_acquaintances, axis=1)
        self.state.trade.active_friend_slot[...] = active_slots
        self.state.trade.active_friend_id[...] = xp.take_along_axis(self.state.friend_id, active_slots, axis=1)

    def _select_trade_candidates(self, *, allow_stock_trade: bool) -> None:
        xp = self.backend.xp
        # Reference barter scores all known friend and good pairs.
        # These candidate buffers are for snapshots and future calibrated pruning.
        demand_signal = self.state.need * self.state.purchase_price
        if allow_stock_trade:
            stock_room = xp.maximum(self.state.stock_limit - self.state.stock, 0.0)
            demand_signal = demand_signal + (stock_room * self.state.sales_price * self.config.leisure_stock_trade_bias)
        supply_signal = self.state.stock * self.state.sales_price * (1.0 + self.state.talent_mask)
        self.state.trade.candidate_need_good[...] = self.backend.topk_indices(
            demand_signal,
            self.config.demand_candidates,
            axis=1,
        )
        self.state.trade.candidate_offer_good[...] = self.backend.topk_indices(
            supply_signal,
            self.config.supply_candidates,
            axis=1,
        )

    def _score_trade_proposals(self, *, allow_stock_trade: bool) -> int:
        if self.backend.metadata.name == "cuda":
            return self._score_trade_proposals_blocked(allow_stock_trade=allow_stock_trade)

        xp = self.backend.xp
        row_index = xp.arange(self.config.population, dtype=xp.int32)
        best_score = xp.zeros((self.config.population,), dtype=xp.float32)
        best_friend_slot = xp.full((self.config.population,), -1, dtype=xp.int32)
        best_target_agent = xp.full((self.config.population,), -1, dtype=xp.int32)
        best_need_good = xp.full((self.config.population,), -1, dtype=xp.int32)
        best_offer_good = xp.full((self.config.population,), -1, dtype=xp.int32)
        best_quantity = xp.zeros((self.config.population,), dtype=xp.float32)
        fallback_receive_transparency = xp.asarray(self.config.initial_transparency, dtype=xp.float32)

        for friend_slot in range(self.config.acquaintances):
            friend_index = self.state.friend_id[:, friend_slot]
            has_friend = friend_index >= 0
            safe_friend_index = xp.where(has_friend, friend_index, 0)
            friend_stock = self.state.stock[safe_friend_index]
            friend_need = self.state.need[safe_friend_index]
            friend_stock_limit = self.state.stock_limit[safe_friend_index]
            friend_purchase_price = self.state.purchase_price[safe_friend_index]
            friend_sales_price = self.state.sales_price[safe_friend_index]
            receive_transparency = self.state.transparency[row_index, friend_slot]

            for need_good in range(self.config.goods):
                self_need = self.state.need[:, need_good]
                self_stock_room = xp.maximum(self.state.stock_limit[:, need_good] - self.state.stock[:, need_good], 0.0)
                self_interest = self_need + self_stock_room if allow_stock_trade else self_need
                friend_stock_for_need = friend_stock[:, need_good]
                self_purchase_price = self.state.purchase_price[:, need_good]
                transparency_for_need = receive_transparency[:, need_good]

                for offer_good in range(self.config.goods):
                    if offer_good == need_good:
                        continue

                    self_stock_offer = self.state.stock[:, offer_good]
                    self_sales_price = self.state.sales_price[:, offer_good]
                    friend_need_for_offer = friend_need[:, offer_good]
                    friend_stock_room = xp.maximum(friend_stock_limit[:, offer_good] - friend_stock[:, offer_good], 0.0)
                    friend_interest = friend_need_for_offer + friend_stock_room if allow_stock_trade else friend_need_for_offer
                    friend_purchase_for_offer = friend_purchase_price[:, offer_good]
                    friend_sales_for_need = friend_sales_price[:, need_good]

                    quantity = xp.minimum(
                        self_interest,
                        xp.minimum(self_stock_offer, xp.minimum(friend_stock_for_need, friend_interest)),
                    )
                    friend_term = (friend_purchase_for_offer / xp.maximum(friend_sales_for_need, 1e-6)) * transparency_for_need
                    self_term = self_sales_price / xp.maximum(self_purchase_price * fallback_receive_transparency, 1e-6)
                    exchange_index = friend_term - self_term
                    score = quantity * exchange_index

                    valid = has_friend & (quantity > 0.0) & (exchange_index > 0.0)
                    better = valid & (score > best_score)

                    best_score = xp.where(better, score, best_score)
                    best_friend_slot = xp.where(better, friend_slot, best_friend_slot)
                    best_target_agent = xp.where(better, friend_index, best_target_agent)
                    best_need_good = xp.where(better, need_good, best_need_good)
                    best_offer_good = xp.where(better, offer_good, best_offer_good)
                    best_quantity = xp.where(better, quantity, best_quantity)

        self.state.trade.proposal_friend_slot[...] = best_friend_slot
        self.state.trade.proposal_target_agent[...] = best_target_agent
        self.state.trade.proposal_need_good[...] = best_need_good
        self.state.trade.proposal_offer_good[...] = best_offer_good
        self.state.trade.proposal_quantity[...] = best_quantity
        self.state.trade.proposal_score[...] = best_score
        return int(self.backend.to_scalar(xp.sum(best_score > 0.0)))

    def _score_trade_proposals_blocked(self, *, allow_stock_trade: bool) -> int:
        if self.backend.metadata.name == "cuda" and hasattr(self.backend, "score_trade_block"):
            return self._score_trade_proposals_blocked_cuda(allow_stock_trade=allow_stock_trade)

        xp = self.backend.xp
        population = self.config.population
        goods = self.config.goods
        acquaintances = self.config.acquaintances
        friend_block_size = min(self.config.cuda_friend_block, acquaintances)
        goods_block_size = min(self.config.cuda_goods_block, goods)
        initial_transparency = xp.asarray(self.config.initial_transparency, dtype=xp.float32)

        best_score = xp.zeros((population,), dtype=xp.float32)
        best_friend_slot = xp.full((population,), -1, dtype=xp.int32)
        best_target_agent = xp.full((population,), -1, dtype=xp.int32)
        best_need_good = xp.full((population,), -1, dtype=xp.int32)
        best_offer_good = xp.full((population,), -1, dtype=xp.int32)
        best_quantity = xp.zeros((population,), dtype=xp.float32)

        self_interest_full = self.state.need
        if allow_stock_trade:
            self_interest_full = self_interest_full + xp.maximum(self.state.stock_limit - self.state.stock, 0.0)

        self_stock_full = self.state.stock
        self_purchase_full = self.state.purchase_price
        self_sales_full = self.state.sales_price

        for friend_start in range(0, acquaintances, friend_block_size):
            friend_end = min(friend_start + friend_block_size, acquaintances)
            local_friend_count = friend_end - friend_start
            friend_index_block = self.state.friend_id[:, friend_start:friend_end]
            has_friend = friend_index_block >= 0
            safe_friend_index = xp.where(has_friend, friend_index_block, 0)
            friend_stock = self.state.stock[safe_friend_index]
            friend_need = self.state.need[safe_friend_index]
            friend_stock_limit = self.state.stock_limit[safe_friend_index]
            friend_purchase = self.state.purchase_price[safe_friend_index]
            friend_sales = self.state.sales_price[safe_friend_index]
            receive_transparency = self.state.transparency[:, friend_start:friend_end, :]

            for need_start in range(0, goods, goods_block_size):
                need_end = min(need_start + goods_block_size, goods)
                need_width = need_end - need_start
                self_interest_need = self_interest_full[:, need_start:need_end]
                friend_stock_need = friend_stock[:, :, need_start:need_end]
                self_purchase_need = self_purchase_full[:, need_start:need_end]
                transparency_need = receive_transparency[:, :, need_start:need_end]
                friend_sales_need = friend_sales[:, :, need_start:need_end]

                for offer_start in range(0, goods, goods_block_size):
                    offer_end = min(offer_start + goods_block_size, goods)
                    offer_width = offer_end - offer_start
                    self_stock_offer = self_stock_full[:, offer_start:offer_end]
                    self_sales_offer = self_sales_full[:, offer_start:offer_end]
                    friend_need_offer = friend_need[:, :, offer_start:offer_end]
                    friend_purchase_offer = friend_purchase[:, :, offer_start:offer_end]
                    if allow_stock_trade:
                        friend_offer_stock_room = xp.maximum(
                            friend_stock_limit[:, :, offer_start:offer_end] - friend_stock[:, :, offer_start:offer_end],
                            0.0,
                        )
                        friend_interest_offer = friend_need_offer + friend_offer_stock_room
                    else:
                        friend_interest_offer = friend_need_offer

                    quantity = xp.minimum(self_interest_need[:, None, :, None], self_stock_offer[:, None, None, :])
                    quantity = xp.minimum(quantity, friend_stock_need[:, :, :, None])
                    quantity = xp.minimum(quantity, friend_interest_offer[:, :, None, :])

                    friend_term = (
                        friend_purchase_offer[:, :, None, :] / xp.maximum(friend_sales_need[:, :, :, None], 1e-6)
                    ) * transparency_need[:, :, :, None]
                    self_term = self_sales_offer[:, None, None, :] / xp.maximum(
                        self_purchase_need[:, None, :, None] * initial_transparency,
                        1e-6,
                    )
                    exchange_index = friend_term - self_term
                    valid = has_friend[:, :, None, None] & (quantity > 0.0) & (exchange_index > 0.0)

                    if need_start < offer_end and offer_start < need_end:
                        need_idx = xp.arange(need_start, need_end, dtype=xp.int32)
                        offer_idx = xp.arange(offer_start, offer_end, dtype=xp.int32)
                        diagonal = need_idx[:, None] == offer_idx[None, :]
                        valid = valid & (~diagonal[None, None, :, :])

                    score = xp.where(valid, quantity * exchange_index, 0.0)
                    flat_width = local_friend_count * need_width * offer_width
                    flat_score = score.reshape(population, flat_width)
                    block_best_score = xp.max(flat_score, axis=1)
                    block_best_index = xp.argmax(flat_score, axis=1)
                    flat_quantity = quantity.reshape(population, flat_width)
                    block_best_quantity = xp.take_along_axis(flat_quantity, block_best_index[:, None], axis=1).reshape(-1)

                    local_friend = (block_best_index // (need_width * offer_width)).astype(xp.int32)
                    local_remainder = block_best_index % (need_width * offer_width)
                    local_need = (local_remainder // offer_width).astype(xp.int32)
                    local_offer = (local_remainder % offer_width).astype(xp.int32)

                    block_friend_slot = friend_start + local_friend
                    block_target_agent = xp.take_along_axis(friend_index_block, local_friend[:, None], axis=1).reshape(-1)
                    block_need_good = need_start + local_need
                    block_offer_good = offer_start + local_offer

                    better = block_best_score > best_score
                    best_score = xp.where(better, block_best_score, best_score)
                    best_friend_slot = xp.where(better, block_friend_slot, best_friend_slot)
                    best_target_agent = xp.where(better, block_target_agent, best_target_agent)
                    best_need_good = xp.where(better, block_need_good, best_need_good)
                    best_offer_good = xp.where(better, block_offer_good, best_offer_good)
                    best_quantity = xp.where(better, block_best_quantity, best_quantity)

        self.state.trade.proposal_friend_slot[...] = best_friend_slot
        self.state.trade.proposal_target_agent[...] = best_target_agent
        self.state.trade.proposal_need_good[...] = best_need_good
        self.state.trade.proposal_offer_good[...] = best_offer_good
        self.state.trade.proposal_quantity[...] = best_quantity
        self.state.trade.proposal_score[...] = best_score
        return int(self.backend.to_scalar(xp.sum(best_score > 0.0)))

    def _score_trade_proposals_blocked_cuda(self, *, allow_stock_trade: bool) -> int:
        xp = self.backend.xp
        population = self.config.population
        goods = self.config.goods
        acquaintances = self.config.acquaintances
        friend_block_size = min(self.config.cuda_friend_block, acquaintances)
        goods_block_size = min(self.config.cuda_goods_block, goods)

        best_score = xp.zeros((population,), dtype=xp.float32)
        best_friend_slot = xp.full((population,), -1, dtype=xp.int32)
        best_target_agent = xp.full((population,), -1, dtype=xp.int32)
        best_need_good = xp.full((population,), -1, dtype=xp.int32)
        best_offer_good = xp.full((population,), -1, dtype=xp.int32)
        best_quantity = xp.zeros((population,), dtype=xp.float32)

        self_interest_full = self.state.need
        if allow_stock_trade:
            self_interest_full = self_interest_full + xp.maximum(self.state.stock_limit - self.state.stock, 0.0)

        self_stock_full = self.state.stock
        self_purchase_full = self.state.purchase_price
        self_sales_full = self.state.sales_price

        need_blocks: list[tuple[int, int, object, object]] = []
        for need_start in range(0, goods, goods_block_size):
            need_end = min(need_start + goods_block_size, goods)
            need_blocks.append(
                (
                    need_start,
                    need_end,
                    xp.ascontiguousarray(self_interest_full[:, need_start:need_end]),
                    xp.ascontiguousarray(self_purchase_full[:, need_start:need_end]),
                )
            )

        offer_blocks: list[tuple[int, int, object, object]] = []
        for offer_start in range(0, goods, goods_block_size):
            offer_end = min(offer_start + goods_block_size, goods)
            offer_blocks.append(
                (
                    offer_start,
                    offer_end,
                    xp.ascontiguousarray(self_stock_full[:, offer_start:offer_end]),
                    xp.ascontiguousarray(self_sales_full[:, offer_start:offer_end]),
                )
            )

        for friend_start in range(0, acquaintances, friend_block_size):
            friend_end = min(friend_start + friend_block_size, acquaintances)
            friend_index_block = xp.ascontiguousarray(self.state.friend_id[:, friend_start:friend_end])
            safe_friend_index = xp.where(friend_index_block >= 0, friend_index_block, 0)
            friend_stock = self.state.stock[safe_friend_index]
            friend_need = self.state.need[safe_friend_index]
            friend_stock_limit = self.state.stock_limit[safe_friend_index]
            friend_purchase = self.state.purchase_price[safe_friend_index]
            friend_sales = self.state.sales_price[safe_friend_index]
            receive_transparency = self.state.transparency[:, friend_start:friend_end, :]

            for need_start, need_end, self_interest_need, self_purchase_need in need_blocks:
                friend_stock_need = xp.ascontiguousarray(friend_stock[:, :, need_start:need_end])
                transparency_need = xp.ascontiguousarray(receive_transparency[:, :, need_start:need_end])
                friend_sales_need = xp.ascontiguousarray(friend_sales[:, :, need_start:need_end])

                for offer_start, offer_end, self_stock_offer, self_sales_offer in offer_blocks:
                    friend_need_offer = friend_need[:, :, offer_start:offer_end]
                    friend_purchase_offer = xp.ascontiguousarray(friend_purchase[:, :, offer_start:offer_end])
                    if allow_stock_trade:
                        friend_offer_stock_room = xp.maximum(
                            friend_stock_limit[:, :, offer_start:offer_end] - friend_stock[:, :, offer_start:offer_end],
                            0.0,
                        )
                        friend_interest_offer = xp.ascontiguousarray(friend_need_offer + friend_offer_stock_room)
                    else:
                        friend_interest_offer = xp.ascontiguousarray(friend_need_offer)

                    self.backend.score_trade_block(
                        friend_start=friend_start,
                        need_start=need_start,
                        offer_start=offer_start,
                        friend_index_block=friend_index_block,
                        self_interest_need=self_interest_need,
                        self_stock_offer=self_stock_offer,
                        self_purchase_need=self_purchase_need,
                        self_sales_offer=self_sales_offer,
                        friend_stock_need=friend_stock_need,
                        friend_interest_offer=friend_interest_offer,
                        friend_purchase_offer=friend_purchase_offer,
                        friend_sales_need=friend_sales_need,
                        transparency_need=transparency_need,
                        best_score=best_score,
                        best_friend_slot=best_friend_slot,
                        best_target_agent=best_target_agent,
                        best_need_good=best_need_good,
                        best_offer_good=best_offer_good,
                        best_quantity=best_quantity,
                        initial_transparency=self.config.initial_transparency,
                    )

        self.state.trade.proposal_friend_slot[...] = best_friend_slot
        self.state.trade.proposal_target_agent[...] = best_target_agent
        self.state.trade.proposal_need_good[...] = best_need_good
        self.state.trade.proposal_offer_good[...] = best_offer_good
        self.state.trade.proposal_quantity[...] = best_quantity
        self.state.trade.proposal_score[...] = best_score
        return int(self.backend.to_scalar(xp.sum(best_score > 0.0)))

    def _commit_trades(self) -> tuple[int, float, float]:
        resolved = self.backend.resolve_trade_proposals(
            stock=self.state.stock,
            need=self.state.need,
            stock_limit=self.state.stock_limit,
            target_agent=self.state.trade.proposal_target_agent,
            need_good=self.state.trade.proposal_need_good,
            offer_good=self.state.trade.proposal_offer_good,
            quantity=self.state.trade.proposal_quantity,
            score=self.state.trade.proposal_score,
        )

        self.state.trade.accepted_mask[...] = resolved.accepted_mask
        self.state.trade.accepted_quantity[...] = resolved.accepted_quantity

        accepted_mask = self.backend.to_numpy(resolved.accepted_mask).astype(np.bool_, copy=False)
        accepted_quantity = self.backend.to_numpy(resolved.accepted_quantity).astype(np.float32, copy=False)
        accepted_count = int(np.count_nonzero(accepted_mask))
        accepted_volume = float(accepted_quantity[accepted_mask].sum()) if accepted_count else 0.0
        inventory_trade_volume = float(
            self.backend.to_scalar(
                self.backend.xp.sum(resolved.proposer_stock_added) + self.backend.xp.sum(resolved.target_stock_added)
            )
        )

        self.state.stock = resolved.stock
        self.state.need = resolved.need

        if accepted_count == 0:
            return 0, 0.0, 0.0

        committed = self.backend.commit_resolved_trades(
            stock=resolved.stock,
            need=resolved.need,
            recent_sales=self.state.recent_sales,
            recent_purchases=self.state.recent_purchases,
            recent_inventory_inflow=self.state.recent_inventory_inflow,
            friend_id=self.state.friend_id,
            friend_activity=self.state.friend_activity,
            transparency=self.state.transparency,
            proposal_friend_slot=self.state.trade.proposal_friend_slot,
            proposal_target_agent=self.state.trade.proposal_target_agent,
            proposal_need_good=self.state.trade.proposal_need_good,
            proposal_offer_good=self.state.trade.proposal_offer_good,
            accepted_mask=resolved.accepted_mask,
            accepted_quantity=resolved.accepted_quantity,
            proposer_stock_added=resolved.proposer_stock_added,
            target_stock_added=resolved.target_stock_added,
            initial_transparency=self.config.initial_transparency,
        )

        self.state.stock = committed.stock
        self.state.need = committed.need
        self.state.recent_sales = committed.recent_sales
        self.state.recent_purchases = committed.recent_purchases
        self.state.recent_inventory_inflow = committed.recent_inventory_inflow
        self.state.friend_activity = committed.friend_activity
        self.state.friend_id = committed.friend_id
        self.state.transparency = committed.transparency
        return accepted_count, accepted_volume, inventory_trade_volume

    def _introduce_random_contacts(self) -> None:
        candidate_ids = self.backend.plan_contact_candidates(
            friend_id=self.state.friend_id,
            seed=self.config.seed + 1,
            cycle=self.cycle,
        )
        self.backend.apply_contact_candidates(
            friend_id=self.state.friend_id,
            friend_activity=self.state.friend_activity,
            transparency=self.state.transparency,
            candidate_ids=candidate_ids,
            initial_activity=2.0,
            initial_transparency=self.config.initial_transparency,
        )


    def _apply_leisure_demand(self) -> bool:
        xp = self.backend.xp
        utilized_time = xp.maximum(self.state.cycle_time_budget - self.state.time_remaining, 1e-6)
        needs_increment = self.state.cycle_time_budget / utilized_time
        extra_multiplier = xp.clip(needs_increment - 1.0, 0.0, self.config.max_leisure_extra_multiplier)
        if float(self.backend.to_scalar(xp.max(extra_multiplier))) <= 0.0:
            return False

        production_cost = 1.0 / xp.maximum(self.state.efficiency, 1e-6)
        average_cost_by_good = xp.mean(production_cost, axis=0)
        mean_cost = xp.maximum(xp.mean(average_cost_by_good), 1e-6)
        elasticity_weights = mean_cost / xp.maximum(average_cost_by_good, 1e-6)
        elasticity_weights = elasticity_weights / xp.maximum(xp.mean(elasticity_weights), 1e-6)

        extra_need = self.state.base_need * extra_multiplier[:, None] * elasticity_weights[None, :]
        extra_need_total = float(self.backend.to_scalar(xp.sum(extra_need)))
        if extra_need_total <= 0.0:
            return False

        self.state.need += extra_need
        self._cycle_need_total += extra_need_total
        self._leisure_extra_need_total += extra_need_total
        return True

    def _find_friend_slot(self, friend_row: np.ndarray, agent_id: int) -> int:
        matches = np.nonzero(friend_row == agent_id)[0]
        if matches.size == 0:
            return -1
        return int(matches[0])

    def _produce_for_needs(self) -> None:
        xp = self.backend.xp
        required_time = self.state.need / self.state.efficiency
        total_required_time = xp.sum(required_time, axis=1)
        scale = xp.ones_like(total_required_time)
        limited = total_required_time > self.state.time_remaining
        safe_required = xp.maximum(total_required_time, 1e-6)
        scale = xp.where(limited, self.state.time_remaining / safe_required, scale)
        produced = self.state.need * scale[:, None]
        time_spent = xp.sum(produced / self.state.efficiency, axis=1)

        self._production_total += float(self.backend.to_scalar(xp.sum(produced)))
        self.state.need -= produced
        self.state.recent_production += produced
        self.state.time_remaining -= time_spent
        self.state.time_remaining = xp.maximum(self.state.time_remaining, 0.0)

    def _produce_surplus(self) -> None:
        xp = self.backend.xp
        row_index = xp.arange(self.config.population, dtype=xp.int32)
        stock_room = xp.maximum(self.state.stock_limit - self.state.stock, 0.0)
        talented_goods = self.state.talent_mask > 0.0
        eligible_goods = talented_goods & (stock_room > 0.0)
        eligible_any = xp.any(eligible_goods, axis=1)
        surplus_priority = xp.where(eligible_goods, self.state.efficiency * self.state.sales_price, -1.0)
        chosen_good = xp.argmax(surplus_priority, axis=1)
        chosen_efficiency = self.state.efficiency[row_index, chosen_good]
        chosen_stock_room = stock_room[row_index, chosen_good]
        max_units = self.state.time_remaining * chosen_efficiency
        surplus_units = xp.where(eligible_any, xp.minimum(max_units, chosen_stock_room), 0.0)

        self._production_total += float(self.backend.to_scalar(xp.sum(surplus_units)))
        self._surplus_output_total += float(self.backend.to_scalar(xp.sum(surplus_units)))
        self.state.stock[row_index, chosen_good] += surplus_units
        self.state.recent_production[row_index, chosen_good] += surplus_units
        self.state.time_remaining -= xp.where(eligible_any, surplus_units / xp.maximum(chosen_efficiency, 1e-6), 0.0)
        self.state.time_remaining = xp.maximum(self.state.time_remaining, 0.0)

    def _update_efficiency_from_learning(self) -> None:
        xp = self.backend.xp
        window = xp.asarray(self.config.learning_window, dtype=xp.float32)
        normalized_production = self.state.recent_production / xp.maximum(self.state.base_need * window, 1e-6)
        learning_multiplier = xp.sqrt(xp.maximum(normalized_production, 1.0))
        self.state.learned_efficiency[...] = self.config.initial_efficiency * learning_multiplier
        self.state.efficiency[...] = xp.maximum(self.state.innate_efficiency, self.state.learned_efficiency)

    def _apply_spoilage(self) -> None:
        xp = self.backend.xp
        overflow = xp.maximum(self.state.stock - self.state.stock_limit, 0.0)
        spoiled = overflow * self.config.spoilage_rate
        self.state.stock -= spoiled

    def _update_private_values(self) -> None:
        xp = self.backend.xp
        production_cost = 1.0 / xp.maximum(self.state.efficiency, 1e-6)
        small_stock = self.state.stock <= 0.0
        normal_stock = (self.state.stock > 0.0) & (self.state.stock <= self.state.stock_limit)
        large_stock = self.state.stock > self.state.stock_limit
        sales_high = self.state.recent_sales > self.state.recent_production
        sales_small = self.state.recent_sales <= 0.0
        purchase_volume_considerable = self.state.recent_purchases > 0.0

        purchase_price = self.state.purchase_price
        purchase_price = xp.where(
            small_stock,
            xp.minimum(xp.maximum(purchase_price, production_cost), xp.maximum(production_cost, self.state.sales_price)),
            purchase_price,
        )
        purchase_price = xp.where(
            small_stock & sales_high,
            xp.maximum(purchase_price, production_cost),
            purchase_price,
        )
        purchase_price = xp.where(
            normal_stock & sales_small,
            xp.minimum(purchase_price, self.state.sales_price),
            purchase_price,
        )
        purchase_price = xp.where(
            normal_stock & sales_small,
            xp.maximum(purchase_price, production_cost),
            purchase_price,
        )
        purchase_price = xp.where(
            large_stock & sales_small,
            xp.minimum(purchase_price, production_cost),
            purchase_price,
        )

        sales_price = self.state.sales_price
        sales_price = xp.where(
            small_stock & sales_high,
            xp.maximum(sales_price, production_cost),
            sales_price,
        )
        sales_price = xp.where(
            small_stock & sales_high & purchase_volume_considerable,
            xp.maximum(sales_price, purchase_price),
            sales_price,
        )
        sales_price = xp.where(
            normal_stock & sales_small,
            production_cost,
            sales_price,
        )
        sales_price = xp.where(
            large_stock,
            xp.minimum(sales_price, production_cost),
            sales_price,
        )

        self.state.purchase_price = xp.maximum(purchase_price, 0.05)
        self.state.sales_price = xp.maximum(sales_price, 0.05)

