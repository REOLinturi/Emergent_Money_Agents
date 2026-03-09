from __future__ import annotations

import numpy as np

from .backend import create_backend
from .backend.base import BaseBackend
from .config import SimulationConfig
from .initialization import create_initial_state
from .metrics import MetricsSnapshot, compute_metrics
from .state import SimulationState


class SimulationEngine:
    def __init__(self, config: SimulationConfig, backend: BaseBackend, state: SimulationState | None = None) -> None:
        self.config = config
        self.backend = backend
        self.state = state or create_initial_state(config, backend)
        self.cycle = 0
        self.history: list[MetricsSnapshot] = []
        self._cycle_need_total = self._base_need_total()
        self._proposed_trade_count = 0
        self._accepted_trade_count = 0
        self._accepted_trade_volume = 0.0
        self._network_rng = np.random.default_rng(config.seed + 1)

    @classmethod
    def create(cls, config: SimulationConfig | None = None, backend_name: str = "numpy") -> "SimulationEngine":
        resolved_config = config or SimulationConfig()
        backend = create_backend(backend_name)
        state = create_initial_state(resolved_config, backend)
        return cls(config=resolved_config, backend=backend, state=state)

    def step(self) -> MetricsSnapshot:
        self._start_cycle()
        self._run_basic_round()
        self._run_leisure_round()
        self._update_efficiency_from_learning()
        self._apply_spoilage()
        self._update_private_values()
        self.backend.synchronize()

        self.cycle += 1
        snapshot = self.snapshot_metrics()
        self.history.append(snapshot)
        return snapshot

    def snapshot_metrics(self) -> MetricsSnapshot:
        return compute_metrics(
            cycle=self.cycle,
            state=self.state,
            backend=self.backend,
            cycle_need_total=self._cycle_need_total,
            proposed_trade_count=self._proposed_trade_count,
            accepted_trade_count=self._accepted_trade_count,
            accepted_trade_volume=self._accepted_trade_volume,
        )

    def _base_need_total(self) -> float:
        return float(self.backend.to_scalar(self.backend.xp.sum(self.state.base_need)))

    def _start_cycle(self) -> None:
        self.state.need[...] = self.state.base_need
        self.state.time_remaining[...] = self.state.cycle_time_budget
        self.state.friend_activity *= self.config.activity_discount
        self.state.recent_production *= self.config.activity_discount
        self.state.recent_sales *= self.config.activity_discount
        self.state.recent_purchases *= self.config.activity_discount
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
        accepted_count, accepted_volume = self._commit_trades()
        self._accepted_trade_count += accepted_count
        self._accepted_trade_volume += accepted_volume
        self._produce_for_needs()

    def _consume_from_stock(self) -> None:
        xp = self.backend.xp
        consumed = xp.minimum(self.state.need, self.state.stock)
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

    def _commit_trades(self) -> tuple[int, float]:
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

        if accepted_count == 0:
            return 0, 0.0

        committed = self.backend.commit_resolved_trades(
            stock=self.state.stock,
            need=self.state.need,
            recent_sales=self.state.recent_sales,
            recent_purchases=self.state.recent_purchases,
            friend_id=self.state.friend_id,
            friend_activity=self.state.friend_activity,
            transparency=self.state.transparency,
            proposal_friend_slot=self.state.trade.proposal_friend_slot,
            proposal_target_agent=self.state.trade.proposal_target_agent,
            proposal_need_good=self.state.trade.proposal_need_good,
            proposal_offer_good=self.state.trade.proposal_offer_good,
            accepted_mask=resolved.accepted_mask,
            accepted_quantity=resolved.accepted_quantity,
            initial_transparency=self.config.initial_transparency,
        )

        self.state.stock[...] = committed.stock
        self.state.need[...] = committed.need
        self.state.recent_sales[...] = committed.recent_sales
        self.state.recent_purchases[...] = committed.recent_purchases
        self.state.friend_activity[...] = committed.friend_activity
        self.state.friend_id[...] = committed.friend_id
        self.state.transparency[...] = committed.transparency
        return accepted_count, accepted_volume

    def _introduce_random_contacts(self) -> None:
        friend_id = self.backend.to_numpy(self.state.friend_id).astype(np.int32, copy=True)
        friend_activity = self.backend.to_numpy(self.state.friend_activity).astype(np.float32, copy=True)
        transparency = self.backend.to_numpy(self.state.transparency).astype(np.float32, copy=True)

        for agent_id in range(self.config.population):
            self._assign_friend_candidate(
                friend_id=friend_id,
                friend_activity=friend_activity,
                transparency=transparency,
                agent_id=agent_id,
                suggested_id=None,
                initial_activity=2.0,
            )

        self.state.friend_id[...] = self.backend.asarray(friend_id, dtype=np.int32)
        self.state.friend_activity[...] = self.backend.asarray(friend_activity, dtype=np.float32)
        self.state.transparency[...] = self.backend.asarray(transparency, dtype=np.float32)

    def _assign_friend_candidate(
        self,
        *,
        friend_id: np.ndarray,
        friend_activity: np.ndarray,
        transparency: np.ndarray,
        agent_id: int,
        suggested_id: int | None,
        initial_activity: float,
    ) -> int:
        candidate_id = suggested_id
        if candidate_id is None:
            candidate_id = self._pick_random_candidate(agent_id=agent_id, current_friends=friend_id[agent_id])
        if candidate_id is None:
            return -1

        existing_slot = self._find_friend_slot(friend_id[agent_id], candidate_id)
        if existing_slot >= 0:
            friend_activity[agent_id, existing_slot] = max(friend_activity[agent_id, existing_slot], initial_activity)
            return existing_slot

        target_slot = self._select_friend_slot(friend_id[agent_id], friend_activity[agent_id])
        if target_slot < 0:
            return -1

        friend_id[agent_id, target_slot] = candidate_id
        friend_activity[agent_id, target_slot] = initial_activity
        transparency[agent_id, target_slot, :] = self.config.initial_transparency
        return target_slot

    def _pick_random_candidate(self, *, agent_id: int, current_friends: np.ndarray) -> int | None:
        if self.config.population <= 1:
            return None

        known = {int(friend) for friend in current_friends.tolist() if int(friend) >= 0}
        known.add(agent_id)
        if len(known) >= self.config.population:
            return None

        max_attempts = max(2 * self.config.acquaintances, 8)
        for _ in range(max_attempts):
            candidate = int(self._network_rng.integers(0, self.config.population - 1))
            if candidate >= agent_id:
                candidate += 1
            if candidate not in known:
                return candidate

        for candidate in range(self.config.population):
            if candidate not in known:
                return candidate
        return None

    def _select_friend_slot(self, friend_row: np.ndarray, activity_row: np.ndarray) -> int:
        empty_slots = np.flatnonzero(friend_row < 0)
        if empty_slots.size > 0:
            return int(empty_slots[0])
        return int(np.argmin(activity_row))

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
