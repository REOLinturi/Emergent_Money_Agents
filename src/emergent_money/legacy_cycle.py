from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import sqrt

import numpy as np

from .config import SimulationConfig
from .hybrid_batching import NegotiationCandidate, schedule_conflict_free_batches
from .legacy_search_backend import (
    ExchangePlanRequest,
    ExchangePlanResult,
    ExchangeSearchBackend,
    ExchangeSearchRequest,
    build_exchange_search_backend,
    execute_exchange_planning,
    execute_exchange_search,
)
from .legacy_cycle_native import build_native_legacy_cycle_backend
from .state import ROLE_CONSUMER, ROLE_PRODUCER, ROLE_RETAILER, SimulationState

_EPSILON = 1e-6
_SURPLUS_DEAL = 1
_CONSUMPTION_DEAL = 2


@dataclass(slots=True, frozen=True)
class ExchangeOption:
    score: float
    friend_slot: int
    friend_id: int
    offer_good: int


@dataclass(slots=True, frozen=True)
class PlannedHybridExchange:
    deal_type: int
    agent_id: int
    need_good: int
    max_need: float
    exchange: ExchangeOption
    execution_plan: ExchangePlanResult | None = None

    def participants(self) -> tuple[int, int]:
        return (self.agent_id, self.exchange.friend_id)

    @property
    def priority(self) -> float:
        return float(self.exchange.score)

    @property
    def planned_quantity(self) -> float:
        if self.execution_plan is None:
            return 0.0
        return float(self.execution_plan.max_exchange)


@dataclass(slots=True, frozen=True)
class HybridExchangeBatchPlan:
    seed: int
    batches: tuple[tuple[PlannedHybridExchange, ...], ...]
    dropped: tuple[PlannedHybridExchange, ...]
    candidate_agent_ids: tuple[int, ...] = ()
    no_candidate_reasons: dict[str, int] = field(default_factory=dict)

    @property
    def scheduled_count(self) -> int:
        return sum(len(batch) for batch in self.batches)


@dataclass(slots=True, frozen=True)
class HybridWaveDiagnostics:
    stage: str
    frontier_start: int
    frontier_agents: int
    wave_offset: int
    incoming_active_agents: int
    candidate_agents: int
    no_candidate_agents: int
    no_candidate_reasons: dict[str, int]
    scheduled_exchanges: int
    scheduled_quantity_total: float
    dropped_exchanges: int
    executed_exchanges: int
    executed_quantity_total: float
    execution_failure_reasons: dict[str, int]
    exhausted_retry_agents: int
    remaining_active_agents: int


@dataclass(slots=True, frozen=True)
class ExchangeExecutionResult:
    executed: bool
    exhausted_gift: bool
    failure_reason: str | None = None
    executed_quantity: float = 0.0


class LegacyCycleRunner:
    def __init__(self, engine, exchange_search_backend: ExchangeSearchBackend | None = None) -> None:
        self.engine = engine
        self.config: SimulationConfig = engine.config
        self.state: SimulationState = engine.state
        if engine.backend.metadata.name != "numpy":
            raise RuntimeError("Exact legacy mechanics currently require the NumPy backend")
        self.market = self.state.market
        self.period_length = float(self.config.cycle_time_budget)
        self._friend_slot_maps = self._build_friend_slot_maps()
        self._exchange_search = exchange_search_backend or build_exchange_search_backend()
        self._native_cycle = build_native_legacy_cycle_backend(self)
        self._hybrid_wave_diagnostics: list[HybridWaveDiagnostics] = []

    def run(self) -> None:
        self._reset_cycle_state()
        if self._uses_experimental_hybrid_exchange():
            self._run_cycle_with_experimental_hybrid()
        else:
            for agent_id in range(self.config.population):
                self._run_agent_cycle(agent_id)
        self._finalize_cycle_after_agent_loop()

    def _finalize_cycle_after_agent_loop(self) -> None:
        self._evaluate_market_prices()
        self.market.losers = int(np.count_nonzero(self.state.period_time_debt < (-1.0 * self.period_length)))
        self.engine._prepare_trade_frontier()

    def _reset_cycle_state(self) -> None:
        state = self.state
        market = self.market

        state.time_remaining[...] = self.period_length
        state.periodic_spoilage[...] = 0.0
        state.spoilage[...] = 0.0
        state.need[...] = 0.0
        state.produced_this_period[...] = 0.0
        state.sold_this_period[...] = 0.0
        state.purchased_this_period[...] = 0.0
        state.trade.active_friend_slot[...] = -1
        state.trade.active_friend_id[...] = -1
        state.trade.candidate_need_good[...] = 0
        state.trade.candidate_offer_good[...] = 0
        state.trade.proposal_friend_slot[...] = -1
        state.trade.proposal_target_agent[...] = -1
        state.trade.proposal_need_good[...] = -1
        state.trade.proposal_offer_good[...] = -1
        state.trade.proposal_quantity[...] = 0.0
        state.trade.proposal_score[...] = 0.0
        state.trade.accepted_mask[...] = False
        state.trade.accepted_quantity[...] = 0.0

        market.produced_this_period[...] = 0.0
        market.periodic_tce_cost[...] = 0.0
        market.periodic_spoilage[...] = 0.0
        market.cost_of_tce_in_time[...] = 0.0
        market.cost_of_spoilage_in_time[...] = 0.0
        market.consumer_count[...] = 0
        market.retailer_count[...] = 0
        market.producer_count[...] = 0
        market.total_cost_of_tce_in_time = 0.0
        market.total_cost_of_spoilage_in_time = 0.0
        market.losers = 0

        self.engine._cycle_need_total = 0.0
        self.engine._proposed_trade_count = 0
        self.engine._accepted_trade_count = 0
        self.engine._accepted_trade_volume = 0.0
        self.engine._production_total = 0.0
        self.engine._surplus_output_total = 0.0
        self.engine._stock_consumption_total = 0.0
        self.engine._leisure_extra_need_total = 0.0
        self.engine._inventory_trade_volume = 0.0
        self.engine.exact_cycle_diagnostics = None
        self._hybrid_wave_diagnostics = []

    def _run_agent_cycle(self, agent_id: int) -> None:
        self._prepare_agent_for_consumption(agent_id)
        self._satisfy_needs_by_exchange(agent_id)
        self._complete_agent_cycle_after_consumption(agent_id)

    def _uses_experimental_hybrid_consumption(self) -> bool:
        return bool(
            self.config.experimental_hybrid_consumption_stage
            and self.config.experimental_hybrid_batches > 0
        )

    def _uses_experimental_hybrid_surplus(self) -> bool:
        return bool(
            self.config.experimental_hybrid_surplus_stage
            and self.config.experimental_hybrid_batches > 0
        )

    def _uses_experimental_hybrid_exchange(self) -> bool:
        return bool(
            self.config.experimental_hybrid_batches > 0
            and (
                self.config.experimental_hybrid_consumption_stage
                or self.config.experimental_hybrid_surplus_stage
            )
        )

    def _run_cycle_with_experimental_hybrid(self) -> None:
        frontier_size = self._resolved_experimental_hybrid_frontier_size()
        # Mirror the sequential search's retry budget so failed replans cannot livelock a frontier wave.
        max_attempts = self.config.goods * self.config.acquaintances
        consumption_stage = self._uses_experimental_hybrid_consumption()
        surplus_stage = self._uses_experimental_hybrid_surplus() and not self.config.experimental_hybrid_rolling_frontier
        if consumption_stage:
            if self.config.experimental_hybrid_rolling_frontier:
                self._run_cycle_with_rolling_experimental_consumption(frontier_size, max_attempts)
            else:
                self._run_cycle_with_fixed_experimental_consumption(frontier_size, max_attempts, surplus_stage)
        elif surplus_stage:
            self._run_fixed_experimental_surplus_only(frontier_size, max_attempts)
        else:
            for agent_id in range(self.config.population):
                self._run_agent_cycle(agent_id)
        self._finalize_hybrid_cycle_diagnostics(
            frontier_size,
            consumption_stage=consumption_stage,
            surplus_stage=surplus_stage,
        )

    def _run_cycle_with_fixed_experimental_consumption(
        self,
        frontier_size: int,
        max_attempts: int,
        surplus_stage: bool,
    ) -> None:
        for frontier_start in range(0, self.config.population, frontier_size):
            frontier_agent_ids = tuple(range(frontier_start, min(frontier_start + frontier_size, self.config.population)))
            for agent_id in frontier_agent_ids:
                self._prepare_agent_for_consumption(agent_id)

            self._run_fixed_hybrid_exchange_stage(
                frontier_start=frontier_start,
                frontier_agent_ids=frontier_agent_ids,
                max_attempts=max_attempts,
                deal_type=_CONSUMPTION_DEAL,
                stage='consumption',
            )

            if surplus_stage:
                for agent_id in frontier_agent_ids:
                    self._advance_agent_to_surplus_stage(agent_id)
                self._run_fixed_hybrid_exchange_stage(
                    frontier_start=frontier_start,
                    frontier_agent_ids=frontier_agent_ids,
                    max_attempts=max_attempts,
                    deal_type=_SURPLUS_DEAL,
                    stage='surplus',
                )
                for agent_id in frontier_agent_ids:
                    self._complete_agent_period_after_surplus(agent_id)
                continue

            for agent_id in frontier_agent_ids:
                self._complete_agent_cycle_after_consumption(agent_id)

    def _run_fixed_experimental_surplus_only(self, frontier_size: int, max_attempts: int) -> None:
        for frontier_start in range(0, self.config.population, frontier_size):
            frontier_agent_ids = tuple(range(frontier_start, min(frontier_start + frontier_size, self.config.population)))
            for agent_id in frontier_agent_ids:
                self._prepare_agent_for_consumption(agent_id)
                self._satisfy_needs_by_exchange(agent_id)
                self._advance_agent_to_surplus_stage(agent_id)

            self._run_fixed_hybrid_exchange_stage(
                frontier_start=frontier_start,
                frontier_agent_ids=frontier_agent_ids,
                max_attempts=max_attempts,
                deal_type=_SURPLUS_DEAL,
                stage='surplus',
            )

            for agent_id in frontier_agent_ids:
                self._complete_agent_period_after_surplus(agent_id)

    def _run_fixed_hybrid_exchange_stage(
        self,
        *,
        frontier_start: int,
        frontier_agent_ids: tuple[int, ...],
        max_attempts: int,
        deal_type: int,
        stage: str,
    ) -> None:
        frontier_agent_set = set(frontier_agent_ids)
        blocked_partner_ids = frontier_agent_set if self.config.experimental_hybrid_block_frontier_partners else set()
        active_agent_ids = tuple(
            agent_id
            for agent_id in frontier_agent_ids
            if self._has_remaining_hybrid_stage_need(agent_id, deal_type)
        )
        attempts_remaining = {agent_id: max_attempts for agent_id in active_agent_ids}
        wave_offset = 0
        while active_agent_ids:
            incoming_active_agents = len(active_agent_ids)
            plan = self._plan_hybrid_batches_for_stage(
                deal_type=deal_type,
                batch_count=self.config.experimental_hybrid_batches,
                proposer_ids=active_agent_ids,
                blocked_partner_ids=blocked_partner_ids,
                one_candidate_per_agent=True,
                seed_offset=(frontier_start * self.config.experimental_hybrid_seed_stride) + wave_offset,
            )
            if plan.scheduled_count <= 0:
                self._record_hybrid_wave_diagnostics(
                    stage=stage,
                    frontier_start=frontier_start,
                    frontier_agents=len(frontier_agent_ids),
                    wave_offset=wave_offset,
                    incoming_active_agents=incoming_active_agents,
                    plan=plan,
                    executed_exchanges=0,
                    scheduled_quantity_total=0.0,
                    executed_quantity_total=0.0,
                    execution_failure_reasons={},
                    exhausted_retry_agents=0,
                    remaining_active_agents=0,
                )
                break
            for agent_id in plan.candidate_agent_ids:
                attempts_remaining[agent_id] -= 1
            executed_exchanges, execution_failure_reasons, scheduled_quantity_total, executed_quantity_total = self._execute_hybrid_batch_plan(plan)
            candidate_agent_set = set(plan.candidate_agent_ids)
            exhausted_retry_agents = sum(
                1
                for agent_id in candidate_agent_set
                if attempts_remaining[agent_id] <= 0 and self._has_remaining_hybrid_stage_need(agent_id, deal_type)
            )
            next_active_agent_ids = tuple(
                agent_id
                for agent_id in active_agent_ids
                if attempts_remaining[agent_id] > 0
                and agent_id in candidate_agent_set
                and self._has_remaining_hybrid_stage_need(agent_id, deal_type)
            )
            self._record_hybrid_wave_diagnostics(
                stage=stage,
                frontier_start=frontier_start,
                frontier_agents=len(frontier_agent_ids),
                wave_offset=wave_offset,
                incoming_active_agents=incoming_active_agents,
                plan=plan,
                executed_exchanges=executed_exchanges,
                scheduled_quantity_total=scheduled_quantity_total,
                executed_quantity_total=executed_quantity_total,
                execution_failure_reasons=execution_failure_reasons,
                exhausted_retry_agents=exhausted_retry_agents,
                remaining_active_agents=len(next_active_agent_ids),
            )
            active_agent_ids = next_active_agent_ids
            wave_offset += 1

    def _run_cycle_with_rolling_experimental_consumption(self, frontier_size: int, max_attempts: int) -> None:
        next_agent_id = 0
        attempts_remaining: dict[int, int] = {}
        active_agent_ids: list[int] = []
        wave_offset = 0

        def refill_active_frontier() -> None:
            nonlocal next_agent_id
            while len(active_agent_ids) < frontier_size and next_agent_id < self.config.population:
                agent_id = next_agent_id
                next_agent_id += 1
                self._prepare_agent_for_consumption(agent_id)
                attempts_remaining[agent_id] = max_attempts
                active_agent_ids.append(agent_id)

        refill_active_frontier()
        while active_agent_ids:
            active_tuple = tuple(active_agent_ids)
            frontier_start = active_tuple[0]
            blocked_partner_ids = set(active_tuple) if self.config.experimental_hybrid_block_frontier_partners else set()
            incoming_active_agents = len(active_tuple)
            plan = self.plan_experimental_consumption_batches(
                batch_count=self.config.experimental_hybrid_batches,
                proposer_ids=active_tuple,
                blocked_partner_ids=blocked_partner_ids,
                one_candidate_per_agent=True,
                seed_offset=(frontier_start * self.config.experimental_hybrid_seed_stride) + wave_offset,
            )
            if plan.scheduled_count <= 0:
                retiring_agent_ids = tuple(active_agent_ids)
                self._record_hybrid_wave_diagnostics(
                    stage='consumption',
                    frontier_start=frontier_start,
                    frontier_agents=len(active_tuple),
                    wave_offset=wave_offset,
                    incoming_active_agents=incoming_active_agents,
                    plan=plan,
                    executed_exchanges=0,
                    scheduled_quantity_total=0.0,
                    executed_quantity_total=0.0,
                    execution_failure_reasons={},
                    exhausted_retry_agents=0,
                    remaining_active_agents=0,
                )
                active_agent_ids.clear()
                for agent_id in retiring_agent_ids:
                    self._complete_agent_cycle_after_consumption(agent_id)
                refill_active_frontier()
                wave_offset += 1
                continue

            for agent_id in plan.candidate_agent_ids:
                attempts_remaining[agent_id] -= 1
            executed_exchanges, execution_failure_reasons, scheduled_quantity_total, executed_quantity_total = self._execute_hybrid_batch_plan(plan)
            candidate_agent_set = set(plan.candidate_agent_ids)
            exhausted_retry_agents = sum(
                1
                for agent_id in candidate_agent_set
                if attempts_remaining[agent_id] <= 0 and self._has_unmet_consumption_need(agent_id)
            )
            continuing_agent_ids: list[int] = []
            retiring_agent_ids: list[int] = []
            for agent_id in active_agent_ids:
                if (
                    attempts_remaining[agent_id] > 0
                    and agent_id in candidate_agent_set
                    and self._has_unmet_consumption_need(agent_id)
                ):
                    continuing_agent_ids.append(agent_id)
                else:
                    retiring_agent_ids.append(agent_id)
            active_agent_ids[:] = continuing_agent_ids
            for agent_id in retiring_agent_ids:
                self._complete_agent_cycle_after_consumption(agent_id)
            refill_active_frontier()
            self._record_hybrid_wave_diagnostics(
                stage='consumption',
                frontier_start=frontier_start,
                frontier_agents=len(active_tuple),
                wave_offset=wave_offset,
                incoming_active_agents=incoming_active_agents,
                plan=plan,
                executed_exchanges=executed_exchanges,
                scheduled_quantity_total=scheduled_quantity_total,
                executed_quantity_total=executed_quantity_total,
                execution_failure_reasons=execution_failure_reasons,
                exhausted_retry_agents=exhausted_retry_agents,
                remaining_active_agents=len(active_agent_ids),
            )
            wave_offset += 1

    def _resolved_experimental_hybrid_frontier_size(self) -> int:
        if self.config.experimental_hybrid_frontier_size > 0:
            return self.config.experimental_hybrid_frontier_size
        return max(1, self.config.experimental_hybrid_batches)

    def _has_unmet_consumption_need(self, agent_id: int) -> bool:
        return bool(np.any(self.state.need[agent_id] >= 1.0))

    def _has_remaining_hybrid_stage_need(self, agent_id: int, deal_type: int) -> bool:
        for need_good in range(self.config.goods):
            if self._hybrid_exchange_need(agent_id=agent_id, need_good=need_good, deal_type=deal_type) > 0.0:
                return True
        return False

    def _plan_hybrid_batches_for_stage(
        self,
        *,
        deal_type: int,
        batch_count: int,
        proposer_ids: tuple[int, ...],
        blocked_partner_ids: set[int],
        one_candidate_per_agent: bool,
        seed_offset: int,
    ) -> HybridExchangeBatchPlan:
        if deal_type == _SURPLUS_DEAL:
            return self.plan_experimental_surplus_batches(
                batch_count=batch_count,
                proposer_ids=proposer_ids,
                blocked_partner_ids=blocked_partner_ids,
                one_candidate_per_agent=one_candidate_per_agent,
                seed_offset=seed_offset,
            )
        return self.plan_experimental_consumption_batches(
            batch_count=batch_count,
            proposer_ids=proposer_ids,
            blocked_partner_ids=blocked_partner_ids,
            one_candidate_per_agent=one_candidate_per_agent,
            seed_offset=seed_offset,
        )

    def _uses_experimental_native_stage_math(self) -> bool:
        return bool(
            self.config.experimental_native_stage_math
            and not self._uses_experimental_hybrid_exchange()
            and self._native_cycle is not None
        )

    def _uses_experimental_native_exchange_stage(self) -> bool:
        return bool(
            self.config.experimental_native_exchange_stage
            and getattr(self.engine, '_allow_rejected_native_exchange_stage', False)
            and not self._uses_experimental_hybrid_exchange()
            and self._native_cycle is not None
            and self._native_cycle.supports_run_exchange_stage
        )

    def _prepare_agent_for_consumption(self, agent_id: int) -> None:
        if self._uses_experimental_native_stage_math() and self._native_cycle.supports_prepare_agent_for_consumption:
            stock_before = self.state.stock[agent_id].copy()
            self._native_cycle.prepare_agent_for_consumption(agent_id=agent_id)
            cycle_need = self._baseline_cycle_need(agent_id)
            consumed = np.minimum(stock_before, cycle_need)
            stock_consumed_total = float(np.sum(consumed))
            cycle_need_total = float(np.sum(cycle_need))
            self.engine._cycle_need_total += cycle_need_total
            self.engine._stock_consumption_total += stock_consumed_total
            return
        stock_level = self._evaluate_stock(agent_id)
        self._update_needs_level(agent_id, stock_level)
        self._set_cycle_needs(agent_id)
        self._consume_surplus(agent_id)

    def _complete_agent_cycle_after_consumption(self, agent_id: int) -> None:
        self._advance_agent_to_surplus_stage(agent_id)
        self._make_surplus_deals(agent_id)
        self._complete_agent_period_after_surplus(agent_id)

    def _advance_agent_to_surplus_stage(self, agent_id: int) -> None:
        self._produce_need(agent_id)
        self.state.period_failure[agent_id] = self.state.time_remaining[agent_id] < 0.0
        self._add_random_friend(agent_id)

        if self.state.period_time_debt[agent_id] < 0.0:
            half_debt = self.state.period_time_debt[agent_id] / 2.0
            self.state.time_remaining[agent_id] += half_debt
            self.state.period_time_debt[agent_id] = half_debt

        self._surplus_production(agent_id)
        self._run_leisure_round(agent_id)

    def _complete_agent_period_after_surplus(self, agent_id: int) -> None:
        self.state.period_time_debt[agent_id] += self.state.time_remaining[agent_id]
        if self.state.period_time_debt[agent_id] > 0.0:
            self.state.period_time_debt[agent_id] = 0.0
        if (
            self._uses_experimental_native_stage_math()
            and self._native_cycle.supports_leisure_production
            and self._native_cycle.supports_end_agent_period
        ):
            produced_total = self._native_cycle.leisure_production(agent_id=agent_id)
            self.engine._production_total += produced_total
            self.engine._surplus_output_total += produced_total
            self._native_cycle.end_agent_period(cycle=self.engine.cycle, agent_id=agent_id)
            return
        self._leisure_production(agent_id)
        self._end_agent_period(agent_id)

    def _evaluate_stock(self, agent_id: int) -> float:
        elastic_need = self.market.elastic_need * self.state.needs_level[agent_id]
        wealth_minus_needs = self.period_length
        total_needs_value = 0.0

        for good_id in range(self.config.goods):
            surplus_value = self.state.stock[agent_id, good_id] - (elastic_need[good_id] * self.config.max_needs_increase)
            if surplus_value < 0.0:
                if self.state.purchased_last_period[agent_id, good_id] > (-1.0 * surplus_value):
                    surplus_value *= self.state.purchase_price[agent_id, good_id]
                else:
                    surplus_value /= max(float(self.state.efficiency[agent_id, good_id]), _EPSILON)
            else:
                if self.state.recent_sales[agent_id, good_id] > surplus_value:
                    cap = self.config.stock_limit_multiplier * (
                        (self.state.sold_this_period[agent_id, good_id] - self.state.sold_last_period[agent_id, good_id])
                        + elastic_need[good_id]
                    )
                    surplus_value = min(surplus_value, cap)
                    surplus_value *= self.state.sales_price[agent_id, good_id]
                else:
                    surplus_value = min(surplus_value, elastic_need[good_id] * self.config.max_needs_increase)
                    surplus_value *= min(
                        float(self.state.purchase_price[agent_id, good_id]),
                        1.0 / max(float(self.state.efficiency[agent_id, good_id]), _EPSILON),
                    )
            wealth_minus_needs += surplus_value

            if self.state.recent_purchases[agent_id, good_id] > elastic_need[good_id]:
                total_needs_value += self.state.purchase_price[agent_id, good_id] * elastic_need[good_id]
            else:
                total_needs_value += elastic_need[good_id] / max(float(self.state.efficiency[agent_id, good_id]), _EPSILON)

        if total_needs_value <= _EPSILON:
            return 1.0
        return float((total_needs_value + wealth_minus_needs) / total_needs_value)

    def _update_needs_level(self, agent_id: int, stock_level: float) -> None:
        previous_level = float(self.state.needs_level[agent_id])
        debt = float(self.state.period_time_debt[agent_id])
        if stock_level < self.config.max_needs_reduction or bool(self.state.period_failure[agent_id]):
            self.state.needs_level[agent_id] *= self.config.max_needs_reduction
        elif debt > ((1.0 - self.config.max_needs_increase) * self.period_length) and stock_level > self.config.max_needs_increase:
            self.state.needs_level[agent_id] *= self.config.max_needs_increase
        elif stock_level > self.config.small_needs_increase:
            self.state.needs_level[agent_id] *= self.config.small_needs_increase
        else:
            self.state.needs_level[agent_id] *= self.config.small_needs_reduction

        if self.state.needs_level[agent_id] < 1.0:
            self.state.needs_level[agent_id] = 1.0
        if debt < (-1.0 * self.period_length):
            self.state.needs_level[agent_id] = 1.0

        level_ratio = float(self.state.needs_level[agent_id]) / max(previous_level, _EPSILON)
        self.state.recent_needs_increment[agent_id] = (
            level_ratio + (self.config.history * float(self.state.recent_needs_increment[agent_id]))
        ) / float(self.config.history + 1)

    def _set_cycle_needs(self, agent_id: int) -> None:
        need_row = self._baseline_cycle_need(agent_id)
        self.state.need[agent_id] = need_row
        self.engine._cycle_need_total += float(np.sum(need_row))

    def _baseline_cycle_need(self, agent_id: int) -> np.ndarray:
        if self.config.basic_round_elastic and self.state.needs_level[agent_id] >= self.config.small_needs_increase:
            return self.market.elastic_need * self.state.needs_level[agent_id]
        return self.market.elastic_need.copy()

    def _compute_leisure_extra_need(self, agent_id: int) -> np.ndarray | None:
        remaining_time = float(self.state.time_remaining[agent_id])
        if remaining_time <= self.config.leisure_time:
            return None

        utilized_time = max(self.period_length - remaining_time, 1.0)
        raw_increment = self.period_length / utilized_time
        capped_increment = min(
            raw_increment,
            float(self.state.recent_needs_increment[agent_id]) * self.config.max_needs_increase,
        )
        extra_multiplier = min(
            max(capped_increment - 1.0, 0.0),
            self.config.max_leisure_extra_multiplier,
        )
        if extra_multiplier <= 0.0:
            return None

        baseline_need = self._baseline_cycle_need(agent_id)
        extra_need = baseline_need * extra_multiplier
        if float(np.sum(extra_need)) <= 0.0:
            return None
        return extra_need

    def _run_leisure_round(self, agent_id: int) -> None:
        if self._uses_experimental_native_stage_math() and self._native_cycle.supports_prepare_leisure_round:
            expected_extra_need = self._compute_leisure_extra_need(agent_id)
            if expected_extra_need is None:
                return
            stock_before = self.state.stock[agent_id].copy()
            need_before = self.state.need[agent_id].copy()
            consumed = np.minimum(stock_before, need_before + expected_extra_need)
            has_extra, _extra_need_total, _stock_consumed_total = self._native_cycle.prepare_leisure_round(agent_id=agent_id)
            if not has_extra:
                return
            extra_need_total = float(np.sum(expected_extra_need))
            stock_consumed_total = float(np.sum(consumed))
            self.engine._cycle_need_total += extra_need_total
            self.engine._leisure_extra_need_total += extra_need_total
            self.engine._stock_consumption_total += stock_consumed_total
            self._satisfy_needs_by_exchange(agent_id)
            self._produce_need(agent_id)
            self._surplus_production(agent_id)
            return

        remaining_time = float(self.state.time_remaining[agent_id])
        extra_need = self._compute_leisure_extra_need(agent_id)
        if extra_need is None:
            return

        utilized_time = max(self.period_length - remaining_time, 1.0)
        capped_increment = min(
            self.period_length / utilized_time,
            float(self.state.recent_needs_increment[agent_id]) * self.config.max_needs_increase,
        )
        self.state.recent_needs_increment[agent_id] = (
            capped_increment + (self.config.history * float(self.state.recent_needs_increment[agent_id]))
        ) / float(self.config.history + 1)

        self.state.need[agent_id] += extra_need
        self.engine._cycle_need_total += float(np.sum(extra_need))
        self.engine._leisure_extra_need_total += float(np.sum(extra_need))
        self._consume_surplus(agent_id)
        self._satisfy_needs_by_exchange(agent_id)
        self._produce_need(agent_id)
        self._surplus_production(agent_id)

    def _consume_surplus(self, agent_id: int) -> None:
        consumed = np.minimum(self.state.stock[agent_id], self.state.need[agent_id])
        self.state.stock[agent_id] -= consumed
        self.state.need[agent_id] -= consumed
        self.engine._stock_consumption_total += float(np.sum(consumed))

    def _satisfy_needs_by_exchange(self, agent_id: int) -> None:
        if self._uses_experimental_native_exchange_stage():
            proposed_count, accepted_count, accepted_volume, inventory_trade_volume = self._native_cycle.run_exchange_stage(
                agent_id=agent_id,
                deal_type=_CONSUMPTION_DEAL,
            )
            self.engine._proposed_trade_count += proposed_count
            self.engine._accepted_trade_count += accepted_count
            self.engine._accepted_trade_volume += accepted_volume
            self.engine._inventory_trade_volume += inventory_trade_volume
            return

        max_attempts = self.config.goods * self.config.acquaintances

        for need_good in range(self.config.goods):
            if self.state.stock[agent_id, need_good] >= (
                self.market.elastic_need[need_good] * self.state.needs_level[agent_id]
            ):
                continue

            forbidden_gifts: set[int] = set()
            attempts = 0
            while self.state.need[agent_id, need_good] >= 1.0 and attempts < max_attempts:
                exchange, execution_plan, _ = self._plan_best_exchange_with_reason(
                    agent_id,
                    need_good,
                    float(self.state.need[agent_id, need_good]),
                    forbidden_gifts,
                )
                if exchange is None:
                    break
                self.engine._proposed_trade_count += 1
                result = self._execute_exchange(
                    deal_type=_CONSUMPTION_DEAL,
                    agent_id=agent_id,
                    need_good=need_good,
                    max_need=float(self.state.need[agent_id, need_good]),
                    exchange=exchange,
                    execution_plan=execution_plan,
                )
                if not result.executed or result.exhausted_gift:
                    forbidden_gifts.add(exchange.offer_good)
                attempts += 1

    def _make_surplus_deals(self, agent_id: int) -> None:
        if self._uses_experimental_native_exchange_stage():
            proposed_count, accepted_count, accepted_volume, inventory_trade_volume = self._native_cycle.run_exchange_stage(
                agent_id=agent_id,
                deal_type=_SURPLUS_DEAL,
            )
            self.engine._proposed_trade_count += proposed_count
            self.engine._accepted_trade_count += accepted_count
            self.engine._accepted_trade_volume += accepted_volume
            self.engine._inventory_trade_volume += inventory_trade_volume
            return

        max_attempts = self.config.goods * self.config.acquaintances

        for need_good in range(self.config.goods):
            if not (
                self.state.recent_sales[agent_id, need_good]
                > (
                    self.state.recent_production[agent_id, need_good]
                    - self.market.elastic_need[need_good]
                )
                and self.state.stock[agent_id, need_good]
                < (
                    self.state.stock_limit[agent_id, need_good]
                    - self.market.elastic_need[need_good]
                )
            ):
                continue

            forbidden_gifts: set[int] = set()
            attempts = 0
            while (
                self.state.stock[agent_id, need_good]
                < self.state.stock_limit[agent_id, need_good]
                and attempts < max_attempts
            ):
                remaining_room = float(
                    self.state.stock_limit[agent_id, need_good]
                    - self.state.stock[agent_id, need_good]
                )
                exchange, execution_plan, _ = self._plan_best_exchange_with_reason(
                    agent_id,
                    need_good,
                    max(remaining_room, 0.0),
                    forbidden_gifts,
                )
                if exchange is None:
                    break
                self.engine._proposed_trade_count += 1
                result = self._execute_exchange(
                    deal_type=_SURPLUS_DEAL,
                    agent_id=agent_id,
                    need_good=need_good,
                    max_need=max(remaining_room, 0.0),
                    exchange=exchange,
                    execution_plan=execution_plan,
                )
                if not result.executed or result.exhausted_gift:
                    forbidden_gifts.add(exchange.offer_good)
                attempts += 1

    def plan_experimental_consumption_batches(
        self,
        *,
        batch_count: int | None = None,
        proposer_ids: tuple[int, ...] | None = None,
        blocked_partner_ids: set[int] | None = None,
        one_candidate_per_agent: bool = False,
        seed_offset: int = 0,
    ) -> HybridExchangeBatchPlan:
        return self._plan_experimental_exchange_batches(
            deal_type=_CONSUMPTION_DEAL,
            batch_count=batch_count,
            proposer_ids=proposer_ids,
            blocked_partner_ids=blocked_partner_ids,
            one_candidate_per_agent=one_candidate_per_agent,
            seed_offset=seed_offset,
        )

    def execute_experimental_consumption_batches(
        self,
        *,
        batch_count: int | None = None,
        proposer_ids: tuple[int, ...] | None = None,
        blocked_partner_ids: set[int] | None = None,
        one_candidate_per_agent: bool = False,
        seed_offset: int = 0,
    ) -> HybridExchangeBatchPlan:
        plan = self.plan_experimental_consumption_batches(
            batch_count=batch_count,
            proposer_ids=proposer_ids,
            blocked_partner_ids=blocked_partner_ids,
            one_candidate_per_agent=one_candidate_per_agent,
            seed_offset=seed_offset,
        )
        self._execute_hybrid_batch_plan(plan)
        return plan

    def plan_experimental_surplus_batches(
        self,
        *,
        batch_count: int | None = None,
        proposer_ids: tuple[int, ...] | None = None,
        blocked_partner_ids: set[int] | None = None,
        one_candidate_per_agent: bool = False,
        seed_offset: int = 0,
    ) -> HybridExchangeBatchPlan:
        return self._plan_experimental_exchange_batches(
            deal_type=_SURPLUS_DEAL,
            batch_count=batch_count,
            proposer_ids=proposer_ids,
            blocked_partner_ids=blocked_partner_ids,
            one_candidate_per_agent=one_candidate_per_agent,
            seed_offset=seed_offset,
        )

    def execute_experimental_surplus_batches(
        self,
        *,
        batch_count: int | None = None,
        proposer_ids: tuple[int, ...] | None = None,
        blocked_partner_ids: set[int] | None = None,
        one_candidate_per_agent: bool = False,
        seed_offset: int = 0,
    ) -> HybridExchangeBatchPlan:
        plan = self.plan_experimental_surplus_batches(
            batch_count=batch_count,
            proposer_ids=proposer_ids,
            blocked_partner_ids=blocked_partner_ids,
            one_candidate_per_agent=one_candidate_per_agent,
            seed_offset=seed_offset,
        )
        self._execute_hybrid_batch_plan(plan)
        return plan

    def _plan_experimental_exchange_batches(
        self,
        *,
        deal_type: int,
        batch_count: int | None = None,
        proposer_ids: tuple[int, ...] | None = None,
        blocked_partner_ids: set[int] | None = None,
        one_candidate_per_agent: bool = False,
        seed_offset: int = 0,
    ) -> HybridExchangeBatchPlan:
        resolved_batch_count = batch_count
        if resolved_batch_count is None:
            resolved_batch_count = self.config.experimental_hybrid_batches or 1
        if resolved_batch_count <= 0:
            raise ValueError("batch_count must be positive")

        candidates, no_candidate_reasons = self._collect_experimental_exchange_candidates(
            deal_type=deal_type,
            proposer_ids=proposer_ids,
            blocked_partner_ids=blocked_partner_ids,
            one_candidate_per_agent=one_candidate_per_agent,
        )
        candidate_agent_ids = tuple(dict.fromkeys(candidate.agent_id for candidate in candidates))
        if not candidates:
            return HybridExchangeBatchPlan(
                seed=self._hybrid_stage_seed(deal_type, seed_offset=seed_offset),
                batches=(),
                dropped=(),
                candidate_agent_ids=(),
                no_candidate_reasons=no_candidate_reasons,
            )

        scheduler_candidates: list[NegotiationCandidate] = []
        candidate_lookup: dict[int, PlannedHybridExchange] = {}
        for candidate in candidates:
            scheduled_candidate = NegotiationCandidate(
                proposer_id=candidate.agent_id,
                partner_id=candidate.exchange.friend_id,
                priority=candidate.priority,
            )
            scheduler_candidates.append(scheduled_candidate)
            candidate_lookup[id(scheduled_candidate)] = candidate

        scheduled = schedule_conflict_free_batches(
            scheduler_candidates,
            batch_count=resolved_batch_count,
            seed=self._hybrid_stage_seed(deal_type, seed_offset=seed_offset),
            preserve_input_order=self.config.experimental_hybrid_preserve_proposer_order,
        )
        return HybridExchangeBatchPlan(
            seed=scheduled.seed,
            batches=tuple(
                tuple(candidate_lookup[id(item)] for item in batch)
                for batch in scheduled.batches
            ),
            dropped=tuple(candidate_lookup[id(item)] for item in scheduled.dropped),
            candidate_agent_ids=candidate_agent_ids,
            no_candidate_reasons=no_candidate_reasons,
        )

    def _execute_hybrid_batch_plan(self, plan: HybridExchangeBatchPlan) -> tuple[int, dict[str, int], float, float]:
        executed_count = 0
        execution_failure_reasons: dict[str, int] = {}
        scheduled_quantity_total = 0.0
        executed_quantity_total = 0.0
        for batch in plan.batches:
            for candidate in batch:
                self.engine._proposed_trade_count += 1
                scheduled_quantity_total += float(candidate.planned_quantity)
                result = self._execute_exchange(
                    deal_type=candidate.deal_type,
                    agent_id=candidate.agent_id,
                    need_good=candidate.need_good,
                    max_need=candidate.max_need,
                    exchange=candidate.exchange,
                    execution_plan=candidate.execution_plan,
                )
                if result.executed:
                    executed_count += 1
                    executed_quantity_total += float(result.executed_quantity)
                else:
                    self._increment_reason_count(execution_failure_reasons, result.failure_reason)
        return executed_count, execution_failure_reasons, scheduled_quantity_total, executed_quantity_total

    def _record_hybrid_wave_diagnostics(
        self,
        *,
        stage: str,
        frontier_start: int,
        frontier_agents: int,
        wave_offset: int,
        incoming_active_agents: int,
        plan: HybridExchangeBatchPlan,
        executed_exchanges: int,
        scheduled_quantity_total: float,
        executed_quantity_total: float,
        execution_failure_reasons: dict[str, int],
        exhausted_retry_agents: int,
        remaining_active_agents: int,
    ) -> None:
        self._hybrid_wave_diagnostics.append(
            HybridWaveDiagnostics(
                stage=stage,
                frontier_start=frontier_start,
                frontier_agents=frontier_agents,
                wave_offset=wave_offset,
                incoming_active_agents=incoming_active_agents,
                candidate_agents=len(plan.candidate_agent_ids),
                no_candidate_agents=max(incoming_active_agents - len(plan.candidate_agent_ids), 0),
                no_candidate_reasons=dict(plan.no_candidate_reasons),
                scheduled_exchanges=plan.scheduled_count,
                scheduled_quantity_total=scheduled_quantity_total,
                dropped_exchanges=len(plan.dropped),
                executed_exchanges=executed_exchanges,
                executed_quantity_total=executed_quantity_total,
                execution_failure_reasons=dict(execution_failure_reasons),
                exhausted_retry_agents=exhausted_retry_agents,
                remaining_active_agents=remaining_active_agents,
            )
        )

    def _finalize_hybrid_cycle_diagnostics(self, frontier_size: int, *, consumption_stage: bool, surplus_stage: bool) -> None:
        wave_rows = [asdict(item) for item in self._hybrid_wave_diagnostics]
        wave_count = len(wave_rows)
        scheduled_exchanges_total = sum(item['scheduled_exchanges'] for item in wave_rows)
        scheduled_quantity_total = sum(item['scheduled_quantity_total'] for item in wave_rows)
        dropped_exchanges_total = sum(item['dropped_exchanges'] for item in wave_rows)
        executed_exchanges_total = sum(item['executed_exchanges'] for item in wave_rows)
        executed_quantity_total = sum(item['executed_quantity_total'] for item in wave_rows)
        candidate_agents_total = sum(item['candidate_agents'] for item in wave_rows)
        no_candidate_agents_total = sum(item['no_candidate_agents'] for item in wave_rows)
        retry_exhausted_agents_total = sum(item['exhausted_retry_agents'] for item in wave_rows)
        stalled_waves_total = sum(
            1
            for item in wave_rows
            if item['candidate_agents'] <= 0 or item['scheduled_exchanges'] <= 0
        )
        no_candidate_reasons_total: dict[str, int] = {}
        execution_failure_reasons_total: dict[str, int] = {}
        stage_wave_counts: dict[str, int] = {}
        for item in wave_rows:
            self._merge_reason_counts(no_candidate_reasons_total, item['no_candidate_reasons'])
            self._merge_reason_counts(execution_failure_reasons_total, item['execution_failure_reasons'])
            stage = str(item['stage'])
            stage_wave_counts[stage] = stage_wave_counts.get(stage, 0) + 1
        execution_failures_total = sum(execution_failure_reasons_total.values())
        self.engine.exact_cycle_diagnostics = {
            'mode': 'experimental_hybrid_exchange',
            'frontier_size': frontier_size,
            'consumption_stage': consumption_stage,
            'surplus_stage': surplus_stage,
            'block_frontier_partners': self.config.experimental_hybrid_block_frontier_partners,
            'preserve_proposer_order': self.config.experimental_hybrid_preserve_proposer_order,
            'rolling_frontier': self.config.experimental_hybrid_rolling_frontier,
            'stage_wave_counts': stage_wave_counts,
            'frontier_count': (self.config.population + frontier_size - 1) // frontier_size,
            'wave_count': wave_count,
            'candidate_agents_total': candidate_agents_total,
            'no_candidate_agents_total': no_candidate_agents_total,
            'no_candidate_reasons_total': no_candidate_reasons_total,
            'scheduled_exchanges_total': scheduled_exchanges_total,
            'scheduled_quantity_total': scheduled_quantity_total,
            'dropped_exchanges_total': dropped_exchanges_total,
            'scheduler_conflict_exchanges_total': dropped_exchanges_total,
            'executed_exchanges_total': executed_exchanges_total,
            'executed_quantity_total': executed_quantity_total,
            'execution_failures_total': execution_failures_total,
            'execution_failure_reasons_total': execution_failure_reasons_total,
            'retry_exhausted_agents_total': retry_exhausted_agents_total,
            'stalled_waves_total': stalled_waves_total,
            'waves': wave_rows,
        }

    @staticmethod
    def _increment_reason_count(reason_counts: dict[str, int], reason: str | None, amount: int = 1) -> None:
        if not reason or amount <= 0:
            return
        reason_counts[reason] = reason_counts.get(reason, 0) + amount

    def _merge_reason_counts(self, target: dict[str, int], source: dict[str, int]) -> None:
        for reason, count in source.items():
            self._increment_reason_count(target, reason, count)

    def _collect_experimental_exchange_candidates(
        self,
        *,
        deal_type: int,
        proposer_ids: tuple[int, ...] | None = None,
        blocked_partner_ids: set[int] | None = None,
        one_candidate_per_agent: bool = False,
    ) -> tuple[list[PlannedHybridExchange], dict[str, int]]:
        candidates: list[PlannedHybridExchange] = []
        no_candidate_reasons: dict[str, int] = {}
        resolved_proposer_ids = proposer_ids or tuple(range(self.config.population))
        blocked_partner_ids = blocked_partner_ids or set()

        for agent_id in resolved_proposer_ids:
            if one_candidate_per_agent:
                candidate, reason = self._collect_first_experimental_exchange_candidate(
                    agent_id=agent_id,
                    deal_type=deal_type,
                    blocked_partner_ids=blocked_partner_ids,
                )
                if candidate is not None:
                    candidates.append(candidate)
                else:
                    self._increment_reason_count(no_candidate_reasons, reason)
                continue

            for need_good in range(self.config.goods):
                max_need = self._hybrid_exchange_need(agent_id=agent_id, need_good=need_good, deal_type=deal_type)
                if max_need <= 0.0:
                    continue
                exchange = self._find_best_exchange(
                    agent_id,
                    need_good,
                    set(),
                    blocked_friend_ids=blocked_partner_ids,
                )
                if exchange is None:
                    continue
                candidates.append(
                    PlannedHybridExchange(
                        deal_type=deal_type,
                        agent_id=agent_id,
                        need_good=need_good,
                        max_need=max_need,
                        exchange=exchange,
                    )
                )
        return candidates, no_candidate_reasons

    def _collect_first_experimental_exchange_candidate(
        self,
        *,
        agent_id: int,
        deal_type: int,
        blocked_partner_ids: set[int],
    ) -> tuple[PlannedHybridExchange | None, str | None]:
        candidate_reasons: dict[str, int] = {}
        base_offer_goods = self._collect_base_offer_goods(agent_id)
        if not base_offer_goods:
            return None, 'no_offer_goods'
        saw_positive_need = False
        for need_good in range(self.config.goods):
            max_need = self._hybrid_exchange_need(agent_id=agent_id, need_good=need_good, deal_type=deal_type)
            if max_need <= 0.0:
                continue
            saw_positive_need = True
            forbidden_gifts: set[int] = set()
            attempts = 0
            while attempts < self.config.goods:
                exchange, execution_plan, reason = self._plan_best_exchange_with_reason(
                    agent_id,
                    need_good,
                    max_need,
                    forbidden_gifts,
                    blocked_friend_ids=blocked_partner_ids,
                    base_offer_goods=base_offer_goods,
                )
                if exchange is None:
                    self._increment_reason_count(candidate_reasons, reason)
                    break
                if execution_plan is not None:
                    return PlannedHybridExchange(
                        deal_type=deal_type,
                        agent_id=agent_id,
                        need_good=need_good,
                        max_need=max_need,
                        exchange=exchange,
                        execution_plan=execution_plan,
                    ), None
                self._increment_reason_count(candidate_reasons, reason)
                forbidden_gifts.add(exchange.offer_good)
                attempts += 1
        if not saw_positive_need:
            return None, 'no_unmet_need'
        for reason in (
            'rounding_buffer_below_min',
            'partner_capacity_below_min',
            'partner_need_below_min',
            'friend_supply_below_min',
            'offer_surplus_below_min',
            'blocked_partner_only',
            'no_matching_partner',
            'no_offer_goods',
            'no_known_partner',
        ):
            if candidate_reasons.get(reason, 0) > 0:
                return None, reason
        return None, 'no_candidate'

    def _hybrid_exchange_need(self, *, agent_id: int, need_good: int, deal_type: int) -> float:
        if deal_type == _CONSUMPTION_DEAL:
            if self.state.stock[agent_id, need_good] >= (
                self.market.elastic_need[need_good] * self.state.needs_level[agent_id]
            ):
                return 0.0
            max_need = float(self.state.need[agent_id, need_good])
            if max_need < 1.0:
                return 0.0
            return max_need

        if not (
            self.state.recent_sales[agent_id, need_good]
            > (
                self.state.recent_production[agent_id, need_good]
                - self.market.elastic_need[need_good]
            )
            and self.state.stock[agent_id, need_good]
            < (
                self.state.stock_limit[agent_id, need_good]
                - self.market.elastic_need[need_good]
            )
        ):
            return 0.0
        return max(float(self.state.stock_limit[agent_id, need_good] - self.state.stock[agent_id, need_good]), 0.0)

    def _hybrid_stage_seed(self, deal_type: int, *, seed_offset: int = 0) -> int:
        return int(
            self.config.seed
            + (self.engine.cycle * self.config.experimental_hybrid_seed_stride)
            + seed_offset
            + deal_type
        )

    def _collect_base_offer_goods(self, agent_id: int) -> tuple[int, ...]:
        elastic_need = self.market.elastic_need
        my_stock = self.state.stock[agent_id]
        my_needs_level = self.state.needs_level[agent_id]
        offer_goods: list[int] = []
        for offer_good in range(self.config.goods):
            if my_stock[offer_good] <= (elastic_need[offer_good] * my_needs_level + 1.0):
                continue
            offer_goods.append(offer_good)
        return tuple(offer_goods)

    def _build_exchange_plan_request(
        self,
        agent_id: int,
        need_good: int,
        max_need: float,
        forbidden_gifts: set[int],
        *,
        blocked_friend_ids: set[int] | None = None,
        base_offer_goods: tuple[int, ...] | None = None,
    ) -> tuple[ExchangePlanRequest | None, str | None]:
        search_request, reason = self._build_exchange_search_request(
            agent_id,
            need_good,
            forbidden_gifts,
            blocked_friend_ids=blocked_friend_ids,
            base_offer_goods=base_offer_goods,
        )
        if search_request is None:
            return None, reason
        return ExchangePlanRequest(
            search_request=search_request,
            max_need=max_need,
            min_trade_quantity=self.config.min_trade_quantity,
            trade_rounding_buffer=self.config.trade_rounding_buffer,
        ), None

    def _plan_best_exchange_with_reason(
        self,
        agent_id: int,
        need_good: int,
        max_need: float,
        forbidden_gifts: set[int],
        *,
        blocked_friend_ids: set[int] | None = None,
        base_offer_goods: tuple[int, ...] | None = None,
    ) -> tuple[ExchangeOption | None, ExchangePlanResult | None, str | None]:
        request, reason = self._build_exchange_plan_request(
            agent_id,
            need_good,
            max_need,
            forbidden_gifts,
            blocked_friend_ids=blocked_friend_ids,
            base_offer_goods=base_offer_goods,
        )
        if request is None:
            return None, None, reason

        outcome = execute_exchange_planning(self._exchange_search, request)
        if outcome is None:
            return None, None, 'no_matching_partner'
        exchange = ExchangeOption(
            score=outcome.search_result.score,
            friend_slot=outcome.search_result.friend_slot,
            friend_id=outcome.search_result.friend_id,
            offer_good=outcome.search_result.offer_good,
        )
        return exchange, outcome.plan_result, outcome.failure_reason

    def _find_best_exchange(
        self,
        agent_id: int,
        need_good: int,
        forbidden_gifts: set[int],
        *,
        blocked_friend_ids: set[int] | None = None,
        base_offer_goods: tuple[int, ...] | None = None,
    ) -> ExchangeOption | None:
        exchange, _ = self._find_best_exchange_with_reason(
            agent_id,
            need_good,
            forbidden_gifts,
            blocked_friend_ids=blocked_friend_ids,
            base_offer_goods=base_offer_goods,
        )
        return exchange

    def _build_exchange_search_request(
        self,
        agent_id: int,
        need_good: int,
        forbidden_gifts: set[int],
        *,
        blocked_friend_ids: set[int] | None = None,
        base_offer_goods: tuple[int, ...] | None = None,
    ) -> tuple[ExchangeSearchRequest | None, str | None]:
        state = self.state
        elastic_need = self.market.elastic_need
        my_stock = state.stock[agent_id]
        my_sales_price = state.sales_price[agent_id]
        my_purchase_price = state.purchase_price[agent_id]
        my_role = state.role[agent_id]
        my_transparency = state.transparency[agent_id]
        original_friend_ids = state.friend_id[agent_id]
        friend_ids = original_friend_ids
        if blocked_friend_ids:
            friend_ids = friend_ids.copy()
            for friend_slot, friend_id_raw in enumerate(friend_ids):
                if int(friend_id_raw) in blocked_friend_ids:
                    friend_ids[friend_slot] = -1
        my_needs_level = state.needs_level[agent_id]
        if base_offer_goods is None:
            base_offer_goods = self._collect_base_offer_goods(agent_id)
        candidate_offer_goods = [
            offer_good
            for offer_good in base_offer_goods
            if offer_good != need_good and offer_good not in forbidden_gifts
        ]
        if not candidate_offer_goods:
            return None, 'no_offer_goods'

        if not bool(np.any(friend_ids >= 0)):
            if blocked_friend_ids and bool(np.any(original_friend_ids >= 0)):
                return None, 'blocked_partner_only'
            return None, 'no_known_partner'

        reciprocal_slots = np.full((self.config.acquaintances,), -1, dtype=np.int32)
        for friend_slot, friend_id_raw in enumerate(friend_ids):
            friend_id = int(friend_id_raw)
            if friend_id >= 0:
                reciprocal_slots[friend_slot] = self._find_friend_slot(friend_id, agent_id)

        return ExchangeSearchRequest(
            goods=self.config.goods,
            need_good=need_good,
            initial_transparency=self.config.initial_transparency,
            elastic_need=elastic_need,
            candidate_offer_goods=np.asarray(candidate_offer_goods, dtype=np.int32),
            friend_ids=friend_ids,
            reciprocal_slots=reciprocal_slots,
            my_stock=my_stock,
            my_sales_price=my_sales_price,
            my_purchase_price=my_purchase_price,
            my_role=my_role,
            my_transparency=my_transparency,
            my_needs_level=float(my_needs_level),
            stock=state.stock,
            role=state.role,
            stock_limit=state.stock_limit,
            purchase_price=state.purchase_price,
            sales_price=state.sales_price,
            needs_level=state.needs_level,
            transparency=state.transparency,
        ), None

    def _find_best_exchange_with_reason(
        self,
        agent_id: int,
        need_good: int,
        forbidden_gifts: set[int],
        *,
        blocked_friend_ids: set[int] | None = None,
        base_offer_goods: tuple[int, ...] | None = None,
    ) -> tuple[ExchangeOption | None, str | None]:
        request, reason = self._build_exchange_search_request(
            agent_id,
            need_good,
            forbidden_gifts,
            blocked_friend_ids=blocked_friend_ids,
            base_offer_goods=base_offer_goods,
        )
        if request is None:
            return None, reason

        result = execute_exchange_search(self._exchange_search, request)
        if result is None:
            return None, 'no_matching_partner'
        return ExchangeOption(
            score=result.score,
            friend_slot=result.friend_slot,
            friend_id=result.friend_id,
            offer_good=result.offer_good,
        ), None

    def _execute_exchange(
        self,
        *,
        deal_type: int,
        agent_id: int,
        need_good: int,
        max_need: float,
        exchange: ExchangeOption,
        execution_plan: ExchangePlanResult | None = None,
    ) -> ExchangeExecutionResult:
        state = self.state
        market = self.market
        friend_id = exchange.friend_id
        offer_good = exchange.offer_good
        friend_slot = exchange.friend_slot
        my_stock = state.stock[agent_id]
        friend_stock = state.stock[friend_id]
        my_need = state.need[agent_id]
        my_recent_sales = state.recent_sales[agent_id]
        friend_recent_sales = state.recent_sales[friend_id]
        my_recent_purchases = state.recent_purchases[agent_id]
        friend_recent_purchases = state.recent_purchases[friend_id]
        my_sold_this_period = state.sold_this_period[agent_id]
        friend_sold_this_period = state.sold_this_period[friend_id]
        my_purchased_this_period = state.purchased_this_period[agent_id]
        friend_purchased_this_period = state.purchased_this_period[friend_id]
        my_sales_price = state.sales_price[agent_id]
        friend_sales_price = state.sales_price[friend_id]
        my_purchase_price = state.purchase_price[agent_id]
        friend_purchase_price = state.purchase_price[friend_id]
        friend_stock_limit = state.stock_limit[friend_id]
        my_transparency_row = state.transparency[agent_id, friend_slot]
        my_needs_level = state.needs_level[agent_id]
        friend_needs_level = state.needs_level[friend_id]
        my_need_purchase_price = my_purchase_price[need_good]
        my_offer_sales_price = my_sales_price[offer_good]
        friend_need_sales_price = friend_sales_price[need_good]
        friend_offer_purchase_price = friend_purchase_price[offer_good]

        if execution_plan is None:
            need_transparency = my_transparency_row[need_good]
            reciprocal_slot = self._find_friend_slot(friend_id, agent_id)
            receiving_transparency = self.config.initial_transparency
            if reciprocal_slot >= 0:
                receiving_transparency = state.transparency[friend_id, reciprocal_slot, offer_good]

            switch_average = (
                (
                    friend_need_sales_price
                    / max(friend_offer_purchase_price * need_transparency, _EPSILON)
                )
                + ((my_need_purchase_price * receiving_transparency) / max(my_offer_sales_price, _EPSILON))
            ) / 2.0

            max_exchange = (
                (my_stock[offer_good] - (my_needs_level * market.elastic_need[offer_good]))
                * receiving_transparency
            ) / max(switch_average, _EPSILON)
            if max_exchange <= self.config.min_trade_quantity:
                return ExchangeExecutionResult(False, True, 'offer_surplus_below_min', 0.0)

            max_exchange = min(max_exchange, max_need)
            friend_supply = (
                friend_stock[need_good] - (friend_needs_level * market.elastic_need[need_good])
            ) * need_transparency
            max_exchange = min(max_exchange, friend_supply)
            if max_exchange <= self.config.min_trade_quantity:
                return ExchangeExecutionResult(False, True, 'friend_supply_below_min', 0.0)

            if state.role[friend_id, offer_good] == ROLE_RETAILER:
                stock_capacity = friend_stock_limit[offer_good] - friend_stock[offer_good]
                max_exchange = min(max_exchange, stock_capacity / max(switch_average, _EPSILON))
                if max_exchange <= self.config.min_trade_quantity:
                    return ExchangeExecutionResult(False, True, 'partner_capacity_below_min', 0.0)
            else:
                immediate_need = (friend_needs_level * market.elastic_need[offer_good]) - friend_stock[offer_good]
                max_exchange = min(max_exchange, immediate_need / max(switch_average, _EPSILON))
                if max_exchange <= self.config.min_trade_quantity:
                    return ExchangeExecutionResult(False, True, 'partner_need_below_min', 0.0)

            max_exchange -= self.config.trade_rounding_buffer
            max_exchange = float(np.float32(max_exchange))
            if max_exchange < self.config.min_trade_quantity:
                return ExchangeExecutionResult(False, True, 'rounding_buffer_below_min', 0.0)
        else:
            reciprocal_slot = execution_plan.reciprocal_slot
            max_exchange = float(execution_plan.max_exchange)
            switch_average = float(execution_plan.switch_average)
            need_transparency = float(execution_plan.need_transparency)
            receiving_transparency = float(execution_plan.receiving_transparency)

        if reciprocal_slot < 0:
            reciprocal_slot = self._ensure_friend_link(friend_id, agent_id)

        remaining_need_after_trade = max_need - max_exchange
        my_gift_out = (max_exchange * switch_average) / max(receiving_transparency, _EPSILON)
        friend_need_out = max_exchange / max(need_transparency, _EPSILON)
        friend_gift_in = max_exchange * switch_average

        if deal_type == _SURPLUS_DEAL:
            my_stock[need_good] += max_exchange
            state.recent_inventory_inflow[agent_id, need_good] += max_exchange
            self.engine._inventory_trade_volume += float(max_exchange)
        else:
            my_need[need_good] -= max_exchange
            if my_need[need_good] < self.config.min_trade_quantity:
                my_need[need_good] = 0.0

        my_recent_sales[offer_good] += friend_gift_in
        my_sold_this_period[offer_good] += friend_gift_in
        my_recent_purchases[need_good] += max_exchange
        my_purchased_this_period[need_good] += max_exchange
        my_stock[offer_good] -= my_gift_out
        if my_stock[offer_good] < 0.0:
            my_stock[offer_good] = 0.0

        friend_stock[need_good] -= friend_need_out
        if friend_stock[need_good] < 0.0:
            friend_stock[need_good] = 0.0
        friend_stock[offer_good] += friend_gift_in
        state.recent_inventory_inflow[friend_id, offer_good] += friend_gift_in
        self.engine._inventory_trade_volume += float(friend_gift_in)

        market.periodic_tce_cost[need_good] += max(friend_need_out - max_exchange, 0.0)
        market.periodic_tce_cost[offer_good] += max(my_gift_out - friend_gift_in, 0.0)

        friend_recent_sales[need_good] += friend_need_out
        friend_sold_this_period[need_good] += friend_need_out
        friend_recent_purchases[offer_good] += friend_gift_in
        friend_purchased_this_period[offer_good] += friend_gift_in

        state.purchase_times[agent_id, need_good] += 1
        state.sales_times[agent_id, offer_good] += 1
        state.purchase_times[friend_id, offer_good] += 1
        state.sales_times[friend_id, need_good] += 1

        state.friend_activity[agent_id, friend_slot] += 1.0
        state.friend_purchased[agent_id, friend_slot, need_good] += 1.0
        state.friend_sold[agent_id, friend_slot, offer_good] += 1.0
        state.friend_activity[friend_id, reciprocal_slot] += 1.0
        state.friend_purchased[friend_id, reciprocal_slot, offer_good] += 1.0
        state.friend_sold[friend_id, reciprocal_slot, need_good] += 1.0

        exchange_value_to_me = my_need_purchase_price / max(
            (switch_average / max(receiving_transparency, _EPSILON)) * my_offer_sales_price,
            _EPSILON,
        )
        exchange_value_to_friend = (
            friend_offer_purchase_price * switch_average
        ) / max(
            friend_need_sales_price / max(need_transparency, _EPSILON),
            _EPSILON,
        )
        my_value_correction = sqrt(max(float(exchange_value_to_me), 0.0))
        friend_value_correction = sqrt(max(float(exchange_value_to_friend), 0.0))
        if my_value_correction > 1.0 and friend_value_correction > 1.0:
            state.sum_period_purchase_value[agent_id, need_good] += (
                my_need_purchase_price / my_value_correction
            )
            state.sum_period_sales_value[agent_id, offer_good] += (
                my_offer_sales_price * my_value_correction
            )
            state.sum_period_purchase_value[friend_id, offer_good] += (
                friend_offer_purchase_price / friend_value_correction
            )
            state.sum_period_sales_value[friend_id, need_good] += (
                friend_need_sales_price * friend_value_correction
            )

        state.trade.proposal_friend_slot[agent_id] = friend_slot
        state.trade.proposal_target_agent[agent_id] = friend_id
        state.trade.proposal_need_good[agent_id] = need_good
        state.trade.proposal_offer_good[agent_id] = offer_good
        state.trade.proposal_quantity[agent_id] = max_exchange
        state.trade.proposal_score[agent_id] = exchange.score
        state.trade.accepted_mask[agent_id] = True
        state.trade.accepted_quantity[agent_id] = max_exchange

        self.engine._accepted_trade_count += 1
        self.engine._accepted_trade_volume += float(max_exchange)

        exhausted_gift = remaining_need_after_trade >= 1.0 and my_stock[offer_good] < 1.0
        return ExchangeExecutionResult(True, exhausted_gift, None, float(max_exchange))

    def _produce_need(self, agent_id: int) -> None:
        if self._uses_experimental_native_stage_math() and self._native_cycle.supports_produce_need:
            pending = self.state.need[agent_id].copy()
            if float(np.sum(pending)) <= 0.0:
                return
            time_before = np.float32(self.state.time_remaining[agent_id])
            timeout_before = int(self.state.timeout[agent_id])
            time_spent = np.sum(pending / np.maximum(self.state.efficiency[agent_id], _EPSILON))
            expected_time_remaining = np.float32(time_before - time_spent)
            self._native_cycle.produce_need(agent_id=agent_id)
            self.state.time_remaining[agent_id] = expected_time_remaining
            if expected_time_remaining < 0.0:
                self.state.timeout[agent_id] = timeout_before + 1
            else:
                self.state.timeout[agent_id] = timeout_before
            self.engine._production_total += float(np.sum(pending))
            return
        pending = self.state.need[agent_id].copy()
        if float(np.sum(pending)) <= 0.0:
            return
        time_spent = np.sum(pending / np.maximum(self.state.efficiency[agent_id], _EPSILON))
        self.state.time_remaining[agent_id] -= time_spent
        self.state.recent_production[agent_id] += pending
        self.state.produced_this_period[agent_id] += pending
        self.engine._production_total += float(np.sum(pending))
        self.state.need[agent_id] = 0.0
        if self.state.time_remaining[agent_id] < 0.0:
            self.state.timeout[agent_id] += 1

    def _surplus_production(self, agent_id: int) -> None:
        base_need_floor = float(self.state.base_need[agent_id, 0])
        while self.state.time_remaining[agent_id] >= 1.0:
            selected_good = -1
            selected_limit = 0.0
            best_index = 0.0
            for good_id in range(self.config.goods):
                if self.state.talent_mask[agent_id, good_id] <= 0.0:
                    continue
                production_limit = self.state.stock_limit[agent_id, good_id] - self.state.stock[agent_id, good_id]
                if production_limit <= 1.0:
                    continue
                production_profitable = (
                    (self.state.purchase_times[agent_id, good_id] == 0 and self.state.stock[agent_id, good_id] > (self.config.stock_limit_multiplier * base_need_floor))
                    or ((1.0 / max(float(self.state.efficiency[agent_id, good_id]), _EPSILON)) <= (self.config.price_hike * float(self.state.sales_price[agent_id, good_id])))
                )
                if not production_profitable:
                    continue
                production_index = float(self.state.efficiency[agent_id, good_id]) - (1.0 / max(float(self.state.sales_price[agent_id, good_id]), _EPSILON))
                if production_index >= best_index:
                    best_index = production_index
                    selected_good = good_id
                    selected_limit = float(production_limit)

            if selected_good < 0:
                break

            max_production = float(self.state.efficiency[agent_id, selected_good] * self.state.time_remaining[agent_id])
            produced = min(max_production, selected_limit)
            if produced <= 0.0:
                break
            self.state.stock[agent_id, selected_good] += produced
            self.state.produced_this_period[agent_id, selected_good] += produced
            self.state.recent_production[agent_id, selected_good] += produced
            self.engine._production_total += produced
            self.engine._surplus_output_total += produced

            if max_production < selected_limit:
                self.state.time_remaining[agent_id] = 0.0
            else:
                self.state.time_remaining[agent_id] -= self.config.switch_time + (produced / max(float(self.state.efficiency[agent_id, selected_good]), _EPSILON))

    def _leisure_production(self, agent_id: int) -> None:
        while self.state.time_remaining[agent_id] >= 1.0:
            selected_good = -1
            selected_limit = 0.0
            best_index = 0.0
            for good_id in range(self.config.goods):
                if self.state.talent_mask[agent_id, good_id] > 0.0:
                    continue
                production_limit = self.state.stock_limit[agent_id, good_id] - self.state.stock[agent_id, good_id]
                if production_limit > 1.0 and float(self.state.purchase_price[agent_id, good_id]) >= best_index:
                    best_index = float(self.state.purchase_price[agent_id, good_id])
                    selected_good = good_id
                    selected_limit = float(production_limit)
            if selected_good < 0:
                break

            produced = min(float(self.state.time_remaining[agent_id]), selected_limit)
            if produced <= 0.0:
                break
            self.state.stock[agent_id, selected_good] += produced
            self.state.produced_this_period[agent_id, selected_good] += produced
            self.state.recent_production[agent_id, selected_good] += produced
            self.engine._production_total += produced
            self.engine._surplus_output_total += produced
            self.state.time_remaining[agent_id] -= produced

    def _end_agent_period(self, agent_id: int) -> None:
        self.state.periodic_spoilage[agent_id] = 0.0

        for good_id in range(self.config.goods):
            target_stock_limit = (
                (self.config.stock_limit_multiplier * (self.market.elastic_need[good_id] * self.state.needs_level[agent_id]))
                + self.state.recent_sales[agent_id, good_id]
            )
            lower_limit = self.config.max_stocklimit_decrease * float(self.state.previous_stock_limit[agent_id, good_id])
            upper_limit = self.config.max_stocklimit_increase * float(self.state.previous_stock_limit[agent_id, good_id])
            self.state.stock_limit[agent_id, good_id] = float(np.clip(target_stock_limit, lower_limit, upper_limit))
            self.state.previous_stock_limit[agent_id, good_id] = self.state.stock_limit[agent_id, good_id]

            if self.state.talent_mask[agent_id, good_id] > 0.0:
                previous_efficiency = float(self.state.efficiency[agent_id, good_id])
                learned_efficiency = sqrt(
                    max((float(self.state.recent_production[agent_id, good_id]) + 1.0) / (self.config.history * float(self.state.base_need[agent_id, good_id])), _EPSILON)
                )
                learned_efficiency = max(learned_efficiency, self.config.gifted_efficiency_floor)
                learned_efficiency = float(np.clip(
                    learned_efficiency,
                    previous_efficiency * self.config.max_efficiency_downgrade,
                    previous_efficiency * self.config.max_efficiency_upgrade,
                ))
                learned_efficiency = max(learned_efficiency, self.config.gifted_efficiency_floor)
                self.state.learned_efficiency[agent_id, good_id] = learned_efficiency
                self.state.efficiency[agent_id, good_id] = learned_efficiency
            else:
                self.state.learned_efficiency[agent_id, good_id] = self.config.initial_efficiency
                self.state.efficiency[agent_id, good_id] = self.config.initial_efficiency

            self.state.role[agent_id, good_id] = ROLE_CONSUMER
            recent_produced = float(self.state.recent_production[agent_id, good_id])
            recent_purchased = float(self.state.recent_purchases[agent_id, good_id])
            recent_sold = float(self.state.recent_sales[agent_id, good_id])
            if recent_produced > recent_purchased and recent_sold > ((recent_produced + recent_purchased) / 2.0):
                self.state.role[agent_id, good_id] = ROLE_PRODUCER
            elif recent_produced < recent_purchased and recent_sold > ((recent_produced + recent_purchased) / 2.0):
                self.state.role[agent_id, good_id] = ROLE_RETAILER

            self._adjust_purchase_price(agent_id, good_id)
            self._adjust_sales_price(agent_id, good_id)
            self.state.spoilage[agent_id, good_id] = 0.0
            if self.state.stock[agent_id, good_id] > (self.config.stock_limit_multiplier * self.market.elastic_need[good_id]):
                if self.state.stock[agent_id, good_id] > (self.config.stock_spoil_threshold * self.state.stock_limit[agent_id, good_id]):
                    if self.state.stock[agent_id, good_id] > self.state.stock_limit[agent_id, good_id]:
                        spoiled = (self.state.stock[agent_id, good_id] - self.state.stock_limit[agent_id, good_id]) * self.config.spoilage_rate
                        self.state.spoilage[agent_id, good_id] = spoiled
                        self.state.stock[agent_id, good_id] -= spoiled
                        self.state.periodic_spoilage[agent_id] += spoiled
                        self.market.periodic_spoilage[good_id] += spoiled

            self.state.recent_production[agent_id, good_id] *= self.config.activity_discount
            self.state.recent_sales[agent_id, good_id] *= self.config.activity_discount
            self.state.recent_purchases[agent_id, good_id] *= self.config.activity_discount
            self.state.recent_inventory_inflow[agent_id, good_id] *= self.config.activity_discount
            self.state.purchase_times[agent_id, good_id] = 0
            self.state.sales_times[agent_id, good_id] = 0
            self.state.sum_period_purchase_value[agent_id, good_id] = 0.0
            self.state.sum_period_sales_value[agent_id, good_id] = 0.0
            self.state.produced_last_period[agent_id, good_id] = self.state.produced_this_period[agent_id, good_id]
            self.state.sold_last_period[agent_id, good_id] = self.state.sold_this_period[agent_id, good_id]
            self.state.purchased_last_period[agent_id, good_id] = self.state.purchased_this_period[agent_id, good_id]

        self._calibrate_friend_transparency(agent_id)

    def _adjust_purchase_price(self, agent_id: int, good_id: int) -> None:
        production_cost = 1.0 / max(float(self.state.efficiency[agent_id, good_id]), _EPSILON)
        surplus = float(self.state.stock[agent_id, good_id])
        elastic_need = float(self.market.elastic_need[good_id])
        stock_limit = float(self.state.stock_limit[agent_id, good_id])
        role = int(self.state.role[agent_id, good_id])

        scarcity_case = int(surplus > elastic_need) + int(surplus > (stock_limit + elastic_need))
        if scarcity_case == 0:
            if role == ROLE_CONSUMER:
                if self.state.recent_purchases[agent_id, good_id] < (self.state.recent_production[agent_id, good_id] + 1.0):
                    self.state.purchase_price[agent_id, good_id] = production_cost
                elif self.state.purchase_times[agent_id, good_id] == 0 and self.state.purchase_price[agent_id, good_id] < production_cost:
                    self.state.purchase_price[agent_id, good_id] *= self.config.price_leap
            elif role == ROLE_RETAILER:
                if (
                    self.state.purchased_this_period[agent_id, good_id] < (self.state.sold_this_period[agent_id, good_id] + 1.0)
                    or self.state.purchased_this_period[agent_id, good_id] < (stock_limit / max(float(self.config.history), 1.0))
                ):
                    self.state.purchase_price[agent_id, good_id] *= self.config.price_hike
        elif scarcity_case == 1:
            if role == ROLE_RETAILER:
                if self.state.purchase_times[agent_id, good_id] > 1 and self.state.purchased_this_period[agent_id, good_id] > (stock_limit / 2.0):
                    self.state.purchase_price[agent_id, good_id] = (
                        self.state.sum_period_purchase_value[agent_id, good_id] + (self.config.history * self.state.purchase_price[agent_id, good_id])
                    ) / float(self.state.purchase_times[agent_id, good_id] + self.config.history)
                if self.state.purchase_price[agent_id, good_id] > self.state.sales_price[agent_id, good_id]:
                    self.state.purchase_price[agent_id, good_id] = self.config.price_reduction * self.state.sales_price[agent_id, good_id]
            elif role == ROLE_PRODUCER:
                if self.state.purchased_this_period[agent_id, good_id] > self.state.produced_this_period[agent_id, good_id]:
                    self.state.purchase_price[agent_id, good_id] *= self.config.price_reduction
        else:
            if role == ROLE_CONSUMER:
                self.state.purchase_price[agent_id, good_id] *= self.config.price_reduction
            elif role in {ROLE_RETAILER, ROLE_PRODUCER} and self.state.purchase_times[agent_id, good_id] > 0:
                self.state.purchase_price[agent_id, good_id] *= self.config.price_reduction

        self.state.purchase_price[agent_id, good_id] = max(self.state.purchase_price[agent_id, good_id], 0.05)

    def _adjust_sales_price(self, agent_id: int, good_id: int) -> None:
        production_cost = 1.0 / max(float(self.state.efficiency[agent_id, good_id]), _EPSILON)
        previous_sales_price = float(self.state.sales_price[agent_id, good_id])
        surplus = float(self.state.stock[agent_id, good_id])
        elastic_need = float(self.market.elastic_need[good_id])
        stock_limit = float(self.state.stock_limit[agent_id, good_id])
        role = int(self.state.role[agent_id, good_id])

        scarcity_case = int(surplus > elastic_need) + int(surplus > (stock_limit + elastic_need))
        if scarcity_case == 0:
            if role == ROLE_CONSUMER:
                target = min(production_cost, float(self.state.purchase_price[agent_id, good_id]))
                if self.state.sales_price[agent_id, good_id] < target:
                    self.state.sales_price[agent_id, good_id] = self.config.price_hike * target
            elif role == ROLE_RETAILER:
                if self.state.sales_times[agent_id, good_id] > 1 and self.state.sold_this_period[agent_id, good_id] > (stock_limit / 2.0):
                    blended_price = (
                        self.state.sum_period_sales_value[agent_id, good_id] + self.state.sales_price[agent_id, good_id]
                    ) / float(self.state.sales_times[agent_id, good_id] + 1)
                    self.state.sales_price[agent_id, good_id] = min(blended_price, self.config.price_leap * previous_sales_price)
                if self.state.sales_price[agent_id, good_id] < self.state.purchase_price[agent_id, good_id]:
                    self.state.sales_price[agent_id, good_id] = self.state.purchase_price[agent_id, good_id]
                if self.state.sales_times[agent_id, good_id] == 0:
                    self.state.sales_price[agent_id, good_id] = self.config.price_hike * self.state.purchase_price[agent_id, good_id]
            elif role == ROLE_PRODUCER and self.state.sales_price[agent_id, good_id] < production_cost:
                self.state.sales_price[agent_id, good_id] = self.config.price_hike * production_cost
        elif scarcity_case == 1:
            if role == ROLE_CONSUMER:
                self.state.sales_price[agent_id, good_id] = max(production_cost, float(self.state.purchase_price[agent_id, good_id]))
                if surplus > (stock_limit / 2.0):
                    self.state.sales_price[agent_id, good_id] = min(
                        float(self.state.purchase_price[agent_id, good_id]) * 2.0,
                        production_cost,
                    )
            elif role == ROLE_RETAILER:
                if self.state.sold_this_period[agent_id, good_id] < elastic_need:
                    self.state.sales_price[agent_id, good_id] *= self.config.price_reduction
            elif role == ROLE_PRODUCER:
                if self.state.sold_this_period[agent_id, good_id] < elastic_need:
                    self.state.sales_price[agent_id, good_id] *= self.config.price_reduction
                if self.state.sales_price[agent_id, good_id] < production_cost:
                    self.state.sales_price[agent_id, good_id] = self.config.price_hike * production_cost
        else:
            if role == ROLE_CONSUMER:
                self.state.sales_price[agent_id, good_id] *= self.config.price_reduction
            elif role == ROLE_RETAILER:
                if self.state.sold_this_period[agent_id, good_id] < elastic_need:
                    if self.state.sales_price[agent_id, good_id] > self.state.purchase_price[agent_id, good_id]:
                        self.state.sales_price[agent_id, good_id] = self.state.purchase_price[agent_id, good_id]
                    else:
                        self.state.sales_price[agent_id, good_id] *= self.config.price_reduction
            elif role == ROLE_PRODUCER:
                if self.state.sales_price[agent_id, good_id] > production_cost:
                    self.state.sales_price[agent_id, good_id] = self.config.price_hike * production_cost
                if self.state.sold_this_period[agent_id, good_id] < elastic_need:
                    self.state.sales_price[agent_id, good_id] = self.config.price_hike * production_cost

        self.state.sales_price[agent_id, good_id] = max(self.state.sales_price[agent_id, good_id], 0.05)

    def _calibrate_friend_transparency(self, agent_id: int) -> None:
        for friend_slot in range(self.config.acquaintances):
            for good_id in range(self.config.goods):
                transparency = self.config.initial_transparency
                transactions = float(self.state.friend_activity[agent_id, friend_slot])
                if transactions > 0.0:
                    transparency += ((1.0 - transparency) * 0.7) * (transactions / (transactions + self.config.goods))
                purchased = float(self.state.friend_purchased[agent_id, friend_slot, good_id])
                transparency += ((1.0 - transparency) * 0.7) * ((10.0 * purchased) / max((10.0 * purchased) + (self.engine.cycle + 1), _EPSILON))
                recent_purchased = float(self.state.recent_purchases[agent_id, good_id])
                transparency += ((1.0 - transparency) * 0.7) * (
                    recent_purchased / max(recent_purchased + (10.0 * self.config.history), _EPSILON)
                )
                if self.state.talent_mask[agent_id, good_id] > 0.0:
                    transparency += (1.0 - transparency) * 0.5
                self.state.transparency[agent_id, friend_slot, good_id] = min(float(transparency), 1.0)
            if self.state.friend_activity[agent_id, friend_slot] > 1.0:
                self.state.friend_activity[agent_id, friend_slot] *= 0.9

    def _evaluate_market_prices(self) -> None:
        self.market.price_average = 0.0
        self.market.total_cost_of_spoilage_in_time = 0.0
        self.market.total_cost_of_tce_in_time = 0.0
        self.market.consumer_count[...] = 0
        self.market.retailer_count[...] = 0
        self.market.producer_count[...] = 0
        self.market.produced_this_period[...] = np.sum(self.state.produced_this_period, axis=0)

        weighted_price_total = 0.0
        total_recent_market_production = 0.0
        for good_id in range(self.config.goods):
            self.market.consumer_count[good_id] = int(np.count_nonzero(self.state.role[:, good_id] == ROLE_CONSUMER))
            self.market.retailer_count[good_id] = int(np.count_nonzero(self.state.role[:, good_id] == ROLE_RETAILER))
            self.market.producer_count[good_id] = int(np.count_nonzero(self.state.role[:, good_id] == ROLE_PRODUCER))
            produced_now = float(self.market.produced_this_period[good_id])
            sum_production_cost = float(np.sum(self.state.produced_this_period[:, good_id] / np.maximum(self.state.efficiency[:, good_id], _EPSILON)))
            market_price = float(self.market.average_price[good_id])
            if produced_now > (self.config.population * float(self.market.elastic_need[good_id])) and produced_now > 0.0:
                total_weight = produced_now + float(self.market.recent_production[good_id])
                market_price = (
                    (produced_now / max(total_weight, _EPSILON)) * (sum_production_cost / produced_now)
                    + (float(self.market.recent_production[good_id]) / max(total_weight, _EPSILON)) * float(self.market.average_price[good_id])
                )
            self.market.average_price[good_id] = market_price
            self.market.recent_production[good_id] += produced_now
            total_recent_market_production += float(self.market.recent_production[good_id])
            weighted_price_total += market_price * float(self.market.recent_production[good_id])

        if total_recent_market_production > 0.0:
            self.market.price_average = weighted_price_total / total_recent_market_production
        else:
            self.market.price_average = float(np.mean(self.market.average_price))

        self.market.previous_elastic_need[...] = self.market.elastic_need
        if self.config.price_demand_elasticity == 0:
            updated_elastic_need = np.array(self.state.base_need[0], copy=True)
        else:
            price_ratio = self.market.price_average / np.maximum(self.market.average_price, _EPSILON)
            exponent = 1 if self.config.price_demand_elasticity == 1 else 2
            updated_elastic_need = np.array(self.state.base_need[0], copy=True) * np.power(price_ratio, exponent)

        total_elastic = float(np.sum(updated_elastic_need))
        if total_elastic > 0.0:
            updated_elastic_need *= self.period_length / total_elastic
        updated_elastic_need = np.minimum(updated_elastic_need, self.market.previous_elastic_need * self.config.max_rise_in_elastic_need)
        updated_elastic_need = np.maximum(updated_elastic_need, self.market.previous_elastic_need * self.config.max_drop_in_elastic_need)
        total_elastic_limited = float(np.sum(updated_elastic_need))
        if total_elastic_limited > 0.0:
            updated_elastic_need *= self.period_length / total_elastic_limited
        self.market.elastic_need[...] = updated_elastic_need.astype(np.float32)

        self.market.cost_of_spoilage_in_time[...] = self.market.periodic_spoilage * self.market.average_price
        self.market.cost_of_tce_in_time[...] = self.market.periodic_tce_cost * self.market.average_price
        self.market.total_cost_of_spoilage_in_time = float(np.sum(self.market.cost_of_spoilage_in_time))
        self.market.total_cost_of_tce_in_time = float(np.sum(self.market.cost_of_tce_in_time))
        self.market.recent_production[...] = self.market.recent_production * self.config.activity_discount

    def _add_random_friend(self, agent_id: int) -> None:
        if self.config.population <= 1:
            return
        candidate = self._sample_random_agent(agent_id, salt=17)
        if candidate < 0:
            return
        self._ensure_friend_link(agent_id, candidate, initial_transactions=2.0)

    def _build_friend_slot_maps(self) -> list[dict[int, int]]:
        friend_slot_maps: list[dict[int, int]] = []
        for agent_id in range(self.config.population):
            slot_map: dict[int, int] = {}
            for friend_slot, friend_id in enumerate(self.state.friend_id[agent_id]):
                resolved_friend_id = int(friend_id)
                if resolved_friend_id >= 0 and resolved_friend_id not in slot_map:
                    slot_map[resolved_friend_id] = friend_slot
            friend_slot_maps.append(slot_map)
        return friend_slot_maps

    def _ensure_friend_link(self, agent_id: int, friend_id: int, initial_transactions: float = 2.0) -> int:
        existing_slot = self._find_friend_slot(agent_id, friend_id)
        if existing_slot >= 0:
            return existing_slot
        empty_slots = np.flatnonzero(self.state.friend_id[agent_id] < 0)
        if empty_slots.size > 0:
            target_slot = int(empty_slots[0])
        else:
            target_slot = int(np.argmin(self.state.friend_activity[agent_id]))
        previous_friend_id = int(self.state.friend_id[agent_id, target_slot])
        if previous_friend_id >= 0 and self._friend_slot_maps[agent_id].get(previous_friend_id) == target_slot:
            del self._friend_slot_maps[agent_id][previous_friend_id]
        self.state.friend_id[agent_id, target_slot] = friend_id
        self.state.friend_activity[agent_id, target_slot] = initial_transactions
        self.state.friend_purchased[agent_id, target_slot, :] = 0.0
        self.state.friend_sold[agent_id, target_slot, :] = 0.0
        self.state.transparency[agent_id, target_slot, :] = self.config.initial_transparency
        self._friend_slot_maps[agent_id][friend_id] = target_slot
        return target_slot

    def _find_friend_slot(self, agent_id: int, friend_id: int) -> int:
        return self._friend_slot_maps[agent_id].get(friend_id, -1)

    def _sync_friend_slot_maps(self, agent_ids: list[int] | tuple[int, ...]) -> None:
        for raw_agent_id in agent_ids:
            agent_id = int(raw_agent_id)
            if agent_id < 0 or agent_id >= self.config.population:
                continue
            slot_map: dict[int, int] = {}
            for friend_slot, friend_id in enumerate(self.state.friend_id[agent_id]):
                resolved_friend_id = int(friend_id)
                if resolved_friend_id >= 0 and resolved_friend_id not in slot_map:
                    slot_map[resolved_friend_id] = friend_slot
            self._friend_slot_maps[agent_id] = slot_map

    def _sample_random_agent(self, agent_id: int, *, salt: int) -> int:
        rng_seed = self.config.seed + ((self.engine.cycle + 1) * 1_000_003) + (agent_id * 97) + salt
        rng = np.random.default_rng(rng_seed)
        for _ in range(max(8, self.config.acquaintances * 2)):
            candidate = int(rng.integers(0, self.config.population))
            if candidate == agent_id or self._find_friend_slot(agent_id, candidate) >= 0:
                continue
            return candidate
        for candidate in range(self.config.population):
            if candidate != agent_id and self._find_friend_slot(agent_id, candidate) < 0:
                return candidate
        return -1


def run_legacy_cycle(engine) -> None:
    from .legacy_cycle_native import can_use_native_legacy_cycle, run_native_legacy_cycle

    if can_use_native_legacy_cycle(engine) and run_native_legacy_cycle(engine):
        return
    LegacyCycleRunner(engine).run()
