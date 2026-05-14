from __future__ import annotations

import numpy as np
import pytest

from emergent_money.config import SimulationConfig
from emergent_money.engine import SimulationEngine
from emergent_money import legacy_cycle_native
from emergent_money.legacy_cycle import LegacyCycleRunner, run_legacy_cycle
from emergent_money.state import ROLE_CONSUMER, ROLE_RETAILER


class _FakeNativeCycleModule:
    def __init__(self) -> None:
        self.calls = 0
        self.last_engine = None

    def run_exact_cycle(self, engine) -> None:
        self.calls += 1
        self.last_engine = engine


def test_run_legacy_cycle_keeps_native_bridge_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = _FakeNativeCycleModule()
    engine = SimulationEngine.create(
        config=SimulationConfig(population=4, goods=2, acquaintances=1, active_acquaintances=1, demand_candidates=1, supply_candidates=1),
        backend_name='numpy',
    )
    marker = {'calls': 0}

    def fake_run(self) -> None:
        marker['calls'] += 1

    monkeypatch.setattr(legacy_cycle_native, '_load_native_search_module', lambda: fake_module)
    monkeypatch.setattr(LegacyCycleRunner, 'run', fake_run)

    run_legacy_cycle(engine)

    assert fake_module.calls == 0
    assert marker['calls'] == 1


def test_run_legacy_cycle_falls_back_to_python_when_native_cycle_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = SimulationEngine.create(
        config=SimulationConfig(population=4, goods=2, acquaintances=1, active_acquaintances=1, demand_candidates=1, supply_candidates=1),
        backend_name='numpy',
    )
    marker = {'calls': 0}

    def fake_run(self) -> None:
        marker['calls'] += 1

    monkeypatch.setattr(legacy_cycle_native, '_load_native_search_module', lambda: None)
    monkeypatch.setattr(LegacyCycleRunner, 'run', fake_run)

    run_legacy_cycle(engine)

    assert marker['calls'] == 1


def test_native_cycle_bridge_is_disabled_for_experimental_hybrid(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = _FakeNativeCycleModule()
    config = SimulationConfig(
        population=4,
        goods=2,
        acquaintances=1,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
        experimental_hybrid_batches=1,
        experimental_hybrid_consumption_stage=True,
    )
    engine = SimulationEngine.create(config=config, backend_name='numpy')
    marker = {'calls': 0}

    def fake_run(self) -> None:
        marker['calls'] += 1

    monkeypatch.setattr(legacy_cycle_native, '_load_native_search_module', lambda: fake_module)
    monkeypatch.setattr(LegacyCycleRunner, 'run', fake_run)

    run_legacy_cycle(engine)

    assert fake_module.calls == 0
    assert marker['calls'] == 1


def test_native_cycle_bridge_is_disabled_for_session_clearing_phenomenon(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = _FakeNativeCycleModule()
    config = SimulationConfig(
        population=4,
        goods=2,
        acquaintances=1,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
        experimental_session_clearing_phenomenon_exchange=True,
    )
    engine = SimulationEngine.create(config=config, backend_name='numpy')
    marker = {'calls': 0}

    def fake_run(self) -> None:
        marker['calls'] += 1

    monkeypatch.setattr(legacy_cycle_native, '_load_native_search_module', lambda: fake_module)
    monkeypatch.setattr(LegacyCycleRunner, 'run', fake_run)

    run_legacy_cycle(engine)

    assert fake_module.calls == 0
    assert marker['calls'] == 1


def test_run_legacy_cycle_uses_native_bridge_for_exchange_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = _FakeNativeCycleModule()
    config = SimulationConfig(
        population=4,
        goods=2,
        acquaintances=1,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
        experimental_native_exchange_stage=True,
    )
    engine = SimulationEngine.create(config=config, backend_name='numpy')
    marker = {'calls': 0}

    def fake_run(self) -> None:
        marker['calls'] += 1

    monkeypatch.setattr(legacy_cycle_native, '_load_native_search_module', lambda: fake_module)
    monkeypatch.setattr(LegacyCycleRunner, 'run', fake_run)

    run_legacy_cycle(engine)

    assert fake_module.calls == 1
    assert marker['calls'] == 0


def test_run_legacy_cycle_can_disable_native_bridge_for_diagnostics(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = _FakeNativeCycleModule()
    config = SimulationConfig(
        population=4,
        goods=2,
        acquaintances=1,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
        experimental_native_stage_math=True,
        experimental_agent_basket_planning=True,
        experimental_disable_native_cycle_bridge=True,
    )
    engine = SimulationEngine.create(config=config, backend_name='numpy')
    marker = {'calls': 0}

    def fake_run(self) -> None:
        marker['calls'] += 1

    monkeypatch.setattr(legacy_cycle_native, '_load_native_search_module', lambda: fake_module)
    monkeypatch.setattr(LegacyCycleRunner, 'run', fake_run)

    run_legacy_cycle(engine)

    assert fake_module.calls == 0
    assert marker['calls'] == 1


def test_run_legacy_cycle_keeps_native_bridge_disabled_without_native_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = _FakeNativeCycleModule()
    config = SimulationConfig(
        population=4,
        goods=2,
        acquaintances=1,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
    )
    engine = SimulationEngine.create(config=config, backend_name='numpy')
    marker = {'calls': 0}

    def fake_run(self) -> None:
        marker['calls'] += 1

    monkeypatch.setattr(legacy_cycle_native, '_load_native_search_module', lambda: fake_module)
    monkeypatch.setattr(LegacyCycleRunner, 'run', fake_run)

    run_legacy_cycle(engine)

    assert fake_module.calls == 0
    assert marker['calls'] == 1


def test_agent_basket_planning_uses_whole_native_cycle_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = _FakeNativeCycleModule()
    config = SimulationConfig(
        population=4,
        goods=2,
        acquaintances=1,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
        experimental_native_stage_math=True,
        experimental_agent_basket_planning=True,
    )
    engine = SimulationEngine.create(config=config, backend_name='numpy')
    marker = {'calls': 0}

    def fake_run(self) -> None:
        marker['calls'] += 1

    monkeypatch.setattr(legacy_cycle_native, '_load_native_search_module', lambda: fake_module)
    monkeypatch.setattr(LegacyCycleRunner, 'run', fake_run)

    run_legacy_cycle(engine)

    assert fake_module.calls == 1
    assert marker['calls'] == 0


def test_full_native_agent_basket_cycle_runs_end_to_end() -> None:
    config = SimulationConfig(
        population=16,
        goods=4,
        acquaintances=4,
        active_acquaintances=4,
        demand_candidates=2,
        supply_candidates=2,
        seed=2009,
        experimental_native_stage_math=True,
        experimental_agent_basket_planning=True,
        experimental_session_replan_after_trade=False,
    )
    engine = SimulationEngine.create(config=config, backend_name='numpy')
    runner = LegacyCycleRunner(engine)
    native_backend = _require_native_backend(runner)
    if not native_backend.supports_run_exact_cycle:
        pytest.skip('full native cycle is not available in this build')

    snapshot = engine.step()

    assert engine.cycle == 1
    assert snapshot.production_total > 0.0
    assert engine._cycle_need_total > 0.0
    assert snapshot.fulfilled_need_total >= 0.0
    assert np.isfinite(engine.state.stock).all()
    assert np.isfinite(engine.state.purchase_price).all()
    assert np.isfinite(engine.state.sales_price).all()


def test_native_agent_basket_planner_returns_active_need_candidate() -> None:
    engine, runner = _build_basket_exchange_runner()
    native_cycle = runner._native_cycle
    if native_cycle is None or not native_cycle.supports_plan_exchange_basket:
        pytest.skip('native basket planner is not available')

    candidates = native_cycle.plan_exchange_basket(
        agent_id=0,
        deal_type=2,
        forbidden_offer_by_need=np.zeros((engine.config.goods, engine.config.goods), dtype=np.bool_),
    )

    assert candidates
    score, need_good, friend_slot, friend_id, offer_good = candidates[0]
    assert score > 0.0
    assert (need_good, friend_slot, friend_id, offer_good) == (1, 0, 1, 0)


def test_parallel_phenomenon_planner_returns_execution_ready_candidate() -> None:
    _engine, runner = _build_basket_exchange_runner(experimental_parallel_phenomenon_exchange=True)
    native_cycle = runner._native_cycle
    if native_cycle is None or not native_cycle.supports_plan_parallel_phenomenon_candidates:
        pytest.skip('native parallel phenomenon planner is not available')

    candidates, reasons = runner._collect_experimental_exchange_candidates(
        deal_type=2,
        proposer_ids=(0,),
        blocked_partner_ids=set(),
        one_candidate_per_agent=False,
    )

    assert reasons == {}
    assert candidates
    candidate = candidates[0]
    assert candidate.agent_id == 0
    assert candidate.need_good == 1
    assert candidate.exchange.friend_id == 1
    assert candidate.exchange.offer_good == 0
    assert candidate.execution_plan is not None
    assert candidate.execution_plan.max_exchange > 0.0


def test_parallel_phenomenon_batches_execute_planned_trade() -> None:
    _engine, runner = _build_basket_exchange_runner(experimental_parallel_phenomenon_exchange=True)
    native_cycle = runner._native_cycle
    if native_cycle is None or not native_cycle.supports_plan_parallel_phenomenon_candidates:
        pytest.skip('native parallel phenomenon planner is not available')

    need_before = float(runner.state.need[0, 1])
    stock_before = float(runner.state.stock[0, 0])

    plan = runner.execute_experimental_consumption_batches(
        batch_count=1,
        proposer_ids=(0,),
        blocked_partner_ids=set(),
        one_candidate_per_agent=False,
    )

    assert plan.scheduled_count == 1
    assert runner.engine._accepted_trade_count == 1
    assert float(runner.state.need[0, 1]) < need_before
    assert float(runner.state.stock[0, 0]) < stock_before


def test_session_clearing_phenomenon_stage_executes_local_session() -> None:
    _engine, runner = _build_basket_exchange_runner(
        experimental_session_clearing_phenomenon_exchange=True
    )
    native_cycle = runner._native_cycle
    if native_cycle is None or not native_cycle.supports_run_agent_basket_exchange_stage:
        pytest.skip('native basket exchange stage is not available')

    need_before = float(runner.state.need[0, 1])
    stock_before = float(runner.state.stock[0, 0])

    runner._run_fixed_hybrid_exchange_stage(
        frontier_start=0,
        frontier_agent_ids=(0,),
        max_attempts=runner.config.goods * runner.config.acquaintances,
        deal_type=2,
        stage='consumption',
    )

    assert runner.engine._proposed_trade_count >= 1
    assert runner.engine._accepted_trade_count >= 1
    assert float(runner.state.need[0, 1]) < need_before
    assert float(runner.state.stock[0, 0]) < stock_before
    assert runner._hybrid_wave_diagnostics[-1].stage == 'consumption_session_clearing'


def test_session_clearing_after_trade_replan_can_continue_session() -> None:
    config = SimulationConfig(
        population=3,
        goods=3,
        acquaintances=2,
        active_acquaintances=2,
        demand_candidates=2,
        supply_candidates=2,
        initial_stock_fraction=0.0,
        experimental_session_clearing_phenomenon_exchange=True,
        experimental_session_replan_after_trade=True,
    )
    engine = SimulationEngine.create(config=config, backend_name='numpy')
    runner = LegacyCycleRunner(engine)
    native_cycle = runner._native_cycle
    if native_cycle is None or not native_cycle.supports_run_phenomenon_session_stage:
        pytest.skip('native session-clearing stage is not available')

    state = engine.state
    market = state.market
    state.base_need[...] = 1.0
    market.elastic_need[...] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    state.needs_level[...] = 1.0
    state.friend_id[...] = np.array(
        [
            [1, 2],
            [0, 2],
            [0, 1],
        ],
        dtype=np.int32,
    )
    state.friend_activity[...] = 2.0
    state.transparency[...] = 1.0
    runner._friend_slot_maps = runner._build_friend_slot_maps()
    state.stock[...] = np.array(
        [
            [40.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
        ],
        dtype=np.float32,
    )
    state.stock_limit[...] = 50.0
    state.need[...] = np.array(
        [
            [0.0, 5.0, 5.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    state.purchase_price[...] = np.array(
        [
            [1.0, 3.0, 3.0],
            [3.0, 1.0, 3.0],
            [3.0, 3.0, 1.0],
        ],
        dtype=np.float32,
    )
    state.sales_price[...] = 1.0
    state.role[...] = np.array(
        [
            [ROLE_CONSUMER, ROLE_CONSUMER, ROLE_CONSUMER],
            [ROLE_RETAILER, ROLE_CONSUMER, ROLE_CONSUMER],
            [ROLE_RETAILER, ROLE_CONSUMER, ROLE_CONSUMER],
        ],
        dtype=np.int32,
    )

    runner._run_fixed_hybrid_exchange_stage(
        frontier_start=0,
        frontier_agent_ids=(0,),
        max_attempts=runner.config.goods * runner.config.acquaintances,
        deal_type=2,
        stage='consumption',
    )

    assert runner.engine._accepted_trade_count >= 2
    assert float(runner.state.need[0, 1]) < 5.0
    assert float(runner.state.need[0, 2]) < 5.0


def test_agent_basket_planning_executes_trade_through_native_stage() -> None:
    _engine, runner = _build_basket_exchange_runner(experimental_agent_basket_planning=True)
    native_cycle = runner._native_cycle
    if native_cycle is None or not native_cycle.supports_run_agent_basket_exchange_stage:
        pytest.skip('native basket exchange stage is not available')

    need_before = float(runner.state.need[0, 1])
    stock_before = float(runner.state.stock[0, 0])

    runner._satisfy_needs_by_exchange(0)

    assert runner.engine._proposed_trade_count >= 1
    assert runner.engine._accepted_trade_count >= 1
    assert float(runner.state.need[0, 1]) < need_before
    assert float(runner.state.stock[0, 0]) < stock_before


def test_python_basket_fallback_still_executes_trade() -> None:
    _engine, runner = _build_basket_exchange_runner(experimental_agent_basket_planning=True)
    native_cycle = runner._native_cycle
    if native_cycle is None or not native_cycle.supports_plan_exchange_basket:
        pytest.skip('native basket planner is not available')

    need_before = float(runner.state.need[0, 1])
    stock_before = float(runner.state.stock[0, 0])

    runner._run_agent_basket_exchange_stage_python(0, 2)

    assert runner.engine._proposed_trade_count >= 1
    assert runner.engine._accepted_trade_count >= 1
    assert float(runner.state.need[0, 1]) < need_before
    assert float(runner.state.stock[0, 0]) < stock_before


def _build_basket_exchange_runner(
    *,
    experimental_agent_basket_planning: bool = False,
    experimental_parallel_phenomenon_exchange: bool = False,
    experimental_session_clearing_phenomenon_exchange: bool = False,
) -> tuple[SimulationEngine, LegacyCycleRunner]:
    config = SimulationConfig(
        population=2,
        goods=2,
        acquaintances=1,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
        initial_stock_fraction=0.0,
        experimental_agent_basket_planning=experimental_agent_basket_planning,
        experimental_parallel_phenomenon_exchange=experimental_parallel_phenomenon_exchange,
        experimental_session_clearing_phenomenon_exchange=experimental_session_clearing_phenomenon_exchange,
    )
    engine = SimulationEngine.create(config=config, backend_name='numpy')
    runner = LegacyCycleRunner(engine)
    state = engine.state
    market = state.market

    market.elastic_need[...] = np.array([1.0, 1.0], dtype=np.float32)
    state.needs_level[...] = 1.0
    state.friend_id[...] = np.array([[1], [0]], dtype=np.int32)
    state.friend_activity[...] = 2.0
    state.transparency[...] = 1.0
    runner._friend_slot_maps = runner._build_friend_slot_maps()

    state.stock[...] = np.array(
        [
            [10.0, 0.0],
            [0.0, 10.0],
        ],
        dtype=np.float32,
    )
    state.stock_limit[...] = 12.0
    state.need[...] = np.array(
        [
            [0.0, 5.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )
    state.purchase_price[...] = np.array(
        [
            [1.0, 2.0],
            [2.0, 1.0],
        ],
        dtype=np.float32,
    )
    state.sales_price[...] = 1.0
    state.role[...] = np.array(
        [
            [ROLE_CONSUMER, ROLE_CONSUMER],
            [ROLE_RETAILER, ROLE_CONSUMER],
        ],
        dtype=np.int32,
    )
    return engine, runner



def _build_end_agent_period_engine() -> SimulationEngine:
    config = SimulationConfig(
        population=3,
        goods=3,
        acquaintances=2,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
    )
    engine = SimulationEngine.create(config=config, backend_name='numpy')
    state = engine.state
    market = state.market

    state.stock[...] = np.array(
        [
            [4.0, 15.0, 18.0],
            [2.0, 5.0, 1.0],
            [9.0, 3.0, 2.0],
        ],
        dtype=np.float32,
    )
    state.stock_limit[...] = np.array(
        [
            [6.0, 10.0, 8.0],
            [5.0, 8.0, 4.0],
            [7.0, 6.0, 5.0],
        ],
        dtype=np.float32,
    )
    state.previous_stock_limit[...] = state.stock_limit
    state.recent_sales[...] = np.array(
        [
            [1.0, 8.0, 10.0],
            [1.0, 2.0, 1.0],
            [4.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    state.recent_production[...] = np.array(
        [
            [7.0, 5.0, 9.0],
            [1.0, 2.0, 1.0],
            [2.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    state.recent_purchases[...] = np.array(
        [
            [1.0, 3.0, 2.0],
            [4.0, 1.0, 2.0],
            [0.0, 2.0, 1.0],
        ],
        dtype=np.float32,
    )
    state.recent_inventory_inflow[...] = 2.0
    state.produced_this_period[...] = np.array(
        [
            [2.0, 3.0, 4.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    state.sold_this_period[...] = np.array(
        [
            [0.0, 4.0, 6.0],
            [2.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    state.purchased_this_period[...] = np.array(
        [
            [1.0, 1.0, 0.0],
            [0.0, 2.0, 1.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    state.purchase_times[...] = np.array(
        [
            [0, 2, 1],
            [1, 1, 0],
            [0, 0, 0],
        ],
        dtype=np.int32,
    )
    state.sales_times[...] = np.array(
        [
            [0, 2, 0],
            [1, 0, 0],
            [0, 1, 0],
        ],
        dtype=np.int32,
    )
    state.sum_period_purchase_value[...] = 1.5
    state.sum_period_sales_value[...] = 2.0
    state.talent_mask[...] = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    state.efficiency[...] = np.array(
        [
            [1.6, 1.0, 1.7],
            [1.0, 1.0, 1.0],
            [1.5, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    state.learned_efficiency[...] = state.efficiency
    state.purchase_price[...] = np.array(
        [
            [1.0, 1.5, 1.2],
            [0.9, 1.0, 1.1],
            [1.3, 0.8, 0.7],
        ],
        dtype=np.float32,
    )
    state.sales_price[...] = np.array(
        [
            [1.2, 1.8, 1.6],
            [1.0, 1.1, 1.2],
            [1.4, 0.9, 0.8],
        ],
        dtype=np.float32,
    )
    state.friend_activity[...] = np.array(
        [
            [2.0, 0.5],
            [1.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )
    state.friend_purchased[...] = 0.0
    state.friend_purchased[0, 0, 1] = 3.0
    state.friend_purchased[0, 1, 2] = 1.0
    state.transparency[...] = 0.7
    state.needs_level[...] = np.array([1.2, 1.0, 1.0], dtype=np.float32)
    market.elastic_need[...] = np.array([1.0, 4.0, 9.0], dtype=np.float32)
    market.periodic_spoilage[...] = 0.0
    return engine





def _require_native_backend(runner: LegacyCycleRunner):
    backend = runner._native_cycle
    if backend is None:
        pytest.skip('native cycle backend is not available in this environment')
    return backend


def _build_prepare_agent_engine() -> SimulationEngine:
    config = SimulationConfig(
        population=3,
        goods=3,
        acquaintances=2,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
        seed=2009,
    )
    engine = SimulationEngine.create(config=config, backend_name='numpy')
    state = engine.state
    market = state.market

    state.stock[...] = np.array(
        [
            [4.0, 12.0, 8.0],
            [3.0, 5.0, 6.0],
            [9.0, 2.0, 1.0],
        ],
        dtype=np.float32,
    )
    state.purchase_price[...] = np.array(
        [
            [1.0, 1.3, 2.1],
            [1.1, 0.9, 1.2],
            [1.4, 0.8, 0.7],
        ],
        dtype=np.float32,
    )
    state.sales_price[...] = np.array(
        [
            [1.2, 1.5, 1.8],
            [1.0, 1.1, 1.2],
            [1.3, 0.9, 0.8],
        ],
        dtype=np.float32,
    )
    state.purchased_last_period[...] = np.array(
        [
            [0.0, 2.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    state.recent_sales[...] = np.array(
        [
            [2.0, 6.0, 0.0],
            [0.0, 1.0, 2.0],
            [4.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    state.sold_this_period[...] = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    state.sold_last_period[...] = np.array(
        [
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    state.recent_purchases[...] = np.array(
        [
            [1.0, 2.0, 4.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    state.efficiency[...] = np.array(
        [
            [1.5, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    state.needs_level[...] = np.array([1.3, 1.0, 1.0], dtype=np.float32)
    state.recent_needs_increment[...] = np.array([1.1, 1.0, 1.0], dtype=np.float32)
    state.period_failure[...] = np.array([False, False, False], dtype=np.bool_)
    state.period_time_debt[...] = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    market.elastic_need[...] = np.array([1.0, 4.0, 9.0], dtype=np.float32)
    return engine


def test_native_end_agent_period_matches_python_reference() -> None:
    python_engine = _build_end_agent_period_engine()
    native_engine = _build_end_agent_period_engine()
    python_runner = LegacyCycleRunner(python_engine)
    native_runner = LegacyCycleRunner(native_engine)
    native_backend = _require_native_backend(native_runner)
    if not native_backend.supports_end_agent_period:
        pytest.skip('native end_agent_period is not available in this build')

    python_runner._end_agent_period(0)
    native_backend.end_agent_period(cycle=native_engine.cycle, agent_id=0)

    assert np.allclose(native_engine.state.stock, python_engine.state.stock)
    assert np.allclose(native_engine.state.stock_limit, python_engine.state.stock_limit)
    assert np.allclose(native_engine.state.previous_stock_limit, python_engine.state.previous_stock_limit)
    assert np.allclose(native_engine.state.efficiency, python_engine.state.efficiency)
    assert np.allclose(native_engine.state.learned_efficiency, python_engine.state.learned_efficiency)
    assert np.allclose(native_engine.state.purchase_price, python_engine.state.purchase_price)
    assert np.allclose(native_engine.state.sales_price, python_engine.state.sales_price)
    assert np.allclose(native_engine.state.recent_production, python_engine.state.recent_production)
    assert np.allclose(native_engine.state.recent_sales, python_engine.state.recent_sales)
    assert np.allclose(native_engine.state.recent_purchases, python_engine.state.recent_purchases)
    assert np.allclose(native_engine.state.recent_inventory_inflow, python_engine.state.recent_inventory_inflow)
    assert np.array_equal(native_engine.state.role, python_engine.state.role)
    assert np.allclose(native_engine.state.transparency, python_engine.state.transparency)
    assert np.allclose(native_engine.state.periodic_spoilage, python_engine.state.periodic_spoilage)
    assert np.allclose(native_engine.state.market.periodic_spoilage, python_engine.state.market.periodic_spoilage)


def test_native_end_agent_period_does_not_apply_absent_legacy_price_floor() -> None:
    config = SimulationConfig(
        population=1,
        goods=1,
        acquaintances=1,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
        legacy_price_floor=None,
        use_value_price_floor_fraction=0.0,
    )
    python_engine = SimulationEngine.create(config=config, backend_name='numpy')
    native_engine = SimulationEngine.create(config=config, backend_name='numpy')
    for engine in (python_engine, native_engine):
        state = engine.state
        state.stock[0, 0] = 100.0
        state.stock_limit[0, 0] = 10.0
        state.previous_stock_limit[0, 0] = 10.0
        state.recent_sales[0, 0] = 0.0
        state.recent_production[0, 0] = 0.0
        state.recent_purchases[0, 0] = 0.0
        state.produced_this_period[0, 0] = 0.0
        state.sold_this_period[0, 0] = 0.0
        state.purchased_this_period[0, 0] = 0.0
        state.purchase_times[0, 0] = 1
        state.sales_times[0, 0] = 0
        state.talent_mask[0, 0] = 0.0
        state.efficiency[0, 0] = 1.0
        state.purchase_price[0, 0] = 0.049
        state.sales_price[0, 0] = 0.049
        state.needs_level[0] = 1.0
        state.market.elastic_need[0] = 1.0

    python_runner = LegacyCycleRunner(python_engine)
    native_runner = LegacyCycleRunner(native_engine)
    native_backend = _require_native_backend(native_runner)
    if not native_backend.supports_end_agent_period:
        pytest.skip('native end_agent_period is not available in this build')

    python_runner._end_agent_period(0)
    native_backend.end_agent_period(cycle=native_engine.cycle, agent_id=0)

    assert python_engine.state.purchase_price[0, 0] < 0.05
    assert native_engine.state.purchase_price[0, 0] == pytest.approx(
        float(python_engine.state.purchase_price[0, 0])
    )


def test_native_prepare_agent_for_consumption_matches_python_reference() -> None:
    python_engine = _build_prepare_agent_engine()
    native_engine = _build_prepare_agent_engine()
    python_runner = LegacyCycleRunner(python_engine)
    native_runner = LegacyCycleRunner(native_engine)
    python_engine._cycle_need_total = 0.0
    native_engine._cycle_need_total = 0.0
    python_engine._stock_consumption_total = 0.0
    native_engine._stock_consumption_total = 0.0
    native_backend = _require_native_backend(native_runner)
    if not native_backend.supports_prepare_agent_for_consumption:
        pytest.skip('native prepare_agent_for_consumption is not available in this build')

    python_runner._prepare_agent_for_consumption(0)
    cycle_need_total, stock_consumed_total = native_backend.prepare_agent_for_consumption(agent_id=0)

    assert cycle_need_total == pytest.approx(python_engine._cycle_need_total)
    assert stock_consumed_total == pytest.approx(python_engine._stock_consumption_total)
    assert np.allclose(native_engine.state.need, python_engine.state.need)
    assert np.allclose(native_engine.state.stock, python_engine.state.stock)
    assert np.allclose(native_engine.state.needs_level, python_engine.state.needs_level)
    assert np.allclose(native_engine.state.recent_needs_increment, python_engine.state.recent_needs_increment)


def test_native_produce_need_matches_python_reference() -> None:
    config = SimulationConfig(
        population=2,
        goods=3,
        acquaintances=1,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
        seed=2009,
    )
    python_engine = SimulationEngine.create(config=config, backend_name='numpy')
    native_engine = SimulationEngine.create(config=config, backend_name='numpy')
    python_runner = LegacyCycleRunner(python_engine)
    native_runner = LegacyCycleRunner(native_engine)
    native_backend = _require_native_backend(native_runner)
    if not native_backend.supports_produce_need:
        pytest.skip('native produce_need is not available in this build')

    for engine in (python_engine, native_engine):
        state = engine.state
        state.need[0] = np.array([2.0, 5.0, 3.0], dtype=np.float32)
        state.efficiency[0] = np.array([2.0, 1.25, 4.0], dtype=np.float32)
        state.time_remaining[0] = 9.0
        state.recent_production[0] = 0.0
        state.produced_this_period[0] = 0.0
        state.timeout[0] = 0

    python_runner._produce_need(0)
    produced_total = native_backend.produce_need(agent_id=0)

    assert produced_total == pytest.approx(python_engine._production_total)
    assert np.allclose(native_engine.state.need, python_engine.state.need)
    assert np.allclose(native_engine.state.time_remaining, python_engine.state.time_remaining)
    assert np.allclose(native_engine.state.recent_production, python_engine.state.recent_production)
    assert np.allclose(native_engine.state.produced_this_period, python_engine.state.produced_this_period)
    assert np.array_equal(native_engine.state.timeout, python_engine.state.timeout)


def test_native_produce_need_matches_time_limited_python_reference() -> None:
    config = SimulationConfig(
        population=2,
        goods=3,
        acquaintances=1,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
        seed=2009,
    )
    python_engine = SimulationEngine.create(config=config, backend_name='numpy')
    native_engine = SimulationEngine.create(config=config, backend_name='numpy')
    python_runner = LegacyCycleRunner(python_engine)
    native_runner = LegacyCycleRunner(native_engine)
    native_backend = _require_native_backend(native_runner)
    if not native_backend.supports_produce_need:
        pytest.skip('native produce_need is not available in this build')

    for engine in (python_engine, native_engine):
        state = engine.state
        state.need[0] = np.array([10.0, 20.0, 5.0], dtype=np.float32)
        state.efficiency[0] = np.array([1.0, 2.0, 5.0], dtype=np.float32)
        state.time_remaining[0] = 10.5
        state.recent_production[0] = 0.0
        state.produced_this_period[0] = 0.0
        state.timeout[0] = 0

    python_runner._produce_need(0)
    produced_total = native_backend.produce_need(agent_id=0)

    assert produced_total == pytest.approx(python_engine._production_total)
    assert np.allclose(native_engine.state.need, python_engine.state.need)
    assert np.allclose(native_engine.state.time_remaining, python_engine.state.time_remaining)
    assert np.allclose(native_engine.state.recent_production, python_engine.state.recent_production)
    assert np.allclose(native_engine.state.produced_this_period, python_engine.state.produced_this_period)
    assert np.array_equal(native_engine.state.timeout, python_engine.state.timeout)



def test_experimental_native_stage_math_preserves_short_cycle_outputs() -> None:
    config_python = SimulationConfig(
        population=16,
        goods=4,
        acquaintances=4,
        active_acquaintances=2,
        demand_candidates=2,
        supply_candidates=2,
        seed=2009,
    )
    config_native = SimulationConfig(
        population=16,
        goods=4,
        acquaintances=4,
        active_acquaintances=2,
        demand_candidates=2,
        supply_candidates=2,
        seed=2009,
        experimental_native_stage_math=True,
    )
    python_engine = SimulationEngine.create(config=config_python, backend_name='numpy')
    native_engine = SimulationEngine.create(config=config_native, backend_name='numpy')
    native_runner = LegacyCycleRunner(native_engine)
    native_backend = _require_native_backend(native_runner)
    if not (native_backend.supports_prepare_agent_for_consumption and native_backend.supports_produce_need):
        pytest.skip('native stage math is not available in this build')

    for _ in range(3):
        python_snapshot = python_engine.step()
        native_snapshot = native_engine.step()
        assert native_snapshot.production_total == pytest.approx(python_snapshot.production_total)
        assert native_snapshot.accepted_trade_volume == pytest.approx(python_snapshot.accepted_trade_volume)
        assert native_snapshot.utility_proxy_total == pytest.approx(python_snapshot.utility_proxy_total)

    assert np.array_equal(native_engine.state.stock, python_engine.state.stock)
    assert np.array_equal(native_engine.state.need, python_engine.state.need)
    assert np.array_equal(native_engine.state.time_remaining, python_engine.state.time_remaining)
    assert np.array_equal(native_engine.state.recent_production, python_engine.state.recent_production)
    assert np.array_equal(native_engine.state.produced_this_period, python_engine.state.produced_this_period)
    assert np.array_equal(native_engine.state.recent_needs_increment, python_engine.state.recent_needs_increment)



def _build_surplus_production_engine() -> SimulationEngine:
    config = SimulationConfig(
        population=2,
        goods=3,
        acquaintances=1,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
        seed=2009,
    )
    engine = SimulationEngine.create(config=config, backend_name='numpy')
    state = engine.state
    state.base_need[...] = np.array([[1.0, 4.0, 9.0], [1.0, 4.0, 9.0]], dtype=np.float32)
    state.stock[0] = np.array([6.0, 8.0, 3.0], dtype=np.float32)
    state.stock_limit[0] = np.array([10.0, 12.0, 8.0], dtype=np.float32)
    state.talent_mask[0] = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    state.purchase_times[0] = np.array([0, 1, 0], dtype=np.int32)
    state.efficiency[0] = np.array([1.5, 1.8, 1.0], dtype=np.float32)
    state.sales_price[0] = np.array([0.8, 1.4, 1.0], dtype=np.float32)
    state.time_remaining[0] = 6.0
    state.recent_production[0] = 0.0
    state.produced_this_period[0] = 0.0
    return engine


def _build_leisure_production_engine() -> SimulationEngine:
    config = SimulationConfig(
        population=2,
        goods=3,
        acquaintances=1,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
        seed=2009,
    )
    engine = SimulationEngine.create(config=config, backend_name='numpy')
    state = engine.state
    state.stock[0] = np.array([4.0, 2.0, 1.0], dtype=np.float32)
    state.stock_limit[0] = np.array([7.0, 8.0, 6.0], dtype=np.float32)
    state.talent_mask[0] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    state.purchase_price[0] = np.array([0.9, 1.4, 1.1], dtype=np.float32)
    state.time_remaining[0] = 5.0
    state.recent_production[0] = 0.0
    state.produced_this_period[0] = 0.0
    return engine


def test_native_surplus_production_matches_python_reference() -> None:
    python_engine = _build_surplus_production_engine()
    native_engine = _build_surplus_production_engine()
    python_runner = LegacyCycleRunner(python_engine)
    native_runner = LegacyCycleRunner(native_engine)
    native_backend = _require_native_backend(native_runner)
    if not native_backend.supports_surplus_production:
        pytest.skip('native surplus_production is not available in this build')

    python_runner._surplus_production(0)
    produced_total = native_backend.surplus_production(agent_id=0)

    assert produced_total == pytest.approx(python_engine._production_total)
    assert np.array_equal(native_engine.state.stock, python_engine.state.stock)
    assert np.array_equal(native_engine.state.time_remaining, python_engine.state.time_remaining)
    assert np.array_equal(native_engine.state.recent_production, python_engine.state.recent_production)
    assert np.array_equal(native_engine.state.produced_this_period, python_engine.state.produced_this_period)


def test_native_leisure_production_matches_python_reference() -> None:
    python_engine = _build_leisure_production_engine()
    native_engine = _build_leisure_production_engine()
    python_runner = LegacyCycleRunner(python_engine)
    native_runner = LegacyCycleRunner(native_engine)
    native_backend = _require_native_backend(native_runner)
    if not native_backend.supports_leisure_production:
        pytest.skip('native leisure_production is not available in this build')

    python_runner._leisure_production(0)
    produced_total = native_backend.leisure_production(agent_id=0)

    assert produced_total == pytest.approx(python_engine._production_total)
    assert np.array_equal(native_engine.state.stock, python_engine.state.stock)
    assert np.array_equal(native_engine.state.time_remaining, python_engine.state.time_remaining)
    assert np.array_equal(native_engine.state.recent_production, python_engine.state.recent_production)
    assert np.array_equal(native_engine.state.produced_this_period, python_engine.state.produced_this_period)


def _build_consumption_exchange_stage_engine() -> SimulationEngine:
    config = SimulationConfig(
        population=2,
        goods=2,
        acquaintances=1,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
        seed=2009,
    )
    engine = SimulationEngine.create(config=config, backend_name='numpy')
    state = engine.state
    market = state.market
    state.stock[...] = np.array([[0.0, 4.0], [4.0, 0.0]], dtype=np.float32)
    state.stock_limit[...] = np.array([[4.0, 4.0], [4.0, 4.0]], dtype=np.float32)
    state.need[...] = np.array([[1.5, 0.0], [0.0, 0.0]], dtype=np.float32)
    state.needs_level[...] = np.array([1.0, 1.0], dtype=np.float32)
    state.role[...] = ROLE_CONSUMER
    state.purchase_price[...] = 1.0
    state.sales_price[...] = 1.0
    state.recent_sales[...] = 0.0
    state.recent_purchases[...] = 0.0
    state.sold_this_period[...] = 0.0
    state.purchased_this_period[...] = 0.0
    state.recent_inventory_inflow[...] = 0.0
    state.purchase_times[...] = 0
    state.sales_times[...] = 0
    state.sum_period_purchase_value[...] = 0.0
    state.sum_period_sales_value[...] = 0.0
    state.friend_id[...] = -1
    state.friend_id[0, 0] = 1
    state.friend_activity[...] = 0.0
    state.friend_purchased[...] = 0.0
    state.friend_sold[...] = 0.0
    state.transparency[...] = 0.7
    market.elastic_need[...] = np.array([1.0, 1.0], dtype=np.float32)
    market.periodic_tce_cost[...] = 0.0
    return engine


def _build_surplus_exchange_stage_engine() -> SimulationEngine:
    config = SimulationConfig(
        population=2,
        goods=2,
        acquaintances=1,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
        seed=2009,
    )
    engine = SimulationEngine.create(config=config, backend_name='numpy')
    state = engine.state
    market = state.market
    state.stock[...] = np.array([[0.0, 4.0], [4.0, 0.0]], dtype=np.float32)
    state.stock_limit[...] = np.array([[4.0, 4.0], [4.0, 4.0]], dtype=np.float32)
    state.need[...] = 0.0
    state.needs_level[...] = np.array([1.0, 1.0], dtype=np.float32)
    state.role[...] = ROLE_CONSUMER
    state.purchase_price[...] = 1.0
    state.sales_price[...] = 1.0
    state.recent_sales[...] = 0.0
    state.recent_sales[0, 0] = 2.0
    state.recent_production[...] = 0.0
    state.recent_purchases[...] = 0.0
    state.sold_this_period[...] = 0.0
    state.purchased_this_period[...] = 0.0
    state.recent_inventory_inflow[...] = 0.0
    state.purchase_times[...] = 0
    state.sales_times[...] = 0
    state.sum_period_purchase_value[...] = 0.0
    state.sum_period_sales_value[...] = 0.0
    state.friend_id[...] = -1
    state.friend_id[0, 0] = 1
    state.friend_activity[...] = 0.0
    state.friend_purchased[...] = 0.0
    state.friend_sold[...] = 0.0
    state.transparency[...] = 0.7
    market.elastic_need[...] = np.array([1.0, 1.0], dtype=np.float32)
    market.periodic_tce_cost[...] = 0.0
    return engine


def test_native_consumption_exchange_stage_matches_python_reference() -> None:
    python_engine = _build_consumption_exchange_stage_engine()
    native_engine = _build_consumption_exchange_stage_engine()
    python_runner = LegacyCycleRunner(python_engine)
    native_runner = LegacyCycleRunner(native_engine)
    native_backend = _require_native_backend(native_runner)
    if not native_backend.supports_run_exchange_stage:
        pytest.skip('native exchange stage is not available in this build')

    python_runner._satisfy_needs_by_exchange(0)
    proposed_count, accepted_count, accepted_volume, inventory_trade_volume = native_backend.run_exchange_stage(
        agent_id=0,
        deal_type=2,
    )
    native_engine._proposed_trade_count += proposed_count
    native_engine._accepted_trade_count += accepted_count
    native_engine._accepted_trade_volume += accepted_volume
    native_engine._inventory_trade_volume += inventory_trade_volume

    assert native_engine._proposed_trade_count == python_engine._proposed_trade_count
    assert native_engine._accepted_trade_count == python_engine._accepted_trade_count
    assert native_engine._accepted_trade_volume == pytest.approx(python_engine._accepted_trade_volume)
    assert native_engine._inventory_trade_volume == pytest.approx(python_engine._inventory_trade_volume)
    assert np.allclose(native_engine.state.stock, python_engine.state.stock)
    assert np.allclose(native_engine.state.need, python_engine.state.need)
    assert np.allclose(native_engine.state.recent_sales, python_engine.state.recent_sales)
    assert np.allclose(native_engine.state.recent_purchases, python_engine.state.recent_purchases)
    assert np.allclose(native_engine.state.sold_this_period, python_engine.state.sold_this_period)
    assert np.allclose(native_engine.state.purchased_this_period, python_engine.state.purchased_this_period)
    assert np.allclose(native_engine.state.recent_inventory_inflow, python_engine.state.recent_inventory_inflow)
    assert np.array_equal(native_engine.state.friend_id, python_engine.state.friend_id)
    assert np.allclose(native_engine.state.friend_activity, python_engine.state.friend_activity)
    assert np.allclose(native_engine.state.friend_purchased, python_engine.state.friend_purchased)
    assert np.allclose(native_engine.state.friend_sold, python_engine.state.friend_sold)
    assert native_runner._find_friend_slot(1, 0) == python_runner._find_friend_slot(1, 0)


def test_native_surplus_exchange_stage_matches_python_reference() -> None:
    python_engine = _build_surplus_exchange_stage_engine()
    native_engine = _build_surplus_exchange_stage_engine()
    python_runner = LegacyCycleRunner(python_engine)
    native_runner = LegacyCycleRunner(native_engine)
    native_backend = _require_native_backend(native_runner)
    if not native_backend.supports_run_exchange_stage:
        pytest.skip('native exchange stage is not available in this build')

    python_runner._make_surplus_deals(0)
    proposed_count, accepted_count, accepted_volume, inventory_trade_volume = native_backend.run_exchange_stage(
        agent_id=0,
        deal_type=1,
    )
    native_engine._proposed_trade_count += proposed_count
    native_engine._accepted_trade_count += accepted_count
    native_engine._accepted_trade_volume += accepted_volume
    native_engine._inventory_trade_volume += inventory_trade_volume

    assert native_engine._proposed_trade_count == python_engine._proposed_trade_count
    assert native_engine._accepted_trade_count == python_engine._accepted_trade_count
    assert native_engine._accepted_trade_volume == pytest.approx(python_engine._accepted_trade_volume)
    assert native_engine._inventory_trade_volume == pytest.approx(python_engine._inventory_trade_volume)
    assert np.allclose(native_engine.state.stock, python_engine.state.stock)
    assert np.allclose(native_engine.state.recent_sales, python_engine.state.recent_sales)
    assert np.allclose(native_engine.state.recent_purchases, python_engine.state.recent_purchases)
    assert np.allclose(native_engine.state.recent_inventory_inflow, python_engine.state.recent_inventory_inflow)
    assert np.allclose(native_engine.state.market.periodic_tce_cost, python_engine.state.market.periodic_tce_cost)
    assert np.array_equal(native_engine.state.friend_id, python_engine.state.friend_id)
    assert np.allclose(native_engine.state.friend_activity, python_engine.state.friend_activity)
    assert np.allclose(native_engine.state.friend_purchased, python_engine.state.friend_purchased)
    assert np.allclose(native_engine.state.friend_sold, python_engine.state.friend_sold)
    assert native_runner._find_friend_slot(1, 0) == python_runner._find_friend_slot(1, 0)



def _build_prepare_leisure_engine() -> SimulationEngine:
    config = SimulationConfig(
        population=2,
        goods=3,
        acquaintances=1,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
        seed=2009,
    )
    engine = SimulationEngine.create(config=config, backend_name='numpy')
    state = engine.state
    market = state.market
    state.need[...] = 0.0
    state.stock[...] = np.array([[1.0, 3.0, 2.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    state.needs_level[...] = np.array([2.0, 1.0], dtype=np.float32)
    state.recent_needs_increment[...] = np.array([1.1, 1.0], dtype=np.float32)
    state.time_remaining[...] = np.array([5.0, 0.0], dtype=np.float32)
    market.elastic_need[...] = np.array([1.0, 4.0, 9.0], dtype=np.float32)
    return engine


def test_native_prepare_leisure_round_matches_python_reference() -> None:
    python_engine = _build_prepare_leisure_engine()
    native_engine = _build_prepare_leisure_engine()
    python_runner = LegacyCycleRunner(python_engine)
    native_runner = LegacyCycleRunner(native_engine)
    native_backend = _require_native_backend(native_runner)
    if not native_backend.supports_prepare_leisure_round:
        pytest.skip('native prepare_leisure_round is not available in this build')

    remaining_time = float(python_engine.state.time_remaining[0])
    extra_need = python_runner._compute_leisure_extra_need(0)
    assert extra_need is not None
    utilized_time = max(python_runner.period_length - remaining_time, 1.0)
    capped_increment = min(
        python_runner.period_length / utilized_time,
        float(python_engine.state.recent_needs_increment[0]) * python_runner.config.max_needs_increase,
    )
    python_engine.state.recent_needs_increment[0] = (
        capped_increment + (python_runner.config.history * float(python_engine.state.recent_needs_increment[0]))
    ) / float(python_runner.config.history + 1)
    python_engine.state.need[0] += extra_need
    python_engine._cycle_need_total += float(np.sum(extra_need))
    python_engine._leisure_extra_need_total += float(np.sum(extra_need))
    python_runner._consume_surplus(0)

    has_extra, extra_need_total, stock_consumed_total = native_backend.prepare_leisure_round(agent_id=0)

    assert has_extra is True
    assert extra_need_total == pytest.approx(python_engine._leisure_extra_need_total)
    assert stock_consumed_total == pytest.approx(python_engine._stock_consumption_total)
    assert np.array_equal(native_engine.state.need, python_engine.state.need)
    assert np.array_equal(native_engine.state.stock, python_engine.state.stock)
    assert np.array_equal(native_engine.state.recent_needs_increment, python_engine.state.recent_needs_increment)
