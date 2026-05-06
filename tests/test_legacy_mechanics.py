from __future__ import annotations

import sys
from types import ModuleType

import numpy as np
import pytest

from emergent_money.config import SimulationConfig
from emergent_money.engine import SimulationEngine
from emergent_money.hybrid_batching import batch_is_conflict_free
from emergent_money.legacy_cycle import LegacyCycleRunner
from emergent_money import legacy_search_backend
from emergent_money.legacy_search_backend import ExchangePlanRequest, ExchangeSearchRequest, NativeModuleExchangeSearchBackend, build_exchange_search_backend
from emergent_money.state import ROLE_CONSUMER, ROLE_PRODUCER, ROLE_RETAILER


def test_exact_legacy_step_generates_transaction_cost_waste() -> None:
    config = SimulationConfig(
        population=2,
        goods=2,
        acquaintances=1,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
    )
    engine = SimulationEngine.create(config=config, backend_name="numpy")
    state = engine.state

    state.friend_id[...] = np.array([[1], [0]], dtype=np.int32)
    state.friend_activity[...] = 2.0
    state.transparency[...] = 0.7
    state.stock[...] = np.array(
        [
            [6.0, 0.0],
            [0.0, 14.0],
        ],
        dtype=np.float32,
    )
    state.purchase_price[...] = np.array(
        [
            [1.0, 3.0],
            [3.0, 1.0],
        ],
        dtype=np.float32,
    )
    state.sales_price[...] = 1.0

    snapshot = engine.step()

    assert snapshot.accepted_trade_count > 0
    assert snapshot.periodic_tce_cost_total > 0.0
    assert snapshot.tce_cost_in_time_total > 0.0
    assert float(np.sum(state.recent_purchase_value)) > 0.0
    assert float(np.sum(state.recent_sales_value)) > 0.0


def test_local_liquidity_surplus_buffer_requires_observed_friend_acceptance() -> None:
    config = SimulationConfig(
        population=2,
        goods=2,
        acquaintances=1,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
        experimental_local_liquidity_stock_bias=2.0,
        experimental_local_liquidity_min_sales=1.0,
    )
    engine = SimulationEngine.create(config=config, backend_name="numpy")
    runner = LegacyCycleRunner(engine)
    state = engine.state

    state.friend_id[...] = np.array([[1], [0]], dtype=np.int32)
    state.transparency[...] = 0.8
    state.needs_level[...] = 1.0
    state.stock_limit[0, 1] = 2.0
    state.recent_sales[0, 1] = 4.0
    state.recent_purchases[0, 1] = 4.0

    base_limit = float(state.stock_limit[0, 1])
    assert runner._surplus_target_stock_limit(0, 1) == pytest.approx(base_limit)

    state.friend_sold[0, 0, 1] = 2.0

    assert runner._surplus_target_stock_limit(0, 1) > base_limit


def test_local_liquidity_surplus_buffer_scales_from_observed_intermediary_flow() -> None:
    config = SimulationConfig(
        population=2,
        goods=2,
        acquaintances=1,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
        experimental_local_liquidity_stock_bias=2.0,
        experimental_local_liquidity_min_sales=1.0,
    )
    engine = SimulationEngine.create(config=config, backend_name="numpy")
    runner = LegacyCycleRunner(engine)
    state = engine.state

    state.friend_id[...] = np.array([[1], [0]], dtype=np.int32)
    state.transparency[...] = 1.0
    state.needs_level[...] = 1.0
    state.market.elastic_need[...] = np.array([1.0, 100.0], dtype=np.float32)
    state.stock_limit[0, :] = 2.0

    state.friend_sold[0, 0, 0] = 30.0
    state.recent_sales[0, 0] = 30.0
    state.recent_purchases[0, 0] = 30.0

    assert runner._surplus_target_stock_limit(0, 0) > 50.0
    assert runner._surplus_target_stock_limit(0, 1) == pytest.approx(2.0)


def test_end_agent_period_spoils_stock_above_threshold() -> None:
    config = SimulationConfig(
        population=1,
        goods=2,
        acquaintances=1,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
    )
    engine = SimulationEngine.create(config=config, backend_name="numpy")
    state = engine.state
    runner = LegacyCycleRunner(engine)

    state.stock[0, 1] = 50.0
    state.stock_limit[0, 1] = 10.0
    state.previous_stock_limit[0, 1] = 10.0
    state.recent_sales[0, 1] = 0.0
    state.needs_level[0] = 1.0
    state.time_remaining[0] = 0.0

    runner._end_agent_period(0)

    assert state.spoilage[0, 1] > 0.0
    assert state.periodic_spoilage[0] > 0.0
    assert state.market.periodic_spoilage[1] > 0.0


def test_leisure_round_adds_capped_temporary_extra_need() -> None:
    config = SimulationConfig(
        population=1,
        goods=2,
        acquaintances=1,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
        max_leisure_extra_multiplier=1.0,
    )
    engine = SimulationEngine.create(config=config, backend_name="numpy")
    state = engine.state
    runner = LegacyCycleRunner(engine)

    state.need[...] = 0.0
    state.stock[...] = 0.0
    state.stock_limit[...] = 0.0
    state.previous_stock_limit[...] = 0.0
    state.recent_production[...] = 0.0
    state.produced_this_period[...] = 0.0
    state.talent_mask[...] = 0.0
    state.efficiency[...] = 10.0
    state.learned_efficiency[...] = 10.0
    state.innate_efficiency[...] = 10.0
    state.needs_level[0] = 2.0
    state.recent_needs_increment[0] = 1.1
    state.time_remaining[0] = 4.0

    runner._run_leisure_round(0)

    expected_extra = np.array([1.3, 5.2], dtype=np.float32)
    assert np.allclose(state.recent_production[0], expected_extra)
    assert np.isclose(engine._leisure_extra_need_total, float(expected_extra.sum()))
    assert np.allclose(state.need[0], 0.0)
    assert state.recent_needs_increment[0] > 1.1


def test_exact_need_production_is_limited_by_available_time() -> None:
    config = SimulationConfig(
        population=1,
        goods=2,
        acquaintances=1,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
    )
    engine = SimulationEngine.create(config=config, backend_name="numpy")
    state = engine.state
    runner = LegacyCycleRunner(engine)

    state.need[0] = np.array([10.0, 20.0], dtype=np.float32)
    state.efficiency[0] = np.array([1.0, 2.0], dtype=np.float32)
    state.time_remaining[0] = 10.0
    state.recent_production[0] = 0.0
    state.produced_this_period[0] = 0.0

    runner._produce_need(0)

    assert np.allclose(state.produced_this_period[0], np.array([5.0, 10.0], dtype=np.float32))
    assert np.allclose(state.need[0], np.array([5.0, 10.0], dtype=np.float32))
    assert float(state.time_remaining[0]) == pytest.approx(0.0)
    assert int(state.timeout[0]) == 1


def test_exact_agent_period_fails_when_time_limited_need_remains() -> None:
    config = SimulationConfig(
        population=1,
        goods=1,
        acquaintances=1,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
    )
    engine = SimulationEngine.create(config=config, backend_name="numpy")
    state = engine.state
    runner = LegacyCycleRunner(engine)

    state.need[0, 0] = 10.0
    state.efficiency[0, 0] = 1.0
    state.time_remaining[0] = 5.0
    state.recent_production[0, 0] = 0.0
    state.produced_this_period[0, 0] = 0.0

    runner._advance_agent_to_surplus_stage(0)

    assert bool(state.period_failure[0])
    assert float(state.need[0, 0]) == pytest.approx(5.0)
    assert float(state.time_remaining[0]) == pytest.approx(0.0)


def test_exact_legacy_agent_cycle_skips_extra_demand_round_by_default() -> None:
    config = SimulationConfig(
        population=1,
        goods=2,
        acquaintances=1,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
    )
    engine = SimulationEngine.create(config=config, backend_name="numpy")
    runner = LegacyCycleRunner(engine)
    calls: list[int] = []

    def _record(agent_id: int) -> None:
        calls.append(agent_id)

    runner._run_leisure_round = _record  # type: ignore[method-assign]
    runner._run_agent_cycle(0)

    assert calls == []


def test_exact_legacy_agent_cycle_can_opt_in_extra_demand_round() -> None:
    config = SimulationConfig(
        population=1,
        goods=2,
        acquaintances=1,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
        legacy_extra_demand_round=True,
    )
    engine = SimulationEngine.create(config=config, backend_name="numpy")
    runner = LegacyCycleRunner(engine)
    calls: list[int] = []

    def _record(agent_id: int) -> None:
        calls.append(agent_id)

    runner._run_leisure_round = _record  # type: ignore[method-assign]
    runner._run_agent_cycle(0)

    assert calls == [0]


def test_exact_purchase_price_can_drop_below_old_floor_by_default() -> None:
    config = SimulationConfig(
        population=1,
        goods=1,
        acquaintances=1,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
    )
    engine = SimulationEngine.create(config=config, backend_name="numpy")
    state = engine.state
    runner = LegacyCycleRunner(engine)

    state.role[0, 0] = ROLE_RETAILER
    state.stock[0, 0] = 1000.0
    state.stock_limit[0, 0] = 1.0
    state.purchase_times[0, 0] = 1
    state.purchase_price[0, 0] = 0.051

    runner._adjust_purchase_price(0, 0)

    assert float(state.purchase_price[0, 0]) < 0.05


def test_exact_purchase_price_respects_optional_legacy_floor() -> None:
    config = SimulationConfig(
        population=1,
        goods=1,
        acquaintances=1,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
        legacy_price_floor=0.05,
    )
    engine = SimulationEngine.create(config=config, backend_name="numpy")
    state = engine.state
    runner = LegacyCycleRunner(engine)

    state.role[0, 0] = ROLE_RETAILER
    state.stock[0, 0] = 1000.0
    state.stock_limit[0, 0] = 1.0
    state.purchase_times[0, 0] = 1
    state.purchase_price[0, 0] = 0.051

    runner._adjust_purchase_price(0, 0)

    assert float(state.purchase_price[0, 0]) == pytest.approx(0.05)


def test_use_value_price_floor_anchors_to_focused_cost_when_inventory_is_useful() -> None:
    config = SimulationConfig(
        population=1,
        goods=1,
        acquaintances=1,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
    )
    engine = SimulationEngine.create(config=config, backend_name="numpy")
    state = engine.state
    runner = LegacyCycleRunner(engine)

    state.role[0, 0] = ROLE_RETAILER
    state.efficiency[0, 0] = 2.0
    state.stock[0, 0] = 1.0
    state.stock_limit[0, 0] = 100.0
    state.market.elastic_need[0] = 10.0
    state.purchase_price[0, 0] = 0.0
    state.sales_price[0, 0] = 0.0

    runner._adjust_purchase_price(0, 0)

    expected_floor = 0.5
    assert float(state.purchase_price[0, 0]) == pytest.approx(expected_floor)
    assert float(state.sales_price[0, 0]) == pytest.approx(expected_floor / config.price_reduction)


def test_use_value_price_floor_falls_with_visible_inventory_overhang() -> None:
    config = SimulationConfig(
        population=1,
        goods=1,
        acquaintances=1,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
    )
    engine = SimulationEngine.create(config=config, backend_name="numpy")
    state = engine.state
    runner = LegacyCycleRunner(engine)

    state.efficiency[0, 0] = 2.0
    state.stock[0, 0] = 1000.0
    state.stock_limit[0, 0] = 1.0
    state.market.elastic_need[0] = 1.0
    state.needs_level[0] = 1.0

    survival_fraction = (1.0 - config.spoilage_rate) ** config.history
    visible_capacity = max(
        config.history * float(state.market.elastic_need[0]),
        float(state.stock_limit[0, 0]),
        config.min_trade_quantity,
    )
    expected_floor = 0.5 * ((visible_capacity / survival_fraction) / float(state.stock[0, 0]))

    assert runner._minimum_price_floor(0, 0, production_cost=0.5) == pytest.approx(expected_floor)


def test_retailer_purchase_price_keeps_resale_margin_when_scarce() -> None:
    config = SimulationConfig(
        population=1,
        goods=1,
        acquaintances=1,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
    )
    engine = SimulationEngine.create(config=config, backend_name="numpy")
    state = engine.state
    runner = LegacyCycleRunner(engine)

    state.role[0, 0] = ROLE_RETAILER
    state.stock[0, 0] = 0.0
    state.stock_limit[0, 0] = 100.0
    state.purchase_price[0, 0] = 8.0
    state.sales_price[0, 0] = 8.0
    state.purchased_this_period[0, 0] = 0.0
    state.sold_this_period[0, 0] = 4.0

    runner._adjust_purchase_price(0, 0)

    assert float(state.purchase_price[0, 0]) == pytest.approx(8.0 * config.price_reduction)


def test_retailer_purchase_price_rises_when_bought_goods_sell_through_with_margin() -> None:
    config = SimulationConfig(
        population=1,
        goods=1,
        acquaintances=1,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
    )
    engine = SimulationEngine.create(config=config, backend_name="numpy")
    state = engine.state
    runner = LegacyCycleRunner(engine)

    state.role[0, 0] = ROLE_RETAILER
    state.stock[0, 0] = 0.0
    state.stock_limit[0, 0] = 100.0
    state.purchase_price[0, 0] = 8.0
    state.sales_price[0, 0] = 10.0
    state.purchased_this_period[0, 0] = 4.0
    state.sold_this_period[0, 0] = 4.0

    runner._adjust_purchase_price(0, 0)

    assert float(state.purchase_price[0, 0]) == pytest.approx(8.0 * config.price_hike)


def test_retailer_purchase_price_falls_when_buying_more_than_selling() -> None:
    config = SimulationConfig(
        population=1,
        goods=1,
        acquaintances=1,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
    )
    engine = SimulationEngine.create(config=config, backend_name="numpy")
    state = engine.state
    runner = LegacyCycleRunner(engine)

    state.role[0, 0] = ROLE_RETAILER
    state.stock[0, 0] = 0.0
    state.stock_limit[0, 0] = 100.0
    state.purchase_price[0, 0] = 8.0
    state.sales_price[0, 0] = 10.0
    state.purchased_this_period[0, 0] = 10.0
    state.sold_this_period[0, 0] = 4.0

    runner._adjust_purchase_price(0, 0)

    assert float(state.purchase_price[0, 0]) == pytest.approx(8.0 * config.price_reduction)


def test_retailer_sales_price_falls_when_no_sales_despite_inventory() -> None:
    config = SimulationConfig(
        population=1,
        goods=1,
        acquaintances=1,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
    )
    engine = SimulationEngine.create(config=config, backend_name="numpy")
    state = engine.state
    runner = LegacyCycleRunner(engine)

    state.role[0, 0] = ROLE_RETAILER
    state.stock[0, 0] = 2.0
    state.stock_limit[0, 0] = 100.0
    state.market.elastic_need[0] = 3.0
    state.purchase_price[0, 0] = 8.0
    state.sales_price[0, 0] = 10.0
    state.sales_times[0, 0] = 0
    state.sold_this_period[0, 0] = 0.0

    runner._adjust_sales_price(0, 0)

    assert float(state.sales_price[0, 0]) == pytest.approx(10.0 * config.price_reduction)
    assert float(state.purchase_price[0, 0]) == pytest.approx(8.0)


def test_retailer_sales_price_reduction_keeps_purchase_below_sales() -> None:
    config = SimulationConfig(
        population=1,
        goods=1,
        acquaintances=1,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
    )
    engine = SimulationEngine.create(config=config, backend_name="numpy")
    state = engine.state
    runner = LegacyCycleRunner(engine)

    state.role[0, 0] = ROLE_RETAILER
    state.stock[0, 0] = 2.0
    state.stock_limit[0, 0] = 100.0
    state.purchase_price[0, 0] = 9.7
    state.sales_price[0, 0] = 10.0
    state.sold_this_period[0, 0] = 0.0
    state.market.elastic_need[0] = 1.0

    runner._adjust_sales_price(0, 0)

    expected_sales = 10.0 * config.price_reduction
    assert float(state.sales_price[0, 0]) == pytest.approx(expected_sales)
    assert float(state.purchase_price[0, 0]) == pytest.approx(expected_sales * config.price_reduction)


def test_producer_normal_stock_low_sales_sets_sales_price_to_cost() -> None:
    config = SimulationConfig(
        population=1,
        goods=1,
        acquaintances=1,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
    )
    engine = SimulationEngine.create(config=config, backend_name="numpy")
    state = engine.state
    runner = LegacyCycleRunner(engine)

    state.role[0, 0] = ROLE_PRODUCER
    state.efficiency[0, 0] = 1.0
    state.stock[0, 0] = 2.0
    state.stock_limit[0, 0] = 100.0
    state.market.elastic_need[0] = 1.0
    state.sales_price[0, 0] = 10.0
    state.sold_this_period[0, 0] = 0.0

    runner._adjust_sales_price(0, 0)

    assert float(state.sales_price[0, 0]) == pytest.approx(1.0)


def test_producer_large_unsold_stock_discounts_after_reaching_cost() -> None:
    config = SimulationConfig(
        population=1,
        goods=1,
        acquaintances=1,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
    )
    engine = SimulationEngine.create(config=config, backend_name="numpy")
    state = engine.state
    runner = LegacyCycleRunner(engine)

    state.role[0, 0] = ROLE_PRODUCER
    state.efficiency[0, 0] = 1.0
    state.stock[0, 0] = 200.0
    state.stock_limit[0, 0] = 100.0
    state.market.elastic_need[0] = 1.0
    state.sales_price[0, 0] = 10.0
    state.sold_this_period[0, 0] = 0.0

    runner._adjust_sales_price(0, 0)
    assert float(state.sales_price[0, 0]) == pytest.approx(1.0)

    runner._adjust_sales_price(0, 0)
    assert float(state.sales_price[0, 0]) == pytest.approx(config.price_reduction)


def test_friend_slot_cache_tracks_replacement_without_semantic_drift() -> None:
    config = SimulationConfig(
        population=4,
        goods=2,
        acquaintances=2,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
    )
    engine = SimulationEngine.create(config=config, backend_name="numpy")
    state = engine.state

    state.friend_id[0] = np.array([1, 2], dtype=np.int32)
    state.friend_activity[0] = np.array([4.0, 1.0], dtype=np.float32)
    runner = LegacyCycleRunner(engine)

    assert runner._find_friend_slot(0, 1) == 0
    assert runner._find_friend_slot(0, 2) == 1

    inserted_slot = runner._ensure_friend_link(0, 3, initial_transactions=2.0)

    assert inserted_slot == 1
    assert runner._find_friend_slot(0, 2) == -1
    assert runner._find_friend_slot(0, 3) == 1
    assert int(state.friend_id[0, 1]) == 3


def test_exchange_search_backend_falls_back_to_python_without_native_module(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(legacy_search_backend, "_load_native_search_module", lambda: None)
    backend = build_exchange_search_backend()

    assert backend.name == "python"
    assert backend.is_native is False


def test_native_exchange_search_backend_wraps_tuple_result_contract() -> None:
    class FakeNativeModule:
        @staticmethod
        def find_best_exchange(**kwargs):
            return (1.5, 2, 7, 3)

        @staticmethod
        def plan_best_exchange(**kwargs):
            return (0, 1.5, 2, 7, 3, 1, 2.25, 1.1, 0.7, 0.9)

    backend = NativeModuleExchangeSearchBackend(FakeNativeModule())
    search_request = ExchangeSearchRequest(
        goods=4,
        need_good=1,
        initial_transparency=0.7,
        elastic_need=np.ones((4,), dtype=np.float32),
        candidate_offer_goods=np.array([0, 2, 3], dtype=np.int32),
        friend_ids=np.array([7], dtype=np.int32),
        reciprocal_slots=np.array([0], dtype=np.int32),
        my_stock=np.ones((4,), dtype=np.float32),
        my_sales_price=np.ones((4,), dtype=np.float32),
        my_purchase_price=np.ones((4,), dtype=np.float32),
        my_role=np.full((4,), ROLE_CONSUMER, dtype=np.int32),
        my_transparency=np.ones((1, 4), dtype=np.float32),
        my_needs_level=1.0,
        stock=np.ones((8, 4), dtype=np.float32),
        role=np.full((8, 4), ROLE_CONSUMER, dtype=np.int32),
        stock_limit=np.ones((8, 4), dtype=np.float32),
        purchase_price=np.ones((8, 4), dtype=np.float32),
        sales_price=np.ones((8, 4), dtype=np.float32),
        needs_level=np.ones((8,), dtype=np.float32),
        transparency=np.ones((8, 1, 4), dtype=np.float32),
    )
    result = backend.find_best_exchange(**search_request.as_kwargs())
    planning = backend.plan_best_exchange(
        ExchangePlanRequest(
            search_request=search_request,
            max_need=3.0,
            min_trade_quantity=0.5,
            trade_rounding_buffer=(1.0 / 3.0),
        )
    )

    assert result is not None
    assert result.score == 1.5
    assert result.friend_slot == 2
    assert result.friend_id == 7
    assert result.offer_good == 3
    assert planning is not None
    assert planning.failure_reason is None
    assert planning.plan_result is not None
    assert planning.plan_result.max_exchange == 2.25
    assert planning.plan_result.reciprocal_slot == 1


def test_exchange_search_backend_uses_top_level_native_module_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = ModuleType('_legacy_native_search')
    fake_module.find_best_exchange = lambda **kwargs: (2.0, 1, 5, 0)
    monkeypatch.delitem(sys.modules, 'emergent_money._legacy_native_search', raising=False)
    monkeypatch.setitem(sys.modules, '_legacy_native_search', fake_module)

    backend = build_exchange_search_backend()

    assert backend.name == 'native'
    assert backend.is_native is True


def test_find_best_exchange_uses_backend_seam_for_expected_barter_match() -> None:
    config = SimulationConfig(
        population=2,
        goods=2,
        acquaintances=1,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
    )
    engine = SimulationEngine.create(config=config, backend_name="numpy")
    state = engine.state

    state.friend_id[...] = np.array([[1], [0]], dtype=np.int32)
    state.friend_activity[...] = 1.0
    state.stock[...] = np.array([[10.0, 0.0], [0.0, 10.0]], dtype=np.float32)
    state.stock_limit[...] = np.array([[12.0, 12.0], [12.0, 12.0]], dtype=np.float32)
    state.previous_stock_limit[...] = state.stock_limit
    state.needs_level[...] = 1.0
    state.role[...] = ROLE_CONSUMER
    state.role[1, 0] = ROLE_RETAILER
    state.purchase_price[...] = np.array([[1.0, 2.0], [2.0, 1.0]], dtype=np.float32)
    state.sales_price[...] = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    state.transparency[...] = 1.0
    runner = LegacyCycleRunner(engine)

    result = runner._find_best_exchange(agent_id=0, need_good=1, forbidden_gifts=set())

    assert result is not None
    assert result.friend_slot == 0
    assert result.friend_id == 1
    assert result.offer_good == 0
    assert result.score > 0.0



def _build_disjoint_consumption_runner(
    *,
    experimental_stage: bool = False,
    experimental_surplus_stage: bool = False,
    experimental_batches: int = 1,
    experimental_frontier_size: int = 0,
    block_frontier_partners: bool = True,
    rolling_frontier: bool = False,
) -> tuple[SimulationEngine, LegacyCycleRunner]:
    config = SimulationConfig(
        population=4,
        goods=2,
        acquaintances=1,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
        experimental_hybrid_batches=experimental_batches,
        experimental_hybrid_frontier_size=experimental_frontier_size,
        experimental_hybrid_consumption_stage=experimental_stage,
        experimental_hybrid_surplus_stage=experimental_surplus_stage,
        experimental_hybrid_block_frontier_partners=block_frontier_partners,
        experimental_hybrid_rolling_frontier=rolling_frontier,
    )
    engine = SimulationEngine.create(config=config, backend_name="numpy")
    state = engine.state
    state.friend_id[...] = np.array([[1], [0], [3], [2]], dtype=np.int32)
    state.friend_activity[...] = 1.0
    state.stock[...] = np.array(
        [
            [10.0, 0.0],
            [0.0, 10.0],
            [10.0, 0.0],
            [0.0, 10.0],
        ],
        dtype=np.float32,
    )
    state.stock_limit[...] = 12.0
    state.previous_stock_limit[...] = state.stock_limit
    state.need[...] = np.array(
        [
            [0.0, 4.0],
            [1.0, 0.0],
            [0.0, 4.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )
    state.needs_level[...] = 1.0
    state.role[...] = ROLE_CONSUMER
    state.role[1, 0] = ROLE_RETAILER
    state.role[3, 0] = ROLE_RETAILER
    state.purchase_price[...] = np.array(
        [
            [1.0, 2.0],
            [2.0, 1.0],
            [1.0, 2.0],
            [2.0, 1.0],
        ],
        dtype=np.float32,
    )
    state.sales_price[...] = 1.0
    state.transparency[...] = 1.0
    return engine, LegacyCycleRunner(engine)



def _build_conflicting_consumption_runner(
    *,
    experimental_stage: bool = False,
    experimental_surplus_stage: bool = False,
    experimental_batches: int = 1,
    experimental_frontier_size: int = 0,
    block_frontier_partners: bool = True,
    rolling_frontier: bool = False,
) -> tuple[SimulationEngine, LegacyCycleRunner]:
    config = SimulationConfig(
        population=3,
        goods=2,
        acquaintances=2,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
        experimental_hybrid_batches=experimental_batches,
        experimental_hybrid_frontier_size=experimental_frontier_size,
        experimental_hybrid_consumption_stage=experimental_stage,
        experimental_hybrid_surplus_stage=experimental_surplus_stage,
        experimental_hybrid_block_frontier_partners=block_frontier_partners,
        experimental_hybrid_rolling_frontier=rolling_frontier,
    )
    engine = SimulationEngine.create(config=config, backend_name="numpy")
    state = engine.state
    state.friend_id[...] = np.array(
        [
            [2, -1],
            [2, -1],
            [0, 1],
        ],
        dtype=np.int32,
    )
    state.friend_activity[...] = 1.0
    state.stock[...] = np.array(
        [
            [10.0, 0.0],
            [10.0, 0.0],
            [0.0, 6.0],
        ],
        dtype=np.float32,
    )
    state.stock_limit[...] = 12.0
    state.previous_stock_limit[...] = state.stock_limit
    state.need[...] = np.array(
        [
            [0.0, 4.0],
            [0.0, 4.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )
    state.needs_level[...] = 1.0
    state.role[...] = ROLE_CONSUMER
    state.role[2, 0] = ROLE_RETAILER
    state.purchase_price[...] = np.array(
        [
            [1.0, 2.0],
            [1.0, 2.0],
            [2.0, 1.0],
        ],
        dtype=np.float32,
    )
    state.sales_price[...] = 1.0
    state.transparency[...] = 1.0
    return engine, LegacyCycleRunner(engine)


def _build_disjoint_surplus_runner(
    *,
    experimental_stage: bool = False,
    experimental_surplus_stage: bool = False,
    experimental_batches: int = 1,
    experimental_frontier_size: int = 0,
    block_frontier_partners: bool = True,
    rolling_frontier: bool = False,
) -> tuple[SimulationEngine, LegacyCycleRunner]:
    config = SimulationConfig(
        population=4,
        goods=2,
        acquaintances=1,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
        experimental_hybrid_batches=experimental_batches,
        experimental_hybrid_frontier_size=experimental_frontier_size,
        experimental_hybrid_consumption_stage=experimental_stage,
        experimental_hybrid_surplus_stage=experimental_surplus_stage,
        experimental_hybrid_block_frontier_partners=block_frontier_partners,
        experimental_hybrid_rolling_frontier=rolling_frontier,
    )
    engine = SimulationEngine.create(config=config, backend_name="numpy")
    state = engine.state
    state.friend_id[...] = np.array([[1], [0], [3], [2]], dtype=np.int32)
    state.friend_activity[...] = 1.0
    state.stock[...] = np.array(
        [
            [10.0, 0.0],
            [0.0, 10.0],
            [10.0, 0.0],
            [0.0, 10.0],
        ],
        dtype=np.float32,
    )
    state.stock_limit[...] = 12.0
    state.previous_stock_limit[...] = state.stock_limit
    state.need[...] = 0.0
    state.recent_sales[...] = 0.0
    state.recent_sales[0, 1] = 10.0
    state.recent_sales[2, 1] = 10.0
    state.recent_production[...] = np.array([1.0, 4.0], dtype=np.float32)
    state.recent_production[0, 1] = 0.0
    state.recent_production[2, 1] = 0.0
    state.needs_level[...] = 1.0
    state.role[...] = ROLE_CONSUMER
    state.role[1, 0] = ROLE_RETAILER
    state.role[3, 0] = ROLE_RETAILER
    state.purchase_price[...] = np.array(
        [
            [1.0, 2.0],
            [2.0, 1.0],
            [1.0, 2.0],
            [2.0, 1.0],
        ],
        dtype=np.float32,
    )
    state.sales_price[...] = 1.0
    state.transparency[...] = 1.0
    return engine, LegacyCycleRunner(engine)


def _build_conflicting_surplus_runner(
    *,
    experimental_stage: bool = False,
    experimental_surplus_stage: bool = False,
    experimental_batches: int = 1,
    experimental_frontier_size: int = 0,
    block_frontier_partners: bool = True,
    rolling_frontier: bool = False,
) -> tuple[SimulationEngine, LegacyCycleRunner]:
    config = SimulationConfig(
        population=3,
        goods=2,
        acquaintances=2,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
        experimental_hybrid_batches=experimental_batches,
        experimental_hybrid_frontier_size=experimental_frontier_size,
        experimental_hybrid_consumption_stage=experimental_stage,
        experimental_hybrid_surplus_stage=experimental_surplus_stage,
        experimental_hybrid_block_frontier_partners=block_frontier_partners,
        experimental_hybrid_rolling_frontier=rolling_frontier,
    )
    engine = SimulationEngine.create(config=config, backend_name="numpy")
    state = engine.state
    state.friend_id[...] = np.array(
        [
            [2, -1],
            [2, -1],
            [0, 1],
        ],
        dtype=np.int32,
    )
    state.friend_activity[...] = 1.0
    state.stock[...] = np.array(
        [
            [10.0, 0.0],
            [10.0, 0.0],
            [0.0, 6.0],
        ],
        dtype=np.float32,
    )
    state.stock_limit[...] = 12.0
    state.previous_stock_limit[...] = state.stock_limit
    state.need[...] = 0.0
    state.recent_sales[...] = 0.0
    state.recent_sales[0, 1] = 10.0
    state.recent_sales[1, 1] = 10.0
    state.recent_production[...] = np.array([1.0, 4.0], dtype=np.float32)
    state.recent_production[0, 1] = 0.0
    state.recent_production[1, 1] = 0.0
    state.needs_level[...] = 1.0
    state.role[...] = ROLE_CONSUMER
    state.role[2, 0] = ROLE_RETAILER
    state.purchase_price[...] = np.array(
        [
            [1.0, 2.0],
            [1.0, 2.0],
            [2.0, 1.0],
        ],
        dtype=np.float32,
    )
    state.sales_price[...] = 1.0
    state.transparency[...] = 1.0
    return engine, LegacyCycleRunner(engine)


def _build_surplus_cycle_runner(
    *,
    experimental_stage: bool = False,
    experimental_surplus_stage: bool = False,
    experimental_batches: int = 1,
    experimental_frontier_size: int = 0,
) -> tuple[SimulationEngine, LegacyCycleRunner]:
    config = SimulationConfig(
        population=2,
        goods=2,
        acquaintances=1,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
        experimental_hybrid_batches=experimental_batches,
        experimental_hybrid_frontier_size=experimental_frontier_size,
        experimental_hybrid_consumption_stage=experimental_stage,
        experimental_hybrid_surplus_stage=experimental_surplus_stage,
    )
    engine = SimulationEngine.create(config=config, backend_name="numpy")
    state = engine.state
    state.friend_id[...] = np.array([[1], [0]], dtype=np.int32)
    state.friend_activity[...] = 1.0
    state.stock[...] = np.array([[10.0, 4.0], [0.0, 10.0]], dtype=np.float32)
    state.stock_limit[...] = 12.0
    state.previous_stock_limit[...] = state.stock_limit
    state.need[...] = 0.0
    state.recent_sales[...] = 0.0
    state.recent_sales[0, 1] = 10.0
    state.recent_production[...] = np.array([1.0, 4.0], dtype=np.float32)
    state.recent_production[0, 1] = 0.0
    state.needs_level[...] = 1.0
    state.role[...] = ROLE_CONSUMER
    state.role[1, 0] = ROLE_RETAILER
    state.purchase_price[...] = np.array([[1.0, 2.0], [2.0, 1.0]], dtype=np.float32)
    state.sales_price[...] = 1.0
    state.transparency[...] = 1.0
    return engine, LegacyCycleRunner(engine)


def test_experimental_consumption_batches_match_sequential_trade_count_on_disjoint_fixture() -> None:
    _, runner = _build_disjoint_consumption_runner()

    plan = runner.plan_experimental_consumption_batches(batch_count=1)

    assert plan.scheduled_count == 2
    assert len(plan.dropped) == 2
    assert len(plan.batches) == 1
    assert batch_is_conflict_free(plan.batches[0])

    sequential_engine, sequential_runner = _build_disjoint_consumption_runner()
    for agent_id in range(sequential_engine.config.population):
        sequential_runner._satisfy_needs_by_exchange(agent_id)

    assert sequential_engine._accepted_trade_count == plan.scheduled_count



def test_experimental_consumption_batches_drop_conflicts_and_stay_in_sequential_range() -> None:
    _, runner = _build_conflicting_consumption_runner()

    plan = runner.plan_experimental_consumption_batches(batch_count=1)

    assert plan.scheduled_count == 1
    assert len(plan.dropped) == 1
    assert len(plan.batches) == 1
    assert batch_is_conflict_free(plan.batches[0])

    sequential_engine, sequential_runner = _build_conflicting_consumption_runner()
    for agent_id in range(sequential_engine.config.population):
        sequential_runner._satisfy_needs_by_exchange(agent_id)

    assert sequential_engine._accepted_trade_count == plan.scheduled_count



def test_execute_experimental_consumption_batches_matches_sequential_trade_count_on_disjoint_fixture() -> None:
    engine, runner = _build_disjoint_consumption_runner()

    plan = runner.execute_experimental_consumption_batches(batch_count=1)

    assert engine._accepted_trade_count == plan.scheduled_count == 2

    sequential_engine, sequential_runner = _build_disjoint_consumption_runner()
    for agent_id in range(sequential_engine.config.population):
        sequential_runner._satisfy_needs_by_exchange(agent_id)

    assert sequential_engine._accepted_trade_count == engine._accepted_trade_count


def test_execute_experimental_consumption_batches_matches_sequential_trade_count_on_conflicting_fixture() -> None:
    engine, runner = _build_conflicting_consumption_runner()

    plan = runner.execute_experimental_consumption_batches(batch_count=1)

    assert engine._accepted_trade_count == plan.scheduled_count == 1

    sequential_engine, sequential_runner = _build_conflicting_consumption_runner()
    for agent_id in range(sequential_engine.config.population):
        sequential_runner._satisfy_needs_by_exchange(agent_id)

    assert sequential_engine._accepted_trade_count == engine._accepted_trade_count



def test_experimental_surplus_batches_match_sequential_trade_count_on_disjoint_fixture() -> None:
    _, runner = _build_disjoint_surplus_runner()

    plan = runner.plan_experimental_surplus_batches(batch_count=1)

    assert plan.scheduled_count == 2
    assert len(plan.dropped) == 0
    assert len(plan.batches) == 1
    assert batch_is_conflict_free(plan.batches[0])

    sequential_engine, sequential_runner = _build_disjoint_surplus_runner()
    for agent_id in range(sequential_engine.config.population):
        sequential_runner._make_surplus_deals(agent_id)

    assert sequential_engine._accepted_trade_count == plan.scheduled_count


def test_experimental_surplus_batches_drop_conflicts_and_stay_in_sequential_range() -> None:
    _, runner = _build_conflicting_surplus_runner()

    plan = runner.plan_experimental_surplus_batches(batch_count=1)

    assert plan.scheduled_count == 1
    assert len(plan.dropped) == 1
    assert len(plan.batches) == 1
    assert batch_is_conflict_free(plan.batches[0])

    sequential_engine, sequential_runner = _build_conflicting_surplus_runner()
    for agent_id in range(sequential_engine.config.population):
        sequential_runner._make_surplus_deals(agent_id)

    assert sequential_engine._accepted_trade_count == plan.scheduled_count


def test_execute_experimental_surplus_batches_matches_sequential_trade_count_on_disjoint_fixture() -> None:
    engine, runner = _build_disjoint_surplus_runner()

    plan = runner.execute_experimental_surplus_batches(batch_count=1)

    assert engine._accepted_trade_count == plan.scheduled_count == 2

    sequential_engine, sequential_runner = _build_disjoint_surplus_runner()
    for agent_id in range(sequential_engine.config.population):
        sequential_runner._make_surplus_deals(agent_id)

    assert sequential_engine._accepted_trade_count == engine._accepted_trade_count


def test_execute_experimental_surplus_batches_matches_sequential_trade_count_on_conflicting_fixture() -> None:
    engine, runner = _build_conflicting_surplus_runner()

    plan = runner.execute_experimental_surplus_batches(batch_count=1)

    assert engine._accepted_trade_count == plan.scheduled_count == 1

    sequential_engine, sequential_runner = _build_conflicting_surplus_runner()
    for agent_id in range(sequential_engine.config.population):
        sequential_runner._make_surplus_deals(agent_id)

    assert sequential_engine._accepted_trade_count == engine._accepted_trade_count


def test_exact_cycle_ignores_hybrid_batches_when_opt_in_flag_is_off() -> None:
    baseline_engine, _ = _build_disjoint_consumption_runner(
        experimental_stage=False,
        experimental_batches=0,
    )
    staged_engine, _ = _build_disjoint_consumption_runner(
        experimental_stage=False,
        experimental_batches=1,
    )

    baseline = baseline_engine.step()
    staged = staged_engine.step()

    assert staged.accepted_trade_count == baseline.accepted_trade_count
    assert np.isclose(staged.accepted_trade_volume, baseline.accepted_trade_volume)
    assert np.isclose(staged.production_total, baseline.production_total)


def test_exact_cycle_opt_in_hybrid_consumption_matches_sequential_when_frontier_is_one() -> None:
    sequential_engine, _ = _build_disjoint_consumption_runner(
        experimental_stage=False,
        experimental_batches=0,
        experimental_frontier_size=0,
    )
    hybrid_engine, _ = _build_disjoint_consumption_runner(
        experimental_stage=True,
        experimental_batches=1,
        experimental_frontier_size=1,
    )

    sequential = sequential_engine.step()
    hybrid = hybrid_engine.step()

    assert hybrid.accepted_trade_count == sequential.accepted_trade_count
    assert np.isclose(hybrid.accepted_trade_volume, sequential.accepted_trade_volume)
    assert np.isclose(hybrid.production_total, sequential.production_total)
    assert np.isclose(hybrid.utility_proxy_total, sequential.utility_proxy_total)



def test_exact_cycle_rolling_hybrid_consumption_matches_sequential_when_frontier_is_one() -> None:
    sequential_engine, _ = _build_disjoint_consumption_runner(
        experimental_stage=False,
        experimental_batches=0,
        experimental_frontier_size=0,
    )
    rolling_engine, _ = _build_disjoint_consumption_runner(
        experimental_stage=True,
        experimental_batches=1,
        experimental_frontier_size=1,
        rolling_frontier=True,
    )

    sequential = sequential_engine.step()
    rolling = rolling_engine.step()

    assert rolling.accepted_trade_count == sequential.accepted_trade_count
    assert np.isclose(rolling.accepted_trade_volume, sequential.accepted_trade_volume)
    assert np.isclose(rolling.production_total, sequential.production_total)
    assert np.isclose(rolling.utility_proxy_total, sequential.utility_proxy_total)


def test_exact_cycle_hybrid_consumption_records_wave_diagnostics() -> None:
    sequential_engine, _ = _build_disjoint_consumption_runner(
        experimental_stage=False,
        experimental_batches=0,
        experimental_frontier_size=0,
    )
    hybrid_engine, _ = _build_disjoint_consumption_runner(
        experimental_stage=True,
        experimental_batches=1,
        experimental_frontier_size=1,
    )

    sequential_engine.step()
    hybrid = hybrid_engine.step()

    assert sequential_engine.exact_cycle_diagnostics is None
    diagnostics = hybrid_engine.exact_cycle_diagnostics
    assert diagnostics is not None
    assert diagnostics['mode'] == 'experimental_hybrid_exchange'
    assert diagnostics['frontier_size'] == 1
    assert diagnostics['consumption_stage'] is True
    assert diagnostics['surplus_stage'] is False
    assert diagnostics['block_frontier_partners'] is True
    assert diagnostics['preserve_proposer_order'] is False
    assert diagnostics['rolling_frontier'] is False
    assert diagnostics['wave_count'] > 0
    assert diagnostics['executed_exchanges_total'] == hybrid.accepted_trade_count
    assert diagnostics['scheduled_exchanges_total'] >= diagnostics['executed_exchanges_total']
    assert diagnostics['scheduled_quantity_total'] >= diagnostics['executed_quantity_total']
    assert diagnostics['scheduler_conflict_exchanges_total'] >= 0
    assert diagnostics['execution_failures_total'] >= 0
    assert diagnostics['retry_exhausted_agents_total'] >= 0
    assert isinstance(diagnostics['no_candidate_reasons_total'], dict)
    assert isinstance(diagnostics['execution_failure_reasons_total'], dict)
    assert len(diagnostics['waves']) == diagnostics['wave_count']



def test_exact_cycle_opt_in_hybrid_surplus_matches_sequential_when_frontier_is_one() -> None:
    sequential_engine, _ = _build_surplus_cycle_runner(
        experimental_stage=False,
        experimental_surplus_stage=False,
        experimental_batches=0,
        experimental_frontier_size=0,
    )
    hybrid_engine, _ = _build_surplus_cycle_runner(
        experimental_stage=True,
        experimental_surplus_stage=True,
        experimental_batches=1,
        experimental_frontier_size=1,
    )

    sequential = sequential_engine.step()
    hybrid = hybrid_engine.step()

    assert hybrid.accepted_trade_count == sequential.accepted_trade_count
    assert np.isclose(hybrid.accepted_trade_volume, sequential.accepted_trade_volume)
    assert np.isclose(hybrid.production_total, sequential.production_total)
    assert np.isclose(hybrid.utility_proxy_total, sequential.utility_proxy_total)


def test_exact_cycle_opt_in_surplus_only_hybrid_matches_sequential_when_frontier_is_one() -> None:
    sequential_engine, _ = _build_surplus_cycle_runner(
        experimental_stage=False,
        experimental_surplus_stage=False,
        experimental_batches=0,
        experimental_frontier_size=0,
    )
    hybrid_engine, _ = _build_surplus_cycle_runner(
        experimental_stage=False,
        experimental_surplus_stage=True,
        experimental_batches=1,
        experimental_frontier_size=1,
    )

    sequential = sequential_engine.step()
    hybrid = hybrid_engine.step()

    assert hybrid.accepted_trade_count == sequential.accepted_trade_count
    assert np.isclose(hybrid.accepted_trade_volume, sequential.accepted_trade_volume)
    assert np.isclose(hybrid.production_total, sequential.production_total)
    assert np.isclose(hybrid.utility_proxy_total, sequential.utility_proxy_total)


def test_exact_cycle_hybrid_surplus_records_stage_flag_when_enabled() -> None:
    hybrid_engine, _ = _build_surplus_cycle_runner(
        experimental_stage=True,
        experimental_surplus_stage=True,
        experimental_batches=1,
        experimental_frontier_size=1,
    )

    hybrid_engine.step()

    diagnostics = hybrid_engine.exact_cycle_diagnostics
    assert diagnostics is not None
    assert diagnostics['surplus_stage'] is True
    assert diagnostics['stage_wave_counts'].get('surplus', 0) >= 0


def test_experimental_consumption_batches_can_allow_same_frontier_partners() -> None:
    config = SimulationConfig(
        population=2,
        goods=2,
        acquaintances=1,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
        experimental_hybrid_batches=1,
        experimental_hybrid_frontier_size=2,
        experimental_hybrid_consumption_stage=True,
        experimental_hybrid_block_frontier_partners=False,
    )
    engine = SimulationEngine.create(config=config, backend_name="numpy")
    state = engine.state
    state.friend_id[...] = np.array([[1], [0]], dtype=np.int32)
    state.friend_activity[...] = 1.0
    state.stock[...] = np.array([[10.0, 0.0], [0.0, 10.0]], dtype=np.float32)
    state.stock_limit[...] = 12.0
    state.previous_stock_limit[...] = state.stock_limit
    state.need[...] = np.array([[0.0, 4.0], [4.0, 0.0]], dtype=np.float32)
    state.needs_level[...] = 1.0
    state.role[...] = ROLE_CONSUMER
    state.role[1, 0] = ROLE_RETAILER
    state.purchase_price[...] = np.array([[1.0, 2.0], [2.0, 1.0]], dtype=np.float32)
    state.sales_price[...] = 1.0
    state.transparency[...] = 1.0
    runner = LegacyCycleRunner(engine)

    blocked_plan = runner.plan_experimental_consumption_batches(
        batch_count=1,
        proposer_ids=(0, 1),
        blocked_partner_ids={0, 1},
        one_candidate_per_agent=True,
    )
    allowed_plan = runner.plan_experimental_consumption_batches(
        batch_count=1,
        proposer_ids=(0, 1),
        blocked_partner_ids=set(),
        one_candidate_per_agent=True,
    )

    assert blocked_plan.scheduled_count == 0
    assert allowed_plan.scheduled_count == 1
