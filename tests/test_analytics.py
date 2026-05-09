import numpy as np

from emergent_money.analytics import compute_good_snapshots
from emergent_money.config import SimulationConfig
from emergent_money.engine import SimulationEngine
from emergent_money.state import ROLE_CONSUMER, ROLE_RETAILER


def test_monetary_score_requires_retailer_intermediation_role() -> None:
    config = SimulationConfig(
        population=4,
        goods=2,
        acquaintances=1,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
    )
    engine = SimulationEngine.create(config=config, backend_name="numpy")
    state = engine.state

    state.role[...] = ROLE_CONSUMER
    state.purchase_price[...] = 1.0
    state.sales_price[...] = 1.0
    state.recent_purchases[...] = 0.0
    state.recent_sales[...] = 0.0
    state.recent_inventory_inflow[...] = 0.0
    state.recent_purchase_value[...] = 0.0
    state.recent_sales_value[...] = 0.0
    state.recent_inventory_inflow_value[...] = 0.0

    # Large ordinary stock inflow is not money-like unless the agent is using
    # the good as an intermediary/retailer good.
    state.recent_purchases[:, 0] = np.array([100.0, 90.0, 80.0, 70.0], dtype=np.float32)
    state.recent_sales[:, 0] = np.array([90.0, 80.0, 70.0, 60.0], dtype=np.float32)
    state.recent_inventory_inflow[:, 0] = state.recent_purchases[:, 0]

    state.role[:2, 1] = ROLE_RETAILER
    state.recent_purchases[:2, 1] = np.array([10.0, 8.0], dtype=np.float32)
    state.recent_sales[:2, 1] = np.array([9.0, 8.0], dtype=np.float32)
    state.recent_inventory_inflow[:2, 1] = state.recent_purchases[:2, 1]

    goods = {item.good_id: item for item in compute_good_snapshots(state=state, backend=engine.backend, limit=None)}

    assert goods[0].monetary_score == 0.0
    assert goods[0].value_weighted_monetary_score == 0.0
    assert goods[1].monetary_score > 0.0
    assert goods[1].value_weighted_monetary_score > 0.0


def test_exchange_media_score_requires_relative_tce_and_local_circulation() -> None:
    config = SimulationConfig(
        population=4,
        goods=2,
        acquaintances=2,
        active_acquaintances=2,
        demand_candidates=1,
        supply_candidates=1,
    )
    engine = SimulationEngine.create(config=config, backend_name="numpy")
    state = engine.state

    state.role[...] = ROLE_RETAILER
    state.purchase_price[...] = 1.0
    state.sales_price[...] = 1.0
    state.stock[...] = 0.0
    state.recent_purchases[...] = 0.0
    state.recent_sales[...] = 0.0
    state.recent_inventory_inflow[...] = 0.0
    state.recent_purchase_value[...] = 0.0
    state.recent_sales_value[...] = 0.0
    state.recent_inventory_inflow_value[...] = 0.0
    state.friend_purchased[...] = 0.0
    state.friend_sold[...] = 0.0
    state.friend_id[...] = 0

    # Good 0 circulates broadly and absorbs TCE relative to its own need scale.
    state.market.elastic_need[...] = np.array([1.0, 100.0], dtype=np.float32)
    state.market.periodic_tce_cost[...] = np.array([20.0, 0.0], dtype=np.float32)
    state.stock[:, 0] = 100.0
    state.recent_purchases[:, 0] = 10.0
    state.recent_sales[:, 0] = 8.0
    state.recent_inventory_inflow[:, 0] = state.recent_purchases[:, 0]
    state.friend_purchased[:, 0, 0] = 1.0
    state.friend_sold[:, 1, 0] = 1.0

    # Good 1 can still look like ordinary retailer flow, but it has no TCE
    # loss relative to consumption need and no broad local circulation signal.
    state.stock[:, 1] = 1000.0
    state.recent_purchases[:, 1] = 1000.0
    state.recent_sales[:, 1] = 800.0
    state.recent_inventory_inflow[:, 1] = state.recent_purchases[:, 1]

    goods = {item.good_id: item for item in compute_good_snapshots(state=state, backend=engine.backend, limit=None)}

    assert goods[0].exchange_media_score > 0.0
    assert goods[0].relative_tce_loss > goods[1].relative_tce_loss
    assert goods[0].network_circulation_breadth > goods[1].network_circulation_breadth
    assert goods[0].excess_stock_breadth > 0.0
    assert goods[0].round_trip_breadth > 0.0
    assert goods[0].consumer_flow_share == 0.0
    assert goods[1].monetary_score > 0.0
    assert goods[1].exchange_media_score == 0.0


def test_exchange_media_score_does_not_require_retailer_role() -> None:
    config = SimulationConfig(
        population=4,
        goods=1,
        acquaintances=2,
        active_acquaintances=2,
        demand_candidates=1,
        supply_candidates=1,
    )
    engine = SimulationEngine.create(config=config, backend_name="numpy")
    state = engine.state

    state.role[...] = ROLE_CONSUMER
    state.purchase_price[...] = 1.0
    state.sales_price[...] = 1.0
    state.stock[...] = 100.0
    state.recent_purchases[...] = 10.0
    state.recent_sales[...] = 8.0
    state.recent_inventory_inflow[...] = state.recent_purchases
    state.recent_purchase_value[...] = 0.0
    state.recent_sales_value[...] = 0.0
    state.recent_inventory_inflow_value[...] = 0.0
    state.friend_purchased[...] = 0.0
    state.friend_sold[...] = 0.0
    state.friend_id[...] = 0
    state.friend_purchased[:, 0, 0] = 1.0
    state.friend_sold[:, 1, 0] = 1.0
    state.market.elastic_need[...] = 1.0
    state.market.periodic_tce_cost[...] = 20.0

    goods = compute_good_snapshots(state=state, backend=engine.backend, limit=None)

    assert goods[0].monetary_score == 0.0
    assert goods[0].value_weighted_monetary_score == 0.0
    assert goods[0].exchange_media_score > 0.0
    assert goods[0].consumer_flow_share == 1.0
    assert goods[0].round_trip_turnover_share > 0.0


def test_local_liquidity_reserve_diagnostics_require_local_friend_evidence() -> None:
    config = SimulationConfig(
        population=4,
        goods=2,
        acquaintances=2,
        active_acquaintances=2,
        demand_candidates=1,
        supply_candidates=1,
        experimental_exchange_media_reserve_min_acceptance=1.0,
    )
    engine = SimulationEngine.create(config=config, backend_name="numpy")
    state = engine.state

    state.friend_id[...] = 0
    state.friend_sold[...] = 0.0
    state.transparency[...] = 1.0
    state.recent_purchases[...] = 0.0
    state.recent_sales[...] = 0.0
    state.recent_inventory_inflow[...] = 0.0
    state.recent_production[...] = 0.0
    state.purchase_price[...] = 1.0
    state.sales_price[...] = 1.0

    # Global-looking own flow alone is not enough: the reserve diagnostic must
    # be activated by directly observed friend acceptance.
    state.recent_purchases[:, 0] = 100.0
    state.recent_sales[:, 0] = 90.0
    goods_without_local_evidence = {
        item.good_id: item
        for item in compute_good_snapshots(state=state, backend=engine.backend, config=config, limit=None)
    }
    assert goods_without_local_evidence[0].local_liquidity_score == 0.0
    assert goods_without_local_evidence[0].exchange_media_reserve_score == 0.0

    state.friend_sold[:, 0, 0] = 10.0
    goods_with_local_evidence = {
        item.good_id: item
        for item in compute_good_snapshots(state=state, backend=engine.backend, config=config, limit=None)
    }

    assert goods_with_local_evidence[0].local_liquidity_score > 0.0
    assert goods_with_local_evidence[0].local_liquidity_acceptance_breadth > 0.0
    assert goods_with_local_evidence[0].local_liquidity_visible_acceptance > 0.0
    assert goods_with_local_evidence[0].exchange_media_reserve_score > 0.0
    assert goods_with_local_evidence[0].exchange_media_reserve_gap > 0.0


def test_exchange_media_reserve_diagnostic_obeys_spread_gate() -> None:
    config = SimulationConfig(
        population=4,
        goods=1,
        acquaintances=2,
        active_acquaintances=2,
        demand_candidates=1,
        supply_candidates=1,
        experimental_exchange_media_reserve_min_acceptance=1.0,
    )
    engine = SimulationEngine.create(config=config, backend_name="numpy")
    state = engine.state

    state.friend_id[...] = 0
    state.friend_sold[:, 0, 0] = 10.0
    state.transparency[...] = 1.0
    state.recent_purchases[:, 0] = 10.0
    state.recent_sales[:, 0] = 8.0
    state.recent_inventory_inflow[:, 0] = 10.0
    state.recent_production[:, 0] = 0.0
    state.purchase_price[:, 0] = 2.0
    state.sales_price[:, 0] = 1.0

    goods = compute_good_snapshots(state=state, backend=engine.backend, config=config, limit=None)

    assert goods[0].local_liquidity_score > 0.0
    assert goods[0].exchange_media_reserve_score == 0.0
    assert goods[0].exchange_media_spread_ok_share == 0.0
