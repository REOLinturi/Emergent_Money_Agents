from __future__ import annotations

from dataclasses import asdict
from math import ceil
from typing import Sequence

import numpy as np

from .backend.base import BaseBackend
from .config import SimulationConfig
from .dto import GoodRoleSnapshot, GoodSnapshot, InequalitySnapshot, MarketSnapshot, PhenomenaSnapshot, RecentTrendWindowSnapshot
from .metrics import MetricsSnapshot
from .state import ROLE_CONSUMER, ROLE_PRODUCER, ROLE_RETAILER, SimulationState

_EPSILON = 1e-6


def _good_metric_arrays(state: SimulationState, backend: BaseBackend) -> dict[str, np.ndarray]:
    xp = backend.xp
    base_need = backend.to_numpy(state.base_need[0]).astype(np.float64, copy=False)
    report_good_id = _report_good_ids_from_base_need(base_need)
    stock_total = backend.to_numpy(xp.sum(state.stock, axis=0)).astype(np.float64, copy=False)
    average_efficiency = backend.to_numpy(xp.mean(state.efficiency, axis=0)).astype(np.float64, copy=False)
    elastic_need = backend.to_numpy(state.market.elastic_need).astype(np.float64, copy=False)
    population = int(state.base_need.shape[0])
    consumption_need_total = np.maximum(elastic_need * float(population), _EPSILON)
    periodic_tce_cost = backend.to_numpy(state.market.periodic_tce_cost).astype(np.float64, copy=False)
    purchase_price = backend.to_numpy(state.purchase_price).astype(np.float64, copy=False)
    sales_price = backend.to_numpy(state.sales_price).astype(np.float64, copy=False)
    stock = backend.to_numpy(state.stock).astype(np.float64, copy=False)
    recent_purchases = backend.to_numpy(state.recent_purchases).astype(np.float64, copy=False)
    recent_sales = backend.to_numpy(state.recent_sales).astype(np.float64, copy=False)
    recent_inventory_inflow = backend.to_numpy(state.recent_inventory_inflow).astype(np.float64, copy=False)
    role = backend.to_numpy(state.role).astype(np.int32, copy=False)
    needs_level = backend.to_numpy(state.needs_level).astype(np.float64, copy=False)
    consumer_mask = role == ROLE_CONSUMER
    retailer_mask = role == ROLE_RETAILER
    observed_purchase_value = backend.to_numpy(state.recent_purchase_value).astype(np.float64, copy=False)
    observed_sales_value = backend.to_numpy(state.recent_sales_value).astype(np.float64, copy=False)
    observed_inventory_inflow_value = backend.to_numpy(state.recent_inventory_inflow_value).astype(np.float64, copy=False)
    recent_purchase_total = np.sum(recent_purchases, axis=0, dtype=np.float64)
    recent_sales_total = np.sum(recent_sales, axis=0, dtype=np.float64)
    recent_inventory_inflow_total = np.sum(recent_inventory_inflow, axis=0, dtype=np.float64)
    gross_purchase_breadth = np.mean(recent_purchases > _EPSILON, axis=0, dtype=np.float64)
    gross_sales_breadth = np.mean(recent_sales > _EPSILON, axis=0, dtype=np.float64)
    exchange_turnover_balance = np.divide(
        np.minimum(recent_purchase_total, recent_sales_total),
        np.maximum(np.maximum(recent_purchase_total, recent_sales_total), _EPSILON),
        out=np.zeros_like(recent_purchase_total),
        where=(recent_purchase_total > _EPSILON) | (recent_sales_total > _EPSILON),
    )
    round_trip_breadth = np.mean((recent_purchases > _EPSILON) & (recent_sales > _EPSILON), axis=0, dtype=np.float64)
    round_trip_turnover_share = np.divide(
        np.sum(np.minimum(recent_purchases, recent_sales), axis=0, dtype=np.float64),
        np.maximum(np.sum(np.maximum(recent_purchases, recent_sales), axis=0, dtype=np.float64), _EPSILON),
        out=np.zeros_like(recent_purchase_total),
        where=(recent_purchase_total > _EPSILON) | (recent_sales_total > _EPSILON),
    )
    gross_flow_total = recent_purchase_total + recent_sales_total
    consumer_flow_total = np.sum(np.where(consumer_mask, recent_purchases + recent_sales, 0.0), axis=0, dtype=np.float64)
    consumer_flow_share = np.divide(
        consumer_flow_total,
        np.maximum(gross_flow_total, _EPSILON),
        out=np.zeros_like(gross_flow_total),
        where=gross_flow_total > _EPSILON,
    )
    own_need_scale = needs_level[:, None] * elastic_need[None, :]
    excess_stock = np.maximum(stock - own_need_scale, 0.0)
    excess_stock_total = np.sum(excess_stock, axis=0, dtype=np.float64)
    excess_stock_breadth = np.mean(excess_stock > _EPSILON, axis=0, dtype=np.float64)
    excess_stock_ratio = excess_stock_total / consumption_need_total
    retailer_stock_total = np.sum(np.where(retailer_mask, stock, 0.0), axis=0, dtype=np.float64)
    retailer_stock_share = np.divide(
        retailer_stock_total,
        np.maximum(stock_total, _EPSILON),
        out=np.zeros_like(stock_total),
        where=stock_total > _EPSILON,
    )
    if float(np.sum(observed_purchase_value, dtype=np.float64)) > _EPSILON:
        purchase_value = observed_purchase_value
        sales_value = observed_sales_value
        inventory_inflow_value = observed_inventory_inflow_value
    else:
        # Backward-compatible fallback for checkpoints created before
        # transaction-time observed value accounting existed.
        purchase_value = recent_purchases * purchase_price
        sales_value = recent_sales * sales_price
        inventory_inflow_value = recent_inventory_inflow * purchase_price
    recent_purchase_value_total = np.sum(purchase_value, axis=0, dtype=np.float64)
    recent_sales_value_total = np.sum(sales_value, axis=0, dtype=np.float64)
    recent_inventory_inflow_value_total = np.sum(inventory_inflow_value, axis=0, dtype=np.float64)

    # Monetary-role diagnostics should identify goods accumulated for
    # intermediation, not every good that happens to enter someone's stock.
    merchant_purchases = np.where(retailer_mask, recent_purchases, 0.0)
    merchant_sales = np.where(retailer_mask, recent_sales, 0.0)
    merchant_inventory_inflow = np.where(retailer_mask, recent_inventory_inflow, 0.0)
    merchant_purchase_value = np.where(retailer_mask, purchase_value, 0.0)
    merchant_sales_value = np.where(retailer_mask, sales_value, 0.0)
    merchant_inventory_inflow_value = np.where(retailer_mask, inventory_inflow_value, 0.0)

    merchant_purchase_total = np.sum(merchant_purchases, axis=0, dtype=np.float64)
    merchant_sales_total = np.sum(merchant_sales, axis=0, dtype=np.float64)
    merchant_inventory_inflow_total = np.sum(merchant_inventory_inflow, axis=0, dtype=np.float64)
    merchant_purchase_value_total = np.sum(merchant_purchase_value, axis=0, dtype=np.float64)
    merchant_sales_value_total = np.sum(merchant_sales_value, axis=0, dtype=np.float64)
    merchant_inventory_inflow_value_total = np.sum(merchant_inventory_inflow_value, axis=0, dtype=np.float64)
    purchase_breadth = np.mean(merchant_purchases > _EPSILON, axis=0, dtype=np.float64)
    value_purchase_breadth = np.mean(merchant_purchase_value > _EPSILON, axis=0, dtype=np.float64)

    inventory_acceptance_share = np.divide(
        merchant_inventory_inflow_total,
        np.maximum(merchant_purchase_total, _EPSILON),
        out=np.zeros_like(merchant_inventory_inflow_total),
        where=merchant_purchase_total > _EPSILON,
    )
    turnover_balance = np.divide(
        np.minimum(merchant_purchase_total, merchant_sales_total),
        np.maximum(np.maximum(merchant_purchase_total, merchant_sales_total), _EPSILON),
        out=np.zeros_like(merchant_purchase_total),
        where=(merchant_purchase_total > _EPSILON) | (merchant_sales_total > _EPSILON),
    )

    nonzero_purchase = merchant_purchase_total[merchant_purchase_total > _EPSILON]
    mean_nonzero_purchase = float(nonzero_purchase.mean()) if nonzero_purchase.size else 0.0
    if mean_nonzero_purchase > _EPSILON:
        trade_salience = np.clip(merchant_purchase_total / mean_nonzero_purchase, 0.0, 1.0)
    else:
        trade_salience = np.zeros_like(merchant_purchase_total)

    monetary_score = inventory_acceptance_share * purchase_breadth * trade_salience * turnover_balance

    value_inventory_acceptance_share = np.divide(
        merchant_inventory_inflow_value_total,
        np.maximum(merchant_purchase_value_total, _EPSILON),
        out=np.zeros_like(merchant_inventory_inflow_value_total),
        where=merchant_purchase_value_total > _EPSILON,
    )
    value_turnover_balance = np.divide(
        np.minimum(merchant_purchase_value_total, merchant_sales_value_total),
        np.maximum(np.maximum(merchant_purchase_value_total, merchant_sales_value_total), _EPSILON),
        out=np.zeros_like(merchant_purchase_value_total),
        where=(merchant_purchase_value_total > _EPSILON) | (merchant_sales_value_total > _EPSILON),
    )

    nonzero_purchase_value = merchant_purchase_value_total[merchant_purchase_value_total > _EPSILON]
    mean_nonzero_purchase_value = float(nonzero_purchase_value.mean()) if nonzero_purchase_value.size else 0.0
    if mean_nonzero_purchase_value > _EPSILON:
        value_trade_salience = np.clip(merchant_purchase_value_total / mean_nonzero_purchase_value, 0.0, 1.0)
    else:
        value_trade_salience = np.zeros_like(merchant_purchase_value_total)

    value_weighted_monetary_score = (
        value_inventory_acceptance_share
        * value_purchase_breadth
        * value_trade_salience
        * value_turnover_balance
    )

    # Lightweight exchange-media approximation. It intentionally avoids using
    # the proposal "need/offer" direction as an economic direction: in barter,
    # both sides are exchanging goods. Money-like goods are diagnosed ex post by
    # wide local circulation, stock held beyond own consumption scale, and large
    # transaction-cost loss relative to the good's own consumption need.
    friend_purchased = backend.to_numpy(state.friend_purchased).astype(np.float64, copy=False)
    friend_sold = backend.to_numpy(state.friend_sold).astype(np.float64, copy=False)
    friend_id = backend.to_numpy(state.friend_id)
    if friend_purchased.ndim == 3 and friend_sold.ndim == 3 and friend_id.ndim == 2:
        known_friend = friend_id >= 0
        friend_activity = (friend_purchased > _EPSILON) | (friend_sold > _EPSILON)
        friend_activity &= known_friend[:, :, None]
        network_circulation_breadth = np.mean(np.any(friend_activity, axis=1), axis=0, dtype=np.float64)
    else:
        network_circulation_breadth = np.zeros_like(base_need)

    relative_tce_loss = periodic_tce_cost / consumption_need_total
    relative_trade_flow = ((recent_purchase_total + recent_sales_total) * 0.5) / consumption_need_total
    relative_stock = stock_total / consumption_need_total
    tce_loss_salience = _log_salience(relative_tce_loss)
    trade_flow_salience = _log_salience(relative_trade_flow)
    stock_salience = _log_salience(relative_stock)
    exchange_route_breadth = np.maximum(
        np.sqrt(np.clip(gross_purchase_breadth, 0.0, 1.0) * np.clip(gross_sales_breadth, 0.0, 1.0)),
        0.25 * network_circulation_breadth,
    )
    exchange_media_score = (
        tce_loss_salience
        * np.sqrt(np.clip(trade_flow_salience, 0.0, 1.0))
        * np.sqrt(np.clip(network_circulation_breadth, 0.0, 1.0))
        * exchange_route_breadth
        * stock_salience
        * exchange_turnover_balance
    )
    demand_order = np.argsort(base_need, kind="stable")
    demand_rank = np.empty_like(demand_order)
    demand_rank[demand_order] = np.arange(1, base_need.size + 1, dtype=np.int64)
    rare_cutoff = max(1, ceil(base_need.size / 4.0))
    is_rare = demand_rank <= rare_cutoff

    return {
        "base_need": base_need,
        "report_good_id": report_good_id,
        "stock_total": stock_total,
        "average_efficiency": average_efficiency,
        "recent_purchase_total": recent_purchase_total,
        "recent_sales_total": recent_sales_total,
        "recent_inventory_inflow_total": recent_inventory_inflow_total,
        "recent_purchase_value_total": recent_purchase_value_total,
        "recent_sales_value_total": recent_sales_value_total,
        "recent_inventory_inflow_value_total": recent_inventory_inflow_value_total,
        "merchant_purchase_total": merchant_purchase_total,
        "merchant_sales_total": merchant_sales_total,
        "merchant_inventory_inflow_total": merchant_inventory_inflow_total,
        "merchant_purchase_value_total": merchant_purchase_value_total,
        "merchant_sales_value_total": merchant_sales_value_total,
        "merchant_inventory_inflow_value_total": merchant_inventory_inflow_value_total,
        "purchase_breadth": purchase_breadth,
        "inventory_acceptance_share": inventory_acceptance_share,
        "turnover_balance": turnover_balance,
        "monetary_score": monetary_score,
        "value_purchase_breadth": value_purchase_breadth,
        "value_inventory_acceptance_share": value_inventory_acceptance_share,
        "value_turnover_balance": value_turnover_balance,
        "value_weighted_monetary_score": value_weighted_monetary_score,
        "exchange_media_score": exchange_media_score,
        "relative_tce_loss": relative_tce_loss,
        "relative_trade_flow": relative_trade_flow,
        "relative_stock": relative_stock,
        "network_circulation_breadth": network_circulation_breadth,
        "excess_stock_breadth": excess_stock_breadth,
        "excess_stock_ratio": excess_stock_ratio,
        "round_trip_breadth": round_trip_breadth,
        "round_trip_turnover_share": round_trip_turnover_share,
        "consumer_flow_share": consumer_flow_share,
        "retailer_stock_share": retailer_stock_share,
        "demand_rank": demand_rank,
        "is_rare": is_rare,
    }


def _report_good_ids_from_base_need(base_need: np.ndarray) -> np.ndarray:
    # The canonical demand curve is (report_good_id + 1)^2. This lets the
    # dashboard show g0,g3,... labels for spaced-good experiments without
    # changing the internal dense 0..N-1 array layout.
    candidate = np.rint(np.sqrt(np.maximum(base_need, 0.0)) - 1.0).astype(np.int64)
    reconstructed = (candidate.astype(np.float64) + 1.0) ** 2
    fallback = np.arange(base_need.size, dtype=np.int64)
    return np.where(np.isclose(reconstructed, base_need, rtol=1e-6, atol=1e-6), candidate, fallback)


def compute_monetary_aggregates(state: SimulationState, backend: BaseBackend) -> tuple[float, float, float, float, float, float]:
    arrays = _good_metric_arrays(state, backend)
    monetary_concentration, rare_goods_monetary_share = _score_concentration_and_rare_share(
        arrays["monetary_score"],
        arrays["is_rare"],
    )
    value_weighted_monetary_concentration, value_weighted_rare_goods_monetary_share = _score_concentration_and_rare_share(
        arrays["value_weighted_monetary_score"],
        arrays["is_rare"],
    )
    exchange_media_concentration, rare_goods_exchange_media_share = _score_concentration_and_rare_share(
        arrays["exchange_media_score"],
        arrays["is_rare"],
    )
    return (
        monetary_concentration,
        rare_goods_monetary_share,
        value_weighted_monetary_concentration,
        value_weighted_rare_goods_monetary_share,
        exchange_media_concentration,
        rare_goods_exchange_media_share,
    )


def _score_concentration_and_rare_share(scores: np.ndarray, is_rare: np.ndarray) -> tuple[float, float]:
    total_score = float(scores.sum())
    if total_score <= _EPSILON:
        return 0.0, 0.0

    normalized = scores / total_score
    concentration = float(np.square(normalized).sum())
    rare_share = float(scores[is_rare].sum() / total_score)
    return concentration, rare_share


def _log_salience(values: np.ndarray) -> np.ndarray:
    logged = np.log1p(np.maximum(values.astype(np.float64, copy=False), 0.0))
    scale = float(np.max(logged)) if logged.size else 0.0
    if scale <= _EPSILON:
        return np.zeros_like(logged)
    return np.clip(logged / scale, 0.0, 1.0)


def compute_good_snapshots(
    *,
    state: SimulationState,
    backend: BaseBackend,
    limit: int | None = None,
    sort_by: str = "monetary_score",
) -> list[GoodSnapshot]:
    arrays = _good_metric_arrays(state, backend)
    goods = arrays["base_need"].size
    snapshots = [
        GoodSnapshot(
            good_id=good_id,
            base_need=float(arrays["base_need"][good_id]),
            demand_rank=int(arrays["demand_rank"][good_id]),
            is_rare=bool(arrays["is_rare"][good_id]),
            stock_total=float(arrays["stock_total"][good_id]),
            average_efficiency=float(arrays["average_efficiency"][good_id]),
            recent_purchase_total=float(arrays["recent_purchase_total"][good_id]),
            recent_sales_total=float(arrays["recent_sales_total"][good_id]),
            recent_inventory_inflow_total=float(arrays["recent_inventory_inflow_total"][good_id]),
            recent_purchase_value_total=float(arrays["recent_purchase_value_total"][good_id]),
            recent_sales_value_total=float(arrays["recent_sales_value_total"][good_id]),
            recent_inventory_inflow_value_total=float(arrays["recent_inventory_inflow_value_total"][good_id]),
            purchase_breadth=float(arrays["purchase_breadth"][good_id]),
            inventory_acceptance_share=float(arrays["inventory_acceptance_share"][good_id]),
            turnover_balance=float(arrays["turnover_balance"][good_id]),
            monetary_score=float(arrays["monetary_score"][good_id]),
            value_weighted_monetary_score=float(arrays["value_weighted_monetary_score"][good_id]),
            report_good_id=int(arrays["report_good_id"][good_id]),
            exchange_media_score=float(arrays["exchange_media_score"][good_id]),
            relative_tce_loss=float(arrays["relative_tce_loss"][good_id]),
            relative_trade_flow=float(arrays["relative_trade_flow"][good_id]),
            relative_stock=float(arrays["relative_stock"][good_id]),
            network_circulation_breadth=float(arrays["network_circulation_breadth"][good_id]),
            excess_stock_breadth=float(arrays["excess_stock_breadth"][good_id]),
            excess_stock_ratio=float(arrays["excess_stock_ratio"][good_id]),
            round_trip_breadth=float(arrays["round_trip_breadth"][good_id]),
            round_trip_turnover_share=float(arrays["round_trip_turnover_share"][good_id]),
            consumer_flow_share=float(arrays["consumer_flow_share"][good_id]),
            retailer_stock_share=float(arrays["retailer_stock_share"][good_id]),
        )
        for good_id in range(goods)
    ]

    valid_sort_keys = {
        "monetary_score",
        "value_weighted_monetary_score",
        "recent_purchase_total",
        "recent_purchase_value_total",
        "recent_inventory_inflow_total",
        "recent_inventory_inflow_value_total",
        "stock_total",
        "average_efficiency",
        "base_need",
        "exchange_media_score",
        "relative_tce_loss",
        "relative_trade_flow",
        "relative_stock",
        "network_circulation_breadth",
        "excess_stock_breadth",
        "excess_stock_ratio",
        "round_trip_breadth",
        "round_trip_turnover_share",
        "consumer_flow_share",
        "retailer_stock_share",
    }
    if sort_by not in valid_sort_keys:
        raise ValueError(f"Unsupported sort field: {sort_by}")

    snapshots.sort(key=lambda item: (getattr(item, sort_by), -item.good_id), reverse=True)
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        snapshots = snapshots[:limit]
    return snapshots


def summarize_recent_trends(
    history: Sequence[MetricsSnapshot] | Sequence[MarketSnapshot],
    *,
    windows: Sequence[int] = (50, 100, 200),
) -> list[RecentTrendWindowSnapshot]:
    if not history:
        return []

    latest_cycle = int(history[-1].cycle)
    summaries: list[RecentTrendWindowSnapshot] = []
    for window in windows:
        subset = [item for item in history if item.cycle >= latest_cycle - int(window)]
        if len(subset) < 2:
            continue

        production_series = np.asarray([item.production_total for item in subset], dtype=np.float64)
        utility_series = np.asarray([item.utility_proxy_total for item in subset], dtype=np.float64)
        trade_series = np.asarray([item.accepted_trade_volume for item in subset], dtype=np.float64)
        rare_money_series = np.asarray([item.rare_goods_monetary_share for item in subset], dtype=np.float64)
        value_rare_money_series = np.asarray(
            [item.value_weighted_rare_goods_monetary_share for item in subset],
            dtype=np.float64,
        )
        exchange_media_series = np.asarray(
            [item.rare_goods_exchange_media_share for item in subset],
            dtype=np.float64,
        )

        production_trend = _normalized_slope(production_series)
        utility_trend = _normalized_slope(utility_series)
        trade_trend = _normalized_slope(trade_series)
        rare_goods_monetary_trend = _normalized_slope(rare_money_series)
        value_weighted_rare_goods_monetary_trend = _normalized_slope(value_rare_money_series)
        rare_goods_exchange_media_trend = _normalized_slope(exchange_media_series)
        cycle_strength, dominant_cycle_length = _dominant_autocorrelation(trade_series)

        summaries.append(
            RecentTrendWindowSnapshot(
                window_cycles=int(window),
                sample_count=len(subset),
                from_cycle=int(subset[0].cycle),
                to_cycle=int(subset[-1].cycle),
                production_trend=production_trend,
                utility_trend=utility_trend,
                trade_trend=trade_trend,
                rare_goods_monetary_trend=rare_goods_monetary_trend,
                production_change=_relative_change(production_series),
                utility_change=_relative_change(utility_series),
                trade_change=_relative_change(trade_series),
                rare_goods_monetary_change=_relative_change(rare_money_series),
                cycle_strength=cycle_strength,
                dominant_cycle_length=dominant_cycle_length,
                economy_growing=production_trend > 0.0001,
                utility_growing=utility_trend > 0.0001,
                value_weighted_rare_goods_monetary_trend=value_weighted_rare_goods_monetary_trend,
                value_weighted_rare_goods_monetary_change=_relative_change(value_rare_money_series),
                rare_goods_exchange_media_trend=rare_goods_exchange_media_trend,
                rare_goods_exchange_media_change=_relative_change(exchange_media_series),
            )
        )
    return summaries


def compute_role_snapshots(
    *,
    state: SimulationState,
    backend: BaseBackend,
    limit: int | None = None,
    sort_by: str = 'retailer_count',
) -> list[GoodRoleSnapshot]:
    arrays = _good_metric_arrays(state, backend)
    role = backend.to_numpy(state.role).astype(np.int32, copy=False)
    stock = backend.to_numpy(state.stock).astype(np.float64, copy=False)
    stock_limit = backend.to_numpy(state.stock_limit).astype(np.float64, copy=False)
    recent_purchases = backend.to_numpy(state.recent_purchases).astype(np.float64, copy=False)
    recent_sales = backend.to_numpy(state.recent_sales).astype(np.float64, copy=False)
    recent_inventory_inflow = backend.to_numpy(state.recent_inventory_inflow).astype(np.float64, copy=False)
    recent_production = backend.to_numpy(state.recent_production).astype(np.float64, copy=False)
    efficiency = backend.to_numpy(state.efficiency).astype(np.float64, copy=False)

    retailer_mask = role == ROLE_RETAILER
    snapshots = [
        GoodRoleSnapshot(
            good_id=good_id,
            base_need=float(arrays['base_need'][good_id]),
            demand_rank=int(arrays['demand_rank'][good_id]),
            is_rare=bool(arrays['is_rare'][good_id]),
            consumer_count=int(np.count_nonzero(role[:, good_id] == ROLE_CONSUMER)),
            retailer_count=int(np.count_nonzero(role[:, good_id] == ROLE_RETAILER)),
            producer_count=int(np.count_nonzero(role[:, good_id] == ROLE_PRODUCER)),
            retailer_purchase_total=float(np.sum(np.where(retailer_mask[:, good_id], recent_purchases[:, good_id], 0.0), dtype=np.float64)),
            retailer_sales_total=float(np.sum(np.where(retailer_mask[:, good_id], recent_sales[:, good_id], 0.0), dtype=np.float64)),
            retailer_inventory_inflow_total=float(np.sum(np.where(retailer_mask[:, good_id], recent_inventory_inflow[:, good_id], 0.0), dtype=np.float64)),
            retailer_stock_limit_ratio_mean=_mean_ratio(
                np.where(retailer_mask[:, good_id], stock[:, good_id], 0.0),
                np.where(retailer_mask[:, good_id], stock_limit[:, good_id], 0.0),
                retailer_mask[:, good_id],
            ),
            top1_producer_output_share=_top_output_share(recent_production[:, good_id], top_k=1),
            top_producer_output_focus_share=_top_producer_output_focus_share(
                good_id,
                recent_production[:, good_id],
                recent_production,
            ),
            top_producer_time_share=_top_producer_time_share(
                good_id,
                recent_production[:, good_id],
                recent_production,
                efficiency,
            ),
            report_good_id=int(arrays['report_good_id'][good_id]),
        )
        for good_id in range(arrays['base_need'].size)
    ]

    valid_sort_keys = {
        'retailer_count',
        'producer_count',
        'consumer_count',
        'retailer_inventory_inflow_total',
        'retailer_sales_total',
        'retailer_purchase_total',
        'base_need',
    }
    if sort_by not in valid_sort_keys:
        raise ValueError(f'Unsupported sort field: {sort_by}')

    snapshots.sort(key=lambda item: (getattr(item, sort_by), -item.good_id), reverse=True)
    if limit is not None:
        if limit <= 0:
            raise ValueError('limit must be positive')
        snapshots = snapshots[:limit]
    return snapshots


def compute_inequality_snapshot(
    *,
    state: SimulationState,
    backend: BaseBackend,
    config: SimulationConfig,
) -> InequalitySnapshot:
    stock = backend.to_numpy(state.stock).astype(np.float64, copy=False)
    average_price = backend.to_numpy(state.market.average_price).astype(np.float64, copy=False)
    stock_value = np.sum(stock * average_price[None, :], axis=1, dtype=np.float64)
    stock_value_gini, stock_value_top_decile_share, stock_value_mean, stock_value_median = _distribution_snapshot(stock_value)
    living_standard, smith_cost, aspiration_balance = _compute_living_standard_components(
        state=state,
        backend=backend,
        config=config,
    )
    living_standard_gini, living_standard_top_decile_share, living_standard_mean, living_standard_median = _distribution_snapshot(living_standard)
    smith_cost_gini, smith_cost_top_decile_share, smith_cost_mean, smith_cost_median = _distribution_snapshot(smith_cost)
    aspiration_shortfall = np.maximum(1.0 - aspiration_balance, 0.0)

    produced = backend.to_numpy(state.produced_this_period).astype(np.float64, copy=False)
    efficiency = backend.to_numpy(state.efficiency).astype(np.float64, copy=False)
    produced_by_good = np.sum(produced, axis=0, dtype=np.float64)
    production_time_value = float(np.sum(produced_by_good * average_price, dtype=np.float64))
    direct_production_time = float(np.sum(produced / np.maximum(efficiency, _EPSILON), dtype=np.float64))
    available_time = float(state.base_need.shape[0]) * float(config.cycle_time_budget)
    tce_time_value = float(state.market.total_cost_of_tce_in_time)
    spoilage_time_value = float(state.market.total_cost_of_spoilage_in_time)
    friction_time_value = tce_time_value + spoilage_time_value

    return InequalitySnapshot(
        stock_value_gini=stock_value_gini,
        stock_value_top_decile_share=stock_value_top_decile_share,
        stock_value_mean=stock_value_mean,
        stock_value_median=stock_value_median,
        living_standard_gini=living_standard_gini,
        living_standard_top_decile_share=living_standard_top_decile_share,
        living_standard_mean=living_standard_mean,
        living_standard_median=living_standard_median,
        living_standard_p10=_quantile_value(living_standard, 0.10),
        living_standard_p25=_quantile_value(living_standard, 0.25),
        living_standard_p75=_quantile_value(living_standard, 0.75),
        living_standard_p90=_quantile_value(living_standard, 0.90),
        living_standard_p99=_quantile_value(living_standard, 0.99),
        aspiration_balance_mean=float(np.mean(aspiration_balance, dtype=np.float64)) if aspiration_balance.size else 0.0,
        aspiration_balance_median=float(np.median(aspiration_balance)) if aspiration_balance.size else 0.0,
        aspiration_balance_p10=_quantile_value(aspiration_balance, 0.10),
        aspiration_balance_p90=_quantile_value(aspiration_balance, 0.90),
        aspiration_shortfall_share=float(np.mean(aspiration_balance < 1.0, dtype=np.float64)) if aspiration_balance.size else 0.0,
        aspiration_shortfall_mean=float(np.mean(aspiration_shortfall, dtype=np.float64)) if aspiration_shortfall.size else 0.0,
        aspiration_shortfall_p90=_quantile_value(aspiration_shortfall, 0.90),
        smith_cost_gini=smith_cost_gini,
        smith_cost_top_decile_share=smith_cost_top_decile_share,
        smith_cost_mean=smith_cost_mean,
        smith_cost_median=smith_cost_median,
        smith_cost_p10=_quantile_value(smith_cost, 0.10),
        smith_cost_p25=_quantile_value(smith_cost, 0.25),
        smith_cost_p75=_quantile_value(smith_cost, 0.75),
        smith_cost_p90=_quantile_value(smith_cost, 0.90),
        smith_cost_p99=_quantile_value(smith_cost, 0.99),
        production_time_value=production_time_value,
        direct_production_time=direct_production_time,
        production_time_share_of_budget=_safe_ratio(direct_production_time, available_time),
        tce_share_of_output_value=_safe_ratio(tce_time_value, production_time_value),
        spoilage_share_of_output_value=_safe_ratio(spoilage_time_value, production_time_value),
        friction_share_of_output_value=_safe_ratio(friction_time_value, production_time_value),
        tce_share_of_time_budget=_safe_ratio(tce_time_value, available_time),
        spoilage_share_of_time_budget=_safe_ratio(spoilage_time_value, available_time),
        friction_share_of_time_budget=_safe_ratio(friction_time_value, available_time),
    )


def _relative_change(values: np.ndarray) -> float | None:
    if values.size < 2:
        return None
    start = float(values[0])
    end = float(values[-1])
    if abs(start) <= _EPSILON:
        return None
    return (end - start) / start


def analyze_history(
    history: Sequence[MetricsSnapshot] | Sequence[MarketSnapshot],
    goods: Sequence[GoodSnapshot],
) -> PhenomenaSnapshot:
    if not history:
        return PhenomenaSnapshot(
            cycles_observed=0,
            production_trend=0.0,
            utility_trend=0.0,
            stock_trend=0.0,
            cycle_strength=0.0,
            dominant_cycle_length=None,
            rare_goods_monetary_share=0.0,
            monetary_concentration=0.0,
            economy_growing=False,
            utility_growing=False,
            cycles_detected=False,
            top_monetary_good_ids=[],
            value_weighted_rare_goods_monetary_share=0.0,
            value_weighted_monetary_concentration=0.0,
            top_value_weighted_monetary_good_ids=[],
            rare_goods_exchange_media_share=0.0,
            exchange_media_concentration=0.0,
            top_exchange_media_good_ids=[],
        )

    production_series = np.asarray([item.production_total for item in history], dtype=np.float64)
    utility_series = np.asarray([item.utility_proxy_total for item in history], dtype=np.float64)
    stock_series = np.asarray([item.stock_total for item in history], dtype=np.float64)
    trade_series = np.asarray([item.accepted_trade_volume for item in history], dtype=np.float64)

    production_trend = _normalized_slope(production_series)
    utility_trend = _normalized_slope(utility_series)
    stock_trend = _normalized_slope(stock_series)
    cycle_strength, dominant_cycle_length = _dominant_autocorrelation(trade_series)

    latest = history[-1]
    top_ids = [
        item.report_good_id if item.report_good_id is not None else item.good_id
        for item in sorted(goods, key=lambda item: (item.monetary_score, -item.good_id), reverse=True)[:5]
        if item.monetary_score > _EPSILON
    ]
    top_value_ids = [
        item.report_good_id if item.report_good_id is not None else item.good_id
        for item in sorted(goods, key=lambda item: (item.value_weighted_monetary_score, -item.good_id), reverse=True)[:5]
        if item.value_weighted_monetary_score > _EPSILON
    ]
    top_exchange_ids = [
        item.report_good_id if item.report_good_id is not None else item.good_id
        for item in sorted(goods, key=lambda item: (item.exchange_media_score, -item.good_id), reverse=True)[:5]
        if item.exchange_media_score > _EPSILON
    ]
    return PhenomenaSnapshot(
        cycles_observed=len(history),
        production_trend=production_trend,
        utility_trend=utility_trend,
        stock_trend=stock_trend,
        cycle_strength=cycle_strength,
        dominant_cycle_length=dominant_cycle_length,
        rare_goods_monetary_share=float(latest.rare_goods_monetary_share),
        monetary_concentration=float(latest.monetary_concentration),
        economy_growing=production_trend > 0.0001,
        utility_growing=utility_trend > 0.0001,
        cycles_detected=cycle_strength >= 0.35 and dominant_cycle_length is not None,
        top_monetary_good_ids=top_ids,
        value_weighted_rare_goods_monetary_share=float(latest.value_weighted_rare_goods_monetary_share),
        value_weighted_monetary_concentration=float(latest.value_weighted_monetary_concentration),
        top_value_weighted_monetary_good_ids=top_value_ids,
        rare_goods_exchange_media_share=float(latest.rare_goods_exchange_media_share),
        exchange_media_concentration=float(latest.exchange_media_concentration),
        top_exchange_media_good_ids=top_exchange_ids,
    )


def _normalized_slope(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    x = np.arange(values.size, dtype=np.float64)
    x_centered = x - x.mean()
    denominator = float(np.dot(x_centered, x_centered))
    if denominator <= _EPSILON:
        return 0.0
    y_centered = values - values.mean()
    slope = float(np.dot(x_centered, y_centered) / denominator)
    scale = max(abs(float(values.mean())), _EPSILON)
    return slope / scale


def _dominant_autocorrelation(values: np.ndarray) -> tuple[float, int | None]:
    if values.size < 12:
        return 0.0, None
    deltas = np.diff(values)
    if deltas.size < 8:
        return 0.0, None
    centered = deltas - deltas.mean()
    variance = float(np.dot(centered, centered))
    if variance <= _EPSILON:
        return 0.0, None

    best_lag: int | None = None
    best_strength = 0.0
    max_lag = min(centered.size // 2, 60)
    for lag in range(2, max_lag + 1):
        leading = centered[:-lag]
        trailing = centered[lag:]
        if leading.size < 4:
            continue
        denominator = float(np.linalg.norm(leading) * np.linalg.norm(trailing))
        if denominator <= _EPSILON:
            continue
        strength = float(np.dot(leading, trailing) / denominator)
        if strength > best_strength:
            best_strength = strength
            best_lag = lag
    if best_strength < 0.1:
        return 0.0, None
    return best_strength, best_lag


def _distribution_snapshot(values: np.ndarray) -> tuple[float, float, float, float]:
    if values.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    total_value = float(np.sum(values, dtype=np.float64))
    mean_value = float(np.mean(values, dtype=np.float64))
    median_value = float(np.median(values))
    if total_value <= _EPSILON:
        return 0.0, 0.0, mean_value, median_value

    top_count = max(1, int(ceil(values.size * 0.1)))
    sorted_values = np.sort(values)
    top_share = float(np.sum(sorted_values[-top_count:], dtype=np.float64) / total_value)
    return _gini_coefficient(values), top_share, mean_value, median_value


def _quantile_value(values: np.ndarray, quantile: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.quantile(values, quantile))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= _EPSILON:
        return 0.0
    return float(numerator / denominator)


def _compute_living_standard_values(
    *,
    state: SimulationState,
    backend: BaseBackend,
    config: SimulationConfig,
) -> np.ndarray:
    living_standard, _smith_cost, _aspiration_balance = _compute_living_standard_components(state=state, backend=backend, config=config)
    return living_standard


def _compute_living_standard_components(
    *,
    state: SimulationState,
    backend: BaseBackend,
    config: SimulationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stock = backend.to_numpy(state.stock).astype(np.float64, copy=False)
    base_need = backend.to_numpy(state.base_need).astype(np.float64, copy=False)
    need_remaining = backend.to_numpy(state.need).astype(np.float64, copy=False)
    needs_level = backend.to_numpy(state.needs_level).astype(np.float64, copy=False)
    elastic_need = backend.to_numpy(state.market.elastic_need).astype(np.float64, copy=False)
    purchase_price = backend.to_numpy(state.purchase_price).astype(np.float64, copy=False)
    sales_price = backend.to_numpy(state.sales_price).astype(np.float64, copy=False)
    efficiency = backend.to_numpy(state.efficiency).astype(np.float64, copy=False)
    recent_sales = backend.to_numpy(state.recent_sales).astype(np.float64, copy=False)
    recent_purchases = backend.to_numpy(state.recent_purchases).astype(np.float64, copy=False)
    purchased_last_period = backend.to_numpy(state.purchased_last_period).astype(np.float64, copy=False)
    sold_this_period = backend.to_numpy(state.sold_this_period).astype(np.float64, copy=False)
    sold_last_period = backend.to_numpy(state.sold_last_period).astype(np.float64, copy=False)

    elastic_need_scaled = elastic_need[None, :] * needs_level[:, None]
    efficiency_safe = np.maximum(efficiency, _EPSILON)
    surplus_value = stock - (elastic_need_scaled * config.max_needs_increase)

    valued_surplus = np.empty_like(surplus_value, dtype=np.float64)
    negative_mask = surplus_value < 0.0
    if np.any(negative_mask):
        negative_surplus = surplus_value[negative_mask]
        valued_surplus[negative_mask] = np.where(
            purchased_last_period[negative_mask] > (-1.0 * negative_surplus),
            negative_surplus * purchase_price[negative_mask],
            negative_surplus / efficiency_safe[negative_mask],
        )

    positive_mask = ~negative_mask
    if np.any(positive_mask):
        positive_surplus = surplus_value[positive_mask]
        sales_cap = config.stock_limit_multiplier * (
            (sold_this_period[positive_mask] - sold_last_period[positive_mask]) + elastic_need_scaled[positive_mask]
        )
        self_use_cap = elastic_need_scaled[positive_mask] * config.max_needs_increase
        valued_surplus[positive_mask] = np.where(
            recent_sales[positive_mask] > positive_surplus,
            np.minimum(positive_surplus, sales_cap) * sales_price[positive_mask],
            np.minimum(positive_surplus, self_use_cap)
            * np.minimum(purchase_price[positive_mask], 1.0 / efficiency_safe[positive_mask]),
        )

    wealth_minus_needs = float(config.cycle_time_budget) + np.sum(valued_surplus, axis=1, dtype=np.float64)
    need_value = np.where(
        recent_purchases > elastic_need_scaled,
        purchase_price * elastic_need_scaled,
        elastic_need_scaled / efficiency_safe,
    )
    total_needs_value = np.sum(need_value, axis=1, dtype=np.float64)

    aspiration_balance = np.ones_like(total_needs_value, dtype=np.float64)
    valid_mask = total_needs_value > _EPSILON
    aspiration_balance[valid_mask] = (
        total_needs_value[valid_mask] + wealth_minus_needs[valid_mask]
    ) / total_needs_value[valid_mask]

    baseline_quantity = np.sum(base_need, axis=1, dtype=np.float64)
    fulfilled_quantity = np.maximum(elastic_need_scaled - need_remaining, 0.0)
    achieved_quantity = np.sum(fulfilled_quantity, axis=1, dtype=np.float64)
    living_standard = np.ones_like(baseline_quantity, dtype=np.float64)
    valid_baseline = baseline_quantity > _EPSILON
    living_standard[valid_baseline] = np.maximum(
        achieved_quantity[valid_baseline] / baseline_quantity[valid_baseline],
        0.0,
    )
    return living_standard, total_needs_value, aspiration_balance


def _gini_coefficient(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    sorted_values = np.sort(np.maximum(values.astype(np.float64, copy=False), 0.0))
    total = float(np.sum(sorted_values, dtype=np.float64))
    if total <= _EPSILON:
        return 0.0
    index = np.arange(1, sorted_values.size + 1, dtype=np.float64)
    weighted_sum = float(np.sum(index * sorted_values, dtype=np.float64))
    gini = ((2.0 * weighted_sum) / (sorted_values.size * total)) - ((sorted_values.size + 1.0) / sorted_values.size)
    return max(0.0, min(gini, 1.0))


def _mean_ratio(numerator: np.ndarray, denominator: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    values = np.divide(
        numerator,
        np.maximum(denominator, _EPSILON),
        out=np.zeros_like(numerator, dtype=np.float64),
        where=mask,
    )
    return float(np.mean(values[mask], dtype=np.float64))


def _top_output_share(recent_production_by_good: np.ndarray, *, top_k: int) -> float:
    total = float(np.sum(recent_production_by_good, dtype=np.float64))
    if total <= _EPSILON:
        return 0.0
    sorted_values = np.sort(recent_production_by_good.astype(np.float64, copy=False))
    return float(np.sum(sorted_values[-top_k:], dtype=np.float64) / total)


def _top_producer_time_share(
    good_id: int,
    recent_production_by_good: np.ndarray,
    recent_production: np.ndarray,
    efficiency: np.ndarray,
) -> float:
    if recent_production_by_good.size == 0:
        return 0.0
    top_agent = int(np.argmax(recent_production_by_good))
    top_output = float(recent_production_by_good[top_agent])
    if top_output <= _EPSILON:
        return 0.0
    time_by_good = np.divide(
        recent_production[top_agent],
        np.maximum(efficiency[top_agent], _EPSILON),
        out=np.zeros_like(recent_production[top_agent], dtype=np.float64),
    )
    total_time = float(np.sum(time_by_good, dtype=np.float64))
    if total_time <= _EPSILON:
        return 0.0
    return float(time_by_good[good_id] / total_time)


def _top_producer_output_focus_share(
    good_id: int,
    recent_production_by_good: np.ndarray,
    recent_production: np.ndarray,
) -> float:
    if recent_production_by_good.size == 0:
        return 0.0
    top_agent = int(np.argmax(recent_production_by_good))
    top_output = float(recent_production_by_good[top_agent])
    if top_output <= _EPSILON:
        return 0.0
    total_output = float(np.sum(recent_production[top_agent], dtype=np.float64))
    if total_output <= _EPSILON:
        return 0.0
    return float(top_output / total_output)
