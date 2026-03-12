from __future__ import annotations

from dataclasses import asdict
from math import ceil
from typing import Sequence

import numpy as np

from .backend.base import BaseBackend
from .dto import GoodSnapshot, MarketSnapshot, PhenomenaSnapshot
from .metrics import MetricsSnapshot
from .state import SimulationState

_EPSILON = 1e-6


def _good_metric_arrays(state: SimulationState, backend: BaseBackend) -> dict[str, np.ndarray]:
    xp = backend.xp
    base_need = backend.to_numpy(state.base_need[0]).astype(np.float64, copy=False)
    stock_total = backend.to_numpy(xp.sum(state.stock, axis=0)).astype(np.float64, copy=False)
    average_efficiency = backend.to_numpy(xp.mean(state.efficiency, axis=0)).astype(np.float64, copy=False)
    recent_purchase_total = backend.to_numpy(xp.sum(state.recent_purchases, axis=0)).astype(np.float64, copy=False)
    recent_sales_total = backend.to_numpy(xp.sum(state.recent_sales, axis=0)).astype(np.float64, copy=False)
    recent_inventory_inflow_total = backend.to_numpy(xp.sum(state.recent_inventory_inflow, axis=0)).astype(np.float64, copy=False)
    purchase_breadth = backend.to_numpy(xp.mean((state.recent_purchases > _EPSILON).astype(xp.float32), axis=0)).astype(
        np.float64,
        copy=False,
    )

    inventory_acceptance_share = np.divide(
        recent_inventory_inflow_total,
        np.maximum(recent_purchase_total, _EPSILON),
        out=np.zeros_like(recent_inventory_inflow_total),
        where=recent_purchase_total > _EPSILON,
    )
    turnover_balance = np.divide(
        np.minimum(recent_purchase_total, recent_sales_total),
        np.maximum(np.maximum(recent_purchase_total, recent_sales_total), _EPSILON),
        out=np.zeros_like(recent_purchase_total),
        where=(recent_purchase_total > _EPSILON) | (recent_sales_total > _EPSILON),
    )

    nonzero_purchase = recent_purchase_total[recent_purchase_total > _EPSILON]
    mean_nonzero_purchase = float(nonzero_purchase.mean()) if nonzero_purchase.size else 0.0
    if mean_nonzero_purchase > _EPSILON:
        trade_salience = np.clip(recent_purchase_total / mean_nonzero_purchase, 0.0, 1.0)
    else:
        trade_salience = np.zeros_like(recent_purchase_total)

    monetary_score = inventory_acceptance_share * purchase_breadth * trade_salience * turnover_balance
    demand_order = np.argsort(base_need, kind="stable")
    demand_rank = np.empty_like(demand_order)
    demand_rank[demand_order] = np.arange(1, base_need.size + 1, dtype=np.int64)
    rare_cutoff = max(1, ceil(base_need.size / 4.0))
    is_rare = demand_rank <= rare_cutoff

    return {
        "base_need": base_need,
        "stock_total": stock_total,
        "average_efficiency": average_efficiency,
        "recent_purchase_total": recent_purchase_total,
        "recent_sales_total": recent_sales_total,
        "recent_inventory_inflow_total": recent_inventory_inflow_total,
        "purchase_breadth": purchase_breadth,
        "inventory_acceptance_share": inventory_acceptance_share,
        "turnover_balance": turnover_balance,
        "monetary_score": monetary_score,
        "demand_rank": demand_rank,
        "is_rare": is_rare,
    }


def compute_monetary_aggregates(state: SimulationState, backend: BaseBackend) -> tuple[float, float]:
    arrays = _good_metric_arrays(state, backend)
    scores = arrays["monetary_score"]
    total_score = float(scores.sum())
    if total_score <= _EPSILON:
        return 0.0, 0.0

    normalized = scores / total_score
    concentration = float(np.square(normalized).sum())
    rare_share = float(scores[arrays["is_rare"]].sum() / total_score)
    return concentration, rare_share


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
            purchase_breadth=float(arrays["purchase_breadth"][good_id]),
            inventory_acceptance_share=float(arrays["inventory_acceptance_share"][good_id]),
            turnover_balance=float(arrays["turnover_balance"][good_id]),
            monetary_score=float(arrays["monetary_score"][good_id]),
        )
        for good_id in range(goods)
    ]

    valid_sort_keys = {
        "monetary_score",
        "recent_purchase_total",
        "recent_inventory_inflow_total",
        "stock_total",
        "average_efficiency",
        "base_need",
    }
    if sort_by not in valid_sort_keys:
        raise ValueError(f"Unsupported sort field: {sort_by}")

    snapshots.sort(key=lambda item: (getattr(item, sort_by), -item.good_id), reverse=True)
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        snapshots = snapshots[:limit]
    return snapshots


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
    top_ids = [item.good_id for item in goods[:5] if item.monetary_score > _EPSILON]
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
