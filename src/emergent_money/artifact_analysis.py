from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

_EPSILON = 1.0e-9

_DEFAULT_FIELDS = (
    'production_total',
    'utility_proxy_total',
    'accepted_trade_volume',
    'stock_total',
    'rare_goods_monetary_share',
    'value_weighted_rare_goods_monetary_share',
    'rare_goods_exchange_media_share',
    'living_standard_mean',
    'living_standard_gini',
    'friction_share_of_time_budget',
    'tce_share_of_time_budget',
    'spoilage_share_of_time_budget',
)


def summarize_run_artifact(run_dir: str | Path, *, fields: tuple[str, ...] = _DEFAULT_FIELDS) -> dict[str, Any]:
    metrics_path = Path(run_dir) / 'metrics.jsonl'
    if not metrics_path.exists():
        raise FileNotFoundError(f'metrics.jsonl not found in {run_dir}')

    rows = _load_metrics_rows(metrics_path)
    if not rows:
        raise ValueError(f'no metrics rows found in {metrics_path}')

    summary_path = Path(run_dir) / 'summary.json'
    run_summary = _load_optional_json(summary_path)
    field_summaries = {
        field: _summarize_series(rows, field)
        for field in fields
        if field in rows[-1]
    }
    return {
        'run_dir': str(Path(run_dir)),
        'sample_count': len(rows),
        'first_cycle': int(rows[0]['cycle']),
        'last_cycle': int(rows[-1]['cycle']),
        'config': run_summary.get('config', {}),
        'runtime_seconds': run_summary.get('runtime_seconds'),
        'phenomenon_flags': _phenomenon_flags(field_summaries, last_cycle=int(rows[-1]['cycle'])),
        'fields': field_summaries,
    }


def _load_metrics_rows(metrics_path: Path) -> list[dict[str, Any]]:
    rows_by_cycle: dict[int, dict[str, Any]] = {}
    for line in metrics_path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            row = json.loads(line)
            # Interrupted/resumed runs can replay a few cycles after the last
            # checkpoint. Keep the latest sample for each cycle.
            rows_by_cycle[int(row['cycle'])] = row
    return [rows_by_cycle[cycle] for cycle in sorted(rows_by_cycle)]


def _load_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding='utf-8'))


def _summarize_series(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    cycles = np.asarray([int(row['cycle']) for row in rows], dtype=np.int64)
    values = np.asarray([float(row.get(field, 0.0)) for row in rows], dtype=np.float64)
    max_index = int(np.argmax(values))
    min_index = int(np.argmin(values))
    return {
        'first': float(values[0]),
        'last': float(values[-1]),
        'min': float(values[min_index]),
        'min_cycle': int(cycles[min_index]),
        'max': float(values[max_index]),
        'max_cycle': int(cycles[max_index]),
        'normalized_trend': _normalized_slope(values),
        'relative_change': _relative_change(values),
        'last_to_peak': _safe_ratio(float(values[-1]), float(values[max_index])),
    }


def _phenomenon_flags(field_summaries: dict[str, dict[str, Any]], *, last_cycle: int) -> dict[str, Any]:
    production = field_summaries.get('production_total', {})
    utility = field_summaries.get('utility_proxy_total', {})
    rare_money = field_summaries.get('rare_goods_monetary_share', {})
    value_rare_money = field_summaries.get('value_weighted_rare_goods_monetary_share', {})
    exchange_media = field_summaries.get('rare_goods_exchange_media_share', {})
    friction = field_summaries.get('friction_share_of_time_budget', {})
    living_gini = field_summaries.get('living_standard_gini', {})
    utility_peak_cycle = int(utility.get('max_cycle', 0))
    return {
        'production_grew': float(production.get('relative_change') or 0.0) > 1.0,
        'rare_money_emerged': max(
            float(rare_money.get('max', 0.0)),
            float(value_rare_money.get('max', 0.0)),
            float(exchange_media.get('max', 0.0)),
        ) >= 0.5,
        'utility_peaked_before_end': utility_peak_cycle > 0 and utility_peak_cycle < last_cycle,
        'friction_rose': float(friction.get('last', 0.0)) > (float(friction.get('first', 0.0)) + 0.05),
        'living_standard_inequality_high': float(living_gini.get('last', 0.0)) >= 0.4,
    }


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
    return slope / max(abs(float(values.mean())), _EPSILON)


def _relative_change(values: np.ndarray) -> float | None:
    if values.size < 2:
        return None
    first = float(values[0])
    if abs(first) <= _EPSILON:
        return None
    return float((values[-1] - first) / first)


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if abs(denominator) <= _EPSILON:
        return None
    return float(numerator / denominator)
