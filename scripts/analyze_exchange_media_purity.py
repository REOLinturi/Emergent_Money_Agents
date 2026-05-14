from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from emergent_money.analytics import compute_good_snapshots, compute_role_snapshots  # noqa: E402
from emergent_money.long_run import load_checkpoint  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze whether exchange-media candidates look like ordinary "
            "consumer supply-chain throughput or non-consumption intermediation."
        )
    )
    parser.add_argument("run_dir", help="Run directory containing checkpoint_latest.json/.npz")
    parser.add_argument("--limit", type=int, default=20, help="Number of rows to print per ranking")
    parser.add_argument(
        "--output-prefix",
        default="exchange_media_purity_report",
        help="Output basename inside the run directory",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    engine = load_checkpoint(run_dir)
    goods = compute_good_snapshots(
        state=engine.state,
        backend=engine.backend,
        config=engine.config,
        limit=None,
        sort_by="exchange_media_score",
    )
    roles = {
        item.good_id: item
        for item in compute_role_snapshots(
            state=engine.state,
            backend=engine.backend,
            limit=None,
            sort_by="retailer_count",
        )
    }

    rows = [_row_for_good(good, roles[good.good_id]) for good in goods]
    summary = _summary(rows, cycle=engine.cycle)
    output_json = run_dir / f"{args.output_prefix}.json"
    output_csv = run_dir / f"{args.output_prefix}.csv"
    output_json.write_text(json.dumps({"summary": summary, "goods": rows}, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(output_csv, rows)

    print(f"cycle={engine.cycle}")
    print(
        "rare_exchange_media_share="
        f"{summary['rare_exchange_media_share']:.3f} "
        "rare_intermediation_purity_share="
        f"{summary['rare_intermediation_purity_share']:.3f} "
        "rare_value_money_share="
        f"{summary['rare_value_weighted_monetary_share']:.3f}"
    )
    print(f"report_json={output_json.resolve()}")
    print(f"report_csv={output_csv.resolve()}")

    for ranking in ("exchange_media_score", "intermediation_purity_score", "value_weighted_monetary_score"):
        print(f"\nTOP {args.limit} by {ranking}")
        for row in sorted(rows, key=lambda item: item[ranking], reverse=True)[: args.limit]:
            print(
                f"{row['label']:>4} rare={int(row['is_rare'])} "
                f"class={row['flow_class']:<24} "
                f"em={row['exchange_media_score']:.3f} "
                f"purity={row['intermediation_purity_score']:.3f} "
                f"vMoney={row['value_weighted_monetary_score']:.3f} "
                f"consumer={row['consumer_flow_share']:.3f} "
                f"merchantRT={row['merchant_round_trip_breadth']:.3f} "
                f"retailers={row['retailer_count']}"
            )
    return 0


def _row_for_good(good: Any, role: Any) -> dict[str, Any]:
    row = {
        **asdict(good),
        "label": f"g{good.report_good_id if good.report_good_id is not None else good.good_id}",
        "consumer_count": role.consumer_count,
        "retailer_count": role.retailer_count,
        "producer_count": role.producer_count,
        "retailer_purchase_total": role.retailer_purchase_total,
        "retailer_sales_total": role.retailer_sales_total,
        "retailer_inventory_inflow_total": role.retailer_inventory_inflow_total,
        "retailer_stock_limit_ratio_mean": role.retailer_stock_limit_ratio_mean,
        "top1_producer_output_share": role.top1_producer_output_share,
        "top_producer_output_focus_share": role.top_producer_output_focus_share,
        "top_producer_time_share": role.top_producer_time_share,
    }
    row["flow_class"] = _classify_flow(row)
    row["supply_chain_like"] = _is_supply_chain_like(row)
    row["exchange_media_like"] = _is_exchange_media_like(row)
    row["merchant_money_like"] = _is_merchant_money_like(row)
    return row


def _classify_flow(row: dict[str, Any]) -> str:
    if _is_supply_chain_like(row):
        return "supply-chain-like"
    if _is_exchange_media_like(row) and _is_merchant_money_like(row):
        return "exchange+merchant-money"
    if _is_exchange_media_like(row):
        return "exchange-media-like"
    if _is_merchant_money_like(row):
        return "merchant-money-like"
    if row["non_consumption_flow_share"] >= 0.65:
        return "non-consumption-flow"
    return "weak-or-consumption-flow"


def _is_supply_chain_like(row: dict[str, Any]) -> bool:
    return (
        row["consumer_flow_share"] >= 0.50
        and row["merchant_round_trip_breadth"] < 0.20
        and row["intermediation_purity_score"] < 0.05
    )


def _is_exchange_media_like(row: dict[str, Any]) -> bool:
    return (
        row["exchange_media_score"] >= 0.10
        and row["non_consumption_flow_share"] >= 0.80
        and row["round_trip_turnover_share"] >= 0.40
        and row["network_circulation_breadth"] >= 0.50
    )


def _is_merchant_money_like(row: dict[str, Any]) -> bool:
    return (
        max(row["monetary_score"], row["value_weighted_monetary_score"]) >= 0.10
        and row["merchant_round_trip_breadth"] >= 0.15
        and row["non_consumption_flow_share"] >= 0.75
    )


def _summary(rows: list[dict[str, Any]], *, cycle: int) -> dict[str, Any]:
    exchange_total = sum(row["exchange_media_score"] for row in rows)
    purity_total = sum(row["intermediation_purity_score"] for row in rows)
    money_total = sum(row["value_weighted_monetary_score"] for row in rows)
    return {
        "cycle": int(cycle),
        "goods": len(rows),
        "rare_exchange_media_share": _share(rows, "exchange_media_score", exchange_total, rare=True),
        "rare_intermediation_purity_share": _share(rows, "intermediation_purity_score", purity_total, rare=True),
        "rare_value_weighted_monetary_share": _share(rows, "value_weighted_monetary_score", money_total, rare=True),
        "supply_chain_like_count": sum(1 for row in rows if row["supply_chain_like"]),
        "exchange_media_like_count": sum(1 for row in rows if row["exchange_media_like"]),
        "merchant_money_like_count": sum(1 for row in rows if row["merchant_money_like"]),
        "rare_exchange_media_like_count": sum(1 for row in rows if row["exchange_media_like"] and row["is_rare"]),
        "top_exchange_media": [row["label"] for row in sorted(rows, key=lambda r: r["exchange_media_score"], reverse=True)[:10]],
        "top_intermediation_purity": [
            row["label"] for row in sorted(rows, key=lambda r: r["intermediation_purity_score"], reverse=True)[:10]
        ],
        "top_value_weighted_money": [
            row["label"] for row in sorted(rows, key=lambda r: r["value_weighted_monetary_score"], reverse=True)[:10]
        ],
    }


def _share(rows: list[dict[str, Any]], key: str, total: float, *, rare: bool) -> float:
    if total <= 1e-12:
        return 0.0
    return float(sum(row[key] for row in rows if bool(row["is_rare"]) is rare) / total)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
