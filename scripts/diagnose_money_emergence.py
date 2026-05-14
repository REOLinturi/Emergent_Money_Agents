from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from emergent_money.analytics import compute_good_snapshots, compute_inequality_snapshot  # noqa: E402
from emergent_money.config import SimulationConfig  # noqa: E402
from emergent_money.engine import SimulationEngine  # noqa: E402


@dataclass(frozen=True)
class Variant:
    name: str
    acquaintances: int | None = None
    initial_transparency: float | None = None
    price_demand_elasticity: int | None = None
    reserve_bias: float | None = None
    reserve_min_acceptance: float | None = None
    base_good_id_stride: int | None = None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run small rare-money emergence diagnostics.")
    parser.add_argument("--population", type=int, default=300)
    parser.add_argument("--goods", type=int, default=40)
    parser.add_argument("--base-acquaintances", type=int)
    parser.add_argument("--cycles", type=int, default=60)
    parser.add_argument("--seed", type=int, default=2009)
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--variant", action="append", help="Run only named variants; repeatable.")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "tmp" / f"money_emergence_diagnostics_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = _variants()
    if args.variant:
        wanted = set(args.variant)
        variants = [variant for variant in variants if variant.name in wanted]
        missing = wanted - {variant.name for variant in variants}
        if missing:
            raise SystemExit(f"unknown variant(s): {', '.join(sorted(missing))}")

    results: list[dict[str, Any]] = []
    for variant in variants:
        started = time.perf_counter()
        result = _run_variant(
            variant,
            population=args.population,
            goods=args.goods,
            base_acquaintances=args.base_acquaintances,
            cycles=args.cycles,
            seed=args.seed,
            sample_every=args.sample_every,
        )
        result["runtime_seconds"] = time.perf_counter() - started
        results.append(result)
        print(
            "variant "
            f"{variant.name}: c{result['last_cycle']} "
            f"ls={result['final_living_standard_mean']:.3f} "
            f"rare_money_last={result['final_rare_money_share']:.3f} "
            f"rare_xmedia_last={result['final_rare_exchange_media_share']:.3f} "
            f"rare_xmedia_peak={result['peak_rare_exchange_media_share']:.3f}@{result['peak_rare_exchange_media_cycle']} "
            f"top_xmedia={','.join(result['top_exchange_media_goods'][:5])}"
        )

    (output_dir / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(output_dir / "results.csv", results)
    print(f"diagnostic_output={output_dir.resolve()}")
    return 0


def _variants() -> list[Variant]:
    return [
        Variant("baseline"),
        Variant("friends_12", acquaintances=12),
        Variant("friends_20", acquaintances=20),
        Variant("friends_24", acquaintances=24),
        Variant("friends_30", acquaintances=30),
        Variant("friends_50", acquaintances=50),
        Variant("friends_80", acquaintances=80),
        Variant("friends_100", acquaintances=100),
        Variant("transparency_03", initial_transparency=0.30),
        Variant("transparency_05", initial_transparency=0.50),
        Variant("elasticity_0", price_demand_elasticity=0),
        Variant("elasticity_1", price_demand_elasticity=1),
        Variant("reserve_0", reserve_bias=0.0),
        Variant("reserve_1", reserve_bias=1.0),
        Variant("reserve_min_05", reserve_min_acceptance=0.5),
        Variant("stride_3", base_good_id_stride=3),
    ]


def _run_variant(
    variant: Variant,
    *,
    population: int,
    goods: int,
    base_acquaintances: int | None,
    cycles: int,
    seed: int,
    sample_every: int,
) -> dict[str, Any]:
    acquaintances = variant.acquaintances if variant.acquaintances is not None else (base_acquaintances or min(goods, 40))
    acquaintances = max(1, min(acquaintances, max(population - 1, 1)))
    config = SimulationConfig(
        population=population,
        goods=goods,
        acquaintances=acquaintances,
        active_acquaintances=acquaintances,
        demand_candidates=goods,
        supply_candidates=goods,
        seed=seed,
        base_good_id_stride=variant.base_good_id_stride if variant.base_good_id_stride is not None else 1,
        initial_transparency=variant.initial_transparency if variant.initial_transparency is not None else 0.70,
        price_demand_elasticity=(
            variant.price_demand_elasticity if variant.price_demand_elasticity is not None else 2
        ),
        experimental_native_stage_math=True,
        experimental_agent_basket_planning=True,
        experimental_session_replan_after_trade=True,
        experimental_session_candidate_depth=1,
        experimental_local_liquidity_stock_bias=1.0,
        experimental_aspirational_stock_target=2.0,
        experimental_exchange_media_reserve_bias=variant.reserve_bias if variant.reserve_bias is not None else 0.5,
        experimental_exchange_media_reserve_min_acceptance=(
            variant.reserve_min_acceptance if variant.reserve_min_acceptance is not None else 2.0
        ),
        experimental_exchange_media_reserve_bootstrap_floor=1.0,
    )
    engine = SimulationEngine.create(config=config, backend_name="numpy")

    samples: list[dict[str, float | int]] = []
    for _ in range(cycles):
        metrics = engine.step()
        if metrics.cycle % sample_every == 0 or metrics.cycle == cycles:
            inequality = compute_inequality_snapshot(state=engine.state, backend=engine.backend, config=config)
            samples.append(
                {
                    "cycle": metrics.cycle,
                    "rare_money_share": metrics.rare_goods_monetary_share,
                    "value_rare_money_share": metrics.value_weighted_rare_goods_monetary_share,
                    "rare_exchange_media_share": metrics.rare_goods_exchange_media_share,
                    "living_standard_mean": inequality.living_standard_mean,
                    "living_standard_median": inequality.living_standard_median,
                    "production_total": metrics.production_total,
                    "friction_share_of_output_value": inequality.friction_share_of_output_value,
                    "network_density": metrics.network_density,
                }
            )

    goods_snapshot = compute_good_snapshots(state=engine.state, backend=engine.backend, config=config, limit=None)
    rare_goods = [item for item in goods_snapshot if item.is_rare]
    common_goods = [item for item in goods_snapshot if not item.is_rare]
    top_exchange = sorted(goods_snapshot, key=lambda item: item.exchange_media_score, reverse=True)[:10]
    top_money = sorted(goods_snapshot, key=lambda item: item.monetary_score, reverse=True)[:10]

    final = samples[-1]
    rare_exchange = _series(samples, "rare_exchange_media_share")
    rare_money = _series(samples, "rare_money_share")
    value_rare_money = _series(samples, "value_rare_money_share")
    transparency = _rare_common_transparency(engine)

    return {
        "variant": variant.name,
        "parameters": {
            "population": population,
            "goods": goods,
            "cycles": cycles,
            "seed": seed,
            "acquaintances": config.acquaintances,
            "initial_transparency": config.initial_transparency,
            "price_demand_elasticity": config.price_demand_elasticity,
            "reserve_bias": config.experimental_exchange_media_reserve_bias,
            "reserve_min_acceptance": config.experimental_exchange_media_reserve_min_acceptance,
            "base_good_id_stride": config.base_good_id_stride,
        },
        "last_cycle": int(final["cycle"]),
        "final_rare_money_share": float(final["rare_money_share"]),
        "final_value_rare_money_share": float(final["value_rare_money_share"]),
        "final_rare_exchange_media_share": float(final["rare_exchange_media_share"]),
        "peak_rare_money_share": rare_money["max"],
        "peak_rare_money_cycle": rare_money["cycle"],
        "peak_value_rare_money_share": value_rare_money["max"],
        "peak_value_rare_money_cycle": value_rare_money["cycle"],
        "peak_rare_exchange_media_share": rare_exchange["max"],
        "peak_rare_exchange_media_cycle": rare_exchange["cycle"],
        "final_living_standard_mean": float(final["living_standard_mean"]),
        "final_production_total": float(final["production_total"]),
        "final_friction_share_of_output_value": float(final["friction_share_of_output_value"]),
        "network_density": float(final["network_density"]),
        "rare_top_exchange_media_count": int(sum(1 for item in top_exchange if item.is_rare)),
        "rare_top_money_count": int(sum(1 for item in top_money if item.is_rare)),
        "rare_avg_exchange_media_score": float(np.mean([item.exchange_media_score for item in rare_goods] or [0.0])),
        "common_avg_exchange_media_score": float(np.mean([item.exchange_media_score for item in common_goods] or [0.0])),
        "rare_avg_monetary_score": float(np.mean([item.monetary_score for item in rare_goods] or [0.0])),
        "common_avg_monetary_score": float(np.mean([item.monetary_score for item in common_goods] or [0.0])),
        "rare_avg_endogenous_standardization_score": float(
            np.mean([item.endogenous_standardization_score for item in rare_goods] or [0.0])
        ),
        "common_avg_endogenous_standardization_score": float(
            np.mean([item.endogenous_standardization_score for item in common_goods] or [0.0])
        ),
        "rare_avg_top_seller_breadth_share": float(
            np.mean([item.top_seller_breadth_share for item in rare_goods] or [0.0])
        ),
        "common_avg_top_seller_breadth_share": float(
            np.mean([item.top_seller_breadth_share for item in common_goods] or [0.0])
        ),
        "rare_avg_seller_specialization_score": float(
            np.mean([item.seller_specialization_score for item in rare_goods] or [0.0])
        ),
        "common_avg_seller_specialization_score": float(
            np.mean([item.seller_specialization_score for item in common_goods] or [0.0])
        ),
        "rare_avg_intermediation_purity_score": float(
            np.mean([item.intermediation_purity_score for item in rare_goods] or [0.0])
        ),
        "common_avg_intermediation_purity_score": float(
            np.mean([item.intermediation_purity_score for item in common_goods] or [0.0])
        ),
        "rare_avg_transparency": transparency["rare_avg_transparency"],
        "common_avg_transparency": transparency["common_avg_transparency"],
        "top_exchange_media_goods": [_good_label(item) for item in top_exchange],
        "top_money_goods": [_good_label(item) for item in top_money],
        "samples": samples,
    }


def _series(samples: list[dict[str, float | int]], key: str) -> dict[str, float | int]:
    values = np.asarray([float(sample[key]) for sample in samples], dtype=np.float64)
    cycles = np.asarray([int(sample["cycle"]) for sample in samples], dtype=np.int64)
    index = int(np.argmax(values))
    return {"max": float(values[index]), "cycle": int(cycles[index])}


def _rare_common_transparency(engine: SimulationEngine) -> dict[str, float]:
    state = engine.state
    friend_id = engine.backend.to_numpy(state.friend_id)
    transparency = engine.backend.to_numpy(state.transparency).astype(np.float64, copy=False)
    known = friend_id >= 0
    if not np.any(known):
        return {"rare_avg_transparency": 0.0, "common_avg_transparency": 0.0}
    base_need = engine.backend.to_numpy(state.base_need)[0]
    order = np.argsort(base_need, kind="stable")
    rare_cutoff = max(1, int(np.ceil(base_need.size / 4.0)))
    rare_mask = np.zeros(base_need.size, dtype=bool)
    rare_mask[order[:rare_cutoff]] = True
    known_transparency = transparency[known]
    rare_values = known_transparency[:, rare_mask]
    common_values = known_transparency[:, ~rare_mask]
    return {
        "rare_avg_transparency": float(np.mean(rare_values)) if rare_values.size else 0.0,
        "common_avg_transparency": float(np.mean(common_values)) if common_values.size else 0.0,
    }


def _good_label(item: Any) -> str:
    report_id = item.report_good_id if item.report_good_id is not None else item.good_id
    return f"g{report_id}:r{item.demand_rank}:x{item.exchange_media_score:.3f}:m{item.monetary_score:.3f}"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "variant",
        "last_cycle",
        "final_living_standard_mean",
        "final_rare_money_share",
        "final_value_rare_money_share",
        "final_rare_exchange_media_share",
        "peak_rare_money_share",
        "peak_rare_money_cycle",
        "peak_rare_exchange_media_share",
        "peak_rare_exchange_media_cycle",
        "rare_top_exchange_media_count",
        "rare_top_money_count",
        "rare_avg_exchange_media_score",
        "common_avg_exchange_media_score",
        "rare_avg_monetary_score",
        "common_avg_monetary_score",
        "rare_avg_endogenous_standardization_score",
        "common_avg_endogenous_standardization_score",
        "rare_avg_top_seller_breadth_share",
        "common_avg_top_seller_breadth_share",
        "rare_avg_seller_specialization_score",
        "common_avg_seller_specialization_score",
        "rare_avg_intermediation_purity_score",
        "common_avg_intermediation_purity_score",
        "rare_avg_transparency",
        "common_avg_transparency",
        "network_density",
        "runtime_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


if __name__ == "__main__":
    raise SystemExit(main())
