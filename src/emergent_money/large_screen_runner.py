from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .artifact_analysis import summarize_run_artifact
from .config import SimulationConfig
from .long_run import run_long_simulation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run large checkpointed screening jobs sequentially.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2009, 2011, 2013])
    parser.add_argument("--target-cycle", type=int, default=2000)
    parser.add_argument("--chunk-cycles", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--sleep-seconds", type=float, default=2.0)
    parser.add_argument("--backend", default="numpy")
    parser.add_argument("--population", type=int, default=10000)
    parser.add_argument("--goods", type=int, default=100)
    parser.add_argument("--acquaintances", type=int, default=150)
    parser.add_argument("--active-acquaintances", type=int, default=24)
    parser.add_argument("--demand-candidates", type=int, default=4)
    parser.add_argument("--supply-candidates", type=int, default=4)
    parser.add_argument("--experimental-native-stage-math", action="store_true")
    parser.add_argument("--experimental-native-exchange-stage", action="store_true")
    parser.add_argument("--uncompressed-checkpoint", action="store_true")
    args = parser.parse_args(argv)

    if args.target_cycle <= 0:
        raise ValueError("target-cycle must be positive")
    if args.chunk_cycles <= 0:
        raise ValueError("chunk-cycles must be positive")
    if args.checkpoint_every <= 0:
        raise ValueError("checkpoint-every must be positive")
    if args.sample_every <= 0:
        raise ValueError("sample-every must be positive")

    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    root_log = run_root / "large_screen_runner.log"
    _write_log(root_log, f"runner starting seeds={args.seeds} target_cycle={args.target_cycle}")

    for seed in args.seeds:
        run_dir = _run_dir(args=args, seed=seed)
        run_dir.mkdir(parents=True, exist_ok=True)
        run_log = run_dir / "large_screen_runner.log"
        _write_log(run_log, f"seed starting seed={seed} target_cycle={args.target_cycle}")
        _run_seed(args=args, seed=seed, run_dir=run_dir, run_log=run_log)
        _write_log(root_log, f"seed completed seed={seed} run_dir={run_dir}")

    _write_log(root_log, "runner exiting")
    return 0


def _run_seed(*, args: argparse.Namespace, seed: int, run_dir: Path, run_log: Path) -> None:
    while True:
        current_cycle = _checkpoint_cycle(run_dir)
        if current_cycle >= args.target_cycle:
            _write_log(run_log, f"target reached current_cycle={current_cycle}")
            _write_artifact_summary(run_dir=run_dir, run_log=run_log)
            return

        chunk = min(args.chunk_cycles, args.target_cycle - current_cycle)
        _write_log(run_log, f"starting chunk from_cycle={current_cycle} chunk_cycles={chunk}")
        try:
            started_at = time.perf_counter()
            summary = run_long_simulation(
                cycles=chunk,
                checkpoint_dir=run_dir,
                config=_config_from_args(args, seed) if current_cycle <= 0 else None,
                backend_name=args.backend,
                checkpoint_every=min(args.checkpoint_every, chunk),
                sample_every=min(args.sample_every, chunk),
                resume_from=run_dir if current_cycle > 0 else None,
                compress_checkpoint=not args.uncompressed_checkpoint,
            )
        except Exception as exc:  # pragma: no cover - operational runner
            _write_log(
                run_log,
                f"chunk failed current_cycle={_checkpoint_cycle(run_dir)} error={type(exc).__name__}: {exc}",
            )
            time.sleep(max(args.sleep_seconds, 1.0))
            continue

        latest = summary["latest_market"]
        phenomena = summary["phenomena"]
        elapsed = time.perf_counter() - started_at
        _write_log(
            run_log,
            (
                f"chunk completed start={summary['start_cycle']} end={summary['end_cycle']} "
                f"seconds={elapsed:.2f} utility={latest['utility_proxy_total']:.4f} "
                f"production={latest['production_total']:.2f} rare_money={latest['rare_goods_monetary_share']:.4f}"
            ),
        )
        _write_log(
            run_log,
            (
                f"analysis production_trend={phenomena['production_trend']:.4f} "
                f"utility_trend={phenomena['utility_trend']:.4f} cycle_strength={phenomena['cycle_strength']:.4f} "
                f"dominant_period={phenomena['dominant_cycle_length']} economy_growing={phenomena['economy_growing']} "
                f"utility_growing={phenomena['utility_growing']}"
            ),
        )
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)


def _config_from_args(args: argparse.Namespace, seed: int) -> SimulationConfig:
    return SimulationConfig(
        population=args.population,
        goods=args.goods,
        acquaintances=args.acquaintances,
        active_acquaintances=args.active_acquaintances,
        demand_candidates=args.demand_candidates,
        supply_candidates=args.supply_candidates,
        seed=seed,
        experimental_native_stage_math=args.experimental_native_stage_math,
        experimental_native_exchange_stage=args.experimental_native_exchange_stage,
    )


def _run_dir(*, args: argparse.Namespace, seed: int) -> Path:
    return (
        Path(args.run_root)
        / f"large_fast_{args.population}_{args.goods}_{args.acquaintances}_{args.target_cycle}_seed{seed}"
    )


def _checkpoint_cycle(run_dir: Path) -> int:
    path = run_dir / "checkpoint_latest.json"
    if not path.exists():
        return 0
    return int(json.loads(path.read_text(encoding="utf-8")).get("cycle", 0))


def _write_artifact_summary(*, run_dir: Path, run_log: Path) -> None:
    try:
        summary = summarize_run_artifact(run_dir)
    except Exception as exc:  # pragma: no cover - operational runner
        _write_log(run_log, f"artifact summary failed error={type(exc).__name__}: {exc}")
        return

    summary_json_path = run_dir / "artifact_summary_final.json"
    summary_txt_path = run_dir / "artifact_summary_final.txt"
    summary_json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_txt_path.write_text(_format_artifact_summary(summary), encoding="utf-8")
    _write_log(run_log, f"artifact summary written path={summary_json_path}")


def _format_artifact_summary(summary: dict[str, Any]) -> str:
    flags = summary.get("phenomenon_flags", {})
    fields = summary.get("fields", {})
    lines = [
        (
            f"artifact_summary run={summary.get('run_dir')} samples={summary.get('sample_count')} "
            f"cycles={summary.get('first_cycle')}..{summary.get('last_cycle')} "
            f"production_grew={flags.get('production_grew')} rare_money_emerged={flags.get('rare_money_emerged')} "
            f"utility_peaked_before_end={flags.get('utility_peaked_before_end')} friction_rose={flags.get('friction_rose')} "
            f"living_inequality_high={flags.get('living_standard_inequality_high')}"
        )
    ]
    for name in (
        "production_total",
        "utility_proxy_total",
        "accepted_trade_volume",
        "rare_goods_monetary_share",
        "friction_share_of_time_budget",
        "tce_share_of_time_budget",
        "stock_total",
    ):
        field = fields.get(name)
        if field:
            lines.append(
                (
                    f"{name}: last={_format_optional_float(field['last'])} "
                    f"max={_format_optional_float(field['max'])} "
                    f"max_cycle={field['max_cycle']} "
                    f"last_to_peak={_format_optional_float(field['last_to_peak'])}"
                )
            )
    return "\n".join(lines) + "\n"


def _format_optional_float(value: Any) -> str:
    if value is None:
        return "None"
    return f"{float(value):.6g}"


def _write_log(path: Path, message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")
        handle.flush()


if __name__ == "__main__":
    raise SystemExit(main())
