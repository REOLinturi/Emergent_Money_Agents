from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from .long_run import run_long_simulation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run checkpointed exact chunks continuously in one Python process.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--target-cycle", type=int, default=1000)
    parser.add_argument("--chunk-cycles", type=int, default=1)
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--sleep-seconds", type=float, default=5.0)
    args = parser.parse_args(argv)

    if args.target_cycle <= 0:
        raise ValueError("target-cycle must be positive")
    if args.chunk_cycles <= 0:
        raise ValueError("chunk-cycles must be positive")
    if args.sample_every <= 0:
        raise ValueError("sample-every must be positive")

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "runner.log"
    _write_log(log_path, f"python continuous runner starting target_cycle={args.target_cycle} chunk_cycles={args.chunk_cycles}")

    while True:
        current_cycle = _checkpoint_cycle(run_dir)
        if current_cycle >= args.target_cycle:
            _write_log(log_path, f"target reached current_cycle={current_cycle}")
            break

        chunk = min(args.chunk_cycles, args.target_cycle - current_cycle)
        _write_log(log_path, f"starting chunk from_cycle={current_cycle} chunk_cycles={chunk}")
        try:
            summary = run_long_simulation(
                cycles=chunk,
                checkpoint_dir=run_dir,
                backend_name="numpy",
                checkpoint_every=chunk,
                sample_every=args.sample_every,
                resume_from=run_dir,
            )
        except Exception as exc:  # pragma: no cover - operational runner
            _write_log(log_path, f"chunk failed current_cycle={_checkpoint_cycle(run_dir)} error={type(exc).__name__}: {exc}")
            time.sleep(max(args.sleep_seconds, 1.0))
            continue

        latest = summary["latest_market"]
        phenomena = summary["phenomena"]
        _write_log(
            log_path,
            (
                f"long_run start={summary['start_cycle']} end={summary['end_cycle']} "
                f"seconds={summary['runtime_seconds']:.2f} utility={latest['utility_proxy_total']:.4f} "
                f"production={latest['production_total']:.2f} rare_money={latest['rare_goods_monetary_share']:.4f} "
                f"value_rare_money={latest.get('value_weighted_rare_goods_monetary_share', 0.0):.4f}"
            ),
        )
        _write_log(
            log_path,
            (
                f"analysis production_trend={phenomena['production_trend']:.4f} "
                f"utility_trend={phenomena['utility_trend']:.4f} cycle_strength={phenomena['cycle_strength']:.4f} "
                f"dominant_period={phenomena['dominant_cycle_length']} economy_growing={phenomena['economy_growing']} "
                f"utility_growing={phenomena['utility_growing']}"
            ),
        )
        _write_log(log_path, f"chunk completed end_cycle={summary['end_cycle']}")
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    _write_log(log_path, "python continuous runner exiting")
    return 0


def _checkpoint_cycle(run_dir: Path) -> int:
    path = run_dir / "checkpoint_latest.json"
    if not path.exists():
        return 0
    return int(json.loads(path.read_text(encoding="utf-8")).get("cycle", 0))


def _write_log(path: Path, message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")
        handle.flush()


if __name__ == "__main__":
    raise SystemExit(main())
