from __future__ import annotations

import argparse

from . import available_backend_names
from .config import SimulationConfig
from .engine import SimulationEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Emergent Money scaffold")
    parser.add_argument("--backend", default="numpy", choices=available_backend_names())
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--population", type=int, default=256)
    parser.add_argument("--goods", type=int, default=12)
    parser.add_argument("--acquaintances", type=int, default=24)
    parser.add_argument("--active-acquaintances", type=int, default=8)
    parser.add_argument("--demand-candidates", type=int, default=4)
    parser.add_argument("--supply-candidates", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2009)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = SimulationConfig(
        population=args.population,
        goods=args.goods,
        acquaintances=args.acquaintances,
        active_acquaintances=args.active_acquaintances,
        demand_candidates=args.demand_candidates,
        supply_candidates=args.supply_candidates,
        seed=args.seed,
    )
    engine = SimulationEngine.create(config=config, backend_name=args.backend)

    for _ in range(args.cycles):
        snapshot = engine.step()
        print(
            f"cycle={snapshot.cycle} fulfilled={snapshot.fulfilled_share:.4f} "
            f"unmet={snapshot.unmet_need_total:.2f} stock={snapshot.stock_total:.2f} "
            f"avg_eff={snapshot.average_efficiency:.4f} proposed_trades={snapshot.proposed_trade_count} "
            f"accepted_trades={snapshot.accepted_trade_count} accepted_volume={snapshot.accepted_trade_volume:.2f}"
        )

    return 0
