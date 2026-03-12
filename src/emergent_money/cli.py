from __future__ import annotations

import argparse
from pathlib import Path

from . import available_backend_names
from .config import SimulationConfig
from .dashboard import serve_dashboard
from .drift_compare import run_hybrid_consumption_comparison, run_hybrid_consumption_frontier_sweep
from .native_behavior_compare import run_native_behavior_comparison
from .native_cycle_compare import run_native_cycle_comparison
from .native_exchange_stage_compare import run_native_exchange_stage_comparison
from .native_exchange_stage_trace_compare import run_native_exchange_stage_trace_comparison
from .native_post_period_compare import run_native_post_period_comparison
from .native_search_compare import run_native_search_comparison
from .native_stage_math_trace_compare import run_native_stage_math_trace_comparison
from .long_run import run_long_simulation
from .service import SimulationService


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
    parser.add_argument("--cuda-friend-block", type=int, default=12)
    parser.add_argument("--cuda-goods-block", type=int, default=25)
    parser.add_argument("--experimental-hybrid-batches", type=int, default=0)
    parser.add_argument("--experimental-hybrid-frontier-size", type=int, default=0)
    parser.add_argument("--experimental-hybrid-seed-stride", type=int, default=9973)
    parser.add_argument("--experimental-hybrid-consumption", action="store_true")
    parser.add_argument("--experimental-hybrid-surplus", action="store_true")
    parser.add_argument("--experimental-hybrid-allow-frontier-partners", action="store_true")
    parser.add_argument("--experimental-hybrid-preserve-proposer-order", action="store_true")
    parser.add_argument("--experimental-hybrid-rolling-frontier", action="store_true")
    parser.add_argument("--experimental-native-stage-math", action="store_true")
    parser.add_argument("--experimental-native-exchange-stage", action="store_true")
    parser.add_argument("--seed", type=int, default=2009)
    parser.add_argument("--talent-probability", type=float, default=0.20)
    parser.add_argument("--dashboard", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--batch-cycles", type=int, default=1)
    parser.add_argument("--analysis", action="store_true")
    parser.add_argument("--compare-hybrid-consumption", action="store_true")
    parser.add_argument("--compare-native-search", action="store_true")
    parser.add_argument("--compare-native-cycle", action="store_true")
    parser.add_argument("--compare-native-behavior", action="store_true")
    parser.add_argument("--compare-native-exchange-stage", action="store_true")
    parser.add_argument("--compare-native-exchange-trace", action="store_true")
    parser.add_argument("--compare-native-post-period", action="store_true")
    parser.add_argument("--compare-native-stage-math-trace", action="store_true")
    parser.add_argument("--compare-seeds", nargs='+', type=int)
    parser.add_argument("--compare-frontier-sizes", nargs='+', type=int)
    parser.add_argument("--compare-native-sample-limit", type=int, default=256)
    parser.add_argument("--compare-native-benchmark-iterations", type=int, default=10)
    parser.add_argument("--compare-output")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--resume-from")
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--sample-every", type=int, default=10)
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
        cuda_friend_block=args.cuda_friend_block,
        cuda_goods_block=args.cuda_goods_block,
        experimental_hybrid_batches=args.experimental_hybrid_batches,
        experimental_hybrid_frontier_size=args.experimental_hybrid_frontier_size,
        experimental_hybrid_seed_stride=args.experimental_hybrid_seed_stride,
        experimental_hybrid_consumption_stage=args.experimental_hybrid_consumption,
        experimental_hybrid_surplus_stage=args.experimental_hybrid_surplus,
        experimental_hybrid_block_frontier_partners=not args.experimental_hybrid_allow_frontier_partners,
        experimental_hybrid_preserve_proposer_order=args.experimental_hybrid_preserve_proposer_order,
        experimental_hybrid_rolling_frontier=args.experimental_hybrid_rolling_frontier,
        experimental_native_stage_math=args.experimental_native_stage_math,
        experimental_native_exchange_stage=args.experimental_native_exchange_stage,
        seed=args.seed,
        talent_probability=args.talent_probability,
    )

    if args.dashboard:
        serve_dashboard(
            host=args.host,
            port=args.port,
            config=config,
            backend_name=args.backend,
            batch_cycles=args.batch_cycles,
        )
        return 0

    if args.compare_native_post_period:
        summary = run_native_post_period_comparison(
            cycles=args.cycles,
            seeds=args.compare_seeds or [args.seed],
            config=config,
            backend_name=args.backend,
            output_path=args.compare_output,
        )
        benchmark = summary['benchmark']
        print(
            f"native_post_period seeds={len(summary['seeds'])} cycles={summary['cycles']} "
            f"matched_seeds={summary['matched_seed_count']} mismatches={summary['mismatch_count']} "
            f"reference_seconds={benchmark['reference_seconds']:.6f} target_seconds={benchmark['target_seconds']:.6f} "
            f"speedup={benchmark['speedup_vs_reference']:.4f}"
        )
        if summary['mismatch_examples']:
            first = summary['mismatch_examples'][0]
            example = first['field_mismatch_examples'][0] if first['field_mismatch_examples'] else None
            detail = 'none'
            if example is not None:
                detail = example.get('field', str(example))
            print(
                f"first_post_period_mismatch seed={first['seed']} cycle={first['cycle']} agent={first['agent_id']} field={detail}"
            )
        if args.compare_output:
            print(f"comparison_output={Path(args.compare_output).resolve()}")
        return 0

    if args.compare_native_stage_math_trace:
        summary = run_native_stage_math_trace_comparison(
            cycles=args.cycles,
            seeds=args.compare_seeds or [args.seed],
            config=config,
            backend_name=args.backend,
            output_path=args.compare_output,
        )
        benchmark = summary['benchmark']
        print(
            f"native_stage_math_trace seeds={len(summary['seeds'])} cycles={summary['cycles']} "
            f"matched_seeds={summary['matched_seed_count']} mismatches={summary['mismatch_count']} "
            f"reference_seconds={benchmark['reference_seconds']:.6f} target_seconds={benchmark['target_seconds']:.6f} "
            f"speedup={benchmark['speedup_vs_reference']:.4f}"
        )
        if summary['mismatch_examples']:
            first = summary['mismatch_examples'][0]
            example = first['field_mismatch_examples'][0] if first['field_mismatch_examples'] else None
            detail = 'none'
            if example is not None:
                detail = example.get('field', str(example))
            print(
                f"first_stage_math_mismatch seed={first['seed']} cycle={first['cycle']} agent={first['agent_id']} stage={first['stage']} field={detail}"
            )
        if args.compare_output:
            print(f"comparison_output={Path(args.compare_output).resolve()}")
        return 0

    if args.compare_native_exchange_stage:
        summary = run_native_exchange_stage_comparison(
            cycles=args.cycles,
            seeds=args.compare_seeds or [args.seed],
            config=config,
            backend_name=args.backend,
            output_path=args.compare_output,
        )
        benchmark = summary['benchmark']
        print(
            f"native_exchange_stage seeds={len(summary['seeds'])} cycles={summary['cycles']} "
            f"matched_seeds={summary['matched_seed_count']} mismatches={summary['mismatch_count']} "
            f"reference_seconds={benchmark['reference_seconds']:.6f} target_seconds={benchmark['target_seconds']:.6f} "
            f"speedup={benchmark['speedup_vs_reference']:.4f}"
        )
        if summary['mismatch_examples']:
            first = summary['mismatch_examples'][0]
            example = first['field_mismatch_examples'][0] if first['field_mismatch_examples'] else None
            detail = 'none'
            if example is not None:
                detail = example.get('field', str(example))
            print(
                f"first_exchange_stage_mismatch seed={first['seed']} cycle={first['cycle']} agent={first['agent_id']} stage={first['stage']} field={detail}"
            )
        if args.compare_output:
            print(f"comparison_output={Path(args.compare_output).resolve()}")
        return 0

    if args.compare_native_exchange_trace:
        summary = run_native_exchange_stage_trace_comparison(
            cycles=args.cycles,
            seeds=args.compare_seeds or [args.seed],
            config=config,
            backend_name=args.backend,
            output_path=args.compare_output,
        )
        benchmark = summary['benchmark']
        print(
            f"native_exchange_trace seeds={len(summary['seeds'])} cycles={summary['cycles']} "
            f"matched_seeds={summary['matched_seed_count']} mismatches={summary['mismatch_count']} "
            f"reference_seconds={benchmark['reference_seconds']:.6f} target_seconds={benchmark['target_seconds']:.6f} "
            f"speedup={benchmark['speedup_vs_reference']:.4f}"
        )
        if summary['mismatch_examples']:
            first = summary['mismatch_examples'][0]
            print(
                f"first_trace_mismatch seed={first['seed']} cycle={first['cycle']} stage={first.get('stage', 'unknown')} "
                f"agent={first.get('agent_id', 'unknown')} reason={first['reason']}"
            )
        if args.compare_output:
            print(f"comparison_output={Path(args.compare_output).resolve()}")
        return 0

    if args.compare_native_behavior:
        summary = run_native_behavior_comparison(
            cycles=args.cycles,
            seeds=args.compare_seeds or [args.seed],
            config=config,
            backend_name=args.backend,
            output_path=args.compare_output,
        )
        benchmark = summary['benchmark']
        mean_delta = summary['mean_final_delta']
        print(
            f"native_behavior seeds={len(summary['seeds'])} cycles={summary['cycles']} "
            f"matched_seeds={summary['matched_seed_count']} mismatches={summary['mismatch_count']} "
            f"reference_seconds={benchmark['reference_seconds']:.6f} target_seconds={benchmark['target_seconds']:.6f} "
            f"speedup={benchmark['speedup_vs_reference']:.4f}"
        )
        print(
            f"mean_final_delta trade={mean_delta['accepted_trade_count']:.4f} volume={mean_delta['accepted_trade_volume']:.4f} "
            f"production={mean_delta['production_total']:.4f} utility={mean_delta['utility_proxy_total']:.4f} "
            f"rare_money={mean_delta['rare_goods_monetary_share']:.6f}"
        )
        if summary['mismatch_examples']:
            first = summary['mismatch_examples'][0]
            print(
                f"first_behavioral_mismatch seed={first['seed']} cycle={first['cycle']} deltas={first['deltas']}"
            )
        if args.compare_output:
            print(f"comparison_output={Path(args.compare_output).resolve()}")
        return 0

    if args.compare_native_cycle:
        summary = run_native_cycle_comparison(
            cycles=args.cycles,
            seeds=args.compare_seeds or [args.seed],
            config=config,
            backend_name=args.backend,
            output_path=args.compare_output,
        )
        benchmark = summary['benchmark']
        print(
            f"native_cycle seeds={len(summary['seeds'])} cycles={summary['cycles']} "
            f"matched_seeds={summary['matched_seed_count']} mismatches={summary['mismatch_count']} "
            f"python_seconds={benchmark['python_seconds']:.6f} native_seconds={benchmark['native_seconds']:.6f} "
            f"speedup={benchmark['speedup_vs_python']:.4f}"
        )
        if summary['mismatch_examples']:
            first = summary['mismatch_examples'][0]
            example = first['field_mismatch_examples'][0] if first['field_mismatch_examples'] else None
            detail = 'none'
            if example is not None:
                detail = example.get('field', str(example))
            print(
                f"first_mismatch seed={first['seed']} cycle={first['cycle']} field={detail} "
                f"snapshot_deltas={first['snapshot_deltas']}"
            )
        if args.compare_output:
            print(f"comparison_output={Path(args.compare_output).resolve()}")
        return 0
    if args.compare_native_search:
        summary = run_native_search_comparison(
            cycles=args.cycles,
            seeds=args.compare_seeds or [args.seed],
            config=config,
            backend_name=args.backend,
            sample_limit=args.compare_native_sample_limit,
            benchmark_iterations=args.compare_native_benchmark_iterations,
            output_path=args.compare_output,
        )
        comparison = summary['comparison']
        benchmark = comparison['benchmark']
        print(
            f"native_search cases={comparison['captured_calls']} mismatches={comparison['mismatch_count']} "
            f"python_seconds={benchmark['python_seconds']:.6f} native_seconds={benchmark['native_seconds']:.6f} "
            f"speedup={benchmark['speedup_vs_python']:.4f}"
        )
        if comparison['mismatch_examples']:
            first = comparison['mismatch_examples'][0]
            print(
                f"first_mismatch seed={first['seed']} cycle={first['cycle']} call_index={first['call_index']} "
                f"python={first['python_result']} native={first['native_result']}"
            )
        if args.compare_output:
            print(f"comparison_output={Path(args.compare_output).resolve()}")
        return 0

    if args.compare_hybrid_consumption:
        if args.compare_frontier_sizes:
            summary = run_hybrid_consumption_frontier_sweep(
                cycles=args.cycles,
                seeds=args.compare_seeds or [args.seed],
                config=config,
                frontier_sizes=args.compare_frontier_sizes,
                backend_name=args.backend,
                output_path=args.compare_output,
            )
            recommended = summary['recommended_frontier']
            recommended_nontrivial = summary.get('recommended_nontrivial_frontier')
            recommended_nontrivial_frontier = 'none'
            if recommended_nontrivial is not None:
                recommended_nontrivial_frontier = str(recommended_nontrivial['frontier_size'])
            print(
                f"frontier_sweep cycles={summary['cycles']} seeds={len(summary['seeds'])} frontiers={summary['frontier_sizes']} "
                f"recommended_frontier={recommended['frontier_size']} recommended_nontrivial_frontier={recommended_nontrivial_frontier}"
            )
            for comparison in summary['comparisons']:
                mean_delta = comparison['mean_delta']
                peak_cycles = comparison['peak_delta_cycles']
                print(
                    f"frontier_result frontier={comparison['frontier_size']} "
                    f"trade_delta={mean_delta['accepted_trade_count']:.4f} volume_delta={mean_delta['accepted_trade_volume']:.4f} "
                    f"production_delta={mean_delta['production_total']:.4f} utility_delta={mean_delta['utility_proxy_total']:.4f} "
                    f"peak_volume_delta={peak_cycles['accepted_trade_volume']['delta']:.4f} "
                    f"peak_production_delta={peak_cycles['production_total']['delta']:.4f}"
                )
            if args.compare_output:
                print(f"comparison_output={Path(args.compare_output).resolve()}")
            return 0

        summary = run_hybrid_consumption_comparison(
            cycles=args.cycles,
            seeds=args.compare_seeds or [args.seed],
            config=config,
            backend_name=args.backend,
            output_path=args.compare_output,
        )
        mean_delta = summary['mean_delta']
        print(
            f"comparison cycles={summary['cycles']} seeds={len(summary['seeds'])} "
            f"trade_delta={mean_delta['accepted_trade_count']:.4f} volume_delta={mean_delta['accepted_trade_volume']:.4f} "
            f"production_delta={mean_delta['production_total']:.4f} utility_delta={mean_delta['utility_proxy_total']:.4f} "
            f"rare_money_delta={mean_delta['rare_goods_monetary_share']:.4f}"
        )
        peak_cycles = summary['peak_delta_cycles']
        print(
            f"cycle_peaks trade_volume_cycle={peak_cycles['accepted_trade_volume']['cycle']} "
            f"trade_volume_delta={peak_cycles['accepted_trade_volume']['delta']:.4f} "
            f"production_cycle={peak_cycles['production_total']['cycle']} "
            f"production_delta={peak_cycles['production_total']['delta']:.4f}"
        )
        hybrid_wave = summary.get('hybrid_wave_diagnostics')
        if hybrid_wave and hybrid_wave['cycles_with_diagnostics'] > 0:
            mean_wave = hybrid_wave['mean_wave']
            no_candidate = hybrid_wave['no_candidate_reasons_total']
            execution_fail = hybrid_wave['execution_failure_reasons_total']
            print(
                f"hybrid_wave cycles={hybrid_wave['cycles_with_diagnostics']} waves={hybrid_wave['wave_count_total']} "
                f"scheduled_per_wave={mean_wave['scheduled']:.4f} executed_per_wave={mean_wave['executed']:.4f} "
                f"qty_per_scheduled={mean_wave['mean_scheduled_quantity_per_exchange']:.4f} qty_per_executed={mean_wave['mean_executed_quantity_per_exchange']:.4f} "
                f"conflicts_total={hybrid_wave['scheduler_conflict_exchanges_total']} exec_fail_total={hybrid_wave['execution_failures_total']} "
                f"retry_exhausted_total={hybrid_wave['retry_exhausted_agents_total']} stalled_waves={hybrid_wave['stalled_waves_total']} "
                f"no_candidate={no_candidate} exec_fail={execution_fail}"
            )
            stage_activation = hybrid_wave.get('stage_activation', {})
            if stage_activation:
                print(
                    f"hybrid_stage surplus_activated={stage_activation.get('surplus_activated', False)} "
                    f"stage_cycles={stage_activation.get('stage_cycle_counts', {})} "
                    f"stage_waves={stage_activation.get('stage_wave_counts', {})} "
                    f"first_cycle_overall={stage_activation.get('first_cycle_overall', {})}"
                )
        if args.compare_output:
            print(f"comparison_output={Path(args.compare_output).resolve()}")
        return 0

    if args.checkpoint_dir or args.resume_from:
        checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else Path(args.resume_from)
        if args.resume_from and checkpoint_dir.suffix:
            checkpoint_dir = checkpoint_dir.parent
        summary = run_long_simulation(
            cycles=args.cycles,
            checkpoint_dir=checkpoint_dir,
            config=None if args.resume_from else config,
            backend_name=args.backend,
            checkpoint_every=args.checkpoint_every,
            sample_every=args.sample_every,
            resume_from=args.resume_from,
        )
        latest = summary["latest_market"]
        phenomena = summary["phenomena"]
        print(
            f"long_run start={summary['start_cycle']} end={summary['end_cycle']} "
            f"seconds={summary['runtime_seconds']:.2f} utility={latest['utility_proxy_total']:.4f} "
            f"production={latest['production_total']:.2f} rare_money={latest['rare_goods_monetary_share']:.4f}"
        )
        print(
            f"analysis production_trend={phenomena['production_trend']:.4f} utility_trend={phenomena['utility_trend']:.4f} "
            f"cycle_strength={phenomena['cycle_strength']:.4f} dominant_period={phenomena['dominant_cycle_length']} "
            f"economy_growing={phenomena['economy_growing']} utility_growing={phenomena['utility_growing']}"
        )
        print(f"artifacts summary={summary['artifacts']['summary_json']} checkpoint={summary['artifacts']['checkpoint_json']}")
        return 0

    service = SimulationService.create(config=config, backend_name=args.backend)
    snapshots = service.step(args.cycles)
    for snapshot in snapshots:
        print(
            f"cycle={snapshot.cycle} fulfilled={snapshot.fulfilled_share:.4f} "
            f"utility={snapshot.utility_proxy_total:.2f} production={snapshot.production_total:.2f} "
            f"stock={snapshot.stock_total:.2f} proposed_trades={snapshot.proposed_trade_count} "
            f"accepted_trades={snapshot.accepted_trade_count} accepted_volume={snapshot.accepted_trade_volume:.2f} "
            f"rare_money={snapshot.rare_goods_monetary_share:.4f}"
        )

    if args.analysis:
        phenomena = service.get_phenomena_snapshot(top_goods=5)
        goods = service.get_goods_snapshot(limit=5)
        print(
            f"analysis cycles={phenomena.cycles_observed} production_trend={phenomena.production_trend:.4f} "
            f"utility_trend={phenomena.utility_trend:.4f} cycle_strength={phenomena.cycle_strength:.4f} "
            f"dominant_period={phenomena.dominant_cycle_length} economy_growing={phenomena.economy_growing} "
            f"utility_growing={phenomena.utility_growing}"
        )
        if goods:
            top = ", ".join(
                f"g{item.good_id}:score={item.monetary_score:.3f}:rare={int(item.is_rare)}" for item in goods
            )
            print(f"top_monetary_goods={top}")

    return 0


