from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import SimulationConfig
from .engine import SimulationEngine
from .legacy_cycle import LegacyCycleRunner
from .legacy_search_backend import (
    ExchangePlanRequest,
    ExchangePlanningOutcome,
    ExchangeSearchBackend,
    ExchangeSearchRequest,
    ExchangeSearchResult,
    build_native_exchange_search_backend,
    build_python_exchange_search_backend,
    execute_exchange_search,
)

_SCORE_ATOL = 1e-6


@dataclass(slots=True, frozen=True)
class RecordedExchangeSearchCall:
    seed: int
    cycle: int
    call_index: int
    request: ExchangeSearchRequest
    python_result: ExchangeSearchResult | None


class TracingExchangeSearchBackend:
    name = 'python-tracing'
    is_native = False

    def __init__(
        self,
        delegate: ExchangeSearchBackend,
        *,
        seed: int,
        sample_limit: int,
    ) -> None:
        self._delegate = delegate
        self._seed = seed
        self._sample_limit = sample_limit
        self._cycle = 0
        self._call_index = 0
        self.calls: list[RecordedExchangeSearchCall] = []

    def start_cycle(self, cycle: int) -> None:
        self._cycle = cycle

    def _record_call(
        self,
        request: ExchangeSearchRequest,
        result: ExchangeSearchResult | None,
    ) -> None:
        if len(self.calls) < self._sample_limit:
            self.calls.append(
                RecordedExchangeSearchCall(
                    seed=self._seed,
                    cycle=self._cycle,
                    call_index=self._call_index,
                    request=_copy_request(request),
                    python_result=result,
                )
            )
        self._call_index += 1

    def find_best_exchange(self, **kwargs) -> ExchangeSearchResult | None:
        request = ExchangeSearchRequest(**kwargs)
        result = execute_exchange_search(self._delegate, request)
        self._record_call(request, result)
        return result

    def plan_best_exchange(self, request: ExchangePlanRequest) -> ExchangePlanningOutcome | None:
        outcome = self._delegate.plan_best_exchange(request)
        recorded_result = None if outcome is None else outcome.search_result
        self._record_call(request.search_request, recorded_result)
        return outcome


def run_native_search_comparison(
    *,
    cycles: int,
    seeds: list[int],
    config: SimulationConfig,
    backend_name: str = 'numpy',
    sample_limit: int = 256,
    benchmark_iterations: int = 10,
    output_path: str | Path | None = None,
    python_backend: ExchangeSearchBackend | None = None,
    native_backend: ExchangeSearchBackend | None = None,
) -> dict[str, Any]:
    native_search_backend = native_backend or build_native_exchange_search_backend()
    if native_search_backend is None:
        raise RuntimeError(
            'native exchange search backend is not available; build the optional '
            'Rust module before running compare-native-search'
        )
    python_search_backend = python_backend or build_python_exchange_search_backend()
    capture = capture_exchange_search_calls(
        cycles=cycles,
        seeds=seeds,
        config=config,
        backend_name=backend_name,
        sample_limit=sample_limit,
        python_backend=python_search_backend,
    )
    if capture['captured_calls'] <= 0:
        raise RuntimeError(
            'native exchange search comparison captured no search calls; '
            'increase cycles or use a richer population/network configuration'
        )
    comparison = compare_recorded_exchange_search_calls(
        calls=capture['calls'],
        python_backend=python_search_backend,
        native_backend=native_search_backend,
        benchmark_iterations=benchmark_iterations,
    )
    summary = {
        'cycles': cycles,
        'seeds': seeds,
        'config': asdict(config),
        'capture': {
            'sample_limit': sample_limit,
            'captured_calls': capture['captured_calls'],
            'captured_calls_by_seed': capture['captured_calls_by_seed'],
            'capture_runtime_seconds': capture['capture_runtime_seconds'],
        },
        'comparison': comparison,
    }
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding='utf-8')
    return summary


def capture_exchange_search_calls(
    *,
    cycles: int,
    seeds: list[int],
    config: SimulationConfig,
    backend_name: str = 'numpy',
    sample_limit: int = 256,
    python_backend: ExchangeSearchBackend | None = None,
) -> dict[str, Any]:
    if cycles <= 0:
        raise ValueError('cycles must be positive')
    if not seeds:
        raise ValueError('seeds must not be empty')
    if sample_limit <= 0:
        raise ValueError('sample_limit must be positive')

    search_backend = python_backend or build_python_exchange_search_backend()
    started_at = time.perf_counter()
    calls: list[RecordedExchangeSearchCall] = []
    captured_calls_by_seed: dict[str, int] = {}

    for seed in seeds:
        seed_config = _native_compare_config(config, seed=seed)
        engine = SimulationEngine.create(config=seed_config, backend_name=backend_name)
        tracer = TracingExchangeSearchBackend(search_backend, seed=seed, sample_limit=sample_limit)
        for _ in range(cycles):
            tracer.start_cycle(engine.cycle + 1)
            _run_exact_cycle(engine, exchange_search_backend=tracer)
        calls.extend(tracer.calls)
        captured_calls_by_seed[str(seed)] = len(tracer.calls)

    return {
        'calls': calls,
        'captured_calls': len(calls),
        'captured_calls_by_seed': captured_calls_by_seed,
        'capture_runtime_seconds': time.perf_counter() - started_at,
    }


def compare_recorded_exchange_search_calls(
    *,
    calls: list[RecordedExchangeSearchCall],
    python_backend: ExchangeSearchBackend,
    native_backend: ExchangeSearchBackend,
    benchmark_iterations: int = 10,
) -> dict[str, Any]:
    if not calls:
        raise ValueError('calls must not be empty')
    if benchmark_iterations <= 0:
        raise ValueError('benchmark_iterations must be positive')

    started_at = time.perf_counter()
    mismatches: list[dict[str, Any]] = []
    for call in calls:
        native_result = execute_exchange_search(native_backend, call.request)
        if _results_match(call.python_result, native_result):
            continue
        mismatches.append(
            {
                'seed': call.seed,
                'cycle': call.cycle,
                'call_index': call.call_index,
                'python_result': _result_payload(call.python_result),
                'native_result': _result_payload(native_result),
            }
        )
    validation_runtime_seconds = time.perf_counter() - started_at

    python_seconds = _benchmark_backend(
        backend=python_backend,
        calls=calls,
        benchmark_iterations=benchmark_iterations,
    )
    native_seconds = _benchmark_backend(
        backend=native_backend,
        calls=calls,
        benchmark_iterations=benchmark_iterations,
    )
    speedup = 0.0
    if native_seconds > 0.0:
        speedup = python_seconds / native_seconds

    return {
        'captured_calls': len(calls),
        'mismatch_count': len(mismatches),
        'mismatch_rate': float(len(mismatches)) / float(len(calls)),
        'mismatch_examples': mismatches[:10],
        'validation_runtime_seconds': validation_runtime_seconds,
        'benchmark_iterations': benchmark_iterations,
        'python_backend_name': python_backend.name,
        'native_backend_name': native_backend.name,
        'benchmark': {
            'python_seconds': python_seconds,
            'native_seconds': native_seconds,
            'speedup_vs_python': speedup,
            'python_calls_per_second': _calls_per_second(len(calls), benchmark_iterations, python_seconds),
            'native_calls_per_second': _calls_per_second(len(calls), benchmark_iterations, native_seconds),
        },
    }


def _copy_request(request: ExchangeSearchRequest) -> ExchangeSearchRequest:
    return ExchangeSearchRequest(
        goods=request.goods,
        need_good=request.need_good,
        initial_transparency=request.initial_transparency,
        elastic_need=_copy_array(request.elastic_need),
        candidate_offer_goods=_copy_array(request.candidate_offer_goods),
        friend_ids=_copy_array(request.friend_ids),
        reciprocal_slots=_copy_array(request.reciprocal_slots),
        my_stock=_copy_array(request.my_stock),
        my_sales_price=_copy_array(request.my_sales_price),
        my_purchase_price=_copy_array(request.my_purchase_price),
        my_role=_copy_array(request.my_role),
        my_transparency=_copy_array(request.my_transparency),
        my_needs_level=request.my_needs_level,
        stock=_copy_array(request.stock),
        role=_copy_array(request.role),
        stock_limit=_copy_array(request.stock_limit),
        purchase_price=_copy_array(request.purchase_price),
        sales_price=_copy_array(request.sales_price),
        needs_level=_copy_array(request.needs_level),
        transparency=_copy_array(request.transparency),
    )


def _copy_array(array: np.ndarray) -> np.ndarray:
    return np.array(array, copy=True, order='C')


def _run_exact_cycle(engine: SimulationEngine, *, exchange_search_backend: ExchangeSearchBackend) -> None:
    runner = LegacyCycleRunner(engine, exchange_search_backend=exchange_search_backend)
    runner.run()
    engine.backend.synchronize()
    engine.cycle += 1
    snapshot = engine.snapshot_metrics()
    if engine.config.use_exact_legacy_mechanics:
        engine.state.market.total_stock_previous = snapshot.stock_total
    engine.history.append(snapshot)


def _native_compare_config(config: SimulationConfig, *, seed: int) -> SimulationConfig:
    payload = asdict(config)
    payload['seed'] = seed
    payload['experimental_hybrid_batches'] = 0
    payload['experimental_hybrid_frontier_size'] = 0
    payload['experimental_hybrid_consumption_stage'] = False
    payload['experimental_hybrid_surplus_stage'] = False
    return SimulationConfig(**payload)


def _results_match(
    python_result: ExchangeSearchResult | None,
    native_result: ExchangeSearchResult | None,
) -> bool:
    if python_result is None or native_result is None:
        return python_result is None and native_result is None
    return (
        python_result.friend_slot == native_result.friend_slot
        and python_result.friend_id == native_result.friend_id
        and python_result.offer_good == native_result.offer_good
        and bool(np.isclose(python_result.score, native_result.score, atol=_SCORE_ATOL, rtol=0.0))
    )


def _result_payload(result: ExchangeSearchResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        'score': result.score,
        'friend_slot': result.friend_slot,
        'friend_id': result.friend_id,
        'offer_good': result.offer_good,
    }


def _benchmark_backend(
    *,
    backend: ExchangeSearchBackend,
    calls: list[RecordedExchangeSearchCall],
    benchmark_iterations: int,
) -> float:
    started_at = time.perf_counter()
    for _ in range(benchmark_iterations):
        for call in calls:
            execute_exchange_search(backend, call.request)
    return time.perf_counter() - started_at


def _calls_per_second(call_count: int, iterations: int, runtime_seconds: float) -> float:
    total_calls = call_count * iterations
    if runtime_seconds <= 0.0:
        return 0.0
    return float(total_calls) / runtime_seconds
