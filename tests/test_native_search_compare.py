from __future__ import annotations

import numpy as np

from emergent_money.config import SimulationConfig
from emergent_money.legacy_search_backend import (
    ExchangeSearchRequest,
    ExchangeSearchResult,
    PythonExchangeSearchBackend,
    execute_exchange_search,
)
from emergent_money.native_search_compare import (
    RecordedExchangeSearchCall,
    compare_recorded_exchange_search_calls,
    run_native_search_comparison,
)
from emergent_money.state import ROLE_CONSUMER, ROLE_RETAILER


class _MatchingNativeBackend:
    name = 'native-fake'
    is_native = True

    def __init__(self) -> None:
        self._delegate = PythonExchangeSearchBackend()

    def find_best_exchange(self, **kwargs) -> ExchangeSearchResult | None:
        return self._delegate.find_best_exchange(**kwargs)


class _MismatchingNativeBackend:
    name = 'native-fake'
    is_native = True

    def __init__(self) -> None:
        self._delegate = PythonExchangeSearchBackend()

    def find_best_exchange(self, **kwargs) -> ExchangeSearchResult | None:
        result = self._delegate.find_best_exchange(**kwargs)
        if result is None:
            return None
        return ExchangeSearchResult(
            score=result.score,
            friend_slot=result.friend_slot,
            friend_id=result.friend_id + 1,
            offer_good=result.offer_good,
        )


def test_compare_recorded_exchange_search_calls_accepts_matching_backend() -> None:
    python_backend = PythonExchangeSearchBackend()
    request = _sample_request()
    call = RecordedExchangeSearchCall(
        seed=2009,
        cycle=1,
        call_index=0,
        request=request,
        python_result=execute_exchange_search(python_backend, request),
    )

    summary = compare_recorded_exchange_search_calls(
        calls=[call],
        python_backend=python_backend,
        native_backend=_MatchingNativeBackend(),
        benchmark_iterations=2,
    )

    assert summary['captured_calls'] == 1
    assert summary['mismatch_count'] == 0
    assert summary['benchmark']['python_seconds'] >= 0.0
    assert summary['benchmark']['native_seconds'] >= 0.0


def test_compare_recorded_exchange_search_calls_reports_mismatch_details() -> None:
    python_backend = PythonExchangeSearchBackend()
    request = _sample_request()
    call = RecordedExchangeSearchCall(
        seed=2011,
        cycle=2,
        call_index=3,
        request=request,
        python_result=execute_exchange_search(python_backend, request),
    )

    summary = compare_recorded_exchange_search_calls(
        calls=[call],
        python_backend=python_backend,
        native_backend=_MismatchingNativeBackend(),
        benchmark_iterations=1,
    )

    assert summary['mismatch_count'] == 1
    mismatch = summary['mismatch_examples'][0]
    assert mismatch['seed'] == 2011
    assert mismatch['cycle'] == 2
    assert mismatch['call_index'] == 3
    assert mismatch['python_result'] != mismatch['native_result']


def test_run_native_search_comparison_captures_exact_cycle_calls() -> None:
    summary = run_native_search_comparison(
        cycles=5,
        seeds=[2009],
        config=SimulationConfig(
            population=32,
            goods=6,
            acquaintances=6,
            active_acquaintances=3,
            demand_candidates=2,
            supply_candidates=2,
            initial_stock_fraction=0.0,
        ),
        backend_name='numpy',
        sample_limit=8,
        benchmark_iterations=1,
        python_backend=PythonExchangeSearchBackend(),
        native_backend=_MatchingNativeBackend(),
    )

    assert summary['capture']['captured_calls'] > 0
    assert summary['comparison']['mismatch_count'] == 0
    assert summary['comparison']['benchmark']['native_calls_per_second'] > 0.0


def _sample_request() -> ExchangeSearchRequest:
    return ExchangeSearchRequest(
        goods=2,
        need_good=1,
        initial_transparency=0.7,
        elastic_need=np.array([1.0, 4.0], dtype=np.float32),
        candidate_offer_goods=np.array([0], dtype=np.int32),
        friend_ids=np.array([1], dtype=np.int32),
        reciprocal_slots=np.array([0], dtype=np.int32),
        my_stock=np.array([10.0, 0.0], dtype=np.float32),
        my_sales_price=np.array([1.0, 2.0], dtype=np.float32),
        my_purchase_price=np.array([1.0, 2.0], dtype=np.float32),
        my_role=np.array([ROLE_CONSUMER, ROLE_CONSUMER], dtype=np.int32),
        my_transparency=np.ones((1, 2), dtype=np.float32),
        my_needs_level=1.0,
        stock=np.array(
            [
                [10.0, 0.0],
                [0.0, 10.0],
            ],
            dtype=np.float32,
        ),
        role=np.array(
            [
                [ROLE_CONSUMER, ROLE_CONSUMER],
                [ROLE_RETAILER, ROLE_CONSUMER],
            ],
            dtype=np.int32,
        ),
        stock_limit=np.full((2, 2), 12.0, dtype=np.float32),
        purchase_price=np.array(
            [
                [1.0, 2.0],
                [2.0, 1.0],
            ],
            dtype=np.float32,
        ),
        sales_price=np.ones((2, 2), dtype=np.float32),
        needs_level=np.ones((2,), dtype=np.float32),
        transparency=np.ones((2, 1, 2), dtype=np.float32),
    )
