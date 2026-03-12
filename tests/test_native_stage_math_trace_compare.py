from __future__ import annotations

import pytest

from emergent_money.config import SimulationConfig
from emergent_money.native_stage_math_trace_compare import run_native_stage_math_trace_comparison


def test_run_native_stage_math_trace_comparison_requires_experimental_flag() -> None:
    config = SimulationConfig(
        population=4,
        goods=2,
        acquaintances=1,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
        experimental_native_stage_math=False,
    )

    with pytest.raises(ValueError, match='experimental_native_stage_math=True'):
        run_native_stage_math_trace_comparison(cycles=1, seeds=[2009], config=config)


def test_run_native_stage_math_trace_comparison_requires_native_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    config = SimulationConfig(
        population=4,
        goods=2,
        acquaintances=1,
        active_acquaintances=1,
        demand_candidates=1,
        supply_candidates=1,
        experimental_native_stage_math=True,
    )
    monkeypatch.setattr(
        'emergent_money.native_stage_math_trace_compare._target_runner_has_stage_math',
        lambda runner: False,
    )

    with pytest.raises(RuntimeError, match='native stage-math helpers are not available'):
        run_native_stage_math_trace_comparison(cycles=1, seeds=[2009], config=config)
