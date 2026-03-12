from __future__ import annotations

import json

import numpy as np

from emergent_money.config import SimulationConfig
from emergent_money.engine import SimulationEngine
from emergent_money.long_run import load_checkpoint, run_long_simulation, save_checkpoint


def test_checkpoint_round_trip_preserves_engine_state(tmp_path) -> None:
    config = SimulationConfig(
        population=12,
        goods=5,
        acquaintances=4,
        active_acquaintances=2,
        demand_candidates=2,
        supply_candidates=2,
    )
    engine = SimulationEngine.create(config=config, backend_name="numpy")
    for _ in range(3):
        engine.step()

    save_checkpoint(engine, tmp_path)
    restored = load_checkpoint(tmp_path)

    assert restored.cycle == engine.cycle
    assert len(restored.history) == len(engine.history)
    assert restored.config.population == engine.config.population
    assert np.allclose(restored.backend.to_numpy(restored.state.stock), engine.backend.to_numpy(engine.state.stock))
    assert np.allclose(restored.backend.to_numpy(restored.state.time_remaining), engine.backend.to_numpy(engine.state.time_remaining))
    assert np.array_equal(restored.backend.to_numpy(restored.state.friend_id), engine.backend.to_numpy(engine.state.friend_id))
    assert np.allclose(
        restored.backend.to_numpy(restored.state.market.elastic_need),
        engine.backend.to_numpy(engine.state.market.elastic_need),
    )


def test_long_run_runner_writes_artifacts_and_resumes(tmp_path) -> None:
    config = SimulationConfig(
        population=10,
        goods=4,
        acquaintances=3,
        active_acquaintances=2,
        demand_candidates=2,
        supply_candidates=2,
    )

    summary1 = run_long_simulation(
        cycles=2,
        checkpoint_dir=tmp_path,
        config=config,
        backend_name="numpy",
        checkpoint_every=1,
        sample_every=1,
        top_goods=3,
    )
    assert summary1["start_cycle"] == 0
    assert summary1["end_cycle"] == 2
    assert (tmp_path / "checkpoint_latest.json").exists()
    assert (tmp_path / "checkpoint_latest.npz").exists()

    metrics_path = tmp_path / "metrics.jsonl"
    assert metrics_path.exists()
    assert len(metrics_path.read_text(encoding="utf-8").splitlines()) == 2

    summary2 = run_long_simulation(
        cycles=2,
        checkpoint_dir=tmp_path,
        resume_from=tmp_path,
        checkpoint_every=1,
        sample_every=1,
        top_goods=3,
    )
    assert summary2["start_cycle"] == 2
    assert summary2["end_cycle"] == 4
    assert len(metrics_path.read_text(encoding="utf-8").splitlines()) == 4

    payload = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert payload["end_cycle"] == 4
    assert payload["latest_market"]["cycle"] == 4
