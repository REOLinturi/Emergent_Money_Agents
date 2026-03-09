from __future__ import annotations

import numpy as np

from emergent_money.backend import create_backend
from emergent_money.config import SimulationConfig
from emergent_money.initialization import create_initial_state


def test_initial_state_shapes_and_friend_ids() -> None:
    config = SimulationConfig(
        population=24,
        goods=6,
        acquaintances=5,
        active_acquaintances=3,
        demand_candidates=2,
        supply_candidates=2,
    )
    backend = create_backend("numpy")
    state = create_initial_state(config, backend)

    assert state.base_need.shape == config.agent_good_shape
    assert state.friend_id.shape == config.friend_shape
    assert state.transparency.shape == config.transparency_shape
    assert state.innate_efficiency.shape == config.agent_good_shape
    assert state.learned_efficiency.shape == config.agent_good_shape

    friend_id = backend.to_numpy(state.friend_id)
    self_ids = np.arange(config.population, dtype=np.int32)[:, None]
    assert np.all(friend_id == -1)
    assert not np.any(friend_id == self_ids)


def test_initial_state_contains_binary_mild_talent_advantage() -> None:
    config = SimulationConfig(
        population=64,
        goods=8,
        acquaintances=4,
        active_acquaintances=2,
        demand_candidates=2,
        supply_candidates=2,
        talent_probability=0.5,
        gifted_efficiency_bonus=0.5,
    )
    backend = create_backend("numpy")
    state = create_initial_state(config, backend)

    innate = backend.to_numpy(state.innate_efficiency)
    unique_values = set(np.unique(innate).round(4).tolist())
    assert config.initial_efficiency in unique_values
    assert (config.initial_efficiency + config.gifted_efficiency_bonus) in unique_values
    assert unique_values.issubset({config.initial_efficiency, config.initial_efficiency + config.gifted_efficiency_bonus})
    assert np.allclose(backend.to_numpy(state.learned_efficiency), config.initial_efficiency)