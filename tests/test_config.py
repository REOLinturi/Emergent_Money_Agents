import pytest

from emergent_money.config import SimulationConfig


def test_config_shapes_follow_parameters() -> None:
    config = SimulationConfig(
        population=32,
        goods=8,
        acquaintances=5,
        active_acquaintances=3,
        demand_candidates=2,
        supply_candidates=2,
        cuda_friend_block=2,
        cuda_goods_block=4,
    )

    assert config.agent_good_shape == (32, 8)
    assert config.friend_shape == (32, 5)
    assert config.active_friend_shape == (32, 3)
    assert config.transparency_shape == (32, 5, 8)
    assert config.cuda_friend_block == 2
    assert config.cuda_goods_block == 4


def test_config_rejects_invalid_active_frontier() -> None:
    with pytest.raises(ValueError):
        SimulationConfig(acquaintances=4, active_acquaintances=5)


def test_config_rejects_non_positive_cuda_block_sizes() -> None:
    with pytest.raises(ValueError):
        SimulationConfig(cuda_friend_block=0)

    with pytest.raises(ValueError):
        SimulationConfig(cuda_goods_block=0)