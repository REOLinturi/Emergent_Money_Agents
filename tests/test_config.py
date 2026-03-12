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
        experimental_hybrid_batches=3,
        experimental_hybrid_frontier_size=7,
        experimental_hybrid_seed_stride=101,
        experimental_hybrid_consumption_stage=True,
        experimental_hybrid_surplus_stage=True,
        experimental_hybrid_block_frontier_partners=False,
        experimental_hybrid_preserve_proposer_order=True,
        experimental_hybrid_rolling_frontier=True,
        experimental_native_stage_math=True,
        experimental_native_exchange_stage=True,
    )

    assert config.agent_good_shape == (32, 8)
    assert config.friend_shape == (32, 5)
    assert config.active_friend_shape == (32, 3)
    assert config.transparency_shape == (32, 5, 8)
    assert config.cuda_friend_block == 2
    assert config.cuda_goods_block == 4
    assert config.experimental_hybrid_batches == 3
    assert config.experimental_hybrid_frontier_size == 7
    assert config.experimental_hybrid_seed_stride == 101
    assert config.experimental_hybrid_consumption_stage is True
    assert config.experimental_hybrid_surplus_stage is True
    assert config.experimental_hybrid_block_frontier_partners is False
    assert config.experimental_hybrid_preserve_proposer_order is True
    assert config.experimental_hybrid_rolling_frontier is True
    assert config.experimental_native_stage_math is True
    assert config.experimental_native_exchange_stage is True


def test_config_rejects_invalid_active_frontier() -> None:
    with pytest.raises(ValueError):
        SimulationConfig(acquaintances=4, active_acquaintances=5)


def test_config_rejects_non_positive_cuda_block_sizes() -> None:
    with pytest.raises(ValueError):
        SimulationConfig(cuda_friend_block=0)

    with pytest.raises(ValueError):
        SimulationConfig(cuda_goods_block=0)


def test_config_rejects_invalid_experimental_hybrid_settings() -> None:
    with pytest.raises(ValueError):
        SimulationConfig(experimental_hybrid_batches=-1)

    with pytest.raises(ValueError):
        SimulationConfig(experimental_hybrid_frontier_size=-1)

    with pytest.raises(ValueError):
        SimulationConfig(experimental_hybrid_seed_stride=0)
