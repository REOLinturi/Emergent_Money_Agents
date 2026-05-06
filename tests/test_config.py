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
        base_good_id_offset=0,
        base_good_id_stride=3,
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
        experimental_parallel_phenomenon_exchange=True,
        experimental_session_clearing_phenomenon_exchange=True,
        experimental_native_stage_math=True,
        experimental_native_exchange_stage=True,
        experimental_agent_basket_planning=True,
        experimental_local_liquidity_stock_bias=1.5,
        experimental_local_liquidity_min_sales=3.0,
        experimental_aspirational_stock_target=2.0,
        experimental_session_replan_passes=4,
        experimental_session_replan_after_trade=True,
        experimental_session_candidate_depth=3,
        legacy_price_floor=0.05,
        use_value_price_floor_fraction=0.25,
    )

    assert config.agent_good_shape == (32, 8)
    assert config.friend_shape == (32, 5)
    assert config.active_friend_shape == (32, 3)
    assert config.transparency_shape == (32, 5, 8)
    assert config.base_good_id_stride == 3
    assert config.base_need_vector().tolist() == pytest.approx([1.0, 16.0, 49.0, 100.0, 169.0, 256.0, 361.0, 484.0])
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
    assert config.experimental_parallel_phenomenon_exchange is True
    assert config.experimental_session_clearing_phenomenon_exchange is True
    assert config.experimental_native_stage_math is True
    assert config.experimental_native_exchange_stage is True
    assert config.experimental_agent_basket_planning is True
    assert config.experimental_local_liquidity_stock_bias == pytest.approx(1.5)
    assert config.experimental_local_liquidity_min_sales == pytest.approx(3.0)
    assert config.experimental_aspirational_stock_target == pytest.approx(2.0)
    assert config.experimental_session_replan_passes == 4
    assert config.experimental_session_replan_after_trade is True
    assert config.experimental_session_candidate_depth == 3
    assert config.legacy_price_floor == pytest.approx(0.05)
    assert config.use_value_price_floor_fraction == pytest.approx(0.25)


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


def test_config_rejects_negative_legacy_price_floor() -> None:
    with pytest.raises(ValueError):
        SimulationConfig(legacy_price_floor=-0.01)


def test_config_rejects_invalid_use_value_price_floor_fraction() -> None:
    with pytest.raises(ValueError):
        SimulationConfig(use_value_price_floor_fraction=-0.01)

    with pytest.raises(ValueError):
        SimulationConfig(use_value_price_floor_fraction=1.01)


def test_config_rejects_invalid_local_liquidity_settings() -> None:
    with pytest.raises(ValueError):
        SimulationConfig(experimental_local_liquidity_stock_bias=-0.01)

    with pytest.raises(ValueError):
        SimulationConfig(experimental_local_liquidity_min_sales=-0.01)

    with pytest.raises(ValueError):
        SimulationConfig(experimental_aspirational_stock_target=-0.01)

    with pytest.raises(ValueError):
        SimulationConfig(experimental_session_replan_passes=0)

    with pytest.raises(ValueError):
        SimulationConfig(experimental_session_candidate_depth=0)


def test_config_rejects_invalid_spaced_good_settings() -> None:
    with pytest.raises(ValueError):
        SimulationConfig(base_good_id_offset=-1)

    with pytest.raises(ValueError):
        SimulationConfig(base_good_id_stride=0)
