import pytest

from emergent_money.config import SimulationConfig


def test_session_pairwise_offer_exhaustion_is_default() -> None:
    config = SimulationConfig()

    assert config.experimental_session_pairwise_offer_exhaustion is True


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
        experimental_disable_native_cycle_bridge=True,
        experimental_native_exchange_stage=True,
        experimental_agent_basket_planning=True,
        experimental_local_liquidity_stock_bias=1.5,
        experimental_local_liquidity_min_sales=3.0,
        experimental_aspirational_stock_target=2.0,
        experimental_exchange_media_reserve_bias=0.75,
        experimental_exchange_media_reserve_min_acceptance=4.0,
        experimental_exchange_media_reserve_bootstrap_floor=1.5,
        experimental_storage_class_mode="mod3",
        experimental_poor_storage_spoilage_multiplier=3.0,
        experimental_medium_storage_spoilage_multiplier=1.0,
        experimental_good_storage_spoilage_multiplier=0.1,
        experimental_poor_storage_target_multiplier=0.25,
        experimental_medium_storage_target_multiplier=1.0,
        experimental_good_storage_target_multiplier=3.0,
        experimental_standardization_mode="rare",
        experimental_standardization_strength=0.6,
        experimental_standardization_random_seed=17,
        experimental_transparency_learning_mode="recent-count",
        experimental_endogenous_standardization_strength=0.4,
        experimental_endogenous_standardization_need_power=0.75,
        experimental_session_replan_passes=4,
        experimental_session_replan_after_trade=True,
        experimental_session_disable_replan_cache=True,
        experimental_session_disable_offer_prefilter=True,
        experimental_session_pairwise_offer_exhaustion=True,
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
    assert config.experimental_disable_native_cycle_bridge is True
    assert config.experimental_native_exchange_stage is True
    assert config.experimental_agent_basket_planning is True
    assert config.experimental_local_liquidity_stock_bias == pytest.approx(1.5)
    assert config.experimental_local_liquidity_min_sales == pytest.approx(3.0)
    assert config.experimental_aspirational_stock_target == pytest.approx(2.0)
    assert config.experimental_exchange_media_reserve_bias == pytest.approx(0.75)
    assert config.experimental_exchange_media_reserve_min_acceptance == pytest.approx(4.0)
    assert config.experimental_exchange_media_reserve_bootstrap_floor == pytest.approx(1.5)
    assert config.experimental_storage_class_mode == "mod3"
    assert config.storage_class_codes().tolist() == [0, 1, 2, 0, 1, 2, 0, 1]
    assert config.storage_spoilage_rates().tolist() == pytest.approx([0.3, 0.1, 0.01, 0.3, 0.1, 0.01, 0.3, 0.1])
    assert config.storage_target_multipliers().tolist() == pytest.approx([0.25, 1.0, 3.0, 0.25, 1.0, 3.0, 0.25, 1.0])
    assert config.experimental_standardization_mode == "rare"
    assert config.experimental_standardization_strength == pytest.approx(0.6)
    assert config.experimental_standardization_random_seed == 17
    assert config.exchange_standardization_scores().tolist() == pytest.approx([0.6, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert config.experimental_transparency_learning_mode == "recent-count"
    assert config.experimental_endogenous_standardization_strength == pytest.approx(0.4)
    assert config.experimental_endogenous_standardization_need_power == pytest.approx(0.75)
    assert config.experimental_session_replan_passes == 4
    assert config.experimental_session_replan_after_trade is True
    assert config.experimental_session_disable_replan_cache is True
    assert config.experimental_session_disable_offer_prefilter is True
    assert config.experimental_session_pairwise_offer_exhaustion is True
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
        SimulationConfig(experimental_exchange_media_reserve_bias=-0.01)

    with pytest.raises(ValueError):
        SimulationConfig(experimental_exchange_media_reserve_min_acceptance=-0.01)

    with pytest.raises(ValueError):
        SimulationConfig(experimental_exchange_media_reserve_bootstrap_floor=-0.01)

    with pytest.raises(ValueError):
        SimulationConfig(experimental_storage_class_mode="by-report-id")

    with pytest.raises(ValueError):
        SimulationConfig(experimental_poor_storage_spoilage_multiplier=-0.01)

    with pytest.raises(ValueError):
        SimulationConfig(experimental_good_storage_target_multiplier=-0.01)

    with pytest.raises(ValueError):
        SimulationConfig(experimental_standardization_mode="global-money")

    with pytest.raises(ValueError):
        SimulationConfig(experimental_standardization_strength=-0.01)

    with pytest.raises(ValueError):
        SimulationConfig(experimental_standardization_strength=1.01)

    with pytest.raises(ValueError):
        SimulationConfig(experimental_transparency_learning_mode="cumulative-volume")

    with pytest.raises(ValueError):
        SimulationConfig(experimental_endogenous_standardization_strength=-0.01)

    with pytest.raises(ValueError):
        SimulationConfig(experimental_endogenous_standardization_strength=1.01)

    with pytest.raises(ValueError):
        SimulationConfig(experimental_endogenous_standardization_need_power=-0.01)

    with pytest.raises(ValueError):
        SimulationConfig(experimental_session_replan_passes=0)

    with pytest.raises(ValueError):
        SimulationConfig(experimental_session_candidate_depth=0)


def test_config_rejects_invalid_spaced_good_settings() -> None:
    with pytest.raises(ValueError):
        SimulationConfig(base_good_id_offset=-1)

    with pytest.raises(ValueError):
        SimulationConfig(base_good_id_stride=0)


def test_exchange_standardization_scores_support_experimental_controls() -> None:
    rare = SimulationConfig(goods=8, experimental_standardization_mode="rare", experimental_standardization_strength=0.5)
    common = SimulationConfig(goods=8, experimental_standardization_mode="common", experimental_standardization_strength=0.5)
    rare_gradient = SimulationConfig(
        goods=4,
        experimental_standardization_mode="rare-gradient",
        experimental_standardization_strength=0.8,
    )
    random_a = SimulationConfig(
        goods=8,
        experimental_standardization_mode="random",
        experimental_standardization_strength=0.5,
        experimental_standardization_random_seed=1,
    )
    random_b = SimulationConfig(
        goods=8,
        experimental_standardization_mode="random",
        experimental_standardization_strength=0.5,
        experimental_standardization_random_seed=1,
    )

    assert SimulationConfig(goods=8).exchange_standardization_scores().tolist() == pytest.approx([0.0] * 8)
    assert rare.exchange_standardization_scores().tolist() == pytest.approx([0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert common.exchange_standardization_scores().tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5])
    assert rare_gradient.exchange_standardization_scores().tolist() == pytest.approx([0.8, 0.533333, 0.266667, 0.0], rel=1e-5)
    assert random_a.exchange_standardization_scores().tolist() == pytest.approx(random_b.exchange_standardization_scores().tolist())
    assert sum(score > 0.0 for score in random_a.exchange_standardization_scores()) == 2


def test_storage_class_mod3_uses_run_internal_good_index() -> None:
    config = SimulationConfig(
        goods=6,
        base_good_id_offset=0,
        base_good_id_stride=3,
        spoilage_rate=0.1,
        experimental_storage_class_mode="mod3",
    )

    assert config.base_need_vector().tolist() == pytest.approx([1.0, 16.0, 49.0, 100.0, 169.0, 256.0])
    assert config.storage_class_codes().tolist() == [0, 1, 2, 0, 1, 2]
    assert config.storage_spoilage_rates().tolist() == pytest.approx([0.2, 0.1, 0.025, 0.2, 0.1, 0.025])
    assert config.storage_target_multipliers().tolist() == pytest.approx([0.5, 1.0, 2.0, 0.5, 1.0, 2.0])


def test_storage_class_rare_good_makes_lowest_demand_quartile_well_storable() -> None:
    config = SimulationConfig(
        goods=8,
        base_good_id_offset=0,
        base_good_id_stride=3,
        spoilage_rate=0.1,
        experimental_storage_class_mode="rare-good",
    )

    assert config.base_need_vector().tolist() == pytest.approx([1.0, 16.0, 49.0, 100.0, 169.0, 256.0, 361.0, 484.0])
    assert config.storage_class_codes().tolist() == [2, 2, 1, 1, 1, 1, 1, 1]
    assert config.storage_spoilage_rates().tolist() == pytest.approx([0.025, 0.025, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    assert config.storage_target_multipliers().tolist() == pytest.approx([2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
