from __future__ import annotations

from dataclasses import dataclass

from .legacy_search_backend import _load_native_search_module


@dataclass(slots=True)
class NativeLegacyCycleBackend:
    _native_module: object
    _runner: object
    _config: object
    _state: object
    _market: object

    @property
    def supports_run_exact_cycle(self) -> bool:
        return hasattr(self._native_module, 'run_exact_cycle')

    @property
    def supports_prepare_agent_for_consumption(self) -> bool:
        return hasattr(self._native_module, 'prepare_agent_for_consumption')

    @property
    def supports_produce_need(self) -> bool:
        return hasattr(self._native_module, 'produce_need')

    @property
    def supports_prepare_leisure_round(self) -> bool:
        return hasattr(self._native_module, 'prepare_leisure_round')

    @property
    def supports_run_exchange_stage(self) -> bool:
        return hasattr(self._native_module, 'run_exchange_stage')

    @property
    def supports_surplus_production(self) -> bool:
        return hasattr(self._native_module, 'surplus_production')

    @property
    def supports_leisure_production(self) -> bool:
        return hasattr(self._native_module, 'leisure_production')

    @property
    def supports_end_agent_period(self) -> bool:
        return hasattr(self._native_module, 'end_agent_period')

    def run_exact_cycle(self, engine) -> None:
        self._native_module.run_exact_cycle(engine)

    def prepare_agent_for_consumption(self, *, agent_id: int) -> tuple[float, float]:
        state = self._state
        market = self._market
        config = self._config
        cycle_need_total, stock_consumed_total = self._native_module.prepare_agent_for_consumption(
            agent_id=agent_id,
            goods=config.goods,
            period_length=config.cycle_time_budget,
            history=config.history,
            basic_round_elastic=config.basic_round_elastic,
            stock_limit_multiplier=config.stock_limit_multiplier,
            max_needs_increase=config.max_needs_increase,
            max_needs_reduction=config.max_needs_reduction,
            small_needs_increase=config.small_needs_increase,
            small_needs_reduction=config.small_needs_reduction,
            base_need=state.base_need,
            need=state.need,
            stock=state.stock,
            purchase_price=state.purchase_price,
            sales_price=state.sales_price,
            purchased_last_period=state.purchased_last_period,
            recent_sales=state.recent_sales,
            sold_this_period=state.sold_this_period,
            sold_last_period=state.sold_last_period,
            recent_purchases=state.recent_purchases,
            efficiency=state.efficiency,
            period_failure=state.period_failure,
            period_time_debt=state.period_time_debt,
            needs_level=state.needs_level,
            recent_needs_increment=state.recent_needs_increment,
            market_elastic_need=market.elastic_need,
        )
        return float(cycle_need_total), float(stock_consumed_total)

    def produce_need(self, *, agent_id: int) -> float:
        state = self._state
        produced_total = self._native_module.produce_need(
            agent_id=agent_id,
            goods=self._config.goods,
            efficiency=state.efficiency,
            need=state.need,
            time_remaining=state.time_remaining,
            recent_production=state.recent_production,
            produced_this_period=state.produced_this_period,
            timeout=state.timeout,
        )
        return float(produced_total)

    def prepare_leisure_round(self, *, agent_id: int) -> tuple[bool, float, float]:
        state = self._state
        market = self._market
        config = self._config
        has_extra, extra_need_total, stock_consumed_total = self._native_module.prepare_leisure_round(
            agent_id=agent_id,
            goods=config.goods,
            period_length=config.cycle_time_budget,
            history=config.history,
            leisure_time=config.leisure_time,
            max_needs_increase=config.max_needs_increase,
            max_leisure_extra_multiplier=config.max_leisure_extra_multiplier,
            small_needs_increase=config.small_needs_increase,
            basic_round_elastic=config.basic_round_elastic,
            market_elastic_need=market.elastic_need,
            time_remaining=state.time_remaining,
            needs_level=state.needs_level,
            recent_needs_increment=state.recent_needs_increment,
            need=state.need,
            stock=state.stock,
        )
        return bool(has_extra), float(extra_need_total), float(stock_consumed_total)

    def run_exchange_stage(self, *, agent_id: int, deal_type: int) -> tuple[int, int, float, float]:
        proposed_count, accepted_count, accepted_volume, inventory_trade_volume = self._native_module.run_exchange_stage(
            self._runner,
            agent_id=agent_id,
            deal_type=deal_type,
        )
        return int(proposed_count), int(accepted_count), float(accepted_volume), float(inventory_trade_volume)

    def surplus_production(self, *, agent_id: int) -> float:
        state = self._state
        produced_total = self._native_module.surplus_production(
            agent_id=agent_id,
            goods=self._config.goods,
            switch_time=self._config.switch_time,
            stock_limit_multiplier=self._config.stock_limit_multiplier,
            price_hike=self._config.price_hike,
            base_need=state.base_need,
            stock=state.stock,
            stock_limit=state.stock_limit,
            talent_mask=state.talent_mask,
            purchase_times=state.purchase_times,
            efficiency=state.efficiency,
            sales_price=state.sales_price,
            time_remaining=state.time_remaining,
            recent_production=state.recent_production,
            produced_this_period=state.produced_this_period,
        )
        return float(produced_total)

    def leisure_production(self, *, agent_id: int) -> float:
        state = self._state
        produced_total = self._native_module.leisure_production(
            agent_id=agent_id,
            goods=self._config.goods,
            stock=state.stock,
            stock_limit=state.stock_limit,
            talent_mask=state.talent_mask,
            purchase_price=state.purchase_price,
            time_remaining=state.time_remaining,
            recent_production=state.recent_production,
            produced_this_period=state.produced_this_period,
        )
        return float(produced_total)

    def end_agent_period(self, *, cycle: int, agent_id: int) -> None:
        state = self._state
        market = self._market
        config = self._config
        self._native_module.end_agent_period(
            agent_id=agent_id,
            cycle=cycle,
            goods=config.goods,
            acquaintances=config.acquaintances,
            history=config.history,
            initial_efficiency=config.initial_efficiency,
            gifted_efficiency_floor=config.gifted_efficiency_floor,
            initial_transparency=config.initial_transparency,
            stock_limit_multiplier=config.stock_limit_multiplier,
            activity_discount=config.activity_discount,
            spoilage_rate=config.spoilage_rate,
            stock_spoil_threshold=config.stock_spoil_threshold,
            price_reduction=config.price_reduction,
            price_hike=config.price_hike,
            price_leap=config.price_leap,
            max_stocklimit_decrease=config.max_stocklimit_decrease,
            max_stocklimit_increase=config.max_stocklimit_increase,
            max_efficiency_downgrade=config.max_efficiency_downgrade,
            max_efficiency_upgrade=config.max_efficiency_upgrade,
            base_need=state.base_need,
            stock=state.stock,
            stock_limit=state.stock_limit,
            previous_stock_limit=state.previous_stock_limit,
            efficiency=state.efficiency,
            learned_efficiency=state.learned_efficiency,
            recent_production=state.recent_production,
            recent_sales=state.recent_sales,
            recent_purchases=state.recent_purchases,
            recent_inventory_inflow=state.recent_inventory_inflow,
            produced_this_period=state.produced_this_period,
            produced_last_period=state.produced_last_period,
            sold_this_period=state.sold_this_period,
            sold_last_period=state.sold_last_period,
            purchased_this_period=state.purchased_this_period,
            purchased_last_period=state.purchased_last_period,
            purchase_times=state.purchase_times,
            sales_times=state.sales_times,
            sum_period_purchase_value=state.sum_period_purchase_value,
            sum_period_sales_value=state.sum_period_sales_value,
            spoilage=state.spoilage,
            periodic_spoilage=state.periodic_spoilage,
            talent_mask=state.talent_mask,
            role=state.role,
            purchase_price=state.purchase_price,
            sales_price=state.sales_price,
            friend_activity=state.friend_activity,
            friend_purchased=state.friend_purchased,
            transparency=state.transparency,
            needs_level=state.needs_level,
            market_elastic_need=market.elastic_need,
            market_periodic_spoilage=market.periodic_spoilage,
        )


def build_native_legacy_cycle_backend(runner) -> NativeLegacyCycleBackend | None:
    native_module = _load_native_search_module()
    if native_module is None:
        return None
    return NativeLegacyCycleBackend(
        _native_module=native_module,
        _runner=runner,
        _config=runner.config,
        _state=runner.state,
        _market=runner.market,
    )


def native_exact_cycle_available() -> bool:
    native_module = _load_native_search_module()
    return native_module is not None and hasattr(native_module, 'run_exact_cycle')


def can_use_native_legacy_cycle(engine) -> bool:
    config = engine.config
    if config.experimental_hybrid_consumption_stage or config.experimental_hybrid_surplus_stage:
        return False
    allow_rejected_exchange_stage = bool(getattr(engine, '_allow_rejected_native_exchange_stage', False))
    return bool(
        config.use_exact_legacy_mechanics
        and (
            config.experimental_native_stage_math
            or (config.experimental_native_exchange_stage and allow_rejected_exchange_stage)
        )
    )


def run_native_legacy_cycle(engine) -> bool:
    native_module = _load_native_search_module()
    if native_module is None or not hasattr(native_module, 'run_exact_cycle'):
        return False
    native_module.run_exact_cycle(engine)
    return True
