from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any, Protocol

import numpy as np

from .state import ROLE_PRODUCER, ROLE_RETAILER

_EPSILON = 1e-6
_PLAN_REASON_BY_CODE = {
    1: 'offer_surplus_below_min',
    2: 'friend_supply_below_min',
    3: 'partner_capacity_below_min',
    4: 'partner_need_below_min',
    5: 'rounding_buffer_below_min',
}
_PLAN_CODE_BY_REASON = {reason: code for code, reason in _PLAN_REASON_BY_CODE.items()}


@dataclass(slots=True, frozen=True)
class ExchangeSearchResult:
    score: float
    friend_slot: int
    friend_id: int
    offer_good: int


@dataclass(slots=True, frozen=True)
class ExchangeSearchRequest:
    goods: int
    need_good: int
    initial_transparency: float
    elastic_need: np.ndarray
    candidate_offer_goods: np.ndarray
    friend_ids: np.ndarray
    reciprocal_slots: np.ndarray
    my_stock: np.ndarray
    my_sales_price: np.ndarray
    my_purchase_price: np.ndarray
    my_role: np.ndarray
    my_transparency: np.ndarray
    my_needs_level: float
    stock: np.ndarray
    role: np.ndarray
    stock_limit: np.ndarray
    purchase_price: np.ndarray
    sales_price: np.ndarray
    needs_level: np.ndarray
    transparency: np.ndarray

    def as_kwargs(self) -> dict[str, Any]:
        return {
            'goods': self.goods,
            'need_good': self.need_good,
            'initial_transparency': self.initial_transparency,
            'elastic_need': self.elastic_need,
            'candidate_offer_goods': self.candidate_offer_goods,
            'friend_ids': self.friend_ids,
            'reciprocal_slots': self.reciprocal_slots,
            'my_stock': self.my_stock,
            'my_sales_price': self.my_sales_price,
            'my_purchase_price': self.my_purchase_price,
            'my_role': self.my_role,
            'my_transparency': self.my_transparency,
            'my_needs_level': self.my_needs_level,
            'stock': self.stock,
            'role': self.role,
            'stock_limit': self.stock_limit,
            'purchase_price': self.purchase_price,
            'sales_price': self.sales_price,
            'needs_level': self.needs_level,
            'transparency': self.transparency,
        }


@dataclass(slots=True, frozen=True)
class ExchangePlanRequest:
    search_request: ExchangeSearchRequest
    max_need: float
    min_trade_quantity: float
    trade_rounding_buffer: float

    def as_kwargs(self) -> dict[str, Any]:
        payload = self.search_request.as_kwargs()
        payload['max_need'] = self.max_need
        payload['min_trade_quantity'] = self.min_trade_quantity
        payload['trade_rounding_buffer'] = self.trade_rounding_buffer
        return payload


@dataclass(slots=True, frozen=True)
class ExchangePlanResult:
    score: float
    friend_slot: int
    friend_id: int
    offer_good: int
    reciprocal_slot: int
    max_exchange: float
    switch_average: float
    need_transparency: float
    receiving_transparency: float

    def as_search_result(self) -> ExchangeSearchResult:
        return ExchangeSearchResult(
            score=self.score,
            friend_slot=self.friend_slot,
            friend_id=self.friend_id,
            offer_good=self.offer_good,
        )


@dataclass(slots=True, frozen=True)
class ExchangePlanningOutcome:
    search_result: ExchangeSearchResult
    plan_result: ExchangePlanResult | None
    failure_reason: str | None = None


class ExchangeSearchBackend(Protocol):
    name: str
    is_native: bool

    def find_best_exchange(
        self,
        *,
        goods: int,
        need_good: int,
        initial_transparency: float,
        elastic_need: np.ndarray,
        candidate_offer_goods: np.ndarray,
        friend_ids: np.ndarray,
        reciprocal_slots: np.ndarray,
        my_stock: np.ndarray,
        my_sales_price: np.ndarray,
        my_purchase_price: np.ndarray,
        my_role: np.ndarray,
        my_transparency: np.ndarray,
        my_needs_level: float,
        stock: np.ndarray,
        role: np.ndarray,
        stock_limit: np.ndarray,
        purchase_price: np.ndarray,
        sales_price: np.ndarray,
        needs_level: np.ndarray,
        transparency: np.ndarray,
    ) -> ExchangeSearchResult | None: ...

    def plan_best_exchange(self, request: ExchangePlanRequest) -> ExchangePlanningOutcome | None: ...


class PythonExchangeSearchBackend:
    name = 'python'
    is_native = False

    def find_best_exchange(
        self,
        *,
        goods: int,
        need_good: int,
        initial_transparency: float,
        elastic_need: np.ndarray,
        candidate_offer_goods: np.ndarray,
        friend_ids: np.ndarray,
        reciprocal_slots: np.ndarray,
        my_stock: np.ndarray,
        my_sales_price: np.ndarray,
        my_purchase_price: np.ndarray,
        my_role: np.ndarray,
        my_transparency: np.ndarray,
        my_needs_level: float,
        stock: np.ndarray,
        role: np.ndarray,
        stock_limit: np.ndarray,
        purchase_price: np.ndarray,
        sales_price: np.ndarray,
        needs_level: np.ndarray,
        transparency: np.ndarray,
    ) -> ExchangeSearchResult | None:
        my_need_purchase_price = my_purchase_price[need_good]
        my_need_is_producer = my_role[need_good] == ROLE_PRODUCER
        best_score = -1.0
        best_friend_slot = -1
        best_friend_id = -1
        best_offer_good = -1

        for friend_slot, friend_id_raw in enumerate(friend_ids):
            friend_id = int(friend_id_raw)
            if friend_id < 0:
                continue

            friend_stock = stock[friend_id]
            friend_role = role[friend_id]
            friend_stock_limit = stock_limit[friend_id]
            friend_purchase_price = purchase_price[friend_id]
            friend_sales_price = sales_price[friend_id]
            friend_needs_level = needs_level[friend_id]
            if friend_stock[need_good] <= (elastic_need[need_good] * friend_needs_level + 1.0):
                continue

            reciprocal_slot = int(reciprocal_slots[friend_slot])
            friend_transparency = transparency[friend_id, reciprocal_slot] if reciprocal_slot >= 0 else None
            need_transparency = my_transparency[friend_slot, need_good]
            friend_need_role = float(friend_role[need_good])
            friend_need_sales_price = friend_sales_price[need_good]

            for offer_good_raw in candidate_offer_goods:
                offer_good = int(offer_good_raw)
                if friend_role[offer_good] == ROLE_RETAILER:
                    gift_max_level = float(friend_stock_limit[offer_good]) - 1.0
                else:
                    gift_max_level = float(elastic_need[offer_good] * friend_needs_level) - 1.0
                if friend_stock[offer_good] >= gift_max_level:
                    continue

                receiving_transparency = initial_transparency
                if friend_transparency is not None:
                    receiving_transparency = friend_transparency[offer_good]

                score = (
                    (friend_purchase_price[offer_good] / max(friend_need_sales_price, _EPSILON))
                    * need_transparency
                ) - (
                    my_sales_price[offer_good]
                    / max(my_need_purchase_price * receiving_transparency, _EPSILON)
                )
                score *= float(my_role[offer_good])
                score *= friend_need_role
                if my_need_is_producer:
                    score /= 2.0

                score = float(np.float32(score))
                if score > best_score:
                    best_score = score
                    best_friend_slot = friend_slot
                    best_friend_id = friend_id
                    best_offer_good = offer_good

        if best_score <= 0.0 or best_friend_slot < 0 or best_offer_good < 0:
            return None
        return ExchangeSearchResult(
            score=float(best_score),
            friend_slot=best_friend_slot,
            friend_id=best_friend_id,
            offer_good=best_offer_good,
        )

    def plan_best_exchange(self, request: ExchangePlanRequest) -> ExchangePlanningOutcome | None:
        search_result = self.find_best_exchange(**request.search_request.as_kwargs())
        if search_result is None:
            return None
        return build_exchange_planning_outcome(request, search_result)


class NativeModuleExchangeSearchBackend:
    name = 'native'
    is_native = True

    def __init__(self, native_module) -> None:
        self._native_module = native_module

    def find_best_exchange(
        self,
        *,
        goods: int,
        need_good: int,
        initial_transparency: float,
        elastic_need: np.ndarray,
        candidate_offer_goods: np.ndarray,
        friend_ids: np.ndarray,
        reciprocal_slots: np.ndarray,
        my_stock: np.ndarray,
        my_sales_price: np.ndarray,
        my_purchase_price: np.ndarray,
        my_role: np.ndarray,
        my_transparency: np.ndarray,
        my_needs_level: float,
        stock: np.ndarray,
        role: np.ndarray,
        stock_limit: np.ndarray,
        purchase_price: np.ndarray,
        sales_price: np.ndarray,
        needs_level: np.ndarray,
        transparency: np.ndarray,
    ) -> ExchangeSearchResult | None:
        result = self._native_module.find_best_exchange(
            goods=goods,
            need_good=need_good,
            initial_transparency=initial_transparency,
            elastic_need=elastic_need,
            candidate_offer_goods=candidate_offer_goods,
            friend_ids=friend_ids,
            reciprocal_slots=reciprocal_slots,
            my_stock=my_stock,
            my_sales_price=my_sales_price,
            my_purchase_price=my_purchase_price,
            my_role=my_role,
            my_transparency=my_transparency,
            my_needs_level=my_needs_level,
            stock=stock,
            role=role,
            stock_limit=stock_limit,
            purchase_price=purchase_price,
            sales_price=sales_price,
            needs_level=needs_level,
            transparency=transparency,
        )
        if result is None:
            return None
        return ExchangeSearchResult(
            score=float(result[0]),
            friend_slot=int(result[1]),
            friend_id=int(result[2]),
            offer_good=int(result[3]),
        )

    def plan_best_exchange(self, request: ExchangePlanRequest) -> ExchangePlanningOutcome | None:
        if hasattr(self._native_module, 'plan_best_exchange'):
            result = self._native_module.plan_best_exchange(**request.as_kwargs())
            if result is None:
                return None
            reason_code = int(result[0])
            search_result = ExchangeSearchResult(
                score=float(result[1]),
                friend_slot=int(result[2]),
                friend_id=int(result[3]),
                offer_good=int(result[4]),
            )
            failure_reason = _PLAN_REASON_BY_CODE.get(reason_code)
            plan_result = None
            if reason_code == 0:
                plan_result = ExchangePlanResult(
                    score=search_result.score,
                    friend_slot=search_result.friend_slot,
                    friend_id=search_result.friend_id,
                    offer_good=search_result.offer_good,
                    reciprocal_slot=int(result[5]),
                    max_exchange=float(result[6]),
                    switch_average=float(result[7]),
                    need_transparency=float(result[8]),
                    receiving_transparency=float(result[9]),
                )
            return ExchangePlanningOutcome(
                search_result=search_result,
                plan_result=plan_result,
                failure_reason=failure_reason,
            )

        search_result = self.find_best_exchange(**request.search_request.as_kwargs())
        if search_result is None:
            return None
        return build_exchange_planning_outcome(request, search_result)


def _load_native_search_module():
    for module_name in (
        'emergent_money._legacy_native_search',
        '_legacy_native_search',
    ):
        try:
            return importlib.import_module(module_name)
        except ImportError:
            continue
    return None


def _build_exchange_plan_result(
    request: ExchangePlanRequest,
    search_result: ExchangeSearchResult,
) -> tuple[ExchangePlanResult | None, str | None]:
    search_request = request.search_request
    friend_id = search_result.friend_id
    offer_good = search_result.offer_good
    need_good = search_request.need_good
    friend_slot = search_result.friend_slot
    reciprocal_slot = int(search_request.reciprocal_slots[friend_slot])
    need_transparency = float(search_request.my_transparency[friend_slot, need_good])
    receiving_transparency = float(search_request.initial_transparency)
    if reciprocal_slot >= 0:
        receiving_transparency = float(search_request.transparency[friend_id, reciprocal_slot, offer_good])

    my_need_purchase_price = float(search_request.my_purchase_price[need_good])
    my_offer_sales_price = float(search_request.my_sales_price[offer_good])
    friend_need_sales_price = float(search_request.sales_price[friend_id, need_good])
    friend_offer_purchase_price = float(search_request.purchase_price[friend_id, offer_good])

    switch_average = (
        (
            friend_need_sales_price
            / max(friend_offer_purchase_price * need_transparency, _EPSILON)
        )
        + ((my_need_purchase_price * receiving_transparency) / max(my_offer_sales_price, _EPSILON))
    ) / 2.0

    friend_needs_level = float(search_request.needs_level[friend_id])
    max_exchange = (
        (
            float(search_request.my_stock[offer_good])
            - (float(search_request.elastic_need[offer_good]) * search_request.my_needs_level)
        )
        * receiving_transparency
    ) / max(switch_average, _EPSILON)
    if max_exchange <= request.min_trade_quantity:
        return None, 'offer_surplus_below_min'

    max_exchange = min(max_exchange, request.max_need)
    friend_supply = (
        float(search_request.stock[friend_id, need_good])
        - (friend_needs_level * float(search_request.elastic_need[need_good]))
    ) * need_transparency
    max_exchange = min(max_exchange, friend_supply)
    if max_exchange <= request.min_trade_quantity:
        return None, 'friend_supply_below_min'

    if int(search_request.role[friend_id, offer_good]) == ROLE_RETAILER:
        stock_capacity = float(search_request.stock_limit[friend_id, offer_good] - search_request.stock[friend_id, offer_good])
        max_exchange = min(max_exchange, stock_capacity / max(switch_average, _EPSILON))
        if max_exchange <= request.min_trade_quantity:
            return None, 'partner_capacity_below_min'
    else:
        immediate_need = (
            friend_needs_level * float(search_request.elastic_need[offer_good])
        ) - float(search_request.stock[friend_id, offer_good])
        max_exchange = min(max_exchange, immediate_need / max(switch_average, _EPSILON))
        if max_exchange <= request.min_trade_quantity:
            return None, 'partner_need_below_min'

    max_exchange -= request.trade_rounding_buffer
    max_exchange = float(np.float32(max_exchange))
    if max_exchange < request.min_trade_quantity:
        return None, 'rounding_buffer_below_min'

    return ExchangePlanResult(
        score=search_result.score,
        friend_slot=search_result.friend_slot,
        friend_id=search_result.friend_id,
        offer_good=search_result.offer_good,
        reciprocal_slot=reciprocal_slot,
        max_exchange=max_exchange,
        switch_average=float(switch_average),
        need_transparency=need_transparency,
        receiving_transparency=receiving_transparency,
    ), None


def build_exchange_planning_outcome(
    request: ExchangePlanRequest,
    search_result: ExchangeSearchResult,
) -> ExchangePlanningOutcome:
    plan_result, failure_reason = _build_exchange_plan_result(request, search_result)
    return ExchangePlanningOutcome(
        search_result=search_result,
        plan_result=plan_result,
        failure_reason=failure_reason,
    )


def execute_exchange_search(
    backend: ExchangeSearchBackend,
    request: ExchangeSearchRequest,
) -> ExchangeSearchResult | None:
    return backend.find_best_exchange(**request.as_kwargs())


def execute_exchange_planning(
    backend: ExchangeSearchBackend,
    request: ExchangePlanRequest,
) -> ExchangePlanningOutcome | None:
    return backend.plan_best_exchange(request)


def build_python_exchange_search_backend() -> ExchangeSearchBackend:
    return PythonExchangeSearchBackend()


def build_native_exchange_search_backend() -> ExchangeSearchBackend | None:
    native_module = _load_native_search_module()
    if native_module is None:
        return None
    if not hasattr(native_module, 'find_best_exchange'):
        return None
    return NativeModuleExchangeSearchBackend(native_module)


def build_exchange_search_backend() -> ExchangeSearchBackend:
    native_backend = build_native_exchange_search_backend()
    if native_backend is not None:
        return native_backend
    return build_python_exchange_search_backend()
