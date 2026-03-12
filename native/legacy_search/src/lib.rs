use ndarray::{ArrayView1, ArrayView2, ArrayView3, ArrayViewMut1, ArrayViewMut2, ArrayViewMut3, Axis};
use numpy::{
    PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyReadwriteArray1, PyReadwriteArray2,
    PyReadwriteArray3,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashSet;

const ROLE_CONSUMER: i32 = 10;
const ROLE_RETAILER: i32 = 11;
const ROLE_PRODUCER: i32 = 12;
const EPSILON: f32 = 1.0e-6;
const PLAN_OK: i32 = 0;
const PLAN_OFFER_SURPLUS_BELOW_MIN: i32 = 1;
const PLAN_FRIEND_SUPPLY_BELOW_MIN: i32 = 2;
const PLAN_PARTNER_CAPACITY_BELOW_MIN: i32 = 3;
const PLAN_PARTNER_NEED_BELOW_MIN: i32 = 4;
const PLAN_ROUNDING_BUFFER_BELOW_MIN: i32 = 5;

fn mul_assign_like_numpy_float32(value: f64, factor: f64) -> f64 {
    ((value as f32) * (factor as f32)) as f64
}

fn add_assign_like_numpy_float32(cell: &mut f32, delta: f64) {
    *cell = ((*cell as f32) + (delta as f32)) as f32;
}

#[derive(Clone, Copy)]
struct SearchCandidate {
    score: f32,
    friend_slot: i32,
    friend_id: i32,
    offer_good: i32,
    reciprocal_slot: i32,
}

#[allow(clippy::too_many_arguments)]
fn search_best_exchange_internal(
    goods: usize,
    need_good: usize,
    initial_transparency: f32,
    elastic_need: ArrayView1<'_, f32>,
    candidate_offer_goods: ArrayView1<'_, i32>,
    friend_ids: ArrayView1<'_, i32>,
    reciprocal_slots: ArrayView1<'_, i32>,
    my_sales_price: ArrayView1<'_, f32>,
    my_purchase_price: ArrayView1<'_, f32>,
    my_role: ArrayView1<'_, i32>,
    my_transparency: ArrayView2<'_, f32>,
    stock: ArrayView2<'_, f32>,
    role: ArrayView2<'_, i32>,
    stock_limit: ArrayView2<'_, f32>,
    purchase_price: ArrayView2<'_, f32>,
    sales_price: ArrayView2<'_, f32>,
    needs_level: ArrayView1<'_, f32>,
    transparency: ArrayView3<'_, f32>,
) -> Option<SearchCandidate> {
    if need_good >= goods {
        return None;
    }

    let my_need_purchase_price = my_purchase_price[need_good];
    let my_need_is_producer = my_role[need_good] == ROLE_PRODUCER;
    let mut best_score = -1.0_f32;
    let mut best_candidate: Option<SearchCandidate> = None;

    for (friend_slot, friend_id_raw) in friend_ids.iter().enumerate() {
        let friend_id = *friend_id_raw;
        if friend_id < 0 {
            continue;
        }
        let fid = friend_id as usize;
        let friend_needs_level = needs_level[fid];
        if stock[[fid, need_good]] <= (elastic_need[need_good] * friend_needs_level + 1.0) {
            continue;
        }

        let reciprocal_slot = reciprocal_slots[friend_slot];
        let need_transparency = my_transparency[[friend_slot, need_good]];
        let friend_need_role = role[[fid, need_good]] as f32;
        let friend_need_sales_price = sales_price[[fid, need_good]];

        for offer_good_raw in candidate_offer_goods.iter() {
            let offer_good_i32 = *offer_good_raw;
            if offer_good_i32 < 0 {
                continue;
            }
            let offer_good = offer_good_i32 as usize;
            if offer_good >= goods || offer_good == need_good {
                continue;
            }

            let gift_max_level = if role[[fid, offer_good]] == ROLE_RETAILER {
                stock_limit[[fid, offer_good]] - 1.0
            } else {
                elastic_need[offer_good] * friend_needs_level - 1.0
            };
            if stock[[fid, offer_good]] >= gift_max_level {
                continue;
            }

            let mut receiving_transparency = initial_transparency;
            if reciprocal_slot >= 0 {
                receiving_transparency = transparency[[fid, reciprocal_slot as usize, offer_good]];
            }

            let mut score =
                (purchase_price[[fid, offer_good]] / friend_need_sales_price.max(EPSILON)) * need_transparency;
            score -= my_sales_price[offer_good]
                / (my_need_purchase_price * receiving_transparency).max(EPSILON);
            score *= my_role[offer_good] as f32;
            score *= friend_need_role;
            if my_need_is_producer {
                score /= 2.0;
            }

            if score > best_score {
                best_score = score;
                best_candidate = Some(SearchCandidate {
                    score,
                    friend_slot: friend_slot as i32,
                    friend_id,
                    offer_good: offer_good_i32,
                    reciprocal_slot,
                });
            }
        }
    }

    if best_score <= 0.0 {
        return None;
    }
    best_candidate
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn find_best_exchange(
    goods: usize,
    need_good: usize,
    initial_transparency: f32,
    elastic_need: PyReadonlyArray1<'_, f32>,
    candidate_offer_goods: PyReadonlyArray1<'_, i32>,
    friend_ids: PyReadonlyArray1<'_, i32>,
    reciprocal_slots: PyReadonlyArray1<'_, i32>,
    my_stock: PyReadonlyArray1<'_, f32>,
    my_sales_price: PyReadonlyArray1<'_, f32>,
    my_purchase_price: PyReadonlyArray1<'_, f32>,
    my_role: PyReadonlyArray1<'_, i32>,
    my_transparency: PyReadonlyArray2<'_, f32>,
    my_needs_level: f32,
    stock: PyReadonlyArray2<'_, f32>,
    role: PyReadonlyArray2<'_, i32>,
    stock_limit: PyReadonlyArray2<'_, f32>,
    purchase_price: PyReadonlyArray2<'_, f32>,
    sales_price: PyReadonlyArray2<'_, f32>,
    needs_level: PyReadonlyArray1<'_, f32>,
    transparency: PyReadonlyArray3<'_, f32>,
) -> PyResult<Option<(f32, i32, i32, i32)>> {
    let _ = my_stock;
    let _ = my_needs_level;
    let candidate = search_best_exchange_internal(
        goods,
        need_good,
        initial_transparency,
        elastic_need.as_array(),
        candidate_offer_goods.as_array(),
        friend_ids.as_array(),
        reciprocal_slots.as_array(),
        my_sales_price.as_array(),
        my_purchase_price.as_array(),
        my_role.as_array(),
        my_transparency.as_array(),
        stock.as_array(),
        role.as_array(),
        stock_limit.as_array(),
        purchase_price.as_array(),
        sales_price.as_array(),
        needs_level.as_array(),
        transparency.as_array(),
    );
    Ok(candidate.map(|item| (item.score, item.friend_slot, item.friend_id, item.offer_good)))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn plan_best_exchange(
    goods: usize,
    need_good: usize,
    initial_transparency: f64,
    elastic_need: PyReadonlyArray1<'_, f32>,
    candidate_offer_goods: PyReadonlyArray1<'_, i32>,
    friend_ids: PyReadonlyArray1<'_, i32>,
    reciprocal_slots: PyReadonlyArray1<'_, i32>,
    my_stock: PyReadonlyArray1<'_, f32>,
    my_sales_price: PyReadonlyArray1<'_, f32>,
    my_purchase_price: PyReadonlyArray1<'_, f32>,
    my_role: PyReadonlyArray1<'_, i32>,
    my_transparency: PyReadonlyArray2<'_, f32>,
    my_needs_level: f32,
    stock: PyReadonlyArray2<'_, f32>,
    role: PyReadonlyArray2<'_, i32>,
    stock_limit: PyReadonlyArray2<'_, f32>,
    purchase_price: PyReadonlyArray2<'_, f32>,
    sales_price: PyReadonlyArray2<'_, f32>,
    needs_level: PyReadonlyArray1<'_, f32>,
    transparency: PyReadonlyArray3<'_, f32>,
    max_need: f64,
    min_trade_quantity: f64,
    trade_rounding_buffer: f64,
) -> PyResult<Option<(i32, f32, i32, i32, i32, i32, f64, f64, f64, f64)>> {
    if need_good >= goods {
        return Ok(None);
    }

    let elastic_need = elastic_need.as_array();
    let candidate_offer_goods = candidate_offer_goods.as_array();
    let friend_ids = friend_ids.as_array();
    let reciprocal_slots = reciprocal_slots.as_array();
    let my_stock = my_stock.as_array();
    let my_sales_price = my_sales_price.as_array();
    let my_purchase_price = my_purchase_price.as_array();
    let my_role = my_role.as_array();
    let my_transparency = my_transparency.as_array();
    let stock = stock.as_array();
    let role = role.as_array();
    let stock_limit = stock_limit.as_array();
    let purchase_price = purchase_price.as_array();
    let sales_price = sales_price.as_array();
    let needs_level = needs_level.as_array();
    let transparency = transparency.as_array();

    let Some(candidate) = search_best_exchange_internal(
        goods,
        need_good,
        initial_transparency as f32,
        elastic_need,
        candidate_offer_goods,
        friend_ids,
        reciprocal_slots,
        my_sales_price,
        my_purchase_price,
        my_role,
        my_transparency,
        stock,
        role,
        stock_limit,
        purchase_price,
        sales_price,
        needs_level,
        transparency,
    ) else {
        return Ok(None);
    };

    let fid = candidate.friend_id as usize;
    let offer_good = candidate.offer_good as usize;
    let need_transparency = my_transparency[[candidate.friend_slot as usize, need_good]] as f64;
    let receiving_transparency = if candidate.reciprocal_slot >= 0 {
        transparency[[fid, candidate.reciprocal_slot as usize, offer_good]] as f64
    } else {
        initial_transparency
    };
    let my_need_purchase_price = my_purchase_price[need_good] as f64;
    let my_offer_sales_price = my_sales_price[offer_good] as f64;
    let friend_need_sales_price = sales_price[[fid, need_good]] as f64;
    let friend_offer_purchase_price = purchase_price[[fid, offer_good]] as f64;

    let switch_average = (
        (friend_need_sales_price / (friend_offer_purchase_price * need_transparency).max(EPSILON as f64))
            + ((my_need_purchase_price * receiving_transparency)
                / my_offer_sales_price.max(EPSILON as f64))
    ) / 2.0;

    let friend_needs_level = needs_level[fid] as f64;
    let mut max_exchange = (((my_stock[offer_good] as f64) - ((elastic_need[offer_good] as f64) * (my_needs_level as f64)))
        * receiving_transparency)
        / switch_average.max(EPSILON as f64);
    let reason_code = if max_exchange <= min_trade_quantity {
        PLAN_OFFER_SURPLUS_BELOW_MIN
    } else {
        max_exchange = max_exchange.min(max_need);
        let friend_supply =
            ((stock[[fid, need_good]] as f64) - (friend_needs_level * (elastic_need[need_good] as f64))) * need_transparency;
        max_exchange = max_exchange.min(friend_supply);
        if max_exchange <= min_trade_quantity {
            PLAN_FRIEND_SUPPLY_BELOW_MIN
        } else if role[[fid, offer_good]] == ROLE_RETAILER {
            let stock_capacity = (stock_limit[[fid, offer_good]] - stock[[fid, offer_good]]) as f64;
            max_exchange = max_exchange.min(stock_capacity / switch_average.max(EPSILON as f64));
            if max_exchange <= min_trade_quantity {
                PLAN_PARTNER_CAPACITY_BELOW_MIN
            } else {
                max_exchange = (max_exchange - trade_rounding_buffer) as f32 as f64;
                if max_exchange < min_trade_quantity {
                    PLAN_ROUNDING_BUFFER_BELOW_MIN
                } else {
                    PLAN_OK
                }
            }
        } else {
            let immediate_need = (friend_needs_level * (elastic_need[offer_good] as f64)) - (stock[[fid, offer_good]] as f64);
            max_exchange = max_exchange.min(immediate_need / switch_average.max(EPSILON as f64));
            if max_exchange <= min_trade_quantity {
                PLAN_PARTNER_NEED_BELOW_MIN
            } else {
                max_exchange = (max_exchange - trade_rounding_buffer) as f32 as f64;
                if max_exchange < min_trade_quantity {
                    PLAN_ROUNDING_BUFFER_BELOW_MIN
                } else {
                    PLAN_OK
                }
            }
        }
    };

    Ok(Some((
        reason_code,
        candidate.score,
        candidate.friend_slot,
        candidate.friend_id,
        candidate.offer_good,
        candidate.reciprocal_slot,
        max_exchange,
        switch_average,
        need_transparency,
        receiving_transparency,
    )))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn end_agent_period(
    agent_id: usize,
    cycle: usize,
    goods: usize,
    acquaintances: usize,
    history: i32,
    initial_efficiency: f64,
    gifted_efficiency_floor: f64,
    initial_transparency: f64,
    stock_limit_multiplier: f64,
    activity_discount: f64,
    spoilage_rate: f64,
    stock_spoil_threshold: f64,
    price_reduction: f64,
    price_hike: f64,
    price_leap: f64,
    max_stocklimit_decrease: f64,
    max_stocklimit_increase: f64,
    max_efficiency_downgrade: f64,
    max_efficiency_upgrade: f64,
    base_need: PyReadonlyArray2<'_, f32>,
    mut stock: PyReadwriteArray2<'_, f32>,
    mut stock_limit: PyReadwriteArray2<'_, f32>,
    mut previous_stock_limit: PyReadwriteArray2<'_, f32>,
    mut efficiency: PyReadwriteArray2<'_, f32>,
    mut learned_efficiency: PyReadwriteArray2<'_, f32>,
    mut recent_production: PyReadwriteArray2<'_, f32>,
    mut recent_sales: PyReadwriteArray2<'_, f32>,
    mut recent_purchases: PyReadwriteArray2<'_, f32>,
    mut recent_inventory_inflow: PyReadwriteArray2<'_, f32>,
    mut produced_this_period: PyReadwriteArray2<'_, f32>,
    mut produced_last_period: PyReadwriteArray2<'_, f32>,
    mut sold_this_period: PyReadwriteArray2<'_, f32>,
    mut sold_last_period: PyReadwriteArray2<'_, f32>,
    mut purchased_this_period: PyReadwriteArray2<'_, f32>,
    mut purchased_last_period: PyReadwriteArray2<'_, f32>,
    mut purchase_times: PyReadwriteArray2<'_, i32>,
    mut sales_times: PyReadwriteArray2<'_, i32>,
    mut sum_period_purchase_value: PyReadwriteArray2<'_, f32>,
    mut sum_period_sales_value: PyReadwriteArray2<'_, f32>,
    mut spoilage: PyReadwriteArray2<'_, f32>,
    mut periodic_spoilage: PyReadwriteArray1<'_, f32>,
    talent_mask: PyReadonlyArray2<'_, f32>,
    mut role: PyReadwriteArray2<'_, i32>,
    mut purchase_price: PyReadwriteArray2<'_, f32>,
    mut sales_price: PyReadwriteArray2<'_, f32>,
    mut friend_activity: PyReadwriteArray2<'_, f32>,
    friend_purchased: PyReadonlyArray3<'_, f32>,
    mut transparency: PyReadwriteArray3<'_, f32>,
    needs_level: PyReadonlyArray1<'_, f32>,
    market_elastic_need: PyReadonlyArray1<'_, f32>,
    mut market_periodic_spoilage: PyReadwriteArray1<'_, f32>,
) -> PyResult<()> {
    let epsilon64 = EPSILON as f64;
    let base_need = base_need.as_array();
    let mut stock = stock.as_array_mut();
    let mut stock_limit = stock_limit.as_array_mut();
    let mut previous_stock_limit = previous_stock_limit.as_array_mut();
    let mut efficiency = efficiency.as_array_mut();
    let mut learned_efficiency = learned_efficiency.as_array_mut();
    let mut recent_production = recent_production.as_array_mut();
    let mut recent_sales = recent_sales.as_array_mut();
    let mut recent_purchases = recent_purchases.as_array_mut();
    let mut recent_inventory_inflow = recent_inventory_inflow.as_array_mut();
    let produced_this_period = produced_this_period.as_array_mut();
    let mut produced_last_period = produced_last_period.as_array_mut();
    let sold_this_period = sold_this_period.as_array_mut();
    let mut sold_last_period = sold_last_period.as_array_mut();
    let purchased_this_period = purchased_this_period.as_array_mut();
    let mut purchased_last_period = purchased_last_period.as_array_mut();
    let mut purchase_times = purchase_times.as_array_mut();
    let mut sales_times = sales_times.as_array_mut();
    let mut sum_period_purchase_value = sum_period_purchase_value.as_array_mut();
    let mut sum_period_sales_value = sum_period_sales_value.as_array_mut();
    let mut spoilage = spoilage.as_array_mut();
    let mut periodic_spoilage = periodic_spoilage.as_array_mut();
    let talent_mask = talent_mask.as_array();
    let mut role = role.as_array_mut();
    let mut purchase_price = purchase_price.as_array_mut();
    let mut sales_price = sales_price.as_array_mut();
    let mut friend_activity = friend_activity.as_array_mut();
    let friend_purchased = friend_purchased.as_array();
    let mut transparency = transparency.as_array_mut();
    let needs_level = needs_level.as_array();
    let market_elastic_need = market_elastic_need.as_array();
    let mut market_periodic_spoilage = market_periodic_spoilage.as_array_mut();

    periodic_spoilage[agent_id] = 0.0;

    for good_id in 0..goods {
        let elastic_component = market_elastic_need[good_id] * needs_level[agent_id];
        let target_stock_limit = (stock_limit_multiplier * (elastic_component as f64))
            + (recent_sales[[agent_id, good_id]] as f64);
        let lower_limit = max_stocklimit_decrease * (previous_stock_limit[[agent_id, good_id]] as f64);
        let upper_limit = max_stocklimit_increase * (previous_stock_limit[[agent_id, good_id]] as f64);
        let updated_stock_limit = target_stock_limit.clamp(lower_limit, upper_limit) as f32;
        stock_limit[[agent_id, good_id]] = updated_stock_limit;
        previous_stock_limit[[agent_id, good_id]] = updated_stock_limit;

        if talent_mask[[agent_id, good_id]] > 0.0 {
            let previous_efficiency = efficiency[[agent_id, good_id]] as f64;
            let mut learned = (((recent_production[[agent_id, good_id]] as f64) + 1.0)
                / (((history as f64) * (base_need[[agent_id, good_id]] as f64)).max(epsilon64)))
                .max(epsilon64)
                .sqrt();
            learned = learned.max(gifted_efficiency_floor);
            learned = learned.clamp(
                previous_efficiency * max_efficiency_downgrade,
                previous_efficiency * max_efficiency_upgrade,
            );
            learned = learned.max(gifted_efficiency_floor);
            learned_efficiency[[agent_id, good_id]] = learned as f32;
            efficiency[[agent_id, good_id]] = learned as f32;
        } else {
            learned_efficiency[[agent_id, good_id]] = initial_efficiency as f32;
            efficiency[[agent_id, good_id]] = initial_efficiency as f32;
        }

        let recent_produced = recent_production[[agent_id, good_id]] as f64;
        let recent_purchased_value = recent_purchases[[agent_id, good_id]] as f64;
        let recent_sold = recent_sales[[agent_id, good_id]] as f64;
        let mut role_value = ROLE_CONSUMER;
        if recent_produced > recent_purchased_value
            && recent_sold > ((recent_produced + recent_purchased_value) / 2.0)
        {
            role_value = ROLE_PRODUCER;
        } else if recent_produced < recent_purchased_value
            && recent_sold > ((recent_produced + recent_purchased_value) / 2.0)
        {
            role_value = ROLE_RETAILER;
        }
        role[[agent_id, good_id]] = role_value;

        let production_cost = 1.0_f64 / (efficiency[[agent_id, good_id]] as f64).max(epsilon64);
        let surplus = stock[[agent_id, good_id]] as f64;
        let elastic_need = market_elastic_need[good_id] as f64;
        let stock_limit_value = stock_limit[[agent_id, good_id]] as f64;
        let scarcity_case = i32::from(surplus > elastic_need) + i32::from(surplus > (stock_limit_value + elastic_need));

        let mut purchase_price_value = purchase_price[[agent_id, good_id]] as f64;
        if scarcity_case == 0 {
            if role_value == ROLE_CONSUMER {
                if recent_purchases[[agent_id, good_id]] < (recent_production[[agent_id, good_id]] + 1.0) {
                    purchase_price_value = production_cost;
                } else if purchase_times[[agent_id, good_id]] == 0
                    && purchase_price[[agent_id, good_id]] < production_cost as f32
                {
                    purchase_price_value = mul_assign_like_numpy_float32(purchase_price_value, price_leap);
                }
            } else if role_value == ROLE_RETAILER {
                if purchased_this_period[[agent_id, good_id]] < (sold_this_period[[agent_id, good_id]] + 1.0)
                    || purchased_this_period[[agent_id, good_id]] < (stock_limit[[agent_id, good_id]] / (history as f32).max(1.0))
                {
                    purchase_price_value = mul_assign_like_numpy_float32(purchase_price_value, price_hike);
                }
            }
        } else if scarcity_case == 1 {
            if role_value == ROLE_RETAILER {
                if purchase_times[[agent_id, good_id]] > 1
                    && purchased_this_period[[agent_id, good_id]] > (stock_limit[[agent_id, good_id]] / 2.0)
                {
                    purchase_price_value = ((sum_period_purchase_value[[agent_id, good_id]]
                        + ((history as f32) * purchase_price[[agent_id, good_id]]))
                        / ((purchase_times[[agent_id, good_id]] + history) as f32)) as f64;
                }
                if purchase_price_value > sales_price[[agent_id, good_id]] as f64 {
                    purchase_price_value = mul_assign_like_numpy_float32(sales_price[[agent_id, good_id]] as f64, price_reduction);
                }
            } else if role_value == ROLE_PRODUCER
                && purchased_this_period[[agent_id, good_id]] > produced_this_period[[agent_id, good_id]]
            {
                purchase_price_value = mul_assign_like_numpy_float32(purchase_price_value, price_reduction);
            }
        } else if role_value == ROLE_CONSUMER {
            purchase_price_value = mul_assign_like_numpy_float32(purchase_price_value, price_reduction);
        } else if (role_value == ROLE_RETAILER || role_value == ROLE_PRODUCER)
            && purchase_times[[agent_id, good_id]] > 0
        {
            purchase_price_value = mul_assign_like_numpy_float32(purchase_price_value, price_reduction);
        }
        purchase_price[[agent_id, good_id]] = purchase_price_value.max(0.05) as f32;

        let mut sales_price_value = sales_price[[agent_id, good_id]] as f64;
        let previous_sales_price = sales_price_value;
        if scarcity_case == 0 {
            if role_value == ROLE_CONSUMER {
                let target = production_cost.min(purchase_price[[agent_id, good_id]] as f64);
                if sales_price_value < target {
                    sales_price_value = price_hike * target;
                }
            } else if role_value == ROLE_RETAILER {
                if sales_times[[agent_id, good_id]] > 1
                    && sold_this_period[[agent_id, good_id]] > (stock_limit[[agent_id, good_id]] / 2.0)
                {
                    let blended_price = ((sum_period_sales_value[[agent_id, good_id]]
                        + sales_price[[agent_id, good_id]])
                        / ((sales_times[[agent_id, good_id]] + 1) as f32)) as f64;
                    sales_price_value = blended_price.min(price_leap * previous_sales_price);
                }
                if sales_price_value < purchase_price[[agent_id, good_id]] as f64 {
                    sales_price_value = purchase_price[[agent_id, good_id]] as f64;
                }
                if sales_times[[agent_id, good_id]] == 0 {
                    sales_price_value = mul_assign_like_numpy_float32(purchase_price[[agent_id, good_id]] as f64, price_hike);
                }
            } else if role_value == ROLE_PRODUCER && sales_price_value < production_cost {
                sales_price_value = price_hike * production_cost;
            }
        } else if scarcity_case == 1 {
            if role_value == ROLE_CONSUMER {
                sales_price_value = production_cost.max(purchase_price[[agent_id, good_id]] as f64);
                if surplus > (stock_limit_value / 2.0) {
                    sales_price_value = ((purchase_price[[agent_id, good_id]] as f64) * 2.0).min(production_cost);
                }
            } else if role_value == ROLE_RETAILER {
                if sold_this_period[[agent_id, good_id]] < market_elastic_need[good_id] {
                    sales_price_value = mul_assign_like_numpy_float32(sales_price_value, price_reduction);
                }
            } else if role_value == ROLE_PRODUCER {
                if sold_this_period[[agent_id, good_id]] < market_elastic_need[good_id] {
                    sales_price_value = mul_assign_like_numpy_float32(sales_price_value, price_reduction);
                }
                if sales_price_value < production_cost {
                    sales_price_value = price_hike * production_cost;
                }
            }
        } else if role_value == ROLE_CONSUMER {
            sales_price_value = mul_assign_like_numpy_float32(sales_price_value, price_reduction);
        } else if role_value == ROLE_RETAILER {
            if sold_this_period[[agent_id, good_id]] < market_elastic_need[good_id] {
                if sales_price_value > purchase_price[[agent_id, good_id]] as f64 {
                    sales_price_value = purchase_price[[agent_id, good_id]] as f64;
                } else {
                    sales_price_value = mul_assign_like_numpy_float32(sales_price_value, price_reduction);
                }
            }
        } else if role_value == ROLE_PRODUCER {
            if sales_price_value > production_cost {
                sales_price_value = price_hike * production_cost;
            }
            if sold_this_period[[agent_id, good_id]] < market_elastic_need[good_id] {
                sales_price_value = price_hike * production_cost;
            }
        }
        sales_price[[agent_id, good_id]] = sales_price_value.max(0.05) as f32;

        spoilage[[agent_id, good_id]] = 0.0;
        if (stock[[agent_id, good_id]] as f64) > (stock_limit_multiplier * (market_elastic_need[good_id] as f64)) {
            if (stock[[agent_id, good_id]] as f64) > (stock_spoil_threshold * (stock_limit[[agent_id, good_id]] as f64)) {
                if stock[[agent_id, good_id]] > stock_limit[[agent_id, good_id]] {
                    let spoiled = ((stock[[agent_id, good_id]] as f64) - (stock_limit[[agent_id, good_id]] as f64)) * spoilage_rate;
                    spoilage[[agent_id, good_id]] = spoiled as f32;
                    stock[[agent_id, good_id]] -= spoiled as f32;
                    periodic_spoilage[agent_id] += spoiled as f32;
                    market_periodic_spoilage[good_id] += spoiled as f32;
                }
            }
        }

        recent_production[[agent_id, good_id]] *= activity_discount as f32;
        recent_sales[[agent_id, good_id]] *= activity_discount as f32;
        recent_purchases[[agent_id, good_id]] *= activity_discount as f32;
        recent_inventory_inflow[[agent_id, good_id]] *= activity_discount as f32;
        purchase_times[[agent_id, good_id]] = 0;
        sales_times[[agent_id, good_id]] = 0;
        sum_period_purchase_value[[agent_id, good_id]] = 0.0;
        sum_period_sales_value[[agent_id, good_id]] = 0.0;
        produced_last_period[[agent_id, good_id]] = produced_this_period[[agent_id, good_id]];
        sold_last_period[[agent_id, good_id]] = sold_this_period[[agent_id, good_id]];
        purchased_last_period[[agent_id, good_id]] = purchased_this_period[[agent_id, good_id]];
    }

    for friend_slot in 0..acquaintances {
        for good_id in 0..goods {
            let mut transparency_value = initial_transparency;
            let transactions = friend_activity[[agent_id, friend_slot]] as f64;
            if transactions > 0.0 {
                transparency_value += ((1.0 - transparency_value) * 0.7)
                    * (transactions / (transactions + goods as f64));
            }
            let purchased = friend_purchased[[agent_id, friend_slot, good_id]] as f64;
            transparency_value += ((1.0 - transparency_value) * 0.7)
                * ((10.0 * purchased)
                    / ((10.0 * purchased) + (cycle + 1) as f64).max(epsilon64));
            let recent_purchased_value = recent_purchases[[agent_id, good_id]] as f64;
            transparency_value += ((1.0 - transparency_value) * 0.7)
                * (recent_purchased_value
                    / (recent_purchased_value + (10.0 * history as f64)).max(epsilon64));
            if talent_mask[[agent_id, good_id]] > 0.0 {
                transparency_value += (1.0 - transparency_value) * 0.5;
            }
            transparency[[agent_id, friend_slot, good_id]] = transparency_value.min(1.0) as f32;
        }
        if friend_activity[[agent_id, friend_slot]] > 1.0 {
            friend_activity[[agent_id, friend_slot]] *= 0.9;
        }
    }

    Ok(())
}


#[derive(Default)]
struct NativeStageTotals {
    cycle_need_total: f64,
    production_total: f64,
    surplus_output_total: f64,
    stock_consumption_total: f64,
    leisure_extra_need_total: f64,
}

#[allow(clippy::too_many_arguments)]
fn prepare_agent_for_consumption_internal(
    agent_id: usize,
    goods: usize,
    period_length: f32,
    history: i32,
    basic_round_elastic: bool,
    stock_limit_multiplier: f32,
    max_needs_increase: f32,
    max_needs_reduction: f32,
    small_needs_increase: f32,
    small_needs_reduction: f32,
    _base_need: ArrayView2<'_, f32>,
    mut need: ArrayViewMut2<'_, f32>,
    mut stock: ArrayViewMut2<'_, f32>,
    purchase_price: ArrayView2<'_, f32>,
    sales_price: ArrayView2<'_, f32>,
    purchased_last_period: ArrayView2<'_, f32>,
    recent_sales: ArrayView2<'_, f32>,
    sold_this_period: ArrayView2<'_, f32>,
    sold_last_period: ArrayView2<'_, f32>,
    recent_purchases: ArrayView2<'_, f32>,
    efficiency: ArrayView2<'_, f32>,
    period_failure: ArrayView1<'_, bool>,
    period_time_debt: ArrayView1<'_, f32>,
    mut needs_level: ArrayViewMut1<'_, f32>,
    mut recent_needs_increment: ArrayViewMut1<'_, f32>,
    market_elastic_need: ArrayView1<'_, f32>,
) -> (f64, f64) {
    let current_needs_level = needs_level[agent_id];
    let mut wealth_minus_needs = period_length;
    let mut total_needs_value = 0.0_f32;

    for good_id in 0..goods {
        let elastic_need = market_elastic_need[good_id] * current_needs_level;
        let mut surplus_value = stock[[agent_id, good_id]] - (elastic_need * max_needs_increase);
        if surplus_value < 0.0 {
            if purchased_last_period[[agent_id, good_id]] > (-1.0 * surplus_value) {
                surplus_value *= purchase_price[[agent_id, good_id]];
            } else {
                surplus_value /= efficiency[[agent_id, good_id]].max(EPSILON);
            }
        } else if recent_sales[[agent_id, good_id]] > surplus_value {
            let cap = stock_limit_multiplier
                * ((sold_this_period[[agent_id, good_id]] - sold_last_period[[agent_id, good_id]]) + elastic_need);
            surplus_value = surplus_value.min(cap);
            surplus_value *= sales_price[[agent_id, good_id]];
        } else {
            surplus_value = surplus_value.min(elastic_need * max_needs_increase);
            surplus_value *= purchase_price[[agent_id, good_id]].min(1.0 / efficiency[[agent_id, good_id]].max(EPSILON));
        }
        wealth_minus_needs += surplus_value;

        if recent_purchases[[agent_id, good_id]] > elastic_need {
            total_needs_value += purchase_price[[agent_id, good_id]] * elastic_need;
        } else {
            total_needs_value += elastic_need / efficiency[[agent_id, good_id]].max(EPSILON);
        }
    }

    let stock_level = if total_needs_value <= EPSILON {
        1.0
    } else {
        (total_needs_value + wealth_minus_needs) / total_needs_value
    };

    let previous_level = needs_level[agent_id] as f64;
    let debt = period_time_debt[agent_id] as f64;
    if stock_level < max_needs_reduction || period_failure[agent_id] {
        needs_level[agent_id] *= max_needs_reduction;
    } else if debt > ((1.0 - max_needs_increase as f64) * (period_length as f64))
        && (stock_level as f64) > (max_needs_increase as f64)
    {
        needs_level[agent_id] *= max_needs_increase;
    } else if stock_level > small_needs_increase {
        needs_level[agent_id] *= small_needs_increase;
    } else {
        needs_level[agent_id] *= small_needs_reduction;
    }
    if needs_level[agent_id] < 1.0 {
        needs_level[agent_id] = 1.0;
    }
    if debt < (-1.0 * period_length as f64) {
        needs_level[agent_id] = 1.0;
    }
    let level_ratio = (needs_level[agent_id] as f64) / previous_level.max(EPSILON as f64);
    recent_needs_increment[agent_id] = ((
        level_ratio + ((history as f64) * (recent_needs_increment[agent_id] as f64))
    ) / ((history + 1) as f64)) as f32;

    let mut cycle_need_total = 0.0_f32;
    let mut stock_consumed_total = 0.0_f32;
    let updated_level = needs_level[agent_id];
    for good_id in 0..goods {
        let need_value = if basic_round_elastic && updated_level >= small_needs_increase {
            market_elastic_need[good_id] * updated_level
        } else {
            market_elastic_need[good_id]
        };
        need[[agent_id, good_id]] = need_value;
        cycle_need_total += need_value;

        let consumed = stock[[agent_id, good_id]].min(need[[agent_id, good_id]]);
        stock[[agent_id, good_id]] -= consumed;
        need[[agent_id, good_id]] -= consumed;
        stock_consumed_total += consumed;
    }

    (cycle_need_total as f64, stock_consumed_total as f64)
}

fn produce_need_internal(
    agent_id: usize,
    goods: usize,
    efficiency: ArrayView2<'_, f32>,
    mut need: ArrayViewMut2<'_, f32>,
    mut time_remaining: ArrayViewMut1<'_, f32>,
    mut recent_production: ArrayViewMut2<'_, f32>,
    mut produced_this_period: ArrayViewMut2<'_, f32>,
    mut timeout: ArrayViewMut1<'_, i32>,
) -> f64 {
    let mut produced_total = 0.0_f32;
    let mut time_spent = 0.0_f32;
    for good_id in 0..goods {
        let pending = need[[agent_id, good_id]];
        if pending <= 0.0 {
            continue;
        }
        time_spent += pending / efficiency[[agent_id, good_id]].max(EPSILON);
        recent_production[[agent_id, good_id]] += pending;
        produced_this_period[[agent_id, good_id]] += pending;
        need[[agent_id, good_id]] = 0.0;
        produced_total += pending;
    }
    time_remaining[agent_id] -= time_spent;
    if time_remaining[agent_id] < 0.0 {
        timeout[agent_id] += 1;
    }
    produced_total as f64
}

#[allow(clippy::too_many_arguments)]
fn prepare_leisure_round_internal(
    agent_id: usize,
    goods: usize,
    period_length: f32,
    history: i32,
    leisure_time: f32,
    max_needs_increase: f32,
    max_leisure_extra_multiplier: f32,
    small_needs_increase: f32,
    basic_round_elastic: bool,
    market_elastic_need: ArrayView1<'_, f32>,
    time_remaining: ArrayView1<'_, f32>,
    needs_level: ArrayView1<'_, f32>,
    mut recent_needs_increment: ArrayViewMut1<'_, f32>,
    mut need: ArrayViewMut2<'_, f32>,
    mut stock: ArrayViewMut2<'_, f32>,
) -> (bool, f64, f64) {
    let remaining_time = time_remaining[agent_id] as f64;
    if remaining_time <= leisure_time as f64 {
        return (false, 0.0, 0.0);
    }

    let utilized_time = ((period_length as f64) - remaining_time).max(1.0);
    let raw_increment = (period_length as f64) / utilized_time;
    let capped_increment = raw_increment.min((recent_needs_increment[agent_id] as f64) * (max_needs_increase as f64));
    let extra_multiplier = (capped_increment - 1.0)
        .max(0.0)
        .min(max_leisure_extra_multiplier as f64);
    if extra_multiplier <= 0.0 {
        return (false, 0.0, 0.0);
    }

    recent_needs_increment[agent_id] = ((
        capped_increment + ((history as f64) * (recent_needs_increment[agent_id] as f64))
    ) / ((history + 1) as f64)) as f32;

    let needs_level_value = needs_level[agent_id];
    let apply_elastic = basic_round_elastic && needs_level_value >= small_needs_increase;
    let extra_multiplier_f32 = extra_multiplier as f32;
    let mut extra_need_total = 0.0_f32;
    let mut stock_consumed_total = 0.0_f32;
    for good_id in 0..goods {
        let baseline_need = if apply_elastic {
            market_elastic_need[good_id] * needs_level_value
        } else {
            market_elastic_need[good_id]
        };
        let extra_need = baseline_need * extra_multiplier_f32;
        need[[agent_id, good_id]] += extra_need;
        extra_need_total += extra_need;

        let consumed = stock[[agent_id, good_id]].min(need[[agent_id, good_id]]);
        stock[[agent_id, good_id]] -= consumed;
        need[[agent_id, good_id]] -= consumed;
        stock_consumed_total += consumed;
    }

    if extra_need_total <= 0.0 {
        return (false, 0.0, 0.0);
    }
    (true, extra_need_total as f64, stock_consumed_total as f64)
}

fn add_engine_metric(engine: &Bound<'_, PyAny>, name: &str, delta: f64) -> PyResult<()> {
    if delta == 0.0 {
        return Ok(());
    }
    let current: f64 = engine.getattr(name)?.extract()?;
    engine.setattr(name, current + delta)?;
    Ok(())
}

fn set_period_failure_from_time_remaining(state: &Bound<'_, PyAny>, agent_id: usize) -> PyResult<()> {
    let time_remaining: PyReadonlyArray1<'_, f32> = state.getattr("time_remaining")?.extract()?;
    let mut period_failure: PyReadwriteArray1<'_, bool> = state.getattr("period_failure")?.extract()?;
    let time_remaining = time_remaining.as_array();
    let mut period_failure = period_failure.as_array_mut();
    period_failure[agent_id] = time_remaining[agent_id] < 0.0;
    Ok(())
}

fn apply_half_debt_adjustment(state: &Bound<'_, PyAny>, agent_id: usize) -> PyResult<()> {
    let mut period_time_debt: PyReadwriteArray1<'_, f32> = state.getattr("period_time_debt")?.extract()?;
    let mut time_remaining: PyReadwriteArray1<'_, f32> = state.getattr("time_remaining")?.extract()?;
    let mut period_time_debt = period_time_debt.as_array_mut();
    let mut time_remaining = time_remaining.as_array_mut();
    if period_time_debt[agent_id] < 0.0 {
        let half_debt = period_time_debt[agent_id] / 2.0;
        time_remaining[agent_id] += half_debt;
        period_time_debt[agent_id] = half_debt;
    }
    Ok(())
}

fn finalize_period_debt(state: &Bound<'_, PyAny>, agent_id: usize) -> PyResult<()> {
    let mut period_time_debt: PyReadwriteArray1<'_, f32> = state.getattr("period_time_debt")?.extract()?;
    let time_remaining: PyReadonlyArray1<'_, f32> = state.getattr("time_remaining")?.extract()?;
    let mut period_time_debt = period_time_debt.as_array_mut();
    let time_remaining = time_remaining.as_array();
    period_time_debt[agent_id] += time_remaining[agent_id];
    if period_time_debt[agent_id] > 0.0 {
        period_time_debt[agent_id] = 0.0;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_agent_cycle_owned(
    py: Python<'_>,
    runner: &Bound<'_, PyAny>,
    state: &Bound<'_, PyAny>,
    market: &Bound<'_, PyAny>,
    agent_id: usize,
    goods: usize,
    period_length: f32,
    history: i32,
    basic_round_elastic: bool,
    stock_limit_multiplier: f32,
    max_needs_increase: f32,
    max_needs_reduction: f32,
    small_needs_increase: f32,
    small_needs_reduction: f32,
    leisure_time: f32,
    max_leisure_extra_multiplier: f32,
    totals: &mut NativeStageTotals,
) -> PyResult<()> {
    runner.call_method1("_prepare_agent_for_consumption", (agent_id,))?;
    runner.call_method1("_satisfy_needs_by_exchange", (agent_id,))?;
    runner.call_method1("_advance_agent_to_surplus_stage", (agent_id,))?;
    runner.call_method1("_make_surplus_deals", (agent_id,))?;
    runner.call_method1("_complete_agent_period_after_surplus", (agent_id,))?;
    let _ = (
        py,
        state,
        market,
        goods,
        period_length,
        history,
        basic_round_elastic,
        stock_limit_multiplier,
        max_needs_increase,
        max_needs_reduction,
        small_needs_increase,
        small_needs_reduction,
        leisure_time,
        max_leisure_extra_multiplier,
        totals,
    );
    Ok(())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn prepare_agent_for_consumption(
    agent_id: usize,
    goods: usize,
    period_length: f32,
    history: i32,
    basic_round_elastic: bool,
    stock_limit_multiplier: f32,
    max_needs_increase: f32,
    max_needs_reduction: f32,
    small_needs_increase: f32,
    small_needs_reduction: f32,
    base_need: PyReadonlyArray2<'_, f32>,
    mut need: PyReadwriteArray2<'_, f32>,
    mut stock: PyReadwriteArray2<'_, f32>,
    purchase_price: PyReadonlyArray2<'_, f32>,
    sales_price: PyReadonlyArray2<'_, f32>,
    purchased_last_period: PyReadonlyArray2<'_, f32>,
    recent_sales: PyReadonlyArray2<'_, f32>,
    sold_this_period: PyReadonlyArray2<'_, f32>,
    sold_last_period: PyReadonlyArray2<'_, f32>,
    recent_purchases: PyReadonlyArray2<'_, f32>,
    efficiency: PyReadonlyArray2<'_, f32>,
    period_failure: PyReadonlyArray1<'_, bool>,
    period_time_debt: PyReadonlyArray1<'_, f32>,
    mut needs_level: PyReadwriteArray1<'_, f32>,
    mut recent_needs_increment: PyReadwriteArray1<'_, f32>,
    market_elastic_need: PyReadonlyArray1<'_, f32>,
) -> PyResult<(f64, f64)> {
    let _base_need = base_need.as_array();
    let mut need = need.as_array_mut();
    let mut stock = stock.as_array_mut();
    let purchase_price = purchase_price.as_array();
    let sales_price = sales_price.as_array();
    let purchased_last_period = purchased_last_period.as_array();
    let recent_sales = recent_sales.as_array();
    let sold_this_period = sold_this_period.as_array();
    let sold_last_period = sold_last_period.as_array();
    let recent_purchases = recent_purchases.as_array();
    let efficiency = efficiency.as_array();
    let period_failure = period_failure.as_array();
    let period_time_debt = period_time_debt.as_array();
    let mut needs_level = needs_level.as_array_mut();
    let mut recent_needs_increment = recent_needs_increment.as_array_mut();
    let market_elastic_need = market_elastic_need.as_array();

    let current_needs_level = needs_level[agent_id];
    let mut wealth_minus_needs = period_length;
    let mut total_needs_value = 0.0_f32;

    for good_id in 0..goods {
        let elastic_need = market_elastic_need[good_id] * current_needs_level;
        let mut surplus_value = stock[[agent_id, good_id]] - (elastic_need * max_needs_increase);
        if surplus_value < 0.0 {
            if purchased_last_period[[agent_id, good_id]] > (-1.0 * surplus_value) {
                surplus_value *= purchase_price[[agent_id, good_id]];
            } else {
                surplus_value /= efficiency[[agent_id, good_id]].max(EPSILON);
            }
        } else if recent_sales[[agent_id, good_id]] > surplus_value {
            let cap = stock_limit_multiplier
                * ((sold_this_period[[agent_id, good_id]] - sold_last_period[[agent_id, good_id]]) + elastic_need);
            surplus_value = surplus_value.min(cap);
            surplus_value *= sales_price[[agent_id, good_id]];
        } else {
            surplus_value = surplus_value.min(elastic_need * max_needs_increase);
            surplus_value *= purchase_price[[agent_id, good_id]].min(1.0 / efficiency[[agent_id, good_id]].max(EPSILON));
        }
        wealth_minus_needs += surplus_value;

        if recent_purchases[[agent_id, good_id]] > elastic_need {
            total_needs_value += purchase_price[[agent_id, good_id]] * elastic_need;
        } else {
            total_needs_value += elastic_need / efficiency[[agent_id, good_id]].max(EPSILON);
        }
    }

    let stock_level = if total_needs_value <= EPSILON {
        1.0
    } else {
        (total_needs_value + wealth_minus_needs) / total_needs_value
    };

    let previous_level = needs_level[agent_id] as f64;
    let debt = period_time_debt[agent_id] as f64;
    if stock_level < max_needs_reduction || period_failure[agent_id] {
        needs_level[agent_id] *= max_needs_reduction;
    } else if debt > ((1.0 - max_needs_increase as f64) * (period_length as f64))
        && (stock_level as f64) > (max_needs_increase as f64)
    {
        needs_level[agent_id] *= max_needs_increase;
    } else if stock_level > small_needs_increase {
        needs_level[agent_id] *= small_needs_increase;
    } else {
        needs_level[agent_id] *= small_needs_reduction;
    }
    if needs_level[agent_id] < 1.0 {
        needs_level[agent_id] = 1.0;
    }
    if debt < (-1.0 * period_length as f64) {
        needs_level[agent_id] = 1.0;
    }
    let level_ratio = (needs_level[agent_id] as f64) / previous_level.max(EPSILON as f64);
    recent_needs_increment[agent_id] = ((
        level_ratio + ((history as f64) * (recent_needs_increment[agent_id] as f64))
    ) / ((history + 1) as f64)) as f32;

    let mut cycle_need_total = 0.0_f32;
    let mut stock_consumed_total = 0.0_f32;
    let updated_level = needs_level[agent_id];
    for good_id in 0..goods {
        let need_value = if basic_round_elastic && updated_level >= small_needs_increase {
            market_elastic_need[good_id] * updated_level
        } else {
            market_elastic_need[good_id]
        };
        need[[agent_id, good_id]] = need_value;
        cycle_need_total += need_value;

        let consumed = stock[[agent_id, good_id]].min(need[[agent_id, good_id]]);
        stock[[agent_id, good_id]] -= consumed;
        need[[agent_id, good_id]] -= consumed;
        stock_consumed_total += consumed;
    }

    Ok((cycle_need_total as f64, stock_consumed_total as f64))
}

#[pyfunction]
fn produce_need(
    agent_id: usize,
    goods: usize,
    efficiency: PyReadonlyArray2<'_, f32>,
    mut need: PyReadwriteArray2<'_, f32>,
    mut time_remaining: PyReadwriteArray1<'_, f32>,
    mut recent_production: PyReadwriteArray2<'_, f32>,
    mut produced_this_period: PyReadwriteArray2<'_, f32>,
    mut timeout: PyReadwriteArray1<'_, i32>,
) -> PyResult<f64> {
    let efficiency = efficiency.as_array();
    let mut need = need.as_array_mut();
    let mut time_remaining = time_remaining.as_array_mut();
    let mut recent_production = recent_production.as_array_mut();
    let mut produced_this_period = produced_this_period.as_array_mut();
    let mut timeout = timeout.as_array_mut();

    let mut produced_total = 0.0_f32;
    let mut time_spent = 0.0_f32;
    for good_id in 0..goods {
        let pending = need[[agent_id, good_id]];
        if pending <= 0.0 {
            continue;
        }
        time_spent += pending / efficiency[[agent_id, good_id]].max(EPSILON);
        recent_production[[agent_id, good_id]] += pending;
        produced_this_period[[agent_id, good_id]] += pending;
        need[[agent_id, good_id]] = 0.0;
        produced_total += pending;
    }
    time_remaining[agent_id] -= time_spent;
    if time_remaining[agent_id] < 0.0 {
        timeout[agent_id] += 1;
    }
    Ok(produced_total as f64)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn surplus_production(
    agent_id: usize,
    goods: usize,
    switch_time: f32,
    stock_limit_multiplier: f32,
    price_hike: f32,
    base_need: PyReadonlyArray2<'_, f32>,
    mut stock: PyReadwriteArray2<'_, f32>,
    stock_limit: PyReadonlyArray2<'_, f32>,
    talent_mask: PyReadonlyArray2<'_, f32>,
    purchase_times: PyReadonlyArray2<'_, i32>,
    efficiency: PyReadonlyArray2<'_, f32>,
    sales_price: PyReadonlyArray2<'_, f32>,
    mut time_remaining: PyReadwriteArray1<'_, f32>,
    mut recent_production: PyReadwriteArray2<'_, f32>,
    mut produced_this_period: PyReadwriteArray2<'_, f32>,
) -> PyResult<f64> {
    let base_need = base_need.as_array();
    let mut stock = stock.as_array_mut();
    let stock_limit = stock_limit.as_array();
    let talent_mask = talent_mask.as_array();
    let purchase_times = purchase_times.as_array();
    let efficiency = efficiency.as_array();
    let sales_price = sales_price.as_array();
    let mut time_remaining = time_remaining.as_array_mut();
    let mut recent_production = recent_production.as_array_mut();
    let mut produced_this_period = produced_this_period.as_array_mut();

    let base_need_floor = base_need[[agent_id, 0]];
    let mut produced_total = 0.0_f64;
    while time_remaining[agent_id] >= 1.0 {
        let mut selected_good: Option<usize> = None;
        let mut selected_limit = 0.0_f32;
        let mut best_index = 0.0_f32;
        for good_id in 0..goods {
            if talent_mask[[agent_id, good_id]] <= 0.0 {
                continue;
            }
            let production_limit = stock_limit[[agent_id, good_id]] - stock[[agent_id, good_id]];
            if production_limit <= 1.0 {
                continue;
            }
            let production_profitable =
                (purchase_times[[agent_id, good_id]] == 0
                    && stock[[agent_id, good_id]] > (stock_limit_multiplier * base_need_floor))
                    || ((1.0 / efficiency[[agent_id, good_id]].max(EPSILON))
                        <= (price_hike * sales_price[[agent_id, good_id]]));
            if !production_profitable {
                continue;
            }
            let production_index = efficiency[[agent_id, good_id]]
                - (1.0 / sales_price[[agent_id, good_id]].max(EPSILON));
            if production_index >= best_index {
                best_index = production_index;
                selected_good = Some(good_id);
                selected_limit = production_limit;
            }
        }

        let Some(selected_good) = selected_good else {
            break;
        };

        let max_production = efficiency[[agent_id, selected_good]] * time_remaining[agent_id];
        let produced = max_production.min(selected_limit);
        if produced <= 0.0 {
            break;
        }
        stock[[agent_id, selected_good]] += produced;
        produced_this_period[[agent_id, selected_good]] += produced;
        recent_production[[agent_id, selected_good]] += produced;
        produced_total += produced as f64;

        if max_production < selected_limit {
            time_remaining[agent_id] = 0.0;
        } else {
            time_remaining[agent_id] -= switch_time + (produced / efficiency[[agent_id, selected_good]].max(EPSILON));
        }
    }
    Ok(produced_total)
}


#[pyfunction]
fn leisure_production(
    agent_id: usize,
    goods: usize,
    mut stock: PyReadwriteArray2<'_, f32>,
    stock_limit: PyReadonlyArray2<'_, f32>,
    talent_mask: PyReadonlyArray2<'_, f32>,
    purchase_price: PyReadonlyArray2<'_, f32>,
    mut time_remaining: PyReadwriteArray1<'_, f32>,
    mut recent_production: PyReadwriteArray2<'_, f32>,
    mut produced_this_period: PyReadwriteArray2<'_, f32>,
) -> PyResult<f64> {
    let mut stock = stock.as_array_mut();
    let stock_limit = stock_limit.as_array();
    let talent_mask = talent_mask.as_array();
    let purchase_price = purchase_price.as_array();
    let mut time_remaining = time_remaining.as_array_mut();
    let mut recent_production = recent_production.as_array_mut();
    let mut produced_this_period = produced_this_period.as_array_mut();

    let mut produced_total = 0.0_f64;
    while time_remaining[agent_id] >= 1.0 {
        let mut selected_good: Option<usize> = None;
        let mut selected_limit = 0.0_f32;
        let mut best_index = 0.0_f32;
        for good_id in 0..goods {
            if talent_mask[[agent_id, good_id]] > 0.0 {
                continue;
            }
            let production_limit = stock_limit[[agent_id, good_id]] - stock[[agent_id, good_id]];
            if production_limit > 1.0 && purchase_price[[agent_id, good_id]] >= best_index {
                best_index = purchase_price[[agent_id, good_id]];
                selected_good = Some(good_id);
                selected_limit = production_limit;
            }
        }
        let Some(selected_good) = selected_good else {
            break;
        };

        let produced = time_remaining[agent_id].min(selected_limit);
        if produced <= 0.0 {
            break;
        }
        stock[[agent_id, selected_good]] += produced;
        produced_this_period[[agent_id, selected_good]] += produced;
        recent_production[[agent_id, selected_good]] += produced;
        produced_total += produced as f64;
        time_remaining[agent_id] -= produced;
    }
    Ok(produced_total as f64)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn prepare_leisure_round(
    agent_id: usize,
    goods: usize,
    period_length: f32,
    history: i32,
    leisure_time: f32,
    max_needs_increase: f32,
    max_leisure_extra_multiplier: f32,
    small_needs_increase: f32,
    basic_round_elastic: bool,
    market_elastic_need: PyReadonlyArray1<'_, f32>,
    time_remaining: PyReadonlyArray1<'_, f32>,
    needs_level: PyReadonlyArray1<'_, f32>,
    mut recent_needs_increment: PyReadwriteArray1<'_, f32>,
    mut need: PyReadwriteArray2<'_, f32>,
    mut stock: PyReadwriteArray2<'_, f32>,
) -> PyResult<(bool, f64, f64)> {
    let market_elastic_need = market_elastic_need.as_array();
    let time_remaining = time_remaining.as_array();
    let needs_level = needs_level.as_array();
    let mut recent_needs_increment = recent_needs_increment.as_array_mut();
    let mut need = need.as_array_mut();
    let mut stock = stock.as_array_mut();

    let remaining_time = time_remaining[agent_id] as f64;
    if remaining_time <= leisure_time as f64 {
        return Ok((false, 0.0, 0.0));
    }

    let utilized_time = ((period_length as f64) - remaining_time).max(1.0);
    let raw_increment = (period_length as f64) / utilized_time;
    let capped_increment = raw_increment.min((recent_needs_increment[agent_id] as f64) * (max_needs_increase as f64));
    let extra_multiplier = (capped_increment - 1.0)
        .max(0.0)
        .min(max_leisure_extra_multiplier as f64);
    if extra_multiplier <= 0.0 {
        return Ok((false, 0.0, 0.0));
    }

    recent_needs_increment[agent_id] = ((
        capped_increment + ((history as f64) * (recent_needs_increment[agent_id] as f64))
    ) / ((history + 1) as f64)) as f32;

    let needs_level_value = needs_level[agent_id];
    let apply_elastic = basic_round_elastic && needs_level_value >= small_needs_increase;
    let extra_multiplier_f32 = extra_multiplier as f32;
    let mut extra_need_total = 0.0_f32;
    let mut stock_consumed_total = 0.0_f32;
    for good_id in 0..goods {
        let baseline_need = if apply_elastic {
            market_elastic_need[good_id] * needs_level_value
        } else {
            market_elastic_need[good_id]
        };
        let extra_need = baseline_need * extra_multiplier_f32;
        need[[agent_id, good_id]] += extra_need;
        extra_need_total += extra_need;

        let consumed = stock[[agent_id, good_id]].min(need[[agent_id, good_id]]);
        stock[[agent_id, good_id]] -= consumed;
        need[[agent_id, good_id]] -= consumed;
        stock_consumed_total += consumed;
    }

    if extra_need_total <= 0.0 {
        return Ok((false, 0.0, 0.0));
    }
    Ok((true, extra_need_total as f64, stock_consumed_total as f64))
}


const SURPLUS_DEAL: i32 = 1;
const CONSUMPTION_DEAL: i32 = 2;

#[derive(Clone, Copy)]
struct PlannedExchangeInternal {
    candidate: SearchCandidate,
    reason_code: i32,
    max_exchange: f64,
    switch_average: f64,
    need_transparency: f64,
    receiving_transparency: f64,
}

#[derive(Default)]
struct ExchangeStageTotals {
    proposed_count: i32,
    accepted_count: i32,
    accepted_volume: f64,
    inventory_trade_volume: f64,
}

#[allow(clippy::too_many_arguments)]
fn plan_exchange_stage_candidate(
    agent_id: usize,
    need_good: usize,
    goods: usize,
    initial_transparency: f32,
    max_need: f64,
    min_trade_quantity: f64,
    trade_rounding_buffer: f64,
    elastic_need: ArrayView1<'_, f32>,
    stock: ArrayView2<'_, f32>,
    role: ArrayView2<'_, i32>,
    stock_limit: ArrayView2<'_, f32>,
    purchase_price: ArrayView2<'_, f32>,
    sales_price: ArrayView2<'_, f32>,
    needs_level: ArrayView1<'_, f32>,
    transparency: ArrayView3<'_, f32>,
    friend_ids: ArrayView1<'_, i32>,
    reciprocal_slots: ArrayView1<'_, i32>,
    candidate_offer_goods: ArrayView1<'_, i32>,
) -> Option<PlannedExchangeInternal> {
    let candidate = search_best_exchange_internal(
        goods,
        need_good,
        initial_transparency,
        elastic_need,
        candidate_offer_goods,
        friend_ids,
        reciprocal_slots,
        sales_price.row(agent_id),
        purchase_price.row(agent_id),
        role.row(agent_id),
        transparency.index_axis(Axis(0), agent_id),
        stock,
        role,
        stock_limit,
        purchase_price,
        sales_price,
        needs_level,
        transparency,
    )?;

    let fid = candidate.friend_id as usize;
    let offer_good = candidate.offer_good as usize;
    let need_transparency = transparency[[agent_id, candidate.friend_slot as usize, need_good]] as f64;
    let receiving_transparency = if candidate.reciprocal_slot >= 0 {
        transparency[[fid, candidate.reciprocal_slot as usize, offer_good]] as f64
    } else {
        initial_transparency as f64
    };
    let my_need_purchase_price = purchase_price[[agent_id, need_good]] as f64;
    let my_offer_sales_price = sales_price[[agent_id, offer_good]] as f64;
    let friend_need_sales_price = sales_price[[fid, need_good]] as f64;
    let friend_offer_purchase_price = purchase_price[[fid, offer_good]] as f64;

    let switch_average = (
        (friend_need_sales_price / (friend_offer_purchase_price * need_transparency).max(EPSILON as f64))
            + ((my_need_purchase_price * receiving_transparency)
                / my_offer_sales_price.max(EPSILON as f64))
    ) / 2.0;

    let my_needs_level = needs_level[agent_id] as f64;
    let friend_needs_level = needs_level[fid] as f64;
    let mut max_exchange = (((stock[[agent_id, offer_good]] as f64)
        - ((elastic_need[offer_good] as f64) * my_needs_level))
        * receiving_transparency)
        / switch_average.max(EPSILON as f64);
    let reason_code = if max_exchange <= min_trade_quantity {
        PLAN_OFFER_SURPLUS_BELOW_MIN
    } else {
        max_exchange = max_exchange.min(max_need);
        let friend_supply = (((stock[[fid, need_good]] as f64)
            - (friend_needs_level * (elastic_need[need_good] as f64)))
            * need_transparency);
        max_exchange = max_exchange.min(friend_supply);
        if max_exchange <= min_trade_quantity {
            PLAN_FRIEND_SUPPLY_BELOW_MIN
        } else if role[[fid, offer_good]] == ROLE_RETAILER {
            let stock_capacity = (stock_limit[[fid, offer_good]] - stock[[fid, offer_good]]) as f64;
            max_exchange = max_exchange.min(stock_capacity / switch_average.max(EPSILON as f64));
            if max_exchange <= min_trade_quantity {
                PLAN_PARTNER_CAPACITY_BELOW_MIN
            } else {
                max_exchange = (max_exchange - trade_rounding_buffer) as f32 as f64;
                if max_exchange < min_trade_quantity {
                    PLAN_ROUNDING_BUFFER_BELOW_MIN
                } else {
                    PLAN_OK
                }
            }
        } else {
            let immediate_need = (friend_needs_level * (elastic_need[offer_good] as f64))
                - (stock[[fid, offer_good]] as f64);
            max_exchange = max_exchange.min(immediate_need / switch_average.max(EPSILON as f64));
            if max_exchange <= min_trade_quantity {
                PLAN_PARTNER_NEED_BELOW_MIN
            } else {
                max_exchange = (max_exchange - trade_rounding_buffer) as f32 as f64;
                if max_exchange < min_trade_quantity {
                    PLAN_ROUNDING_BUFFER_BELOW_MIN
                } else {
                    PLAN_OK
                }
            }
        }
    };

    Some(PlannedExchangeInternal {
        candidate,
        reason_code,
        max_exchange,
        switch_average,
        need_transparency,
        receiving_transparency,
    })
}

fn find_friend_slot_scan_internal(friend_ids: ArrayView2<'_, i32>, agent_id: usize, friend_id: i32) -> i32 {
    for friend_slot in 0..friend_ids.ncols() {
        if friend_ids[[agent_id, friend_slot]] == friend_id {
            return friend_slot as i32;
        }
    }
    -1
}

#[allow(clippy::too_many_arguments)]
fn ensure_friend_link_array_only(
    agent_id: usize,
    friend_id: usize,
    acquaintances: usize,
    goods: usize,
    initial_transparency: f32,
    initial_transactions: f32,
    friend_ids: &mut ArrayViewMut2<'_, i32>,
    friend_activity: &mut ArrayViewMut2<'_, f32>,
    friend_purchased: &mut ArrayViewMut3<'_, f32>,
    friend_sold: &mut ArrayViewMut3<'_, f32>,
    transparency: &mut ArrayViewMut3<'_, f32>,
) -> i32 {
    for friend_slot in 0..acquaintances {
        if friend_ids[[friend_id, friend_slot]] == agent_id as i32 {
            return friend_slot as i32;
        }
    }

    let mut target_slot = None;
    for friend_slot in 0..acquaintances {
        if friend_ids[[friend_id, friend_slot]] < 0 {
            target_slot = Some(friend_slot);
            break;
        }
    }
    let target_slot = target_slot.unwrap_or_else(|| {
        let mut best_slot = 0usize;
        let mut best_activity = friend_activity[[friend_id, 0]];
        for friend_slot in 1..acquaintances {
            let activity = friend_activity[[friend_id, friend_slot]];
            if activity < best_activity {
                best_activity = activity;
                best_slot = friend_slot;
            }
        }
        best_slot
    });

    friend_ids[[friend_id, target_slot]] = agent_id as i32;
    friend_activity[[friend_id, target_slot]] = initial_transactions;
    for good_id in 0..goods {
        friend_purchased[[friend_id, target_slot, good_id]] = 0.0;
        friend_sold[[friend_id, target_slot, good_id]] = 0.0;
        transparency[[friend_id, target_slot, good_id]] = initial_transparency;
    }
    target_slot as i32
}

#[pyfunction]
fn run_exchange_stage(py: Python<'_>, runner: PyObject, agent_id: usize, deal_type: i32) -> PyResult<(i32, i32, f64, f64)> {
    let runner = runner.bind(py);
    let state = runner.getattr("state")?;
    let market = runner.getattr("market")?;
    let config = runner.getattr("config")?;

    let goods: usize = config.getattr("goods")?.extract()?;
    let acquaintances: usize = config.getattr("acquaintances")?.extract()?;
    let initial_transparency: f32 = config.getattr("initial_transparency")?.extract()?;
    let min_trade_quantity: f64 = config.getattr("min_trade_quantity")?.extract()?;
    let trade_rounding_buffer: f64 = config.getattr("trade_rounding_buffer")?.extract()?;
    let max_attempts = goods * acquaintances;
    let initial_transactions = 2.0_f32;

    if deal_type != CONSUMPTION_DEAL && deal_type != SURPLUS_DEAL {
        return Err(PyValueError::new_err("deal_type must be 1 (surplus) or 2 (consumption)"));
    }

    let mut totals = ExchangeStageTotals::default();
    let mut changed_agents: HashSet<usize> = HashSet::new();

    {
        let market_elastic_need: PyReadonlyArray1<'_, f32> = market.getattr("elastic_need")?.extract()?;
        let mut market_periodic_tce_cost: PyReadwriteArray1<'_, f32> = market.getattr("periodic_tce_cost")?.extract()?;
        let mut stock: PyReadwriteArray2<'_, f32> = state.getattr("stock")?.extract()?;
        let role: PyReadonlyArray2<'_, i32> = state.getattr("role")?.extract()?;
        let stock_limit: PyReadonlyArray2<'_, f32> = state.getattr("stock_limit")?.extract()?;
        let purchase_price: PyReadonlyArray2<'_, f32> = state.getattr("purchase_price")?.extract()?;
        let sales_price: PyReadonlyArray2<'_, f32> = state.getattr("sales_price")?.extract()?;
        let needs_level: PyReadonlyArray1<'_, f32> = state.getattr("needs_level")?.extract()?;
        let mut transparency: PyReadwriteArray3<'_, f32> = state.getattr("transparency")?.extract()?;
        let mut friend_id: PyReadwriteArray2<'_, i32> = state.getattr("friend_id")?.extract()?;
        let mut need: PyReadwriteArray2<'_, f32> = state.getattr("need")?.extract()?;
        let recent_production: PyReadonlyArray2<'_, f32> = state.getattr("recent_production")?.extract()?;
        let mut recent_sales: PyReadwriteArray2<'_, f32> = state.getattr("recent_sales")?.extract()?;
        let mut recent_purchases: PyReadwriteArray2<'_, f32> = state.getattr("recent_purchases")?.extract()?;
        let mut sold_this_period: PyReadwriteArray2<'_, f32> = state.getattr("sold_this_period")?.extract()?;
        let mut purchased_this_period: PyReadwriteArray2<'_, f32> = state.getattr("purchased_this_period")?.extract()?;
        let mut recent_inventory_inflow: PyReadwriteArray2<'_, f32> = state.getattr("recent_inventory_inflow")?.extract()?;
        let mut purchase_times: PyReadwriteArray2<'_, i32> = state.getattr("purchase_times")?.extract()?;
        let mut sales_times: PyReadwriteArray2<'_, i32> = state.getattr("sales_times")?.extract()?;
        let mut sum_period_purchase_value: PyReadwriteArray2<'_, f32> = state.getattr("sum_period_purchase_value")?.extract()?;
        let mut sum_period_sales_value: PyReadwriteArray2<'_, f32> = state.getattr("sum_period_sales_value")?.extract()?;
        let mut friend_activity: PyReadwriteArray2<'_, f32> = state.getattr("friend_activity")?.extract()?;
        let mut friend_purchased: PyReadwriteArray3<'_, f32> = state.getattr("friend_purchased")?.extract()?;
        let mut friend_sold: PyReadwriteArray3<'_, f32> = state.getattr("friend_sold")?.extract()?;
        let trade = state.getattr("trade")?;
        let mut proposal_friend_slot: PyReadwriteArray1<'_, i32> = trade.getattr("proposal_friend_slot")?.extract()?;
        let mut proposal_target_agent: PyReadwriteArray1<'_, i32> = trade.getattr("proposal_target_agent")?.extract()?;
        let mut proposal_need_good: PyReadwriteArray1<'_, i32> = trade.getattr("proposal_need_good")?.extract()?;
        let mut proposal_offer_good: PyReadwriteArray1<'_, i32> = trade.getattr("proposal_offer_good")?.extract()?;
        let mut proposal_quantity: PyReadwriteArray1<'_, f32> = trade.getattr("proposal_quantity")?.extract()?;
        let mut proposal_score: PyReadwriteArray1<'_, f32> = trade.getattr("proposal_score")?.extract()?;
        let mut accepted_mask: PyReadwriteArray1<'_, bool> = trade.getattr("accepted_mask")?.extract()?;
        let mut accepted_quantity: PyReadwriteArray1<'_, f32> = trade.getattr("accepted_quantity")?.extract()?;

        let mut market_periodic_tce_cost = market_periodic_tce_cost.as_array_mut();
        let mut stock = stock.as_array_mut();
        let role = role.as_array();
        let stock_limit = stock_limit.as_array();
        let purchase_price = purchase_price.as_array();
        let sales_price = sales_price.as_array();
        let needs_level = needs_level.as_array();
        let mut transparency = transparency.as_array_mut();
        let mut friend_id = friend_id.as_array_mut();
        let mut need = need.as_array_mut();
        let recent_production = recent_production.as_array();
        let mut recent_sales = recent_sales.as_array_mut();
        let mut recent_purchases = recent_purchases.as_array_mut();
        let mut sold_this_period = sold_this_period.as_array_mut();
        let mut purchased_this_period = purchased_this_period.as_array_mut();
        let mut recent_inventory_inflow = recent_inventory_inflow.as_array_mut();
        let mut purchase_times = purchase_times.as_array_mut();
        let mut sales_times = sales_times.as_array_mut();
        let mut sum_period_purchase_value = sum_period_purchase_value.as_array_mut();
        let mut sum_period_sales_value = sum_period_sales_value.as_array_mut();
        let mut friend_activity = friend_activity.as_array_mut();
        let mut friend_purchased = friend_purchased.as_array_mut();
        let mut friend_sold = friend_sold.as_array_mut();
        let mut proposal_friend_slot = proposal_friend_slot.as_array_mut();
        let mut proposal_target_agent = proposal_target_agent.as_array_mut();
        let mut proposal_need_good = proposal_need_good.as_array_mut();
        let mut proposal_offer_good = proposal_offer_good.as_array_mut();
        let mut proposal_quantity = proposal_quantity.as_array_mut();
        let mut proposal_score = proposal_score.as_array_mut();
        let mut accepted_mask = accepted_mask.as_array_mut();
        let mut accepted_quantity = accepted_quantity.as_array_mut();
        let elastic_need = market_elastic_need.as_array();

        for need_good in 0..goods {
            if deal_type == CONSUMPTION_DEAL {
                if stock[[agent_id, need_good]] >= (elastic_need[need_good] * needs_level[agent_id]) {
                    continue;
                }
            } else if !(recent_sales[[agent_id, need_good]]
                > (recent_production[[agent_id, need_good]] - elastic_need[need_good])
                && stock[[agent_id, need_good]]
                    < (stock_limit[[agent_id, need_good]] - elastic_need[need_good]))
            {
                continue;
            }

            let mut forbidden_gifts: HashSet<i32> = HashSet::new();
            let mut attempts = 0usize;
            while attempts < max_attempts {
                let max_need = if deal_type == CONSUMPTION_DEAL {
                    let remaining_need = need[[agent_id, need_good]] as f64;
                    if remaining_need < 1.0 {
                        break;
                    }
                    remaining_need
                } else {
                    if stock[[agent_id, need_good]] >= stock_limit[[agent_id, need_good]] {
                        break;
                    }
                    ((stock_limit[[agent_id, need_good]] - stock[[agent_id, need_good]]) as f64).max(0.0)
                };

                let mut candidate_offer_goods: Vec<i32> = Vec::new();
                for offer_good in 0..goods {
                    if offer_good == need_good || forbidden_gifts.contains(&(offer_good as i32)) {
                        continue;
                    }
                    if stock[[agent_id, offer_good]]
                        <= ((elastic_need[offer_good] * needs_level[agent_id]) + 1.0)
                    {
                        continue;
                    }
                    candidate_offer_goods.push(offer_good as i32);
                }
                if candidate_offer_goods.is_empty() {
                    break;
                }

                let friend_ids_row = friend_id.row(agent_id);
                let mut reciprocal_slots = vec![-1_i32; acquaintances];
                let mut has_known_partner = false;
                for friend_slot in 0..acquaintances {
                    let fid = friend_ids_row[friend_slot];
                    if fid >= 0 {
                        has_known_partner = true;
                        reciprocal_slots[friend_slot] =
                            find_friend_slot_scan_internal(friend_id.view(), fid as usize, agent_id as i32);
                    }
                }
                if !has_known_partner {
                    break;
                }

                let plan = plan_exchange_stage_candidate(
                    agent_id,
                    need_good,
                    goods,
                    initial_transparency,
                    max_need,
                    min_trade_quantity,
                    trade_rounding_buffer,
                    elastic_need,
                    stock.view(),
                    role,
                    stock_limit,
                    purchase_price,
                    sales_price,
                    needs_level,
                    transparency.view(),
                    friend_ids_row,
                    ArrayView1::from(&reciprocal_slots[..]),
                    ArrayView1::from(&candidate_offer_goods[..]),
                );
                let Some(plan) = plan else {
                    break;
                };

                totals.proposed_count += 1;
                if plan.reason_code != PLAN_OK {
                    forbidden_gifts.insert(plan.candidate.offer_good);
                    attempts += 1;
                    continue;
                }

                let friend_idx = plan.candidate.friend_id as usize;
                let offer_good = plan.candidate.offer_good as usize;
                let friend_slot = plan.candidate.friend_slot as usize;
                let reciprocal_slot = if plan.candidate.reciprocal_slot >= 0 {
                    plan.candidate.reciprocal_slot as usize
                } else {
                    let ensured_slot = ensure_friend_link_array_only(
                        agent_id,
                        friend_idx,
                        acquaintances,
                        goods,
                        initial_transparency,
                        initial_transactions,
                        &mut friend_id,
                        &mut friend_activity,
                        &mut friend_purchased,
                        &mut friend_sold,
                        &mut transparency,
                    ) as usize;
                    changed_agents.insert(friend_idx);
                    ensured_slot
                };

                let max_exchange = plan.max_exchange;
                let remaining_need_after_trade = max_need - max_exchange;
                let my_gift_out = (max_exchange * plan.switch_average)
                    / plan.receiving_transparency.max(EPSILON as f64);
                let friend_need_out = max_exchange / plan.need_transparency.max(EPSILON as f64);
                let friend_gift_in = max_exchange * plan.switch_average;

                if deal_type == SURPLUS_DEAL {
                    stock[[agent_id, need_good]] += max_exchange as f32;
                    recent_inventory_inflow[[agent_id, need_good]] += max_exchange as f32;
                    totals.inventory_trade_volume += max_exchange;
                } else {
                    need[[agent_id, need_good]] -= max_exchange as f32;
                    if need[[agent_id, need_good]] < min_trade_quantity as f32 {
                        need[[agent_id, need_good]] = 0.0;
                    }
                }

                recent_sales[[agent_id, offer_good]] += friend_gift_in as f32;
                sold_this_period[[agent_id, offer_good]] += friend_gift_in as f32;
                recent_purchases[[agent_id, need_good]] += max_exchange as f32;
                purchased_this_period[[agent_id, need_good]] += max_exchange as f32;
                stock[[agent_id, offer_good]] -= my_gift_out as f32;
                if stock[[agent_id, offer_good]] < 0.0 {
                    stock[[agent_id, offer_good]] = 0.0;
                }

                stock[[friend_idx, need_good]] -= friend_need_out as f32;
                if stock[[friend_idx, need_good]] < 0.0 {
                    stock[[friend_idx, need_good]] = 0.0;
                }
                stock[[friend_idx, offer_good]] += friend_gift_in as f32;
                recent_inventory_inflow[[friend_idx, offer_good]] += friend_gift_in as f32;
                totals.inventory_trade_volume += friend_gift_in;

                add_assign_like_numpy_float32(
                    &mut market_periodic_tce_cost[need_good],
                    (friend_need_out - max_exchange).max(0.0),
                );
                add_assign_like_numpy_float32(
                    &mut market_periodic_tce_cost[offer_good],
                    (my_gift_out - friend_gift_in).max(0.0),
                );

                recent_sales[[friend_idx, need_good]] += friend_need_out as f32;
                sold_this_period[[friend_idx, need_good]] += friend_need_out as f32;
                recent_purchases[[friend_idx, offer_good]] += friend_gift_in as f32;
                purchased_this_period[[friend_idx, offer_good]] += friend_gift_in as f32;

                purchase_times[[agent_id, need_good]] += 1;
                sales_times[[agent_id, offer_good]] += 1;
                purchase_times[[friend_idx, offer_good]] += 1;
                sales_times[[friend_idx, need_good]] += 1;

                friend_activity[[agent_id, friend_slot]] += 1.0;
                friend_purchased[[agent_id, friend_slot, need_good]] += 1.0;
                friend_sold[[agent_id, friend_slot, offer_good]] += 1.0;
                friend_activity[[friend_idx, reciprocal_slot]] += 1.0;
                friend_purchased[[friend_idx, reciprocal_slot, offer_good]] += 1.0;
                friend_sold[[friend_idx, reciprocal_slot, need_good]] += 1.0;

                let my_need_purchase_price = purchase_price[[agent_id, need_good]] as f64;
                let my_offer_sales_price = sales_price[[agent_id, offer_good]] as f64;
                let friend_need_sales_price = sales_price[[friend_idx, need_good]] as f64;
                let friend_offer_purchase_price = purchase_price[[friend_idx, offer_good]] as f64;
                let exchange_value_to_me = my_need_purchase_price
                    / (((plan.switch_average / plan.receiving_transparency.max(EPSILON as f64))
                        * my_offer_sales_price)
                        .max(EPSILON as f64));
                let exchange_value_to_friend = (friend_offer_purchase_price * plan.switch_average)
                    / (friend_need_sales_price / plan.need_transparency.max(EPSILON as f64)).max(EPSILON as f64);
                let my_value_correction = exchange_value_to_me.max(0.0).sqrt();
                let friend_value_correction = exchange_value_to_friend.max(0.0).sqrt();
                if my_value_correction > 1.0 && friend_value_correction > 1.0 {
                    add_assign_like_numpy_float32(
                        &mut sum_period_purchase_value[[agent_id, need_good]],
                        my_need_purchase_price / my_value_correction,
                    );
                    add_assign_like_numpy_float32(
                        &mut sum_period_sales_value[[agent_id, offer_good]],
                        my_offer_sales_price * my_value_correction,
                    );
                    add_assign_like_numpy_float32(
                        &mut sum_period_purchase_value[[friend_idx, offer_good]],
                        friend_offer_purchase_price / friend_value_correction,
                    );
                    add_assign_like_numpy_float32(
                        &mut sum_period_sales_value[[friend_idx, need_good]],
                        friend_need_sales_price * friend_value_correction,
                    );
                }

                proposal_friend_slot[agent_id] = friend_slot as i32;
                proposal_target_agent[agent_id] = friend_idx as i32;
                proposal_need_good[agent_id] = need_good as i32;
                proposal_offer_good[agent_id] = offer_good as i32;
                proposal_quantity[agent_id] = max_exchange as f32;
                proposal_score[agent_id] = plan.candidate.score;
                accepted_mask[agent_id] = true;
                accepted_quantity[agent_id] = max_exchange as f32;

                totals.accepted_count += 1;
                totals.accepted_volume += max_exchange;

                let exhausted_gift = remaining_need_after_trade >= 1.0
                    && stock[[agent_id, offer_good]] < 1.0;
                if exhausted_gift {
                    forbidden_gifts.insert(plan.candidate.offer_good);
                }
                attempts += 1;
            }
        }
    }

    if !changed_agents.is_empty() {
        let mut changed_list: Vec<usize> = changed_agents.into_iter().collect();
        changed_list.sort_unstable();
        runner.call_method1("_sync_friend_slot_maps", (changed_list,))?;
    }

    Ok((
        totals.proposed_count,
        totals.accepted_count,
        totals.accepted_volume,
        totals.inventory_trade_volume,
    ))
}


#[pyfunction]
fn run_exact_cycle(py: Python<'_>, engine: PyObject) -> PyResult<()> {
    let engine = engine.bind(py);
    let legacy_cycle = PyModule::import_bound(py, "emergent_money.legacy_cycle")?;
    let runner_type = legacy_cycle.getattr("LegacyCycleRunner")?;
    let runner = runner_type.call1((engine.clone(),))?;

    let uses_hybrid_exchange: bool = runner.call_method0("_uses_experimental_hybrid_exchange")?.extract()?;
    if uses_hybrid_exchange {
        runner.call_method0("run")?;
        return Ok(());
    }

    runner.call_method0("_reset_cycle_state")?;
    let config = runner.getattr("config")?;
    let state = runner.getattr("state")?;
    let market = runner.getattr("market")?;
    let population: usize = config.getattr("population")?.extract()?;
    let goods: usize = config.getattr("goods")?.extract()?;
    let period_length: f32 = config.getattr("cycle_time_budget")?.extract()?;
    let history: i32 = config.getattr("history")?.extract()?;
    let basic_round_elastic: bool = config.getattr("basic_round_elastic")?.extract()?;
    let stock_limit_multiplier: f32 = config.getattr("stock_limit_multiplier")?.extract()?;
    let max_needs_increase: f32 = config.getattr("max_needs_increase")?.extract()?;
    let max_needs_reduction: f32 = config.getattr("max_needs_reduction")?.extract()?;
    let small_needs_increase: f32 = config.getattr("small_needs_increase")?.extract()?;
    let small_needs_reduction: f32 = config.getattr("small_needs_reduction")?.extract()?;
    let leisure_time: f32 = config.getattr("leisure_time")?.extract()?;
    let max_leisure_extra_multiplier: f32 = config.getattr("max_leisure_extra_multiplier")?.extract()?;

    let mut totals = NativeStageTotals::default();
    for agent_id in 0..population {
        run_agent_cycle_owned(
            py,
            &runner,
            &state,
            &market,
            agent_id,
            goods,
            period_length,
            history,
            basic_round_elastic,
            stock_limit_multiplier,
            max_needs_increase,
            max_needs_reduction,
            small_needs_increase,
            small_needs_reduction,
            leisure_time,
            max_leisure_extra_multiplier,
            &mut totals,
        )?;
    }

    add_engine_metric(engine, "_cycle_need_total", totals.cycle_need_total)?;
    add_engine_metric(engine, "_production_total", totals.production_total)?;
    add_engine_metric(engine, "_surplus_output_total", totals.surplus_output_total)?;
    add_engine_metric(engine, "_stock_consumption_total", totals.stock_consumption_total)?;
    add_engine_metric(engine, "_leisure_extra_need_total", totals.leisure_extra_need_total)?;
    runner.call_method0("_finalize_cycle_after_agent_loop")?;
    Ok(())
}

#[pymodule]
fn _legacy_native_search(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(find_best_exchange, module)?)?;
    module.add_function(wrap_pyfunction!(plan_best_exchange, module)?)?;
    module.add_function(wrap_pyfunction!(prepare_agent_for_consumption, module)?)?;
    module.add_function(wrap_pyfunction!(produce_need, module)?)?;
    module.add_function(wrap_pyfunction!(prepare_leisure_round, module)?)?;
    module.add_function(wrap_pyfunction!(run_exchange_stage, module)?)?;
    module.add_function(wrap_pyfunction!(surplus_production, module)?)?;
    module.add_function(wrap_pyfunction!(leisure_production, module)?)?;
    module.add_function(wrap_pyfunction!(end_agent_period, module)?)?;
    module.add_function(wrap_pyfunction!(run_exact_cycle, module)?)?;
    Ok(())
}
