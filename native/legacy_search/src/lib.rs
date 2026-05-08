use ndarray::{
    ArrayView1, ArrayView2, ArrayView3, ArrayViewMut1, ArrayViewMut2, ArrayViewMut3, Axis,
};
use numpy::{
    PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyReadwriteArray1, PyReadwriteArray2,
    PyReadwriteArray3,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashSet;
use std::time::Instant;

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
const PARALLEL_SEARCH_MIN_CELLS: usize = 8_192;
const PARALLEL_BASKET_MIN_CELLS: usize = 65_536;

fn mul_assign_like_numpy_float32(value: f64, factor: f64) -> f64 {
    ((value as f32) * (factor as f32)) as f64
}

fn add_assign_like_numpy_float32(cell: &mut f32, delta: f64) {
    *cell = ((*cell as f32) + (delta as f32)) as f32;
}

fn exchange_value_to_me_like_numpy_float32(
    my_need_purchase_price: f32,
    my_offer_sales_price: f32,
    switch_average: f64,
    receiving_transparency: f64,
) -> f64 {
    // Python evaluates the float/float ratio first, then NumPy casts the
    // product with the float32 price back to float32.
    let transparency = receiving_transparency.max(EPSILON as f64);
    let denominator =
        (((switch_average / transparency) as f32) * my_offer_sales_price).max(EPSILON);
    (my_need_purchase_price / denominator) as f64
}

fn exchange_value_to_friend_like_numpy_float32(
    friend_offer_purchase_price: f32,
    friend_need_sales_price: f32,
    switch_average: f64,
    need_transparency: f64,
) -> f64 {
    let numerator = friend_offer_purchase_price * (switch_average as f32);
    let denominator =
        (friend_need_sales_price / (need_transparency as f32).max(EPSILON)).max(EPSILON);
    (numerator / denominator) as f64
}

fn div_like_numpy_float32(numerator: f32, denominator: f64) -> f64 {
    (numerator / (denominator as f32)) as f64
}

fn mul_like_numpy_float32(value: f32, factor: f64) -> f64 {
    (value * (factor as f32)) as f64
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
    let friend_count = friend_ids.len();
    let offer_count = candidate_offer_goods.len();
    if friend_count.saturating_mul(offer_count) >= PARALLEL_SEARCH_MIN_CELLS {
        return search_best_exchange_internal_parallel(
            goods,
            need_good,
            initial_transparency,
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
            my_need_purchase_price,
            my_need_is_producer,
        );
    }
    if let (
        Some(elastic_need_slice),
        Some(candidate_offer_goods_slice),
        Some(friend_ids_slice),
        Some(reciprocal_slots_slice),
        Some(my_sales_price_slice),
        Some(my_purchase_price_slice),
        Some(my_role_slice),
        Some(my_transparency_slice),
        Some(stock_slice),
        Some(role_slice),
        Some(stock_limit_slice),
        Some(purchase_price_slice),
        Some(sales_price_slice),
        Some(needs_level_slice),
        Some(transparency_slice),
    ) = (
        elastic_need.as_slice_memory_order(),
        candidate_offer_goods.as_slice_memory_order(),
        friend_ids.as_slice_memory_order(),
        reciprocal_slots.as_slice_memory_order(),
        my_sales_price.as_slice_memory_order(),
        my_purchase_price.as_slice_memory_order(),
        my_role.as_slice_memory_order(),
        my_transparency.as_slice_memory_order(),
        stock.as_slice_memory_order(),
        role.as_slice_memory_order(),
        stock_limit.as_slice_memory_order(),
        purchase_price.as_slice_memory_order(),
        sales_price.as_slice_memory_order(),
        needs_level.as_slice_memory_order(),
        transparency.as_slice_memory_order(),
    ) {
        let (_population, transparency_acquaintances, transparency_goods) = transparency.dim();
        return search_best_exchange_internal_slice(
            goods,
            need_good,
            initial_transparency,
            elastic_need_slice,
            candidate_offer_goods_slice,
            friend_ids_slice,
            reciprocal_slots_slice,
            my_sales_price_slice,
            my_purchase_price_slice,
            my_role_slice,
            my_transparency_slice,
            my_transparency.ncols(),
            stock_slice,
            stock.ncols(),
            role_slice,
            role.ncols(),
            stock_limit_slice,
            stock_limit.ncols(),
            purchase_price_slice,
            purchase_price.ncols(),
            sales_price_slice,
            sales_price.ncols(),
            needs_level_slice,
            transparency_slice,
            transparency_acquaintances,
            transparency_goods,
            my_need_purchase_price,
            my_need_is_producer,
        );
    }

    let mut best_score = -1.0_f32;
    let mut best_candidate: Option<SearchCandidate> = None;

    for (friend_slot, friend_id_raw) in friend_ids.iter().enumerate() {
        let friend_id = *friend_id_raw;
        if friend_id < 0 {
            continue;
        }
        let fid = friend_id as usize;
        let friend_stock = stock.row(fid);
        let friend_role = role.row(fid);
        let friend_stock_limit = stock_limit.row(fid);
        let friend_purchase_price = purchase_price.row(fid);
        let friend_sales_price = sales_price.row(fid);
        let friend_needs_level = needs_level[fid];
        if friend_stock[need_good] <= (elastic_need[need_good] * friend_needs_level + 1.0) {
            continue;
        }

        let reciprocal_slot = reciprocal_slots[friend_slot];
        let need_transparency = my_transparency[[friend_slot, need_good]];
        let friend_need_role = friend_role[need_good] as f32;
        let friend_need_sales_price = friend_sales_price[need_good];

        for offer_good_raw in candidate_offer_goods.iter() {
            let offer_good_i32 = *offer_good_raw;
            if offer_good_i32 < 0 {
                continue;
            }
            let offer_good = offer_good_i32 as usize;
            if offer_good >= goods || offer_good == need_good {
                continue;
            }

            let gift_max_level = if friend_role[offer_good] == ROLE_RETAILER {
                friend_stock_limit[offer_good] - 1.0
            } else {
                elastic_need[offer_good] * friend_needs_level - 1.0
            };
            if friend_stock[offer_good] >= gift_max_level {
                continue;
            }

            let mut receiving_transparency = initial_transparency;
            if reciprocal_slot >= 0 {
                receiving_transparency = transparency[[fid, reciprocal_slot as usize, offer_good]];
            }

            let mut score = (friend_purchase_price[offer_good]
                / friend_need_sales_price.max(EPSILON))
                * need_transparency;
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

#[allow(clippy::too_many_arguments)]
fn search_best_exchange_internal_slice(
    goods: usize,
    need_good: usize,
    initial_transparency: f32,
    elastic_need: &[f32],
    candidate_offer_goods: &[i32],
    friend_ids: &[i32],
    reciprocal_slots: &[i32],
    my_sales_price: &[f32],
    my_purchase_price: &[f32],
    my_role: &[i32],
    my_transparency: &[f32],
    my_transparency_cols: usize,
    stock: &[f32],
    stock_cols: usize,
    role: &[i32],
    role_cols: usize,
    stock_limit: &[f32],
    stock_limit_cols: usize,
    purchase_price: &[f32],
    purchase_price_cols: usize,
    sales_price: &[f32],
    sales_price_cols: usize,
    needs_level: &[f32],
    transparency: &[f32],
    transparency_acquaintances: usize,
    transparency_goods: usize,
    my_need_purchase_price: f32,
    my_need_is_producer: bool,
) -> Option<SearchCandidate> {
    if need_good >= goods
        || need_good >= elastic_need.len()
        || need_good >= my_purchase_price.len()
        || need_good >= my_role.len()
    {
        return None;
    }

    let mut best_score = -1.0_f32;
    let mut best_candidate: Option<SearchCandidate> = None;

    for (friend_slot, friend_id) in friend_ids.iter().copied().enumerate() {
        if friend_id < 0 {
            continue;
        }
        let fid = friend_id as usize;
        if fid >= needs_level.len() {
            continue;
        }
        let friend_stock_offset = fid * stock_cols;
        let friend_role_offset = fid * role_cols;
        let friend_stock_limit_offset = fid * stock_limit_cols;
        let friend_purchase_price_offset = fid * purchase_price_cols;
        let friend_sales_price_offset = fid * sales_price_cols;
        let friend_needs_level = needs_level[fid];
        if stock[friend_stock_offset + need_good]
            <= (elastic_need[need_good] * friend_needs_level + 1.0)
        {
            continue;
        }

        let reciprocal_slot = reciprocal_slots[friend_slot];
        let need_transparency = my_transparency[(friend_slot * my_transparency_cols) + need_good];
        let friend_need_role = role[friend_role_offset + need_good] as f32;
        let friend_need_sales_price = sales_price[friend_sales_price_offset + need_good];

        for offer_good_i32 in candidate_offer_goods.iter().copied() {
            if offer_good_i32 < 0 {
                continue;
            }
            let offer_good = offer_good_i32 as usize;
            if offer_good >= goods || offer_good == need_good {
                continue;
            }

            let gift_max_level = if role[friend_role_offset + offer_good] == ROLE_RETAILER {
                stock_limit[friend_stock_limit_offset + offer_good] - 1.0
            } else {
                elastic_need[offer_good] * friend_needs_level - 1.0
            };
            if stock[friend_stock_offset + offer_good] >= gift_max_level {
                continue;
            }

            let receiving_transparency = if reciprocal_slot >= 0 {
                let transparency_index = ((fid * transparency_acquaintances
                    + reciprocal_slot as usize)
                    * transparency_goods)
                    + offer_good;
                transparency[transparency_index]
            } else {
                initial_transparency
            };

            let mut score = (purchase_price[friend_purchase_price_offset + offer_good]
                / friend_need_sales_price.max(EPSILON))
                * need_transparency;
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

#[allow(clippy::too_many_arguments)]
fn search_best_exchange_internal_parallel(
    goods: usize,
    need_good: usize,
    initial_transparency: f32,
    elastic_need: ArrayView1<'_, f32>,
    candidate_offer_goods: ArrayView1<'_, i32>,
    friend_ids: ArrayView1<'_, i32>,
    reciprocal_slots: ArrayView1<'_, i32>,
    my_sales_price: ArrayView1<'_, f32>,
    _my_purchase_price: ArrayView1<'_, f32>,
    my_role: ArrayView1<'_, i32>,
    my_transparency: ArrayView2<'_, f32>,
    stock: ArrayView2<'_, f32>,
    role: ArrayView2<'_, i32>,
    stock_limit: ArrayView2<'_, f32>,
    purchase_price: ArrayView2<'_, f32>,
    sales_price: ArrayView2<'_, f32>,
    needs_level: ArrayView1<'_, f32>,
    transparency: ArrayView3<'_, f32>,
    my_need_purchase_price: f32,
    my_need_is_producer: bool,
) -> Option<SearchCandidate> {
    let offer_count = candidate_offer_goods.len();
    if offer_count == 0 {
        return None;
    }

    (0..friend_ids.len())
        .into_par_iter()
        .filter_map(|friend_slot| {
            let friend_id = friend_ids[friend_slot];
            if friend_id < 0 {
                return None;
            }
            let fid = friend_id as usize;
            let friend_needs_level = needs_level[fid];
            if stock[[fid, need_good]] <= (elastic_need[need_good] * friend_needs_level + 1.0) {
                return None;
            }

            let mut best_for_friend: Option<(usize, SearchCandidate)> = None;
            for offer_index in 0..offer_count {
                let offer_good_i32 = candidate_offer_goods[offer_index];
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

                let reciprocal_slot = reciprocal_slots[friend_slot];
                let receiving_transparency = if reciprocal_slot >= 0 {
                    transparency[[fid, reciprocal_slot as usize, offer_good]]
                } else {
                    initial_transparency
                };
                let need_transparency = my_transparency[[friend_slot, need_good]];
                let friend_need_role = role[[fid, need_good]] as f32;
                let friend_need_sales_price = sales_price[[fid, need_good]];

                let mut score = (purchase_price[[fid, offer_good]]
                    / friend_need_sales_price.max(EPSILON))
                    * need_transparency;
                score -= my_sales_price[offer_good]
                    / (my_need_purchase_price * receiving_transparency).max(EPSILON);
                score *= my_role[offer_good] as f32;
                score *= friend_need_role;
                if my_need_is_producer {
                    score /= 2.0;
                }
                if score <= 0.0 {
                    continue;
                }

                let linear_index = (friend_slot * offer_count) + offer_index;
                let candidate = SearchCandidate {
                    score,
                    friend_slot: friend_slot as i32,
                    friend_id,
                    offer_good: offer_good_i32,
                    reciprocal_slot,
                };
                match best_for_friend {
                    Some((best_index, best_candidate))
                        if best_candidate.score > score
                            || (best_candidate.score == score && best_index < linear_index) =>
                    {
                        best_for_friend = Some((best_index, best_candidate));
                    }
                    _ => best_for_friend = Some((linear_index, candidate)),
                }
            }

            best_for_friend
        })
        .reduce_with(|left, right| {
            if right.1.score > left.1.score || (right.1.score == left.1.score && right.0 < left.0) {
                right
            } else {
                left
            }
        })
        .map(|(_, candidate)| candidate)
}

#[allow(clippy::too_many_arguments)]
fn search_best_exchange_all_offer_goods_internal(
    goods: usize,
    need_good: usize,
    initial_transparency: f32,
    forbidden_offer_by_need: &[bool],
    available_offer_goods: &[usize],
    friend_ids: ArrayView1<'_, i32>,
    reciprocal_slots: ArrayView1<'_, i32>,
    my_stock: ArrayView1<'_, f32>,
    my_sales_price: ArrayView1<'_, f32>,
    my_purchase_price: ArrayView1<'_, f32>,
    my_role: ArrayView1<'_, i32>,
    my_transparency: ArrayView2<'_, f32>,
    my_needs_level: f32,
    elastic_need: ArrayView1<'_, f32>,
    stock: ArrayView2<'_, f32>,
    role: ArrayView2<'_, i32>,
    stock_limit: ArrayView2<'_, f32>,
    purchase_price: ArrayView2<'_, f32>,
    sales_price: ArrayView2<'_, f32>,
    needs_level: ArrayView1<'_, f32>,
    transparency: ArrayView3<'_, f32>,
    allow_parallel: bool,
) -> Option<SearchCandidate> {
    if need_good >= goods || forbidden_offer_by_need.len() < goods.saturating_mul(goods) {
        return None;
    }

    let my_need_purchase_price = my_purchase_price[need_good];
    let my_need_is_producer = my_role[need_good] == ROLE_PRODUCER;
    let cell_count = friend_ids.len().saturating_mul(available_offer_goods.len());
    if allow_parallel && cell_count >= PARALLEL_SEARCH_MIN_CELLS {
        return search_best_exchange_all_offer_goods_internal_parallel(
            goods,
            need_good,
            initial_transparency,
            forbidden_offer_by_need,
            available_offer_goods,
            friend_ids,
            reciprocal_slots,
            my_stock,
            my_sales_price,
            my_role,
            my_transparency,
            my_needs_level,
            elastic_need,
            stock,
            role,
            stock_limit,
            purchase_price,
            sales_price,
            needs_level,
            transparency,
            my_need_purchase_price,
            my_need_is_producer,
        );
    }

    let mut best: Option<(usize, SearchCandidate)> = None;
    for friend_slot in 0..friend_ids.len() {
        let friend_id = friend_ids[friend_slot];
        if friend_id < 0 {
            continue;
        }
        let fid = friend_id as usize;
        let friend_needs_level = needs_level[fid];
        if stock[[fid, need_good]] <= (elastic_need[need_good] * friend_needs_level + 1.0) {
            continue;
        }

        for &offer_good in available_offer_goods {
            if offer_good >= goods {
                continue;
            }
            if offer_good == need_good || forbidden_offer_by_need[(need_good * goods) + offer_good]
            {
                continue;
            }
            if my_stock[offer_good] <= ((elastic_need[offer_good] * my_needs_level) + 1.0) {
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

            let reciprocal_slot = reciprocal_slots[friend_slot];
            let receiving_transparency = if reciprocal_slot >= 0 {
                transparency[[fid, reciprocal_slot as usize, offer_good]]
            } else {
                initial_transparency
            };
            let need_transparency = my_transparency[[friend_slot, need_good]];
            let friend_need_role = role[[fid, need_good]] as f32;
            let friend_need_sales_price = sales_price[[fid, need_good]];

            let mut score = (purchase_price[[fid, offer_good]]
                / friend_need_sales_price.max(EPSILON))
                * need_transparency;
            score -= my_sales_price[offer_good]
                / (my_need_purchase_price * receiving_transparency).max(EPSILON);
            score *= my_role[offer_good] as f32;
            score *= friend_need_role;
            if my_need_is_producer {
                score /= 2.0;
            }
            if score <= 0.0 {
                continue;
            }

            let linear_index = (friend_slot * goods) + offer_good;
            let candidate = SearchCandidate {
                score,
                friend_slot: friend_slot as i32,
                friend_id,
                offer_good: offer_good as i32,
                reciprocal_slot,
            };
            match best {
                Some((best_index, best_candidate))
                    if best_candidate.score > score
                        || (best_candidate.score == score && best_index < linear_index) =>
                {
                    best = Some((best_index, best_candidate));
                }
                _ => best = Some((linear_index, candidate)),
            }
        }
    }

    best.map(|(_, candidate)| candidate)
}

#[allow(clippy::too_many_arguments)]
fn search_best_exchange_all_offer_goods_internal_parallel(
    goods: usize,
    need_good: usize,
    initial_transparency: f32,
    forbidden_offer_by_need: &[bool],
    available_offer_goods: &[usize],
    friend_ids: ArrayView1<'_, i32>,
    reciprocal_slots: ArrayView1<'_, i32>,
    my_stock: ArrayView1<'_, f32>,
    my_sales_price: ArrayView1<'_, f32>,
    my_role: ArrayView1<'_, i32>,
    my_transparency: ArrayView2<'_, f32>,
    my_needs_level: f32,
    elastic_need: ArrayView1<'_, f32>,
    stock: ArrayView2<'_, f32>,
    role: ArrayView2<'_, i32>,
    stock_limit: ArrayView2<'_, f32>,
    purchase_price: ArrayView2<'_, f32>,
    sales_price: ArrayView2<'_, f32>,
    needs_level: ArrayView1<'_, f32>,
    transparency: ArrayView3<'_, f32>,
    my_need_purchase_price: f32,
    my_need_is_producer: bool,
) -> Option<SearchCandidate> {
    (0..friend_ids.len())
        .into_par_iter()
        .filter_map(|friend_slot| {
            let friend_id = friend_ids[friend_slot];
            if friend_id < 0 {
                return None;
            }
            let fid = friend_id as usize;
            let friend_needs_level = needs_level[fid];
            if stock[[fid, need_good]] <= (elastic_need[need_good] * friend_needs_level + 1.0) {
                return None;
            }

            let mut best_for_friend: Option<(usize, SearchCandidate)> = None;
            for &offer_good in available_offer_goods {
                if offer_good >= goods {
                    continue;
                }
                if offer_good == need_good
                    || forbidden_offer_by_need[(need_good * goods) + offer_good]
                {
                    continue;
                }
                if my_stock[offer_good] <= ((elastic_need[offer_good] * my_needs_level) + 1.0) {
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

                let reciprocal_slot = reciprocal_slots[friend_slot];
                let receiving_transparency = if reciprocal_slot >= 0 {
                    transparency[[fid, reciprocal_slot as usize, offer_good]]
                } else {
                    initial_transparency
                };
                let need_transparency = my_transparency[[friend_slot, need_good]];
                let friend_need_role = role[[fid, need_good]] as f32;
                let friend_need_sales_price = sales_price[[fid, need_good]];

                let mut score = (purchase_price[[fid, offer_good]]
                    / friend_need_sales_price.max(EPSILON))
                    * need_transparency;
                score -= my_sales_price[offer_good]
                    / (my_need_purchase_price * receiving_transparency).max(EPSILON);
                score *= my_role[offer_good] as f32;
                score *= friend_need_role;
                if my_need_is_producer {
                    score /= 2.0;
                }
                if score <= 0.0 {
                    continue;
                }

                let linear_index = (friend_slot * goods) + offer_good;
                let candidate = SearchCandidate {
                    score,
                    friend_slot: friend_slot as i32,
                    friend_id,
                    offer_good: offer_good as i32,
                    reciprocal_slot,
                };
                match best_for_friend {
                    Some((best_index, best_candidate))
                        if best_candidate.score > score
                            || (best_candidate.score == score && best_index < linear_index) =>
                    {
                        best_for_friend = Some((best_index, best_candidate));
                    }
                    _ => best_for_friend = Some((linear_index, candidate)),
                }
            }

            best_for_friend
        })
        .reduce_with(|left, right| {
            if right.1.score > left.1.score || (right.1.score == left.1.score && right.0 < left.0) {
                right
            } else {
                left
            }
        })
        .map(|(_, candidate)| candidate)
}

#[derive(Clone, Copy)]
struct BasketCandidate {
    score: f32,
    need_good: i32,
    friend_slot: i32,
    friend_id: i32,
    offer_good: i32,
    order: u32,
}

#[derive(Clone, Copy)]
struct StaticBasketCandidate {
    score: f32,
    friend_slot: i32,
    offer_good: i32,
    order: u32,
}

impl StaticBasketCandidate {
    fn with_need_good(self, need_good: usize, friend_id: i32) -> BasketCandidate {
        BasketCandidate {
            score: self.score,
            need_good: need_good as i32,
            friend_slot: self.friend_slot,
            friend_id,
            offer_good: self.offer_good,
            order: self.order,
        }
    }
}

fn basket_candidate_order(
    need_good: usize,
    friend_slot: usize,
    offer_good: usize,
    goods: usize,
    acquaintances: usize,
) -> u32 {
    ((need_good * acquaintances) + friend_slot)
        .saturating_mul(goods)
        .saturating_add(offer_good)
        .min(u32::MAX as usize) as u32
}

#[derive(Clone, Copy)]
struct ParallelPhenomenonCandidate {
    score: f32,
    agent_id: i32,
    need_good: i32,
    max_need: f64,
    friend_slot: i32,
    friend_id: i32,
    offer_good: i32,
    reciprocal_slot: i32,
    max_exchange: f64,
    switch_average: f64,
    need_transparency: f64,
    receiving_transparency: f64,
    order: usize,
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn plan_exchange_basket(
    agent_id: usize,
    deal_type: i32,
    goods: usize,
    acquaintances: usize,
    initial_transparency: f32,
    history: i32,
    local_liquidity_stock_bias: f64,
    local_liquidity_min_sales: f64,
    aspirational_stock_target: f64,
    disable_offer_prefilter: bool,
    market_elastic_need: PyReadonlyArray1<'_, f32>,
    forbidden_offer_by_need: PyReadonlyArray2<'_, bool>,
    stock: PyReadonlyArray2<'_, f32>,
    role: PyReadonlyArray2<'_, i32>,
    stock_limit: PyReadonlyArray2<'_, f32>,
    purchase_price: PyReadonlyArray2<'_, f32>,
    sales_price: PyReadonlyArray2<'_, f32>,
    needs_level: PyReadonlyArray1<'_, f32>,
    transparency: PyReadonlyArray3<'_, f32>,
    friend_id: PyReadonlyArray2<'_, i32>,
    need: PyReadonlyArray2<'_, f32>,
    recent_sales: PyReadonlyArray2<'_, f32>,
    recent_purchases: PyReadonlyArray2<'_, f32>,
    recent_inventory_inflow: PyReadonlyArray2<'_, f32>,
    recent_production: PyReadonlyArray2<'_, f32>,
    friend_sold: PyReadonlyArray3<'_, f32>,
) -> PyResult<Vec<(f32, i32, i32, i32, i32)>> {
    if deal_type != CONSUMPTION_DEAL && deal_type != SURPLUS_DEAL {
        return Err(PyValueError::new_err(
            "deal_type must be 1 (surplus) or 2 (consumption)",
        ));
    }

    let elastic_need = market_elastic_need.as_array();
    let forbidden_offer_by_need = forbidden_offer_by_need.as_array();
    let stock = stock.as_array();
    let role = role.as_array();
    let stock_limit = stock_limit.as_array();
    let purchase_price = purchase_price.as_array();
    let sales_price = sales_price.as_array();
    let needs_level = needs_level.as_array();
    let transparency = transparency.as_array();
    let friend_id = friend_id.as_array();
    let need = need.as_array();
    let recent_sales = recent_sales.as_array();
    let recent_purchases = recent_purchases.as_array();
    let recent_inventory_inflow = recent_inventory_inflow.as_array();
    let recent_production = recent_production.as_array();
    let friend_sold = friend_sold.as_array();

    if agent_id >= stock.nrows()
        || goods > stock.ncols()
        || goods > forbidden_offer_by_need.nrows()
        || goods > forbidden_offer_by_need.ncols()
    {
        return Ok(Vec::new());
    }

    let friend_ids_row = friend_id.row(agent_id);
    if friend_ids_row.len() != acquaintances {
        return Err(PyValueError::new_err(
            "acquaintances must match friend_id row length",
        ));
    }

    let mut reciprocal_slots = vec![-1_i32; acquaintances];
    let mut has_known_partner = false;
    for friend_slot in 0..acquaintances {
        let fid = friend_ids_row[friend_slot];
        if fid >= 0 {
            has_known_partner = true;
            reciprocal_slots[friend_slot] =
                find_friend_slot_scan_internal(friend_id, fid as usize, agent_id as i32);
        }
    }
    if !has_known_partner {
        return Ok(Vec::new());
    }

    let my_stock = stock.row(agent_id);
    let my_sales_price = sales_price.row(agent_id);
    let my_purchase_price = purchase_price.row(agent_id);
    let my_role = role.row(agent_id);
    let my_transparency = transparency.index_axis(Axis(0), agent_id);
    let my_needs_level = needs_level[agent_id];
    let available_offer_goods = collect_available_offer_goods(
        agent_id,
        goods,
        elastic_need,
        stock,
        needs_level,
        disable_offer_prefilter,
    );
    if available_offer_goods.is_empty() {
        return Ok(Vec::new());
    }
    let reciprocal_slots_view = ArrayView1::from(&reciprocal_slots[..]);

    let forbidden_offer_slice = forbidden_offer_by_need.as_slice_memory_order();
    let basket_cell_count = goods.saturating_mul(goods).saturating_mul(acquaintances);
    let mut candidates: Vec<BasketCandidate> = if let Some(forbidden_offer_slice) =
        forbidden_offer_slice
    {
        if basket_cell_count >= PARALLEL_BASKET_MIN_CELLS {
            (0..goods)
                .into_par_iter()
                .filter_map(|need_good| {
                    plan_one_basket_candidate_from_cached_links(
                        agent_id,
                        need_good,
                        deal_type,
                        goods,
                        acquaintances,
                        initial_transparency,
                        history,
                        local_liquidity_stock_bias,
                        local_liquidity_min_sales,
                        aspirational_stock_target,
                        forbidden_offer_slice,
                        &available_offer_goods,
                        elastic_need,
                        stock,
                        role,
                        stock_limit,
                        purchase_price,
                        sales_price,
                        needs_level,
                        transparency,
                        friend_ids_row,
                        reciprocal_slots_view,
                        need,
                        recent_sales,
                        recent_purchases,
                        recent_inventory_inflow,
                        recent_production,
                        friend_id,
                        friend_sold,
                        my_stock,
                        my_sales_price,
                        my_purchase_price,
                        my_role,
                        my_transparency,
                        my_needs_level,
                        false,
                    )
                })
                .collect()
        } else {
            (0..goods)
                .filter_map(|need_good| {
                    plan_one_basket_candidate_from_cached_links(
                        agent_id,
                        need_good,
                        deal_type,
                        goods,
                        acquaintances,
                        initial_transparency,
                        history,
                        local_liquidity_stock_bias,
                        local_liquidity_min_sales,
                        aspirational_stock_target,
                        forbidden_offer_slice,
                        &available_offer_goods,
                        elastic_need,
                        stock,
                        role,
                        stock_limit,
                        purchase_price,
                        sales_price,
                        needs_level,
                        transparency,
                        friend_ids_row,
                        reciprocal_slots_view,
                        need,
                        recent_sales,
                        recent_purchases,
                        recent_inventory_inflow,
                        recent_production,
                        friend_id,
                        friend_sold,
                        my_stock,
                        my_sales_price,
                        my_purchase_price,
                        my_role,
                        my_transparency,
                        my_needs_level,
                        true,
                    )
                })
                .collect()
        }
    } else {
        (0..goods)
            .filter_map(|need_good| {
                if basket_stage_max_need(
                    agent_id,
                    need_good,
                    deal_type,
                    history,
                    local_liquidity_stock_bias,
                    local_liquidity_min_sales,
                    aspirational_stock_target,
                    elastic_need,
                    stock,
                    stock_limit,
                    needs_level,
                    need,
                    recent_sales,
                    recent_purchases,
                    recent_inventory_inflow,
                    recent_production,
                    friend_id,
                    friend_sold,
                    transparency,
                ) <= 0.0
                {
                    return None;
                }

                let mut candidate_offer_goods: Vec<i32> =
                    Vec::with_capacity(goods.saturating_sub(1));
                for offer_good in 0..goods {
                    if offer_good == need_good || forbidden_offer_by_need[[need_good, offer_good]] {
                        continue;
                    }
                    if my_stock[offer_good] <= ((elastic_need[offer_good] * my_needs_level) + 1.0) {
                        continue;
                    }
                    candidate_offer_goods.push(offer_good as i32);
                }
                if candidate_offer_goods.is_empty() {
                    return None;
                }

                let search_candidate = search_best_exchange_internal(
                    goods,
                    need_good,
                    initial_transparency,
                    elastic_need,
                    ArrayView1::from(&candidate_offer_goods[..]),
                    friend_ids_row,
                    reciprocal_slots_view,
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
                )?;

                let order = basket_candidate_order(
                    need_good,
                    search_candidate.friend_slot as usize,
                    search_candidate.offer_good as usize,
                    goods,
                    acquaintances,
                );
                Some(BasketCandidate {
                    score: search_candidate.score,
                    need_good: need_good as i32,
                    friend_slot: search_candidate.friend_slot,
                    friend_id: search_candidate.friend_id,
                    offer_good: search_candidate.offer_good,
                    order,
                })
            })
            .collect()
    };

    candidates.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.order.cmp(&right.order))
    });

    Ok(candidates
        .into_iter()
        .map(|candidate| {
            (
                candidate.score,
                candidate.need_good,
                candidate.friend_slot,
                candidate.friend_id,
                candidate.offer_good,
            )
        })
        .collect())
}

#[allow(clippy::too_many_arguments)]
fn plan_agent_parallel_phenomenon_candidates(
    proposer_order: usize,
    agent_id: usize,
    deal_type: i32,
    goods: usize,
    acquaintances: usize,
    initial_transparency: f32,
    initial_transparency_for_execution: f64,
    history: i32,
    local_liquidity_stock_bias: f64,
    local_liquidity_min_sales: f64,
    aspirational_stock_target: f64,
    min_trade_quantity: f64,
    trade_rounding_buffer: f64,
    blocked_partner_mask: &[bool],
    elastic_need: ArrayView1<'_, f32>,
    stock: ArrayView2<'_, f32>,
    role: ArrayView2<'_, i32>,
    stock_limit: ArrayView2<'_, f32>,
    purchase_price: ArrayView2<'_, f32>,
    sales_price: ArrayView2<'_, f32>,
    needs_level: ArrayView1<'_, f32>,
    transparency: ArrayView3<'_, f32>,
    friend_id: ArrayView2<'_, i32>,
    need: ArrayView2<'_, f32>,
    recent_sales: ArrayView2<'_, f32>,
    recent_purchases: ArrayView2<'_, f32>,
    recent_inventory_inflow: ArrayView2<'_, f32>,
    recent_production: ArrayView2<'_, f32>,
    friend_sold: ArrayView3<'_, f32>,
    one_candidate_per_agent: bool,
) -> Vec<ParallelPhenomenonCandidate> {
    if agent_id >= stock.nrows() || goods > stock.ncols() || acquaintances > friend_id.ncols() {
        return Vec::new();
    }

    let mut agent_friend_ids = vec![-1_i32; acquaintances];
    let mut reciprocal_slots = vec![-1_i32; acquaintances];
    let mut has_known_partner = false;
    for friend_slot in 0..acquaintances {
        let fid = friend_id[[agent_id, friend_slot]];
        if fid < 0 {
            continue;
        }
        let friend_idx = fid as usize;
        if blocked_partner_mask
            .get(friend_idx)
            .copied()
            .unwrap_or(false)
        {
            continue;
        }
        has_known_partner = true;
        agent_friend_ids[friend_slot] = fid;
        reciprocal_slots[friend_slot] =
            find_friend_slot_scan_internal(friend_id, friend_idx, agent_id as i32);
    }
    if !has_known_partner {
        return Vec::new();
    }

    let friend_ids_view = ArrayView1::from(&agent_friend_ids[..]);
    let reciprocal_slots_view = ArrayView1::from(&reciprocal_slots[..]);
    let my_stock = stock.row(agent_id);
    let my_sales_price = sales_price.row(agent_id);
    let my_purchase_price = purchase_price.row(agent_id);
    let my_role = role.row(agent_id);
    let my_transparency = transparency.index_axis(Axis(0), agent_id);
    let my_needs_level = needs_level[agent_id];
    let available_offer_goods =
        collect_available_offer_goods(agent_id, goods, elastic_need, stock, needs_level, false);
    if available_offer_goods.is_empty() {
        return Vec::new();
    }
    let mut forbidden_offer_by_need = vec![false; goods.saturating_mul(goods)];
    let mut candidates: Vec<ParallelPhenomenonCandidate> = Vec::new();

    for need_good in 0..goods {
        let max_need = basket_stage_max_need(
            agent_id,
            need_good,
            deal_type,
            history,
            local_liquidity_stock_bias,
            local_liquidity_min_sales,
            aspirational_stock_target,
            elastic_need,
            stock,
            stock_limit,
            needs_level,
            need,
            recent_sales,
            recent_purchases,
            recent_inventory_inflow,
            recent_production,
            friend_id,
            friend_sold,
            transparency,
        );
        if max_need <= 0.0 {
            continue;
        }

        let mut attempts = 0usize;
        while attempts < goods {
            let Some(candidate) = plan_one_basket_candidate_from_cached_links(
                agent_id,
                need_good,
                deal_type,
                goods,
                acquaintances,
                initial_transparency,
                history,
                local_liquidity_stock_bias,
                local_liquidity_min_sales,
                aspirational_stock_target,
                &forbidden_offer_by_need,
                &available_offer_goods,
                elastic_need,
                stock,
                role,
                stock_limit,
                purchase_price,
                sales_price,
                needs_level,
                transparency,
                friend_ids_view,
                reciprocal_slots_view,
                need,
                recent_sales,
                recent_purchases,
                recent_inventory_inflow,
                recent_production,
                friend_id,
                friend_sold,
                my_stock,
                my_sales_price,
                my_purchase_price,
                my_role,
                my_transparency,
                my_needs_level,
                false,
            ) else {
                break;
            };

            if candidate.offer_good < 0 {
                break;
            }
            let offer_good = candidate.offer_good as usize;
            if offer_good >= goods {
                break;
            }
            let forbidden_index = (need_good * goods) + offer_good;
            let plan = plan_specific_exchange_candidate(
                agent_id,
                need_good,
                initial_transparency_for_execution,
                max_need,
                min_trade_quantity,
                trade_rounding_buffer,
                elastic_need,
                stock,
                role,
                stock_limit,
                purchase_price,
                sales_price,
                needs_level,
                transparency,
                friend_id,
                candidate,
            );

            let Some(plan) = plan else {
                forbidden_offer_by_need[forbidden_index] = true;
                attempts += 1;
                continue;
            };
            if plan.reason_code != PLAN_OK {
                forbidden_offer_by_need[forbidden_index] = true;
                attempts += 1;
                continue;
            }

            let order = proposer_order
                .saturating_mul(goods)
                .saturating_mul(acquaintances)
                .saturating_mul(goods)
                .saturating_add(candidate.order as usize);
            candidates.push(ParallelPhenomenonCandidate {
                score: plan.candidate.score,
                agent_id: agent_id as i32,
                need_good: need_good as i32,
                max_need,
                friend_slot: plan.candidate.friend_slot,
                friend_id: plan.candidate.friend_id,
                offer_good: plan.candidate.offer_good,
                reciprocal_slot: plan.candidate.reciprocal_slot,
                max_exchange: plan.max_exchange,
                switch_average: plan.switch_average,
                need_transparency: plan.need_transparency,
                receiving_transparency: plan.receiving_transparency,
                order,
            });
            break;
        }
    }

    candidates.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.order.cmp(&right.order))
    });
    if one_candidate_per_agent && candidates.len() > 1 {
        candidates.truncate(1);
    }
    candidates
}

#[allow(clippy::too_many_arguments)]
fn plan_agent_parallel_phenomenon_candidates_cached_links(
    proposer_order: usize,
    agent_id: usize,
    deal_type: i32,
    goods: usize,
    acquaintances: usize,
    initial_transparency: f32,
    initial_transparency_for_execution: f64,
    history: i32,
    local_liquidity_stock_bias: f64,
    local_liquidity_min_sales: f64,
    aspirational_stock_target: f64,
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
    friend_id: ArrayView2<'_, i32>,
    need: ArrayView2<'_, f32>,
    recent_sales: ArrayView2<'_, f32>,
    recent_purchases: ArrayView2<'_, f32>,
    recent_inventory_inflow: ArrayView2<'_, f32>,
    recent_production: ArrayView2<'_, f32>,
    friend_sold: ArrayView3<'_, f32>,
    agent_friend_ids: ArrayView1<'_, i32>,
    reciprocal_slots_view: ArrayView1<'_, i32>,
    one_candidate_per_agent: bool,
) -> Vec<ParallelPhenomenonCandidate> {
    if agent_id >= stock.nrows() || goods > stock.ncols() || acquaintances > agent_friend_ids.len()
    {
        return Vec::new();
    }
    if !agent_friend_ids.iter().any(|friend_id| *friend_id >= 0) {
        return Vec::new();
    }

    let my_stock = stock.row(agent_id);
    let my_sales_price = sales_price.row(agent_id);
    let my_purchase_price = purchase_price.row(agent_id);
    let my_role = role.row(agent_id);
    let my_transparency = transparency.index_axis(Axis(0), agent_id);
    let my_needs_level = needs_level[agent_id];
    let available_offer_goods =
        collect_available_offer_goods(agent_id, goods, elastic_need, stock, needs_level, false);
    if available_offer_goods.is_empty() {
        return Vec::new();
    }
    let mut forbidden_offer_by_need = vec![false; goods.saturating_mul(goods)];
    let mut candidates: Vec<ParallelPhenomenonCandidate> = Vec::new();

    for need_good in 0..goods {
        let max_need = basket_stage_max_need(
            agent_id,
            need_good,
            deal_type,
            history,
            local_liquidity_stock_bias,
            local_liquidity_min_sales,
            aspirational_stock_target,
            elastic_need,
            stock,
            stock_limit,
            needs_level,
            need,
            recent_sales,
            recent_purchases,
            recent_inventory_inflow,
            recent_production,
            friend_id,
            friend_sold,
            transparency,
        );
        if max_need <= 0.0 {
            continue;
        }

        let mut attempts = 0usize;
        while attempts < goods {
            let Some(candidate) = plan_one_basket_candidate_from_cached_links(
                agent_id,
                need_good,
                deal_type,
                goods,
                acquaintances,
                initial_transparency,
                history,
                local_liquidity_stock_bias,
                local_liquidity_min_sales,
                aspirational_stock_target,
                &forbidden_offer_by_need,
                &available_offer_goods,
                elastic_need,
                stock,
                role,
                stock_limit,
                purchase_price,
                sales_price,
                needs_level,
                transparency,
                agent_friend_ids,
                reciprocal_slots_view,
                need,
                recent_sales,
                recent_purchases,
                recent_inventory_inflow,
                recent_production,
                friend_id,
                friend_sold,
                my_stock,
                my_sales_price,
                my_purchase_price,
                my_role,
                my_transparency,
                my_needs_level,
                false,
            ) else {
                break;
            };

            if candidate.offer_good < 0 {
                break;
            }
            let offer_good = candidate.offer_good as usize;
            if offer_good >= goods {
                break;
            }
            let forbidden_index = (need_good * goods) + offer_good;
            let plan = plan_specific_exchange_candidate(
                agent_id,
                need_good,
                initial_transparency_for_execution,
                max_need,
                min_trade_quantity,
                trade_rounding_buffer,
                elastic_need,
                stock,
                role,
                stock_limit,
                purchase_price,
                sales_price,
                needs_level,
                transparency,
                friend_id,
                candidate,
            );

            let Some(plan) = plan else {
                forbidden_offer_by_need[forbidden_index] = true;
                attempts += 1;
                continue;
            };
            if plan.reason_code != PLAN_OK {
                forbidden_offer_by_need[forbidden_index] = true;
                attempts += 1;
                continue;
            }

            let order = proposer_order
                .saturating_mul(goods)
                .saturating_mul(acquaintances)
                .saturating_mul(goods)
                .saturating_add(candidate.order as usize);
            candidates.push(ParallelPhenomenonCandidate {
                score: plan.candidate.score,
                agent_id: agent_id as i32,
                need_good: need_good as i32,
                max_need,
                friend_slot: plan.candidate.friend_slot,
                friend_id: plan.candidate.friend_id,
                offer_good: plan.candidate.offer_good,
                reciprocal_slot: plan.candidate.reciprocal_slot,
                max_exchange: plan.max_exchange,
                switch_average: plan.switch_average,
                need_transparency: plan.need_transparency,
                receiving_transparency: plan.receiving_transparency,
                order,
            });
            break;
        }
    }

    candidates.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.order.cmp(&right.order))
    });
    if one_candidate_per_agent && candidates.len() > 1 {
        candidates.truncate(1);
    }
    candidates
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn plan_parallel_phenomenon_candidates(
    deal_type: i32,
    goods: usize,
    acquaintances: usize,
    initial_transparency: f64,
    history: i32,
    local_liquidity_stock_bias: f64,
    local_liquidity_min_sales: f64,
    aspirational_stock_target: f64,
    min_trade_quantity: f64,
    trade_rounding_buffer: f64,
    proposer_ids: PyReadonlyArray1<'_, i32>,
    blocked_partner_ids: PyReadonlyArray1<'_, i32>,
    market_elastic_need: PyReadonlyArray1<'_, f32>,
    stock: PyReadonlyArray2<'_, f32>,
    role: PyReadonlyArray2<'_, i32>,
    stock_limit: PyReadonlyArray2<'_, f32>,
    purchase_price: PyReadonlyArray2<'_, f32>,
    sales_price: PyReadonlyArray2<'_, f32>,
    needs_level: PyReadonlyArray1<'_, f32>,
    transparency: PyReadonlyArray3<'_, f32>,
    friend_id: PyReadonlyArray2<'_, i32>,
    need: PyReadonlyArray2<'_, f32>,
    recent_sales: PyReadonlyArray2<'_, f32>,
    recent_purchases: PyReadonlyArray2<'_, f32>,
    recent_inventory_inflow: PyReadonlyArray2<'_, f32>,
    recent_production: PyReadonlyArray2<'_, f32>,
    friend_sold: PyReadonlyArray3<'_, f32>,
    one_candidate_per_agent: bool,
) -> PyResult<Vec<(f32, i32, i32, f64, i32, i32, i32, i32, f64, f64, f64, f64)>> {
    if deal_type != CONSUMPTION_DEAL && deal_type != SURPLUS_DEAL {
        return Err(PyValueError::new_err(
            "deal_type must be 1 (surplus) or 2 (consumption)",
        ));
    }

    let proposer_ids = proposer_ids.as_array();
    let blocked_partner_ids = blocked_partner_ids.as_array();
    let elastic_need = market_elastic_need.as_array();
    let stock = stock.as_array();
    let role = role.as_array();
    let stock_limit = stock_limit.as_array();
    let purchase_price = purchase_price.as_array();
    let sales_price = sales_price.as_array();
    let needs_level = needs_level.as_array();
    let transparency = transparency.as_array();
    let friend_id = friend_id.as_array();
    let need = need.as_array();
    let recent_sales = recent_sales.as_array();
    let recent_purchases = recent_purchases.as_array();
    let recent_inventory_inflow = recent_inventory_inflow.as_array();
    let recent_production = recent_production.as_array();
    let friend_sold = friend_sold.as_array();

    if goods > stock.ncols()
        || goods > role.ncols()
        || goods > purchase_price.ncols()
        || goods > sales_price.ncols()
        || goods > elastic_need.len()
        || acquaintances > friend_id.ncols()
    {
        return Ok(Vec::new());
    }

    let mut blocked_partner_mask = vec![false; stock.nrows()];
    for raw_partner_id in blocked_partner_ids.iter() {
        if *raw_partner_id >= 0 {
            let partner_id = *raw_partner_id as usize;
            if partner_id < blocked_partner_mask.len() {
                blocked_partner_mask[partner_id] = true;
            }
        }
    }

    let proposer_list: Vec<(usize, usize)> = proposer_ids
        .iter()
        .enumerate()
        .filter_map(|(order, raw_agent_id)| {
            if *raw_agent_id < 0 {
                return None;
            }
            let agent_id = *raw_agent_id as usize;
            if agent_id >= stock.nrows() {
                return None;
            }
            Some((order, agent_id))
        })
        .collect();
    if proposer_list.is_empty() {
        return Ok(Vec::new());
    }

    let initial_transparency_for_execution = initial_transparency;
    let initial_transparency = initial_transparency as f32;
    let mut candidates: Vec<ParallelPhenomenonCandidate> = proposer_list
        .par_iter()
        .flat_map_iter(|(proposer_order, agent_id)| {
            plan_agent_parallel_phenomenon_candidates(
                *proposer_order,
                *agent_id,
                deal_type,
                goods,
                acquaintances,
                initial_transparency,
                initial_transparency_for_execution,
                history,
                local_liquidity_stock_bias,
                local_liquidity_min_sales,
                aspirational_stock_target,
                min_trade_quantity,
                trade_rounding_buffer,
                &blocked_partner_mask,
                elastic_need,
                stock,
                role,
                stock_limit,
                purchase_price,
                sales_price,
                needs_level,
                transparency,
                friend_id,
                need,
                recent_sales,
                recent_purchases,
                recent_inventory_inflow,
                recent_production,
                friend_sold,
                one_candidate_per_agent,
            )
        })
        .collect();

    candidates.sort_by(|left, right| left.order.cmp(&right.order));
    Ok(candidates
        .into_iter()
        .map(|candidate| {
            (
                candidate.score,
                candidate.agent_id,
                candidate.need_good,
                candidate.max_need,
                candidate.friend_slot,
                candidate.friend_id,
                candidate.offer_good,
                candidate.reciprocal_slot,
                candidate.max_exchange,
                candidate.switch_average,
                candidate.need_transparency,
                candidate.receiving_transparency,
            )
        })
        .collect())
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
    Ok(candidate.map(|item| {
        (
            item.score,
            item.friend_slot,
            item.friend_id,
            item.offer_good,
        )
    }))
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

    let switch_average = ((friend_need_sales_price
        / (friend_offer_purchase_price * need_transparency).max(EPSILON as f64))
        + ((my_need_purchase_price * receiving_transparency)
            / my_offer_sales_price.max(EPSILON as f64)))
        / 2.0;

    let friend_needs_level = needs_level[fid] as f64;
    let mut max_exchange = (((my_stock[offer_good] as f64)
        - ((elastic_need[offer_good] as f64) * (my_needs_level as f64)))
        * receiving_transparency)
        / switch_average.max(EPSILON as f64);
    let reason_code = if max_exchange <= min_trade_quantity {
        PLAN_OFFER_SURPLUS_BELOW_MIN
    } else {
        max_exchange = max_exchange.min(max_need);
        let friend_supply = ((stock[[fid, need_good]] as f64)
            - (friend_needs_level * (elastic_need[need_good] as f64)))
            * need_transparency;
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
    min_trade_quantity: f64,
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
    mut recent_purchase_value: PyReadwriteArray2<'_, f32>,
    mut recent_sales_value: PyReadwriteArray2<'_, f32>,
    mut recent_inventory_inflow_value: PyReadwriteArray2<'_, f32>,
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
    use_value_price_floor_fraction: f64,
    legacy_price_floor: Option<f64>,
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
    let mut recent_purchase_value = recent_purchase_value.as_array_mut();
    let mut recent_sales_value = recent_sales_value.as_array_mut();
    let mut recent_inventory_inflow_value = recent_inventory_inflow_value.as_array_mut();
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
        let lower_limit =
            max_stocklimit_decrease * (previous_stock_limit[[agent_id, good_id]] as f64);
        let upper_limit =
            max_stocklimit_increase * (previous_stock_limit[[agent_id, good_id]] as f64);
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
        let mut price_floor = 0.0_f64;
        if use_value_price_floor_fraction > 0.0 {
            let needs_level_value = (needs_level[agent_id] as f64).max(1.0);
            let visible_need = elastic_need * needs_level_value * (history as f64);
            let visible_capacity = visible_need
                .max(stock_limit_value)
                .max(min_trade_quantity.max(epsilon64));
            let survival_fraction = (1.0 - spoilage_rate).powi(history.max(1)).max(epsilon64);
            let spoilage_adjusted_capacity = visible_capacity / survival_fraction;
            let stock_value = surplus.max(0.0);
            let inventory_factor = if stock_value > spoilage_adjusted_capacity {
                spoilage_adjusted_capacity / stock_value.max(epsilon64)
            } else {
                1.0
            };
            price_floor =
                (production_cost * use_value_price_floor_fraction * inventory_factor).max(0.0);
        }
        if let Some(floor) = legacy_price_floor {
            price_floor = price_floor.max(floor);
        }
        let scarcity_case = i32::from(surplus > elastic_need)
            + i32::from(surplus > (stock_limit_value + elastic_need));

        let mut purchase_price_value = purchase_price[[agent_id, good_id]] as f64;
        if scarcity_case == 0 {
            if role_value == ROLE_CONSUMER {
                if recent_purchases[[agent_id, good_id]]
                    < (recent_production[[agent_id, good_id]] + 1.0)
                {
                    purchase_price_value = production_cost;
                } else if purchase_times[[agent_id, good_id]] == 0
                    && purchase_price[[agent_id, good_id]] < production_cost as f32
                {
                    purchase_price_value =
                        mul_assign_like_numpy_float32(purchase_price_value, price_leap);
                }
            } else if role_value == ROLE_RETAILER {
                let purchased = purchased_this_period[[agent_id, good_id]] as f64;
                let sold = sold_this_period[[agent_id, good_id]] as f64;
                let tolerance = min_trade_quantity.max(EPSILON as f64);
                let sell_through = sold > 0.0 && sold + tolerance >= purchased;
                let has_room_for_hike = purchase_price_value * (price_hike as f64)
                    < sales_price[[agent_id, good_id]] as f64;
                if sell_through && has_room_for_hike {
                    purchase_price_value =
                        mul_assign_like_numpy_float32(purchase_price_value, price_hike);
                } else if purchased > sold + tolerance {
                    purchase_price_value =
                        mul_assign_like_numpy_float32(purchase_price_value, price_reduction);
                }
            }
        } else if scarcity_case == 1 {
            if role_value == ROLE_RETAILER {
                if purchase_times[[agent_id, good_id]] > 1
                    && purchased_this_period[[agent_id, good_id]]
                        > (stock_limit[[agent_id, good_id]] / 2.0)
                {
                    purchase_price_value = ((sum_period_purchase_value[[agent_id, good_id]]
                        + ((history as f32) * purchase_price[[agent_id, good_id]]))
                        / ((purchase_times[[agent_id, good_id]] + history) as f32))
                        as f64;
                }
                if purchase_price_value > sales_price[[agent_id, good_id]] as f64 {
                    purchase_price_value = mul_assign_like_numpy_float32(
                        sales_price[[agent_id, good_id]] as f64,
                        price_reduction,
                    );
                }
                let purchased = purchased_this_period[[agent_id, good_id]] as f64;
                let sold = sold_this_period[[agent_id, good_id]] as f64;
                if purchased > sold + min_trade_quantity.max(EPSILON as f64) {
                    purchase_price_value =
                        mul_assign_like_numpy_float32(purchase_price_value, price_reduction);
                }
            } else if role_value == ROLE_PRODUCER
                && purchased_this_period[[agent_id, good_id]]
                    > produced_this_period[[agent_id, good_id]]
            {
                purchase_price_value =
                    mul_assign_like_numpy_float32(purchase_price_value, price_reduction);
            }
        } else if role_value == ROLE_CONSUMER {
            purchase_price_value =
                mul_assign_like_numpy_float32(purchase_price_value, price_reduction);
        } else if (role_value == ROLE_RETAILER || role_value == ROLE_PRODUCER)
            && purchase_times[[agent_id, good_id]] > 0
        {
            purchase_price_value =
                mul_assign_like_numpy_float32(purchase_price_value, price_reduction);
        }
        if price_floor > 0.0 {
            purchase_price_value = purchase_price_value.max(price_floor);
            let sales_floor = if role_value == ROLE_RETAILER {
                price_floor / (price_reduction as f64).max(epsilon64)
            } else {
                price_floor
            };
            if (sales_price[[agent_id, good_id]] as f64) < sales_floor {
                sales_price[[agent_id, good_id]] = sales_floor as f32;
            }
        }
        if role_value == ROLE_RETAILER
            && purchase_price_value >= sales_price[[agent_id, good_id]] as f64
        {
            let discounted_purchase = mul_assign_like_numpy_float32(
                sales_price[[agent_id, good_id]] as f64,
                price_reduction,
            );
            if discounted_purchase >= price_floor {
                purchase_price_value = discounted_purchase;
            } else {
                purchase_price_value = price_floor;
                let sales_floor = price_floor / (price_reduction as f64).max(epsilon64);
                if (sales_price[[agent_id, good_id]] as f64) < sales_floor {
                    sales_price[[agent_id, good_id]] = sales_floor as f32;
                }
            }
        }
        purchase_price[[agent_id, good_id]] = purchase_price_value as f32;

        let mut sales_price_value = sales_price[[agent_id, good_id]] as f64;
        let previous_sales_price = sales_price_value;
        if scarcity_case == 0 {
            if role_value == ROLE_CONSUMER {
                let target = production_cost.min(purchase_price[[agent_id, good_id]] as f64);
                if sales_price_value < target {
                    sales_price_value = price_hike * target;
                }
            } else if role_value == ROLE_RETAILER {
                let purchased = purchased_this_period[[agent_id, good_id]] as f64;
                let sold = sold_this_period[[agent_id, good_id]] as f64;
                let tolerance = min_trade_quantity.max(EPSILON as f64);
                let sell_through = sold > 0.0 && sold + tolerance >= purchased;
                if sales_times[[agent_id, good_id]] > 1 && sell_through {
                    let blended_price = ((sum_period_sales_value[[agent_id, good_id]]
                        + sales_price[[agent_id, good_id]])
                        / ((sales_times[[agent_id, good_id]] + 1) as f32))
                        as f64;
                    sales_price_value = blended_price.min(price_leap * previous_sales_price);
                }
                if sales_price_value < purchase_price[[agent_id, good_id]] as f64 {
                    sales_price_value = purchase_price[[agent_id, good_id]] as f64;
                }
                if sales_times[[agent_id, good_id]] == 0 && surplus > tolerance {
                    sales_price_value =
                        mul_assign_like_numpy_float32(sales_price_value, price_reduction);
                }
            } else if role_value == ROLE_PRODUCER && sales_price_value < production_cost {
                sales_price_value = price_hike * production_cost;
            }
        } else if scarcity_case == 1 {
            if role_value == ROLE_CONSUMER {
                sales_price_value = production_cost.max(purchase_price[[agent_id, good_id]] as f64);
                if surplus > (stock_limit_value / 2.0) {
                    sales_price_value =
                        ((purchase_price[[agent_id, good_id]] as f64) * 2.0).min(production_cost);
                }
            } else if role_value == ROLE_RETAILER {
                if sold_this_period[[agent_id, good_id]] < market_elastic_need[good_id] {
                    sales_price_value =
                        mul_assign_like_numpy_float32(sales_price_value, price_reduction);
                }
            } else if role_value == ROLE_PRODUCER {
                if sold_this_period[[agent_id, good_id]] < market_elastic_need[good_id] {
                    sales_price_value = production_cost;
                } else if sales_price_value < production_cost {
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
                    sales_price_value =
                        mul_assign_like_numpy_float32(sales_price_value, price_reduction);
                }
            }
        } else if role_value == ROLE_PRODUCER {
            if sold_this_period[[agent_id, good_id]] < market_elastic_need[good_id] {
                if sold_this_period[[agent_id, good_id]] <= min_trade_quantity as f32 {
                    if sales_price_value > production_cost {
                        sales_price_value = production_cost;
                    } else {
                        sales_price_value =
                            mul_assign_like_numpy_float32(sales_price_value, price_reduction);
                    }
                } else {
                    sales_price_value =
                        mul_assign_like_numpy_float32(sales_price_value, price_reduction);
                }
            }
        }
        if price_floor > 0.0 {
            let sales_floor = if role_value == ROLE_RETAILER {
                price_floor / (price_reduction as f64).max(epsilon64)
            } else {
                price_floor
            };
            sales_price_value = sales_price_value.max(sales_floor);
        }
        if role_value == ROLE_RETAILER
            && purchase_price[[agent_id, good_id]] as f64 >= sales_price_value
        {
            let discounted_purchase =
                mul_assign_like_numpy_float32(sales_price_value, price_reduction);
            if discounted_purchase >= price_floor {
                purchase_price[[agent_id, good_id]] = discounted_purchase as f32;
            } else {
                purchase_price[[agent_id, good_id]] = price_floor as f32;
                sales_price_value =
                    sales_price_value.max(price_floor / (price_reduction as f64).max(epsilon64));
            }
        }
        sales_price[[agent_id, good_id]] = sales_price_value as f32;

        spoilage[[agent_id, good_id]] = 0.0;
        if (stock[[agent_id, good_id]] as f64)
            > (stock_limit_multiplier * (market_elastic_need[good_id] as f64))
        {
            if (stock[[agent_id, good_id]] as f64)
                > (stock_spoil_threshold * (stock_limit[[agent_id, good_id]] as f64))
            {
                if stock[[agent_id, good_id]] > stock_limit[[agent_id, good_id]] {
                    let spoiled = ((stock[[agent_id, good_id]] as f64)
                        - (stock_limit[[agent_id, good_id]] as f64))
                        * spoilage_rate;
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
        recent_purchase_value[[agent_id, good_id]] *= activity_discount as f32;
        recent_sales_value[[agent_id, good_id]] *= activity_discount as f32;
        recent_inventory_inflow_value[[agent_id, good_id]] *= activity_discount as f32;
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
                * ((10.0 * purchased) / ((10.0 * purchased) + (cycle + 1) as f64).max(epsilon64));
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

#[derive(Default)]
struct BasketProfile {
    sessions: u64,
    active_sessions: u64,
    attempts: u64,
    proposed: u64,
    accepted: u64,
    static_builds: u64,
    static_index_builds: u64,
    first_valid_calls: u64,
    full_plan_calls: u64,
    max_need_calls: u64,
    specific_plan_calls: u64,
    reciprocal_scan_ns: u128,
    static_build_ns: u128,
    static_index_ns: u128,
    first_valid_ns: u128,
    full_plan_ns: u128,
    max_need_ns: u128,
    specific_plan_ns: u128,
    execute_ns: u128,
}

impl BasketProfile {
    fn from_env() -> Option<Self> {
        match std::env::var("EM_PROFILE_BASKET") {
            Ok(value) if value != "0" && !value.eq_ignore_ascii_case("false") => {
                Some(Self::default())
            }
            _ => None,
        }
    }

    fn ms(ns: u128) -> f64 {
        ns as f64 / 1_000_000.0
    }

    fn print(&self, cycle: usize, population: usize, goods: usize, acquaintances: usize) {
        eprintln!(
            concat!(
                "EM_PROFILE_BASKET cycle={} population={} goods={} acquaintances={} ",
                "sessions={} active_sessions={} attempts={} proposed={} accepted={} ",
                "static_builds={} static_index_builds={} first_valid_calls={} ",
                "full_plan_calls={} max_need_calls={} specific_plan_calls={} ",
                "reciprocal_ms={:.3} static_build_ms={:.3} static_index_ms={:.3} ",
                "first_valid_ms={:.3} full_plan_ms={:.3} max_need_ms={:.3} ",
                "specific_plan_ms={:.3} execute_ms={:.3}"
            ),
            cycle,
            population,
            goods,
            acquaintances,
            self.sessions,
            self.active_sessions,
            self.attempts,
            self.proposed,
            self.accepted,
            self.static_builds,
            self.static_index_builds,
            self.first_valid_calls,
            self.full_plan_calls,
            self.max_need_calls,
            self.specific_plan_calls,
            Self::ms(self.reciprocal_scan_ns),
            Self::ms(self.static_build_ns),
            Self::ms(self.static_index_ns),
            Self::ms(self.first_valid_ns),
            Self::ms(self.full_plan_ns),
            Self::ms(self.max_need_ns),
            Self::ms(self.specific_plan_ns),
            Self::ms(self.execute_ns),
        );
    }
}

fn splitmix64_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

fn sample_new_friend_internal(
    agent_id: usize,
    population: usize,
    acquaintances: usize,
    seed: u64,
    cycle: usize,
    salt: u64,
    friend_id: ArrayView2<'_, i32>,
) -> i32 {
    if population <= 1 {
        return -1;
    }
    let mut rng_state = seed
        .wrapping_add(((cycle + 1) as u64).wrapping_mul(1_000_003))
        .wrapping_add((agent_id as u64).wrapping_mul(97))
        .wrapping_add(salt);
    let draws = acquaintances.saturating_mul(2).max(8);
    for _ in 0..draws {
        let candidate = (splitmix64_next(&mut rng_state) % (population as u64)) as usize;
        if candidate != agent_id
            && find_friend_slot_scan_internal(friend_id, agent_id, candidate as i32) < 0
        {
            return candidate as i32;
        }
    }
    for candidate in 0..population {
        if candidate != agent_id
            && find_friend_slot_scan_internal(friend_id, agent_id, candidate as i32) < 0
        {
            return candidate as i32;
        }
    }
    -1
}

#[allow(clippy::too_many_arguments)]
fn add_random_friend_internal(
    agent_id: usize,
    population: usize,
    acquaintances: usize,
    goods: usize,
    seed: u64,
    cycle: usize,
    initial_transparency: f32,
    initial_transactions: f32,
    friend_id: &mut ArrayViewMut2<'_, i32>,
    friend_activity: &mut ArrayViewMut2<'_, f32>,
    friend_purchased: &mut ArrayViewMut3<'_, f32>,
    friend_sold: &mut ArrayViewMut3<'_, f32>,
    transparency: &mut ArrayViewMut3<'_, f32>,
) {
    let candidate = sample_new_friend_internal(
        agent_id,
        population,
        acquaintances,
        seed,
        cycle,
        17,
        friend_id.view(),
    );
    if candidate < 0 {
        return;
    }
    ensure_friend_link_array_only(
        agent_id,
        candidate as usize,
        acquaintances,
        goods,
        initial_transparency,
        initial_transactions,
        friend_id,
        friend_activity,
        friend_purchased,
        friend_sold,
        transparency,
    );
}

#[allow(clippy::too_many_arguments)]
fn surplus_production_internal(
    agent_id: usize,
    goods: usize,
    switch_time: f32,
    stock_limit_multiplier: f32,
    price_hike: f32,
    base_need: ArrayView2<'_, f32>,
    mut stock: ArrayViewMut2<'_, f32>,
    stock_limit: ArrayView2<'_, f32>,
    talent_mask: ArrayView2<'_, f32>,
    purchase_times: ArrayView2<'_, i32>,
    efficiency: ArrayView2<'_, f32>,
    sales_price: ArrayView2<'_, f32>,
    mut time_remaining: ArrayViewMut1<'_, f32>,
    mut recent_production: ArrayViewMut2<'_, f32>,
    mut produced_this_period: ArrayViewMut2<'_, f32>,
) -> f64 {
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
            let production_profitable = (purchase_times[[agent_id, good_id]] == 0
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
            time_remaining[agent_id] -=
                switch_time + (produced / efficiency[[agent_id, selected_good]].max(EPSILON));
        }
    }
    produced_total
}

#[allow(clippy::too_many_arguments)]
fn leisure_production_internal(
    agent_id: usize,
    goods: usize,
    mut stock: ArrayViewMut2<'_, f32>,
    stock_limit: ArrayView2<'_, f32>,
    talent_mask: ArrayView2<'_, f32>,
    purchase_price: ArrayView2<'_, f32>,
    mut time_remaining: ArrayViewMut1<'_, f32>,
    mut recent_production: ArrayViewMut2<'_, f32>,
    mut produced_this_period: ArrayViewMut2<'_, f32>,
) -> f64 {
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
    produced_total
}

#[allow(clippy::too_many_arguments)]
fn end_agent_period_internal(
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
    min_trade_quantity: f64,
    max_stocklimit_decrease: f64,
    max_stocklimit_increase: f64,
    max_efficiency_downgrade: f64,
    max_efficiency_upgrade: f64,
    base_need: ArrayView2<'_, f32>,
    mut stock: ArrayViewMut2<'_, f32>,
    mut stock_limit: ArrayViewMut2<'_, f32>,
    mut previous_stock_limit: ArrayViewMut2<'_, f32>,
    mut efficiency: ArrayViewMut2<'_, f32>,
    mut learned_efficiency: ArrayViewMut2<'_, f32>,
    mut recent_production: ArrayViewMut2<'_, f32>,
    mut recent_sales: ArrayViewMut2<'_, f32>,
    mut recent_purchases: ArrayViewMut2<'_, f32>,
    mut recent_inventory_inflow: ArrayViewMut2<'_, f32>,
    mut recent_purchase_value: ArrayViewMut2<'_, f32>,
    mut recent_sales_value: ArrayViewMut2<'_, f32>,
    mut recent_inventory_inflow_value: ArrayViewMut2<'_, f32>,
    produced_this_period: ArrayViewMut2<'_, f32>,
    mut produced_last_period: ArrayViewMut2<'_, f32>,
    sold_this_period: ArrayViewMut2<'_, f32>,
    mut sold_last_period: ArrayViewMut2<'_, f32>,
    purchased_this_period: ArrayViewMut2<'_, f32>,
    mut purchased_last_period: ArrayViewMut2<'_, f32>,
    mut purchase_times: ArrayViewMut2<'_, i32>,
    mut sales_times: ArrayViewMut2<'_, i32>,
    mut sum_period_purchase_value: ArrayViewMut2<'_, f32>,
    mut sum_period_sales_value: ArrayViewMut2<'_, f32>,
    mut spoilage: ArrayViewMut2<'_, f32>,
    mut periodic_spoilage: ArrayViewMut1<'_, f32>,
    talent_mask: ArrayView2<'_, f32>,
    mut role: ArrayViewMut2<'_, i32>,
    mut purchase_price: ArrayViewMut2<'_, f32>,
    mut sales_price: ArrayViewMut2<'_, f32>,
    mut friend_activity: ArrayViewMut2<'_, f32>,
    friend_purchased: ArrayView3<'_, f32>,
    mut transparency: ArrayViewMut3<'_, f32>,
    needs_level: ArrayView1<'_, f32>,
    market_elastic_need: ArrayView1<'_, f32>,
    mut market_periodic_spoilage: ArrayViewMut1<'_, f32>,
    use_value_price_floor_fraction: f64,
    legacy_price_floor: Option<f64>,
) {
    let epsilon64 = EPSILON as f64;
    periodic_spoilage[agent_id] = 0.0;

    for good_id in 0..goods {
        let elastic_component = market_elastic_need[good_id] * needs_level[agent_id];
        let target_stock_limit = (stock_limit_multiplier * (elastic_component as f64))
            + (recent_sales[[agent_id, good_id]] as f64);
        let lower_limit =
            max_stocklimit_decrease * (previous_stock_limit[[agent_id, good_id]] as f64);
        let upper_limit =
            max_stocklimit_increase * (previous_stock_limit[[agent_id, good_id]] as f64);
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
        let mut price_floor = 0.0_f64;
        if use_value_price_floor_fraction > 0.0 {
            let needs_level_value = (needs_level[agent_id] as f64).max(1.0);
            let visible_need = elastic_need * needs_level_value * (history as f64);
            let visible_capacity = visible_need
                .max(stock_limit_value)
                .max(min_trade_quantity.max(epsilon64));
            let survival_fraction = (1.0 - spoilage_rate).powi(history.max(1)).max(epsilon64);
            let spoilage_adjusted_capacity = visible_capacity / survival_fraction;
            let stock_value = surplus.max(0.0);
            let inventory_factor = if stock_value > spoilage_adjusted_capacity {
                spoilage_adjusted_capacity / stock_value.max(epsilon64)
            } else {
                1.0
            };
            price_floor =
                (production_cost * use_value_price_floor_fraction * inventory_factor).max(0.0);
        }
        if let Some(floor) = legacy_price_floor {
            price_floor = price_floor.max(floor);
        }
        let scarcity_case = i32::from(surplus > elastic_need)
            + i32::from(surplus > (stock_limit_value + elastic_need));

        let mut purchase_price_value = purchase_price[[agent_id, good_id]] as f64;
        if scarcity_case == 0 {
            if role_value == ROLE_CONSUMER {
                if recent_purchases[[agent_id, good_id]]
                    < (recent_production[[agent_id, good_id]] + 1.0)
                {
                    purchase_price_value = production_cost;
                } else if purchase_times[[agent_id, good_id]] == 0
                    && purchase_price[[agent_id, good_id]] < production_cost as f32
                {
                    purchase_price_value =
                        mul_assign_like_numpy_float32(purchase_price_value, price_leap);
                }
            } else if role_value == ROLE_RETAILER {
                let purchased = purchased_this_period[[agent_id, good_id]] as f64;
                let sold = sold_this_period[[agent_id, good_id]] as f64;
                let tolerance = min_trade_quantity.max(EPSILON as f64);
                let sell_through = sold > 0.0 && sold + tolerance >= purchased;
                let has_room_for_hike = purchase_price_value * (price_hike as f64)
                    < sales_price[[agent_id, good_id]] as f64;
                if sell_through && has_room_for_hike {
                    purchase_price_value =
                        mul_assign_like_numpy_float32(purchase_price_value, price_hike);
                } else if purchased > sold + tolerance {
                    purchase_price_value =
                        mul_assign_like_numpy_float32(purchase_price_value, price_reduction);
                }
            }
        } else if scarcity_case == 1 {
            if role_value == ROLE_RETAILER {
                if purchase_times[[agent_id, good_id]] > 1
                    && purchased_this_period[[agent_id, good_id]]
                        > (stock_limit[[agent_id, good_id]] / 2.0)
                {
                    purchase_price_value = ((sum_period_purchase_value[[agent_id, good_id]]
                        + ((history as f32) * purchase_price[[agent_id, good_id]]))
                        / ((purchase_times[[agent_id, good_id]] + history) as f32))
                        as f64;
                }
                if purchase_price_value > sales_price[[agent_id, good_id]] as f64 {
                    purchase_price_value = mul_assign_like_numpy_float32(
                        sales_price[[agent_id, good_id]] as f64,
                        price_reduction,
                    );
                }
                let purchased = purchased_this_period[[agent_id, good_id]] as f64;
                let sold = sold_this_period[[agent_id, good_id]] as f64;
                if purchased > sold + min_trade_quantity.max(EPSILON as f64) {
                    purchase_price_value =
                        mul_assign_like_numpy_float32(purchase_price_value, price_reduction);
                }
            } else if role_value == ROLE_PRODUCER
                && purchased_this_period[[agent_id, good_id]]
                    > produced_this_period[[agent_id, good_id]]
            {
                purchase_price_value =
                    mul_assign_like_numpy_float32(purchase_price_value, price_reduction);
            }
        } else if role_value == ROLE_CONSUMER {
            purchase_price_value =
                mul_assign_like_numpy_float32(purchase_price_value, price_reduction);
        } else if (role_value == ROLE_RETAILER || role_value == ROLE_PRODUCER)
            && purchase_times[[agent_id, good_id]] > 0
        {
            purchase_price_value =
                mul_assign_like_numpy_float32(purchase_price_value, price_reduction);
        }
        if price_floor > 0.0 {
            purchase_price_value = purchase_price_value.max(price_floor);
            let sales_floor = if role_value == ROLE_RETAILER {
                price_floor / (price_reduction as f64).max(epsilon64)
            } else {
                price_floor
            };
            if (sales_price[[agent_id, good_id]] as f64) < sales_floor {
                sales_price[[agent_id, good_id]] = sales_floor as f32;
            }
        }
        if role_value == ROLE_RETAILER
            && purchase_price_value >= sales_price[[agent_id, good_id]] as f64
        {
            let discounted_purchase = mul_assign_like_numpy_float32(
                sales_price[[agent_id, good_id]] as f64,
                price_reduction,
            );
            if discounted_purchase >= price_floor {
                purchase_price_value = discounted_purchase;
            } else {
                purchase_price_value = price_floor;
                let sales_floor = price_floor / (price_reduction as f64).max(epsilon64);
                if (sales_price[[agent_id, good_id]] as f64) < sales_floor {
                    sales_price[[agent_id, good_id]] = sales_floor as f32;
                }
            }
        }
        purchase_price[[agent_id, good_id]] = purchase_price_value as f32;

        let mut sales_price_value = sales_price[[agent_id, good_id]] as f64;
        let previous_sales_price = sales_price_value;
        if scarcity_case == 0 {
            if role_value == ROLE_CONSUMER {
                let target = production_cost.min(purchase_price[[agent_id, good_id]] as f64);
                if sales_price_value < target {
                    sales_price_value = price_hike * target;
                }
            } else if role_value == ROLE_RETAILER {
                let purchased = purchased_this_period[[agent_id, good_id]] as f64;
                let sold = sold_this_period[[agent_id, good_id]] as f64;
                let tolerance = min_trade_quantity.max(EPSILON as f64);
                let sell_through = sold > 0.0 && sold + tolerance >= purchased;
                if sales_times[[agent_id, good_id]] > 1 && sell_through {
                    let blended_price = ((sum_period_sales_value[[agent_id, good_id]]
                        + sales_price[[agent_id, good_id]])
                        / ((sales_times[[agent_id, good_id]] + 1) as f32))
                        as f64;
                    sales_price_value = blended_price.min(price_leap * previous_sales_price);
                }
                if sales_price_value < purchase_price[[agent_id, good_id]] as f64 {
                    sales_price_value = purchase_price[[agent_id, good_id]] as f64;
                }
                if sales_times[[agent_id, good_id]] == 0 && surplus > tolerance {
                    sales_price_value =
                        mul_assign_like_numpy_float32(sales_price_value, price_reduction);
                }
            } else if role_value == ROLE_PRODUCER && sales_price_value < production_cost {
                sales_price_value = price_hike * production_cost;
            }
        } else if scarcity_case == 1 {
            if role_value == ROLE_CONSUMER {
                sales_price_value = production_cost.max(purchase_price[[agent_id, good_id]] as f64);
                if surplus > (stock_limit_value / 2.0) {
                    sales_price_value =
                        ((purchase_price[[agent_id, good_id]] as f64) * 2.0).min(production_cost);
                }
            } else if role_value == ROLE_RETAILER {
                if sold_this_period[[agent_id, good_id]] < market_elastic_need[good_id] {
                    sales_price_value =
                        mul_assign_like_numpy_float32(sales_price_value, price_reduction);
                }
            } else if role_value == ROLE_PRODUCER {
                if sold_this_period[[agent_id, good_id]] < market_elastic_need[good_id] {
                    sales_price_value = production_cost;
                } else if sales_price_value < production_cost {
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
                    sales_price_value =
                        mul_assign_like_numpy_float32(sales_price_value, price_reduction);
                }
            }
        } else if role_value == ROLE_PRODUCER {
            if sold_this_period[[agent_id, good_id]] < market_elastic_need[good_id] {
                if sold_this_period[[agent_id, good_id]] <= min_trade_quantity as f32 {
                    if sales_price_value > production_cost {
                        sales_price_value = production_cost;
                    } else {
                        sales_price_value =
                            mul_assign_like_numpy_float32(sales_price_value, price_reduction);
                    }
                } else {
                    sales_price_value =
                        mul_assign_like_numpy_float32(sales_price_value, price_reduction);
                }
            }
        }
        if price_floor > 0.0 {
            let sales_floor = if role_value == ROLE_RETAILER {
                price_floor / (price_reduction as f64).max(epsilon64)
            } else {
                price_floor
            };
            sales_price_value = sales_price_value.max(sales_floor);
        }
        if role_value == ROLE_RETAILER
            && purchase_price[[agent_id, good_id]] as f64 >= sales_price_value
        {
            let discounted_purchase =
                mul_assign_like_numpy_float32(sales_price_value, price_reduction);
            if discounted_purchase >= price_floor {
                purchase_price[[agent_id, good_id]] = discounted_purchase as f32;
            } else {
                purchase_price[[agent_id, good_id]] = price_floor as f32;
                sales_price_value =
                    sales_price_value.max(price_floor / (price_reduction as f64).max(epsilon64));
            }
        }
        sales_price[[agent_id, good_id]] = sales_price_value as f32;

        spoilage[[agent_id, good_id]] = 0.0;
        if (stock[[agent_id, good_id]] as f64)
            > (stock_limit_multiplier * (market_elastic_need[good_id] as f64))
        {
            if (stock[[agent_id, good_id]] as f64)
                > (stock_spoil_threshold * (stock_limit[[agent_id, good_id]] as f64))
            {
                if stock[[agent_id, good_id]] > stock_limit[[agent_id, good_id]] {
                    let spoiled = ((stock[[agent_id, good_id]] as f64)
                        - (stock_limit[[agent_id, good_id]] as f64))
                        * spoilage_rate;
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
        recent_purchase_value[[agent_id, good_id]] *= activity_discount as f32;
        recent_sales_value[[agent_id, good_id]] *= activity_discount as f32;
        recent_inventory_inflow_value[[agent_id, good_id]] *= activity_discount as f32;
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
                * ((10.0 * purchased) / ((10.0 * purchased) + (cycle + 1) as f64).max(epsilon64));
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
    lifestyle_promotion_threshold: f32,
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
                * ((sold_this_period[[agent_id, good_id]] - sold_last_period[[agent_id, good_id]])
                    + elastic_need);
            surplus_value = surplus_value.min(cap);
            surplus_value *= sales_price[[agent_id, good_id]];
        } else {
            surplus_value = surplus_value.min(elastic_need * max_needs_increase);
            surplus_value *= purchase_price[[agent_id, good_id]]
                .min(1.0 / efficiency[[agent_id, good_id]].max(EPSILON));
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
    } else if stock_level > lifestyle_promotion_threshold {
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
    recent_needs_increment[agent_id] = ((level_ratio
        + ((history as f64) * (recent_needs_increment[agent_id] as f64)))
        / ((history + 1) as f64)) as f32;

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
    let mut required_time = 0.0_f32;
    for good_id in 0..goods {
        let pending = need[[agent_id, good_id]];
        if pending <= 0.0 {
            continue;
        }
        required_time += pending / efficiency[[agent_id, good_id]].max(EPSILON);
    }
    if required_time <= EPSILON {
        return 0.0;
    }

    let available_time = time_remaining[agent_id].max(0.0);
    let scale = if required_time <= available_time + EPSILON {
        1.0
    } else if available_time > EPSILON {
        available_time / required_time
    } else {
        0.0
    };

    let mut produced_total = 0.0_f32;
    for good_id in 0..goods {
        let pending = need[[agent_id, good_id]];
        if pending <= 0.0 {
            continue;
        }
        let produced = pending * scale;
        recent_production[[agent_id, good_id]] += produced;
        produced_this_period[[agent_id, good_id]] += produced;
        need[[agent_id, good_id]] = pending - produced;
        produced_total += produced;
    }

    if scale >= 1.0 {
        time_remaining[agent_id] = (time_remaining[agent_id] - required_time).max(0.0);
    } else {
        time_remaining[agent_id] = 0.0;
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
    let capped_increment =
        raw_increment.min((recent_needs_increment[agent_id] as f64) * (max_needs_increase as f64));
    let extra_multiplier = (capped_increment - 1.0)
        .max(0.0)
        .min(max_leisure_extra_multiplier as f64);
    if extra_multiplier <= 0.0 {
        return (false, 0.0, 0.0);
    }

    recent_needs_increment[agent_id] = ((capped_increment
        + ((history as f64) * (recent_needs_increment[agent_id] as f64)))
        / ((history + 1) as f64)) as f32;

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

fn add_engine_i64_metric(engine: &Bound<'_, PyAny>, name: &str, delta: i32) -> PyResult<()> {
    if delta == 0 {
        return Ok(());
    }
    let current: i64 = engine.getattr(name)?.extract()?;
    engine.setattr(name, current + delta as i64)?;
    Ok(())
}

fn add_trade_stage_metrics(
    engine: &Bound<'_, PyAny>,
    proposed_count: i32,
    accepted_count: i32,
    accepted_volume: f64,
    inventory_trade_volume: f64,
) -> PyResult<()> {
    add_engine_i64_metric(engine, "_proposed_trade_count", proposed_count)?;
    add_engine_i64_metric(engine, "_accepted_trade_count", accepted_count)?;
    add_engine_metric(engine, "_accepted_trade_volume", accepted_volume)?;
    add_engine_metric(engine, "_inventory_trade_volume", inventory_trade_volume)?;
    Ok(())
}

fn set_period_failure_from_time_remaining(
    state: &Bound<'_, PyAny>,
    agent_id: usize,
) -> PyResult<()> {
    let time_remaining: PyReadonlyArray1<'_, f32> = state.getattr("time_remaining")?.extract()?;
    let mut period_failure: PyReadwriteArray1<'_, bool> =
        state.getattr("period_failure")?.extract()?;
    let time_remaining = time_remaining.as_array();
    let mut period_failure = period_failure.as_array_mut();
    period_failure[agent_id] = time_remaining[agent_id] < 0.0;
    Ok(())
}

fn apply_half_debt_adjustment(state: &Bound<'_, PyAny>, agent_id: usize) -> PyResult<()> {
    let mut period_time_debt: PyReadwriteArray1<'_, f32> =
        state.getattr("period_time_debt")?.extract()?;
    let mut time_remaining: PyReadwriteArray1<'_, f32> =
        state.getattr("time_remaining")?.extract()?;
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
    let mut period_time_debt: PyReadwriteArray1<'_, f32> =
        state.getattr("period_time_debt")?.extract()?;
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
    engine: &Bound<'_, PyAny>,
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
    uses_native_exchange_stage: bool,
    uses_agent_basket_planning: bool,
    totals: &mut NativeStageTotals,
) -> PyResult<()> {
    runner.call_method1("_prepare_agent_for_consumption", (agent_id,))?;
    if uses_agent_basket_planning {
        let (proposed_count, accepted_count, accepted_volume, inventory_trade_volume) =
            run_agent_basket_exchange_stage(
                py,
                runner.clone().unbind(),
                agent_id,
                CONSUMPTION_DEAL,
            )?;
        add_trade_stage_metrics(
            engine,
            proposed_count,
            accepted_count,
            accepted_volume,
            inventory_trade_volume,
        )?;
    } else if uses_native_exchange_stage {
        let (proposed_count, accepted_count, accepted_volume, inventory_trade_volume) =
            run_exchange_stage(py, runner.clone().unbind(), agent_id, CONSUMPTION_DEAL)?;
        add_trade_stage_metrics(
            engine,
            proposed_count,
            accepted_count,
            accepted_volume,
            inventory_trade_volume,
        )?;
    } else {
        runner.call_method1("_satisfy_needs_by_exchange", (agent_id,))?;
    }
    runner.call_method1("_advance_agent_to_surplus_stage", (agent_id,))?;
    if uses_agent_basket_planning {
        let (proposed_count, accepted_count, accepted_volume, inventory_trade_volume) =
            run_agent_basket_exchange_stage(py, runner.clone().unbind(), agent_id, SURPLUS_DEAL)?;
        add_trade_stage_metrics(
            engine,
            proposed_count,
            accepted_count,
            accepted_volume,
            inventory_trade_volume,
        )?;
    } else if uses_native_exchange_stage {
        let (proposed_count, accepted_count, accepted_volume, inventory_trade_volume) =
            run_exchange_stage(py, runner.clone().unbind(), agent_id, SURPLUS_DEAL)?;
        add_trade_stage_metrics(
            engine,
            proposed_count,
            accepted_count,
            accepted_volume,
            inventory_trade_volume,
        )?;
    } else {
        runner.call_method1("_make_surplus_deals", (agent_id,))?;
    }
    runner.call_method1("_complete_agent_period_after_surplus", (agent_id,))?;
    let _ = (
        py,
        engine,
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
        uses_native_exchange_stage,
        uses_agent_basket_planning,
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
    lifestyle_promotion_threshold: f32,
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
                * ((sold_this_period[[agent_id, good_id]] - sold_last_period[[agent_id, good_id]])
                    + elastic_need);
            surplus_value = surplus_value.min(cap);
            surplus_value *= sales_price[[agent_id, good_id]];
        } else {
            surplus_value = surplus_value.min(elastic_need * max_needs_increase);
            surplus_value *= purchase_price[[agent_id, good_id]]
                .min(1.0 / efficiency[[agent_id, good_id]].max(EPSILON));
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
    } else if stock_level > lifestyle_promotion_threshold {
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
    recent_needs_increment[agent_id] = ((level_ratio
        + ((history as f64) * (recent_needs_increment[agent_id] as f64)))
        / ((history + 1) as f64)) as f32;

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

    let mut required_time = 0.0_f32;
    for good_id in 0..goods {
        let pending = need[[agent_id, good_id]];
        if pending <= 0.0 {
            continue;
        }
        required_time += pending / efficiency[[agent_id, good_id]].max(EPSILON);
    }
    if required_time <= EPSILON {
        return Ok(0.0);
    }

    let available_time = time_remaining[agent_id].max(0.0);
    let scale = if required_time <= available_time + EPSILON {
        1.0
    } else if available_time > EPSILON {
        available_time / required_time
    } else {
        0.0
    };

    let mut produced_total = 0.0_f32;
    for good_id in 0..goods {
        let pending = need[[agent_id, good_id]];
        if pending <= 0.0 {
            continue;
        }
        let produced = pending * scale;
        recent_production[[agent_id, good_id]] += produced;
        produced_this_period[[agent_id, good_id]] += produced;
        need[[agent_id, good_id]] = pending - produced;
        produced_total += produced;
    }

    if scale >= 1.0 {
        time_remaining[agent_id] = (time_remaining[agent_id] - required_time).max(0.0);
    } else {
        time_remaining[agent_id] = 0.0;
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
            let production_profitable = (purchase_times[[agent_id, good_id]] == 0
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
            time_remaining[agent_id] -=
                switch_time + (produced / efficiency[[agent_id, selected_good]].max(EPSILON));
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
    let capped_increment =
        raw_increment.min((recent_needs_increment[agent_id] as f64) * (max_needs_increase as f64));
    let extra_multiplier = (capped_increment - 1.0)
        .max(0.0)
        .min(max_leisure_extra_multiplier as f64);
    if extra_multiplier <= 0.0 {
        return Ok((false, 0.0, 0.0));
    }

    recent_needs_increment[agent_id] = ((capped_increment
        + ((history as f64) * (recent_needs_increment[agent_id] as f64)))
        / ((history + 1) as f64)) as f32;

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
    initial_transparency_for_execution: f64,
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
    let need_transparency =
        transparency[[agent_id, candidate.friend_slot as usize, need_good]] as f64;
    let receiving_transparency = if candidate.reciprocal_slot >= 0 {
        transparency[[fid, candidate.reciprocal_slot as usize, offer_good]] as f64
    } else {
        initial_transparency_for_execution
    };
    let my_need_purchase_price = purchase_price[[agent_id, need_good]] as f64;
    let my_offer_sales_price = sales_price[[agent_id, offer_good]] as f64;
    let friend_need_sales_price = sales_price[[fid, need_good]] as f64;
    let friend_offer_purchase_price = purchase_price[[fid, offer_good]] as f64;

    let switch_average = ((friend_need_sales_price
        / (friend_offer_purchase_price * need_transparency).max(EPSILON as f64))
        + ((my_need_purchase_price * receiving_transparency)
            / my_offer_sales_price.max(EPSILON as f64)))
        / 2.0;

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
        let friend_supply = ((stock[[fid, need_good]] as f64)
            - (friend_needs_level * (elastic_need[need_good] as f64)))
            * need_transparency;
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

#[allow(clippy::too_many_arguments)]
fn basket_stage_max_need(
    agent_id: usize,
    need_good: usize,
    deal_type: i32,
    history: i32,
    local_liquidity_stock_bias: f64,
    local_liquidity_min_sales: f64,
    aspirational_stock_target: f64,
    elastic_need: ArrayView1<'_, f32>,
    stock: ArrayView2<'_, f32>,
    stock_limit: ArrayView2<'_, f32>,
    needs_level: ArrayView1<'_, f32>,
    need: ArrayView2<'_, f32>,
    recent_sales: ArrayView2<'_, f32>,
    recent_purchases: ArrayView2<'_, f32>,
    recent_inventory_inflow: ArrayView2<'_, f32>,
    recent_production: ArrayView2<'_, f32>,
    friend_id: ArrayView2<'_, i32>,
    friend_sold: ArrayView3<'_, f32>,
    transparency: ArrayView3<'_, f32>,
) -> f64 {
    if deal_type == CONSUMPTION_DEAL {
        if stock[[agent_id, need_good]] >= (elastic_need[need_good] * needs_level[agent_id]) {
            return 0.0;
        }
        let max_need = need[[agent_id, need_good]] as f64;
        if max_need < 1.0 {
            return 0.0;
        }
        return max_need;
    }

    let base_target_stock_limit = surplus_target_stock_limit(
        agent_id,
        need_good,
        history,
        local_liquidity_stock_bias,
        local_liquidity_min_sales,
        elastic_need,
        stock_limit,
        needs_level,
        recent_sales,
        recent_purchases,
        recent_inventory_inflow,
        recent_production,
        friend_id,
        friend_sold,
        transparency,
    );
    let aspirational_target = aspirational_stock_target_for_good(
        agent_id,
        need_good,
        aspirational_stock_target,
        elastic_need,
        stock_limit,
        needs_level,
    );
    let target_stock_limit = base_target_stock_limit.max(aspirational_target);
    let legacy_surplus_signal = recent_sales[[agent_id, need_good]]
        > (recent_production[[agent_id, need_good]] - elastic_need[need_good]);
    let local_liquidity_signal =
        target_stock_limit > (stock_limit[[agent_id, need_good]] as f64) + (EPSILON as f64);
    let aspirational_stock_signal = aspirational_target > EPSILON as f64
        && (stock[[agent_id, need_good]] as f64) < aspirational_target - (EPSILON as f64);
    if !(legacy_surplus_signal || local_liquidity_signal || aspirational_stock_signal) {
        return 0.0;
    }
    if (stock[[agent_id, need_good]] as f64)
        >= target_stock_limit - (elastic_need[need_good] as f64)
    {
        return 0.0;
    }
    (target_stock_limit - (stock[[agent_id, need_good]] as f64)).max(0.0)
}

fn aspirational_stock_target_for_good(
    agent_id: usize,
    good_id: usize,
    aspirational_stock_target: f64,
    elastic_need: ArrayView1<'_, f32>,
    stock_limit: ArrayView2<'_, f32>,
    needs_level: ArrayView1<'_, f32>,
) -> f64 {
    if aspirational_stock_target <= EPSILON as f64 {
        return 0.0;
    }
    let own_need_target =
        (elastic_need[good_id] as f64) * (needs_level[agent_id] as f64) * aspirational_stock_target;
    own_need_target
        .min(stock_limit[[agent_id, good_id]] as f64)
        .max(0.0)
}

#[allow(clippy::too_many_arguments)]
fn surplus_target_stock_limit(
    agent_id: usize,
    good_id: usize,
    history: i32,
    local_liquidity_stock_bias: f64,
    local_liquidity_min_sales: f64,
    _elastic_need: ArrayView1<'_, f32>,
    stock_limit: ArrayView2<'_, f32>,
    _needs_level: ArrayView1<'_, f32>,
    recent_sales: ArrayView2<'_, f32>,
    recent_purchases: ArrayView2<'_, f32>,
    recent_inventory_inflow: ArrayView2<'_, f32>,
    recent_production: ArrayView2<'_, f32>,
    friend_id: ArrayView2<'_, i32>,
    friend_sold: ArrayView3<'_, f32>,
    transparency: ArrayView3<'_, f32>,
) -> f64 {
    let base_limit = stock_limit[[agent_id, good_id]] as f64;
    if local_liquidity_stock_bias <= EPSILON as f64 {
        return base_limit;
    }
    let liquidity_score = local_liquidity_score(
        agent_id,
        good_id,
        history,
        local_liquidity_min_sales,
        recent_sales,
        recent_purchases,
        recent_inventory_inflow,
        recent_production,
        friend_id,
        friend_sold,
        transparency,
    );
    if liquidity_score <= EPSILON as f64 {
        return base_limit;
    }
    let stock_scale = local_liquidity_stock_scale(
        agent_id,
        good_id,
        local_liquidity_min_sales,
        recent_sales,
        recent_purchases,
        recent_inventory_inflow,
        friend_id,
        friend_sold,
    );
    if stock_scale <= EPSILON as f64 {
        return base_limit;
    }
    base_limit + (local_liquidity_stock_bias * stock_scale * liquidity_score)
}

fn local_liquidity_stock_scale(
    agent_id: usize,
    good_id: usize,
    local_liquidity_min_sales: f64,
    recent_sales: ArrayView2<'_, f32>,
    recent_purchases: ArrayView2<'_, f32>,
    recent_inventory_inflow: ArrayView2<'_, f32>,
    friend_id: ArrayView2<'_, i32>,
    friend_sold: ArrayView3<'_, f32>,
) -> f64 {
    if agent_id >= friend_id.nrows() || good_id >= recent_sales.ncols() {
        return 0.0;
    }
    let mut observed_acceptance = 0.0_f64;
    for friend_slot in 0..friend_id.ncols() {
        if friend_id[[agent_id, friend_slot]] < 0 {
            continue;
        }
        observed_acceptance += friend_sold[[agent_id, friend_slot, good_id]] as f64;
    }

    let own_sales = recent_sales[[agent_id, good_id]] as f64;
    let own_sources = (recent_purchases[[agent_id, good_id]]
        + recent_inventory_inflow[[agent_id, good_id]]) as f64;
    let observed_turnover = own_sales.min(own_sources).max(0.0);
    let bootstrap_floor = local_liquidity_min_sales.max(1.0);
    observed_turnover
        .max(observed_acceptance)
        .max(bootstrap_floor)
}

#[allow(clippy::too_many_arguments)]
fn local_liquidity_score(
    agent_id: usize,
    good_id: usize,
    history: i32,
    local_liquidity_min_sales: f64,
    recent_sales: ArrayView2<'_, f32>,
    recent_purchases: ArrayView2<'_, f32>,
    recent_inventory_inflow: ArrayView2<'_, f32>,
    recent_production: ArrayView2<'_, f32>,
    friend_id: ArrayView2<'_, i32>,
    friend_sold: ArrayView3<'_, f32>,
    transparency: ArrayView3<'_, f32>,
) -> f64 {
    if agent_id >= friend_id.nrows() || good_id >= recent_sales.ncols() {
        return 0.0;
    }
    let mut known_count = 0usize;
    let mut accepting_count = 0usize;
    let mut observed_acceptance = 0.0_f64;
    let mut transparency_sum = 0.0_f64;

    for friend_slot in 0..friend_id.ncols() {
        if friend_id[[agent_id, friend_slot]] < 0 {
            continue;
        }
        known_count += 1;
        let sold_count = friend_sold[[agent_id, friend_slot, good_id]] as f64;
        observed_acceptance += sold_count;
        if sold_count > 0.0 {
            accepting_count += 1;
            transparency_sum += transparency[[agent_id, friend_slot, good_id]] as f64;
        }
    }
    if known_count == 0 || observed_acceptance < local_liquidity_min_sales {
        return 0.0;
    }
    let acceptance_breadth = accepting_count as f64 / known_count as f64;
    if acceptance_breadth <= EPSILON as f64 {
        return 0.0;
    }
    let transparency_score = if accepting_count > 0 {
        transparency_sum / accepting_count as f64
    } else {
        0.0
    };
    let own_sales = recent_sales[[agent_id, good_id]] as f64;
    let own_sources = (recent_purchases[[agent_id, good_id]]
        + recent_inventory_inflow[[agent_id, good_id]]
        + recent_production[[agent_id, good_id]]) as f64;
    let turnover_score = if own_sales > EPSILON as f64 || own_sources > EPSILON as f64 {
        own_sales.min(own_sources) / own_sales.max(own_sources).max(EPSILON as f64)
    } else {
        1.0
    };
    let volume_scale = (local_liquidity_min_sales * history.max(1) as f64).max(1.0);
    let volume_score = (observed_acceptance / volume_scale).min(1.0);
    (transparency_score * acceptance_breadth.sqrt() * turnover_score.max(0.25) * volume_score)
        .clamp(0.0, 1.0)
}

#[allow(clippy::too_many_arguments)]
fn plan_specific_exchange_candidate(
    agent_id: usize,
    need_good: usize,
    initial_transparency_for_execution: f64,
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
    friend_ids: ArrayView2<'_, i32>,
    candidate: BasketCandidate,
) -> Option<PlannedExchangeInternal> {
    if need_good >= stock.ncols() || candidate.offer_good < 0 || candidate.friend_id < 0 {
        return None;
    }
    let offer_good = candidate.offer_good as usize;
    let friend_slot = candidate.friend_slot as usize;
    let fid = candidate.friend_id as usize;
    if offer_good >= stock.ncols()
        || offer_good == need_good
        || fid >= stock.nrows()
        || friend_slot >= friend_ids.ncols()
        || friend_ids[[agent_id, friend_slot]] != candidate.friend_id
    {
        return None;
    }

    let reciprocal_slot = find_friend_slot_scan_internal(friend_ids, fid, agent_id as i32);
    let need_transparency = transparency[[agent_id, friend_slot, need_good]] as f64;
    let receiving_transparency = if reciprocal_slot >= 0 {
        transparency[[fid, reciprocal_slot as usize, offer_good]] as f64
    } else {
        initial_transparency_for_execution
    };
    let my_need_purchase_price = purchase_price[[agent_id, need_good]] as f64;
    let my_offer_sales_price = sales_price[[agent_id, offer_good]] as f64;
    let friend_need_sales_price = sales_price[[fid, need_good]] as f64;
    let friend_offer_purchase_price = purchase_price[[fid, offer_good]] as f64;

    let switch_average = ((friend_need_sales_price
        / (friend_offer_purchase_price * need_transparency).max(EPSILON as f64))
        + ((my_need_purchase_price * receiving_transparency)
            / my_offer_sales_price.max(EPSILON as f64)))
        / 2.0;

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
        let friend_supply = ((stock[[fid, need_good]] as f64)
            - (friend_needs_level * (elastic_need[need_good] as f64)))
            * need_transparency;
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
        candidate: SearchCandidate {
            score: candidate.score,
            friend_slot: candidate.friend_slot,
            friend_id: candidate.friend_id,
            offer_good: candidate.offer_good,
            reciprocal_slot,
        },
        reason_code,
        max_exchange,
        switch_average,
        need_transparency,
        receiving_transparency,
    })
}

#[allow(clippy::too_many_arguments)]
fn plan_agent_basket_candidates_from_views(
    agent_id: usize,
    deal_type: i32,
    goods: usize,
    acquaintances: usize,
    initial_transparency: f32,
    history: i32,
    local_liquidity_stock_bias: f64,
    local_liquidity_min_sales: f64,
    aspirational_stock_target: f64,
    forbidden_offer_by_need: &[bool],
    elastic_need: ArrayView1<'_, f32>,
    stock: ArrayView2<'_, f32>,
    role: ArrayView2<'_, i32>,
    stock_limit: ArrayView2<'_, f32>,
    purchase_price: ArrayView2<'_, f32>,
    sales_price: ArrayView2<'_, f32>,
    needs_level: ArrayView1<'_, f32>,
    transparency: ArrayView3<'_, f32>,
    friend_id: ArrayView2<'_, i32>,
    need: ArrayView2<'_, f32>,
    recent_sales: ArrayView2<'_, f32>,
    recent_purchases: ArrayView2<'_, f32>,
    recent_inventory_inflow: ArrayView2<'_, f32>,
    recent_production: ArrayView2<'_, f32>,
    friend_sold: ArrayView3<'_, f32>,
    candidate_depth: usize,
    disable_offer_prefilter: bool,
) -> Vec<BasketCandidate> {
    if agent_id >= stock.nrows()
        || goods > stock.ncols()
        || forbidden_offer_by_need.len() < goods * goods
    {
        return Vec::new();
    }

    let friend_ids_row = friend_id.row(agent_id);
    if friend_ids_row.len() != acquaintances {
        return Vec::new();
    }

    let mut reciprocal_slots = vec![-1_i32; acquaintances];
    let mut has_known_partner = false;
    for friend_slot in 0..acquaintances {
        let fid = friend_ids_row[friend_slot];
        if fid >= 0 {
            has_known_partner = true;
            reciprocal_slots[friend_slot] =
                find_friend_slot_scan_internal(friend_id, fid as usize, agent_id as i32);
        }
    }
    if !has_known_partner {
        return Vec::new();
    }

    plan_agent_basket_candidates_from_cached_links(
        agent_id,
        deal_type,
        goods,
        acquaintances,
        initial_transparency,
        history,
        local_liquidity_stock_bias,
        local_liquidity_min_sales,
        aspirational_stock_target,
        forbidden_offer_by_need,
        elastic_need,
        stock,
        role,
        stock_limit,
        purchase_price,
        sales_price,
        needs_level,
        transparency,
        friend_ids_row,
        ArrayView1::from(&reciprocal_slots[..]),
        need,
        recent_sales,
        recent_purchases,
        recent_inventory_inflow,
        recent_production,
        friend_id,
        friend_sold,
        candidate_depth,
        disable_offer_prefilter,
    )
}

#[allow(clippy::too_many_arguments)]
fn plan_one_basket_candidate_from_cached_links(
    agent_id: usize,
    need_good: usize,
    deal_type: i32,
    goods: usize,
    acquaintances: usize,
    initial_transparency: f32,
    history: i32,
    local_liquidity_stock_bias: f64,
    local_liquidity_min_sales: f64,
    aspirational_stock_target: f64,
    forbidden_offer_by_need: &[bool],
    available_offer_goods: &[usize],
    elastic_need: ArrayView1<'_, f32>,
    stock: ArrayView2<'_, f32>,
    role: ArrayView2<'_, i32>,
    stock_limit: ArrayView2<'_, f32>,
    purchase_price: ArrayView2<'_, f32>,
    sales_price: ArrayView2<'_, f32>,
    needs_level: ArrayView1<'_, f32>,
    transparency: ArrayView3<'_, f32>,
    friend_ids_row: ArrayView1<'_, i32>,
    reciprocal_slots_view: ArrayView1<'_, i32>,
    need: ArrayView2<'_, f32>,
    recent_sales: ArrayView2<'_, f32>,
    recent_purchases: ArrayView2<'_, f32>,
    recent_inventory_inflow: ArrayView2<'_, f32>,
    recent_production: ArrayView2<'_, f32>,
    friend_id: ArrayView2<'_, i32>,
    friend_sold: ArrayView3<'_, f32>,
    my_stock: ArrayView1<'_, f32>,
    my_sales_price: ArrayView1<'_, f32>,
    my_purchase_price: ArrayView1<'_, f32>,
    my_role: ArrayView1<'_, i32>,
    my_transparency: ArrayView2<'_, f32>,
    my_needs_level: f32,
    allow_inner_parallel: bool,
) -> Option<BasketCandidate> {
    if basket_stage_max_need(
        agent_id,
        need_good,
        deal_type,
        history,
        local_liquidity_stock_bias,
        local_liquidity_min_sales,
        aspirational_stock_target,
        elastic_need,
        stock,
        stock_limit,
        needs_level,
        need,
        recent_sales,
        recent_purchases,
        recent_inventory_inflow,
        recent_production,
        friend_id,
        friend_sold,
        transparency,
    ) <= 0.0
    {
        return None;
    }

    let search_candidate = search_best_exchange_all_offer_goods_internal(
        goods,
        need_good,
        initial_transparency,
        forbidden_offer_by_need,
        available_offer_goods,
        friend_ids_row,
        reciprocal_slots_view,
        my_stock,
        my_sales_price,
        my_purchase_price,
        my_role,
        my_transparency,
        my_needs_level,
        elastic_need,
        stock,
        role,
        stock_limit,
        purchase_price,
        sales_price,
        needs_level,
        transparency,
        allow_inner_parallel,
    )?;

    let order = basket_candidate_order(
        need_good,
        search_candidate.friend_slot as usize,
        search_candidate.offer_good as usize,
        goods,
        acquaintances,
    );
    Some(BasketCandidate {
        score: search_candidate.score,
        need_good: need_good as i32,
        friend_slot: search_candidate.friend_slot,
        friend_id: search_candidate.friend_id,
        offer_good: search_candidate.offer_good,
        order,
    })
}

#[allow(clippy::too_many_arguments)]
fn plan_need_basket_candidates_from_cached_links(
    agent_id: usize,
    need_good: usize,
    deal_type: i32,
    goods: usize,
    acquaintances: usize,
    initial_transparency: f32,
    history: i32,
    local_liquidity_stock_bias: f64,
    local_liquidity_min_sales: f64,
    aspirational_stock_target: f64,
    forbidden_offer_by_need: &[bool],
    available_offer_goods: &[usize],
    elastic_need: ArrayView1<'_, f32>,
    stock: ArrayView2<'_, f32>,
    role: ArrayView2<'_, i32>,
    stock_limit: ArrayView2<'_, f32>,
    purchase_price: ArrayView2<'_, f32>,
    sales_price: ArrayView2<'_, f32>,
    needs_level: ArrayView1<'_, f32>,
    transparency: ArrayView3<'_, f32>,
    friend_ids_row: ArrayView1<'_, i32>,
    reciprocal_slots_view: ArrayView1<'_, i32>,
    need: ArrayView2<'_, f32>,
    recent_sales: ArrayView2<'_, f32>,
    recent_purchases: ArrayView2<'_, f32>,
    recent_inventory_inflow: ArrayView2<'_, f32>,
    recent_production: ArrayView2<'_, f32>,
    friend_id: ArrayView2<'_, i32>,
    friend_sold: ArrayView3<'_, f32>,
    my_stock: ArrayView1<'_, f32>,
    my_sales_price: ArrayView1<'_, f32>,
    my_purchase_price: ArrayView1<'_, f32>,
    my_role: ArrayView1<'_, i32>,
    my_transparency: ArrayView2<'_, f32>,
    my_needs_level: f32,
    candidate_depth: usize,
    allow_inner_parallel: bool,
) -> Vec<BasketCandidate> {
    if basket_stage_max_need(
        agent_id,
        need_good,
        deal_type,
        history,
        local_liquidity_stock_bias,
        local_liquidity_min_sales,
        aspirational_stock_target,
        elastic_need,
        stock,
        stock_limit,
        needs_level,
        need,
        recent_sales,
        recent_purchases,
        recent_inventory_inflow,
        recent_production,
        friend_id,
        friend_sold,
        transparency,
    ) <= 0.0
    {
        return Vec::new();
    }

    let depth_limit = candidate_depth.max(1).min(goods.saturating_sub(1).max(1));
    let mut local_forbidden = forbidden_offer_by_need.to_vec();
    let mut candidates = Vec::with_capacity(depth_limit);
    for _ in 0..depth_limit {
        let Some(search_candidate) = search_best_exchange_all_offer_goods_internal(
            goods,
            need_good,
            initial_transparency,
            &local_forbidden,
            available_offer_goods,
            friend_ids_row,
            reciprocal_slots_view,
            my_stock,
            my_sales_price,
            my_purchase_price,
            my_role,
            my_transparency,
            my_needs_level,
            elastic_need,
            stock,
            role,
            stock_limit,
            purchase_price,
            sales_price,
            needs_level,
            transparency,
            allow_inner_parallel,
        ) else {
            break;
        };
        if search_candidate.offer_good < 0 {
            break;
        }
        let offer_good = search_candidate.offer_good as usize;
        if offer_good >= goods {
            break;
        }
        let forbidden_index = (need_good * goods) + offer_good;
        if local_forbidden
            .get(forbidden_index)
            .copied()
            .unwrap_or(true)
        {
            break;
        }
        let order = basket_candidate_order(
            need_good,
            search_candidate.friend_slot as usize,
            offer_good,
            goods,
            acquaintances,
        );
        candidates.push(BasketCandidate {
            score: search_candidate.score,
            need_good: need_good as i32,
            friend_slot: search_candidate.friend_slot,
            friend_id: search_candidate.friend_id,
            offer_good: search_candidate.offer_good,
            order,
        });
        local_forbidden[forbidden_index] = true;
    }
    candidates
}

#[allow(clippy::too_many_arguments)]
fn build_static_basket_candidate_lists(
    agent_id: usize,
    goods: usize,
    acquaintances: usize,
    initial_transparency: f32,
    role: ArrayView2<'_, i32>,
    purchase_price: ArrayView2<'_, f32>,
    sales_price: ArrayView2<'_, f32>,
    transparency: ArrayView3<'_, f32>,
    friend_ids_row: ArrayView1<'_, i32>,
    reciprocal_slots_view: ArrayView1<'_, i32>,
) -> Vec<Vec<StaticBasketCandidate>> {
    let my_sales_price = sales_price.row(agent_id);
    let my_purchase_price = purchase_price.row(agent_id);
    let my_role = role.row(agent_id);
    let my_transparency = transparency.index_axis(Axis(0), agent_id);

    (0..goods)
        .into_par_iter()
        .map(|need_good| {
            let my_need_purchase_price = my_purchase_price[need_good];
            let my_need_is_producer = my_role[need_good] == ROLE_PRODUCER;
            let mut candidates = Vec::new();
            for friend_slot in 0..acquaintances {
                let friend_id = friend_ids_row[friend_slot];
                if friend_id < 0 {
                    continue;
                }
                let fid = friend_id as usize;
                let reciprocal_slot = reciprocal_slots_view[friend_slot];
                for offer_good in 0..goods {
                    if offer_good == need_good {
                        continue;
                    }
                    let receiving_transparency = if reciprocal_slot >= 0 {
                        transparency[[fid, reciprocal_slot as usize, offer_good]]
                    } else {
                        initial_transparency
                    };
                    let need_transparency = my_transparency[[friend_slot, need_good]];
                    let friend_need_role = role[[fid, need_good]] as f32;
                    let friend_need_sales_price = sales_price[[fid, need_good]];

                    let mut score = (purchase_price[[fid, offer_good]]
                        / friend_need_sales_price.max(EPSILON))
                        * need_transparency;
                    score -= my_sales_price[offer_good]
                        / (my_need_purchase_price * receiving_transparency).max(EPSILON);
                    score *= my_role[offer_good] as f32;
                    score *= friend_need_role;
                    if my_need_is_producer {
                        score /= 2.0;
                    }
                    if score <= 0.0 {
                        continue;
                    }
                    let order = basket_candidate_order(
                        need_good,
                        friend_slot,
                        offer_good,
                        goods,
                        acquaintances,
                    );
                    candidates.push(StaticBasketCandidate {
                        score,
                        friend_slot: friend_slot as i32,
                        offer_good: offer_good as i32,
                        order,
                    });
                }
            }
            candidates.sort_by(|left, right| {
                right
                    .score
                    .partial_cmp(&left.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| left.order.cmp(&right.order))
            });
            candidates
        })
        .collect()
}

fn build_static_basket_candidate_indexes(
    static_candidate_lists: &[Vec<StaticBasketCandidate>],
    goods: usize,
    acquaintances: usize,
) -> (Vec<u128>, Vec<u128>) {
    if goods > 128 {
        let all_needs = u128::MAX;
        return (
            vec![all_needs; goods],
            vec![all_needs; goods.saturating_mul(acquaintances)],
        );
    }

    let mut offer_to_needs = vec![0_u128; goods];
    let mut offer_friend_to_needs = vec![0_u128; goods * acquaintances];

    for (need_good, candidates) in static_candidate_lists.iter().enumerate() {
        let need_mask = 1_u128 << need_good;
        for candidate in candidates {
            if candidate.offer_good < 0 || candidate.friend_slot < 0 {
                continue;
            }
            let offer_good = candidate.offer_good as usize;
            let friend_slot = candidate.friend_slot as usize;
            if offer_good >= goods || friend_slot >= acquaintances {
                continue;
            }

            offer_to_needs[offer_good] |= need_mask;
            offer_friend_to_needs[(offer_good * acquaintances) + friend_slot] |= need_mask;
        }
    }

    (offer_to_needs, offer_friend_to_needs)
}

fn mark_dirty_need_mask(
    mask: u128,
    goods: usize,
    dirty_needs: &mut [bool],
    static_candidate_cursors: &mut [usize],
) {
    if mask == 0 {
        return;
    }
    let limit = goods
        .min(dirty_needs.len())
        .min(static_candidate_cursors.len());
    if goods > 128 && mask == u128::MAX {
        for need_good in 0..limit {
            dirty_needs[need_good] = true;
            static_candidate_cursors[need_good] = 0;
        }
        return;
    }
    for need_good in 0..limit.min(128) {
        if ((mask >> need_good) & 1) != 0 {
            dirty_needs[need_good] = true;
            static_candidate_cursors[need_good] = 0;
        }
    }
}

fn clear_cached_need_mask(
    mask: u128,
    goods: usize,
    cached_candidates: &mut [Option<BasketCandidate>],
) {
    if mask == 0 {
        return;
    }
    let limit = goods.min(cached_candidates.len());
    if goods > 128 && mask == u128::MAX {
        for candidate in cached_candidates.iter_mut().take(limit) {
            *candidate = None;
        }
        return;
    }
    for need_good in 0..limit.min(128) {
        if ((mask >> need_good) & 1) != 0 {
            cached_candidates[need_good] = None;
        }
    }
}

fn forbid_offer_good_for_all_needs(
    forbidden_offer_by_need: &mut [bool],
    goods: usize,
    offer_good: usize,
) -> bool {
    if offer_good >= goods || forbidden_offer_by_need.len() < goods * goods {
        return false;
    }

    let mut changed = false;
    for need_good in 0..goods {
        if need_good == offer_good {
            continue;
        }
        let index = (need_good * goods) + offer_good;
        if !forbidden_offer_by_need[index] {
            forbidden_offer_by_need[index] = true;
            changed = true;
        }
    }
    changed
}

fn offer_good_exhausted_for_agent(
    agent_id: usize,
    offer_good: usize,
    elastic_need: ArrayView1<'_, f32>,
    stock: ArrayView2<'_, f32>,
    needs_level: ArrayView1<'_, f32>,
) -> bool {
    if agent_id >= stock.nrows() || offer_good >= stock.ncols() || offer_good >= elastic_need.len()
    {
        return true;
    }
    stock[[agent_id, offer_good]] <= (elastic_need[offer_good] * needs_level[agent_id] + 1.0)
}

#[allow(clippy::too_many_arguments)]
fn build_session_availability_cache(
    agent_id: usize,
    goods: usize,
    acquaintances: usize,
    agent_friend_ids: &[i32],
    elastic_need: ArrayView1<'_, f32>,
    stock: ArrayView2<'_, f32>,
    role: ArrayView2<'_, i32>,
    stock_limit: ArrayView2<'_, f32>,
    needs_level: ArrayView1<'_, f32>,
) -> (Vec<bool>, Vec<bool>, Vec<bool>) {
    let mut own_offer_available = vec![false; goods];
    let mut friend_supply_available = vec![false; acquaintances.saturating_mul(goods)];
    let mut friend_accept_available = vec![false; acquaintances.saturating_mul(goods)];
    if agent_id >= stock.nrows() {
        return (
            own_offer_available,
            friend_supply_available,
            friend_accept_available,
        );
    }

    let my_needs_level = needs_level[agent_id];
    for good_id in 0..goods {
        own_offer_available[good_id] =
            stock[[agent_id, good_id]] > ((elastic_need[good_id] * my_needs_level) + 1.0);
    }

    for friend_slot in 0..acquaintances {
        let Some(friend_id) = agent_friend_ids.get(friend_slot).copied() else {
            continue;
        };
        if friend_id < 0 {
            continue;
        }
        let friend_idx = friend_id as usize;
        if friend_idx >= stock.nrows() {
            continue;
        }
        let friend_needs_level = needs_level[friend_idx];
        let base_index = friend_slot * goods;
        for good_id in 0..goods {
            friend_supply_available[base_index + good_id] =
                stock[[friend_idx, good_id]] > ((elastic_need[good_id] * friend_needs_level) + 1.0);
            let gift_max_level = if role[[friend_idx, good_id]] == ROLE_RETAILER {
                stock_limit[[friend_idx, good_id]] - 1.0
            } else {
                (elastic_need[good_id] * friend_needs_level) - 1.0
            };
            friend_accept_available[base_index + good_id] =
                stock[[friend_idx, good_id]] < gift_max_level;
        }
    }

    (
        own_offer_available,
        friend_supply_available,
        friend_accept_available,
    )
}

#[allow(clippy::too_many_arguments)]
fn refresh_session_availability_good(
    own_offer_available: &mut [bool],
    friend_supply_available: &mut [bool],
    friend_accept_available: &mut [bool],
    agent_id: usize,
    friend_slot: usize,
    friend_idx: usize,
    good_id: usize,
    goods: usize,
    elastic_need: ArrayView1<'_, f32>,
    stock: ArrayView2<'_, f32>,
    role: ArrayView2<'_, i32>,
    stock_limit: ArrayView2<'_, f32>,
    needs_level: ArrayView1<'_, f32>,
) {
    if good_id >= goods || agent_id >= stock.nrows() || friend_idx >= stock.nrows() {
        return;
    }
    own_offer_available[good_id] =
        stock[[agent_id, good_id]] > ((elastic_need[good_id] * needs_level[agent_id]) + 1.0);

    if friend_slot < friend_supply_available.len().saturating_div(goods.max(1)) {
        let friend_needs_level = needs_level[friend_idx];
        let index = (friend_slot * goods) + good_id;
        if index < friend_supply_available.len() && index < friend_accept_available.len() {
            friend_supply_available[index] =
                stock[[friend_idx, good_id]] > ((elastic_need[good_id] * friend_needs_level) + 1.0);
            let gift_max_level = if role[[friend_idx, good_id]] == ROLE_RETAILER {
                stock_limit[[friend_idx, good_id]] - 1.0
            } else {
                (elastic_need[good_id] * friend_needs_level) - 1.0
            };
            friend_accept_available[index] = stock[[friend_idx, good_id]] < gift_max_level;
        }
    }
}

fn collect_available_offer_goods(
    agent_id: usize,
    goods: usize,
    elastic_need: ArrayView1<'_, f32>,
    stock: ArrayView2<'_, f32>,
    needs_level: ArrayView1<'_, f32>,
    disable_offer_prefilter: bool,
) -> Vec<usize> {
    if agent_id >= stock.nrows() {
        return Vec::new();
    }
    if disable_offer_prefilter {
        return (0..goods)
            .filter(|offer_good| *offer_good < stock.ncols() && *offer_good < elastic_need.len())
            .collect();
    }
    let my_needs_level = needs_level[agent_id];
    (0..goods)
        .filter(|offer_good| {
            *offer_good < stock.ncols()
                && *offer_good < elastic_need.len()
                && stock[[agent_id, *offer_good]]
                    > ((elastic_need[*offer_good] * my_needs_level) + 1.0)
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn first_valid_static_basket_candidate(
    need_good: usize,
    goods: usize,
    start_index: usize,
    forbidden_offer_row: &[bool],
    static_candidates: &[StaticBasketCandidate],
    agent_friend_ids: &[i32],
    own_offer_available: &[bool],
    friend_supply_available: &[bool],
    friend_accept_available: &[bool],
) -> Option<(BasketCandidate, usize)> {
    if forbidden_offer_row.len() < goods
        || own_offer_available.len() < goods
        || friend_supply_available.len() < agent_friend_ids.len().saturating_mul(goods)
        || friend_accept_available.len() < agent_friend_ids.len().saturating_mul(goods)
    {
        return None;
    }
    for (candidate_index, candidate) in static_candidates.iter().enumerate().skip(start_index) {
        let offer_good = candidate.offer_good as usize;
        let friend_slot = candidate.friend_slot as usize;
        if forbidden_offer_row[offer_good] {
            continue;
        }
        if !own_offer_available[offer_good] {
            continue;
        }
        let friend_base = friend_slot * goods;
        if !friend_supply_available[friend_base + need_good]
            || !friend_accept_available[friend_base + offer_good]
        {
            continue;
        }
        return Some((
            candidate.with_need_good(need_good, agent_friend_ids[friend_slot]),
            candidate_index,
        ));
    }
    None
}

#[allow(clippy::too_many_arguments)]
fn plan_agent_basket_candidates_from_cached_links(
    agent_id: usize,
    deal_type: i32,
    goods: usize,
    acquaintances: usize,
    initial_transparency: f32,
    history: i32,
    local_liquidity_stock_bias: f64,
    local_liquidity_min_sales: f64,
    aspirational_stock_target: f64,
    forbidden_offer_by_need: &[bool],
    elastic_need: ArrayView1<'_, f32>,
    stock: ArrayView2<'_, f32>,
    role: ArrayView2<'_, i32>,
    stock_limit: ArrayView2<'_, f32>,
    purchase_price: ArrayView2<'_, f32>,
    sales_price: ArrayView2<'_, f32>,
    needs_level: ArrayView1<'_, f32>,
    transparency: ArrayView3<'_, f32>,
    friend_ids_row: ArrayView1<'_, i32>,
    reciprocal_slots_view: ArrayView1<'_, i32>,
    need: ArrayView2<'_, f32>,
    recent_sales: ArrayView2<'_, f32>,
    recent_purchases: ArrayView2<'_, f32>,
    recent_inventory_inflow: ArrayView2<'_, f32>,
    recent_production: ArrayView2<'_, f32>,
    friend_id: ArrayView2<'_, i32>,
    friend_sold: ArrayView3<'_, f32>,
    candidate_depth: usize,
    disable_offer_prefilter: bool,
) -> Vec<BasketCandidate> {
    if agent_id >= stock.nrows()
        || goods > stock.ncols()
        || forbidden_offer_by_need.len() < goods * goods
        || friend_ids_row.len() != acquaintances
        || reciprocal_slots_view.len() != acquaintances
    {
        return Vec::new();
    }
    if !friend_ids_row.iter().any(|friend_id| *friend_id >= 0) {
        return Vec::new();
    }

    let my_stock = stock.row(agent_id);
    let my_sales_price = sales_price.row(agent_id);
    let my_purchase_price = purchase_price.row(agent_id);
    let my_role = role.row(agent_id);
    let my_transparency = transparency.index_axis(Axis(0), agent_id);
    let my_needs_level = needs_level[agent_id];
    let available_offer_goods = collect_available_offer_goods(
        agent_id,
        goods,
        elastic_need,
        stock,
        needs_level,
        disable_offer_prefilter,
    );
    if available_offer_goods.is_empty() {
        return Vec::new();
    }

    let basket_cell_count = goods.saturating_mul(goods).saturating_mul(acquaintances);
    let candidate_depth = candidate_depth.max(1);
    let mut candidates: Vec<BasketCandidate> = if candidate_depth > 1 {
        if basket_cell_count >= PARALLEL_BASKET_MIN_CELLS {
            (0..goods)
                .into_par_iter()
                .flat_map_iter(|need_good| {
                    plan_need_basket_candidates_from_cached_links(
                        agent_id,
                        need_good,
                        deal_type,
                        goods,
                        acquaintances,
                        initial_transparency,
                        history,
                        local_liquidity_stock_bias,
                        local_liquidity_min_sales,
                        aspirational_stock_target,
                        forbidden_offer_by_need,
                        &available_offer_goods,
                        elastic_need,
                        stock,
                        role,
                        stock_limit,
                        purchase_price,
                        sales_price,
                        needs_level,
                        transparency,
                        friend_ids_row,
                        reciprocal_slots_view,
                        need,
                        recent_sales,
                        recent_purchases,
                        recent_inventory_inflow,
                        recent_production,
                        friend_id,
                        friend_sold,
                        my_stock,
                        my_sales_price,
                        my_purchase_price,
                        my_role,
                        my_transparency,
                        my_needs_level,
                        candidate_depth,
                        false,
                    )
                })
                .collect()
        } else {
            (0..goods)
                .flat_map(|need_good| {
                    plan_need_basket_candidates_from_cached_links(
                        agent_id,
                        need_good,
                        deal_type,
                        goods,
                        acquaintances,
                        initial_transparency,
                        history,
                        local_liquidity_stock_bias,
                        local_liquidity_min_sales,
                        aspirational_stock_target,
                        forbidden_offer_by_need,
                        &available_offer_goods,
                        elastic_need,
                        stock,
                        role,
                        stock_limit,
                        purchase_price,
                        sales_price,
                        needs_level,
                        transparency,
                        friend_ids_row,
                        reciprocal_slots_view,
                        need,
                        recent_sales,
                        recent_purchases,
                        recent_inventory_inflow,
                        recent_production,
                        friend_id,
                        friend_sold,
                        my_stock,
                        my_sales_price,
                        my_purchase_price,
                        my_role,
                        my_transparency,
                        my_needs_level,
                        candidate_depth,
                        true,
                    )
                })
                .collect()
        }
    } else if basket_cell_count >= PARALLEL_BASKET_MIN_CELLS {
        (0..goods)
            .into_par_iter()
            .filter_map(|need_good| {
                plan_one_basket_candidate_from_cached_links(
                    agent_id,
                    need_good,
                    deal_type,
                    goods,
                    acquaintances,
                    initial_transparency,
                    history,
                    local_liquidity_stock_bias,
                    local_liquidity_min_sales,
                    aspirational_stock_target,
                    forbidden_offer_by_need,
                    &available_offer_goods,
                    elastic_need,
                    stock,
                    role,
                    stock_limit,
                    purchase_price,
                    sales_price,
                    needs_level,
                    transparency,
                    friend_ids_row,
                    reciprocal_slots_view,
                    need,
                    recent_sales,
                    recent_purchases,
                    recent_inventory_inflow,
                    recent_production,
                    friend_id,
                    friend_sold,
                    my_stock,
                    my_sales_price,
                    my_purchase_price,
                    my_role,
                    my_transparency,
                    my_needs_level,
                    false,
                )
            })
            .collect()
    } else {
        (0..goods)
            .filter_map(|need_good| {
                plan_one_basket_candidate_from_cached_links(
                    agent_id,
                    need_good,
                    deal_type,
                    goods,
                    acquaintances,
                    initial_transparency,
                    history,
                    local_liquidity_stock_bias,
                    local_liquidity_min_sales,
                    aspirational_stock_target,
                    forbidden_offer_by_need,
                    &available_offer_goods,
                    elastic_need,
                    stock,
                    role,
                    stock_limit,
                    purchase_price,
                    sales_price,
                    needs_level,
                    transparency,
                    friend_ids_row,
                    reciprocal_slots_view,
                    need,
                    recent_sales,
                    recent_purchases,
                    recent_inventory_inflow,
                    recent_production,
                    friend_id,
                    friend_sold,
                    my_stock,
                    my_sales_price,
                    my_purchase_price,
                    my_role,
                    my_transparency,
                    my_needs_level,
                    true,
                )
            })
            .collect()
    };

    candidates.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.order.cmp(&right.order))
    });
    candidates
}

fn find_friend_slot_scan_internal(
    friend_ids: ArrayView2<'_, i32>,
    agent_id: usize,
    friend_id: i32,
) -> i32 {
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
fn run_exchange_stage(
    py: Python<'_>,
    runner: PyObject,
    agent_id: usize,
    deal_type: i32,
) -> PyResult<(i32, i32, f64, f64)> {
    let runner = runner.bind(py);
    let state = runner.getattr("state")?;
    let market = runner.getattr("market")?;
    let config = runner.getattr("config")?;

    let goods: usize = config.getattr("goods")?.extract()?;
    let acquaintances: usize = config.getattr("acquaintances")?.extract()?;
    let initial_transparency_for_execution: f64 =
        config.getattr("initial_transparency")?.extract()?;
    let initial_transparency = initial_transparency_for_execution as f32;
    let history: i32 = config.getattr("history")?.extract()?;
    let local_liquidity_stock_bias: f64 = config
        .getattr("experimental_local_liquidity_stock_bias")?
        .extract()?;
    let local_liquidity_min_sales: f64 = config
        .getattr("experimental_local_liquidity_min_sales")?
        .extract()?;
    let aspirational_stock_target: f64 = config
        .getattr("experimental_aspirational_stock_target")?
        .extract()?;
    let min_trade_quantity: f64 = config.getattr("min_trade_quantity")?.extract()?;
    let trade_rounding_buffer: f64 = config.getattr("trade_rounding_buffer")?.extract()?;
    let max_attempts = goods * acquaintances;
    let initial_transactions = 2.0_f32;

    if deal_type != CONSUMPTION_DEAL && deal_type != SURPLUS_DEAL {
        return Err(PyValueError::new_err(
            "deal_type must be 1 (surplus) or 2 (consumption)",
        ));
    }

    let mut totals = ExchangeStageTotals::default();
    let mut changed_agents: HashSet<usize> = HashSet::new();

    {
        let market_elastic_need: PyReadonlyArray1<'_, f32> =
            market.getattr("elastic_need")?.extract()?;
        let mut market_periodic_tce_cost: PyReadwriteArray1<'_, f32> =
            market.getattr("periodic_tce_cost")?.extract()?;
        let mut stock: PyReadwriteArray2<'_, f32> = state.getattr("stock")?.extract()?;
        let role: PyReadonlyArray2<'_, i32> = state.getattr("role")?.extract()?;
        let stock_limit: PyReadonlyArray2<'_, f32> = state.getattr("stock_limit")?.extract()?;
        let purchase_price: PyReadonlyArray2<'_, f32> =
            state.getattr("purchase_price")?.extract()?;
        let sales_price: PyReadonlyArray2<'_, f32> = state.getattr("sales_price")?.extract()?;
        let needs_level: PyReadonlyArray1<'_, f32> = state.getattr("needs_level")?.extract()?;
        let mut transparency: PyReadwriteArray3<'_, f32> =
            state.getattr("transparency")?.extract()?;
        let mut friend_id: PyReadwriteArray2<'_, i32> = state.getattr("friend_id")?.extract()?;
        let mut need: PyReadwriteArray2<'_, f32> = state.getattr("need")?.extract()?;
        let recent_production: PyReadonlyArray2<'_, f32> =
            state.getattr("recent_production")?.extract()?;
        let mut recent_sales: PyReadwriteArray2<'_, f32> =
            state.getattr("recent_sales")?.extract()?;
        let mut recent_purchases: PyReadwriteArray2<'_, f32> =
            state.getattr("recent_purchases")?.extract()?;
        let mut sold_this_period: PyReadwriteArray2<'_, f32> =
            state.getattr("sold_this_period")?.extract()?;
        let mut purchased_this_period: PyReadwriteArray2<'_, f32> =
            state.getattr("purchased_this_period")?.extract()?;
        let mut recent_inventory_inflow: PyReadwriteArray2<'_, f32> =
            state.getattr("recent_inventory_inflow")?.extract()?;
        let mut recent_purchase_value: PyReadwriteArray2<'_, f32> =
            state.getattr("recent_purchase_value")?.extract()?;
        let mut recent_sales_value: PyReadwriteArray2<'_, f32> =
            state.getattr("recent_sales_value")?.extract()?;
        let mut recent_inventory_inflow_value: PyReadwriteArray2<'_, f32> =
            state.getattr("recent_inventory_inflow_value")?.extract()?;
        let mut purchase_times: PyReadwriteArray2<'_, i32> =
            state.getattr("purchase_times")?.extract()?;
        let mut sales_times: PyReadwriteArray2<'_, i32> =
            state.getattr("sales_times")?.extract()?;
        let mut sum_period_purchase_value: PyReadwriteArray2<'_, f32> =
            state.getattr("sum_period_purchase_value")?.extract()?;
        let mut sum_period_sales_value: PyReadwriteArray2<'_, f32> =
            state.getattr("sum_period_sales_value")?.extract()?;
        let mut friend_activity: PyReadwriteArray2<'_, f32> =
            state.getattr("friend_activity")?.extract()?;
        let mut friend_purchased: PyReadwriteArray3<'_, f32> =
            state.getattr("friend_purchased")?.extract()?;
        let mut friend_sold: PyReadwriteArray3<'_, f32> =
            state.getattr("friend_sold")?.extract()?;
        let trade = state.getattr("trade")?;
        let mut proposal_friend_slot: PyReadwriteArray1<'_, i32> =
            trade.getattr("proposal_friend_slot")?.extract()?;
        let mut proposal_target_agent: PyReadwriteArray1<'_, i32> =
            trade.getattr("proposal_target_agent")?.extract()?;
        let mut proposal_need_good: PyReadwriteArray1<'_, i32> =
            trade.getattr("proposal_need_good")?.extract()?;
        let mut proposal_offer_good: PyReadwriteArray1<'_, i32> =
            trade.getattr("proposal_offer_good")?.extract()?;
        let mut proposal_quantity: PyReadwriteArray1<'_, f32> =
            trade.getattr("proposal_quantity")?.extract()?;
        let mut proposal_score: PyReadwriteArray1<'_, f32> =
            trade.getattr("proposal_score")?.extract()?;
        let mut accepted_mask: PyReadwriteArray1<'_, bool> =
            trade.getattr("accepted_mask")?.extract()?;
        let mut accepted_quantity: PyReadwriteArray1<'_, f32> =
            trade.getattr("accepted_quantity")?.extract()?;

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
        let mut recent_purchase_value = recent_purchase_value.as_array_mut();
        let mut recent_sales_value = recent_sales_value.as_array_mut();
        let mut recent_inventory_inflow_value = recent_inventory_inflow_value.as_array_mut();
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

        let agent_friend_ids: Vec<i32> = (0..acquaintances)
            .map(|friend_slot| friend_id[[agent_id, friend_slot]])
            .collect();
        let mut reciprocal_slots = vec![-1_i32; acquaintances];
        let mut has_known_partner = false;
        for friend_slot in 0..acquaintances {
            let fid = agent_friend_ids[friend_slot];
            if fid >= 0 {
                has_known_partner = true;
                reciprocal_slots[friend_slot] =
                    find_friend_slot_scan_internal(friend_id.view(), fid as usize, agent_id as i32);
            }
        }
        if !has_known_partner {
            return Ok((0, 0, 0.0, 0.0));
        }

        let mut forbidden_gifts = vec![false; goods];
        let mut base_offer_goods: Vec<i32> = Vec::with_capacity(goods);
        let mut candidate_offer_goods: Vec<i32> = Vec::with_capacity(goods);

        for need_good in 0..goods {
            let target_stock_limit = if deal_type == SURPLUS_DEAL {
                let base_target_stock_limit = surplus_target_stock_limit(
                    agent_id,
                    need_good,
                    history,
                    local_liquidity_stock_bias,
                    local_liquidity_min_sales,
                    elastic_need,
                    stock_limit,
                    needs_level,
                    recent_sales.view(),
                    recent_purchases.view(),
                    recent_inventory_inflow.view(),
                    recent_production,
                    friend_id.view(),
                    friend_sold.view(),
                    transparency.view(),
                );
                let aspirational_target = aspirational_stock_target_for_good(
                    agent_id,
                    need_good,
                    aspirational_stock_target,
                    elastic_need,
                    stock_limit,
                    needs_level,
                );
                base_target_stock_limit.max(aspirational_target)
            } else {
                stock_limit[[agent_id, need_good]] as f64
            };
            if deal_type == CONSUMPTION_DEAL {
                if stock[[agent_id, need_good]] >= (elastic_need[need_good] * needs_level[agent_id])
                {
                    continue;
                }
            } else {
                let legacy_surplus_signal = recent_sales[[agent_id, need_good]]
                    > (recent_production[[agent_id, need_good]] - elastic_need[need_good]);
                let local_liquidity_signal = target_stock_limit
                    > (stock_limit[[agent_id, need_good]] as f64) + (EPSILON as f64);
                let aspirational_target = aspirational_stock_target_for_good(
                    agent_id,
                    need_good,
                    aspirational_stock_target,
                    elastic_need,
                    stock_limit,
                    needs_level,
                );
                let aspirational_stock_signal = aspirational_target > EPSILON as f64
                    && (stock[[agent_id, need_good]] as f64)
                        < aspirational_target - (EPSILON as f64);
                if !(legacy_surplus_signal || local_liquidity_signal || aspirational_stock_signal)
                    || (stock[[agent_id, need_good]] as f64)
                        >= target_stock_limit - (elastic_need[need_good] as f64)
                {
                    continue;
                }
            }

            forbidden_gifts.fill(false);
            base_offer_goods.clear();
            for offer_good in 0..goods {
                if stock[[agent_id, offer_good]]
                    <= ((elastic_need[offer_good] * needs_level[agent_id]) + 1.0)
                {
                    continue;
                }
                base_offer_goods.push(offer_good as i32);
            }
            let mut attempts = 0usize;
            while attempts < max_attempts {
                let max_need = if deal_type == CONSUMPTION_DEAL {
                    let remaining_need = need[[agent_id, need_good]] as f64;
                    if remaining_need < 1.0 {
                        break;
                    }
                    remaining_need
                } else {
                    if (stock[[agent_id, need_good]] as f64) >= target_stock_limit {
                        break;
                    }
                    (target_stock_limit - (stock[[agent_id, need_good]] as f64)).max(0.0)
                };

                candidate_offer_goods.clear();
                for offer_good_raw in &base_offer_goods {
                    let offer_good = *offer_good_raw as usize;
                    if offer_good == need_good || forbidden_gifts[offer_good] {
                        continue;
                    }
                    candidate_offer_goods.push(*offer_good_raw);
                }
                if candidate_offer_goods.is_empty() {
                    break;
                }

                let plan = plan_exchange_stage_candidate(
                    agent_id,
                    need_good,
                    goods,
                    initial_transparency,
                    initial_transparency_for_execution,
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
                    ArrayView1::from(&agent_friend_ids[..]),
                    ArrayView1::from(&reciprocal_slots[..]),
                    ArrayView1::from(&candidate_offer_goods[..]),
                );
                let Some(plan) = plan else {
                    break;
                };

                totals.proposed_count += 1;
                if plan.reason_code != PLAN_OK {
                    if plan.candidate.offer_good >= 0 {
                        let offer_good = plan.candidate.offer_good as usize;
                        if offer_good < goods {
                            forbidden_gifts[offer_good] = true;
                        }
                    }
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
                    reciprocal_slots[friend_slot] = ensured_slot as i32;
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

                let my_need_purchase_price = purchase_price[[agent_id, need_good]];
                let my_offer_sales_price = sales_price[[agent_id, offer_good]];
                let friend_need_sales_price = sales_price[[friend_idx, need_good]];
                let friend_offer_purchase_price = purchase_price[[friend_idx, offer_good]];
                let exchange_value_to_me = exchange_value_to_me_like_numpy_float32(
                    my_need_purchase_price,
                    my_offer_sales_price,
                    plan.switch_average,
                    plan.receiving_transparency,
                );
                let exchange_value_to_friend = exchange_value_to_friend_like_numpy_float32(
                    friend_offer_purchase_price,
                    friend_need_sales_price,
                    plan.switch_average,
                    plan.need_transparency,
                );
                let my_value_correction = exchange_value_to_me.max(EPSILON as f64).sqrt();
                let friend_value_correction = exchange_value_to_friend.max(EPSILON as f64).sqrt();
                let my_purchase_unit_value =
                    div_like_numpy_float32(my_need_purchase_price, my_value_correction);
                let my_sales_unit_value =
                    mul_like_numpy_float32(my_offer_sales_price, my_value_correction);
                let friend_purchase_unit_value =
                    div_like_numpy_float32(friend_offer_purchase_price, friend_value_correction);
                let friend_sales_unit_value =
                    mul_like_numpy_float32(friend_need_sales_price, friend_value_correction);

                add_assign_like_numpy_float32(
                    &mut recent_purchase_value[[agent_id, need_good]],
                    max_exchange * my_purchase_unit_value,
                );
                add_assign_like_numpy_float32(
                    &mut recent_sales_value[[agent_id, offer_good]],
                    friend_gift_in * my_sales_unit_value,
                );
                add_assign_like_numpy_float32(
                    &mut recent_purchase_value[[friend_idx, offer_good]],
                    friend_gift_in * friend_purchase_unit_value,
                );
                add_assign_like_numpy_float32(
                    &mut recent_sales_value[[friend_idx, need_good]],
                    friend_need_out * friend_sales_unit_value,
                );
                if deal_type == SURPLUS_DEAL {
                    add_assign_like_numpy_float32(
                        &mut recent_inventory_inflow_value[[agent_id, need_good]],
                        max_exchange * my_purchase_unit_value,
                    );
                }
                add_assign_like_numpy_float32(
                    &mut recent_inventory_inflow_value[[friend_idx, offer_good]],
                    friend_gift_in * friend_purchase_unit_value,
                );

                if my_value_correction > 1.0 && friend_value_correction > 1.0 {
                    add_assign_like_numpy_float32(
                        &mut sum_period_purchase_value[[agent_id, need_good]],
                        my_purchase_unit_value,
                    );
                    add_assign_like_numpy_float32(
                        &mut sum_period_sales_value[[agent_id, offer_good]],
                        my_sales_unit_value,
                    );
                    add_assign_like_numpy_float32(
                        &mut sum_period_purchase_value[[friend_idx, offer_good]],
                        friend_purchase_unit_value,
                    );
                    add_assign_like_numpy_float32(
                        &mut sum_period_sales_value[[friend_idx, need_good]],
                        friend_sales_unit_value,
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

                let exhausted_gift =
                    remaining_need_after_trade >= 1.0 && stock[[agent_id, offer_good]] < 1.0;
                if exhausted_gift {
                    forbidden_gifts[offer_good] = true;
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
#[allow(clippy::too_many_arguments)]
fn execute_planned_parallel_phenomenon_batch(
    py: Python<'_>,
    runner: PyObject,
    deal_type: i32,
    agent_ids: PyReadonlyArray1<'_, i32>,
    need_goods: PyReadonlyArray1<'_, i32>,
    max_needs: PyReadonlyArray1<'_, f64>,
    scores: PyReadonlyArray1<'_, f32>,
    friend_slots: PyReadonlyArray1<'_, i32>,
    friend_ids_input: PyReadonlyArray1<'_, i32>,
    offer_goods: PyReadonlyArray1<'_, i32>,
    reciprocal_slots_input: PyReadonlyArray1<'_, i32>,
    max_exchanges: PyReadonlyArray1<'_, f64>,
    switch_averages: PyReadonlyArray1<'_, f64>,
    need_transparencies: PyReadonlyArray1<'_, f64>,
    receiving_transparencies: PyReadonlyArray1<'_, f64>,
) -> PyResult<(i32, f64, f64, f64)> {
    if deal_type != CONSUMPTION_DEAL && deal_type != SURPLUS_DEAL {
        return Err(PyValueError::new_err(
            "deal_type must be 1 (surplus) or 2 (consumption)",
        ));
    }

    let runner = runner.bind(py);
    let state = runner.getattr("state")?;
    let market = runner.getattr("market")?;
    let config = runner.getattr("config")?;

    let goods: usize = config.getattr("goods")?.extract()?;
    let acquaintances: usize = config.getattr("acquaintances")?.extract()?;
    let initial_transparency: f32 = config.getattr("initial_transparency")?.extract()?;
    let min_trade_quantity: f64 = config.getattr("min_trade_quantity")?.extract()?;
    let initial_transactions = 2.0_f32;

    let agent_ids = agent_ids.as_array();
    let need_goods = need_goods.as_array();
    let max_needs = max_needs.as_array();
    let scores = scores.as_array();
    let friend_slots = friend_slots.as_array();
    let friend_ids_input = friend_ids_input.as_array();
    let offer_goods = offer_goods.as_array();
    let reciprocal_slots_input = reciprocal_slots_input.as_array();
    let max_exchanges = max_exchanges.as_array();
    let switch_averages = switch_averages.as_array();
    let need_transparencies = need_transparencies.as_array();
    let receiving_transparencies = receiving_transparencies.as_array();
    let count = agent_ids.len();
    if need_goods.len() != count
        || max_needs.len() != count
        || scores.len() != count
        || friend_slots.len() != count
        || friend_ids_input.len() != count
        || offer_goods.len() != count
        || reciprocal_slots_input.len() != count
        || max_exchanges.len() != count
        || switch_averages.len() != count
        || need_transparencies.len() != count
        || receiving_transparencies.len() != count
    {
        return Err(PyValueError::new_err(
            "all planned exchange arrays must have the same length",
        ));
    }

    let mut market_periodic_tce_cost: PyReadwriteArray1<'_, f32> =
        market.getattr("periodic_tce_cost")?.extract()?;
    let mut stock: PyReadwriteArray2<'_, f32> = state.getattr("stock")?.extract()?;
    let purchase_price: PyReadonlyArray2<'_, f32> = state.getattr("purchase_price")?.extract()?;
    let sales_price: PyReadonlyArray2<'_, f32> = state.getattr("sales_price")?.extract()?;
    let mut transparency: PyReadwriteArray3<'_, f32> = state.getattr("transparency")?.extract()?;
    let mut friend_id: PyReadwriteArray2<'_, i32> = state.getattr("friend_id")?.extract()?;
    let mut need: PyReadwriteArray2<'_, f32> = state.getattr("need")?.extract()?;
    let mut recent_sales: PyReadwriteArray2<'_, f32> = state.getattr("recent_sales")?.extract()?;
    let mut recent_purchases: PyReadwriteArray2<'_, f32> =
        state.getattr("recent_purchases")?.extract()?;
    let mut sold_this_period: PyReadwriteArray2<'_, f32> =
        state.getattr("sold_this_period")?.extract()?;
    let mut purchased_this_period: PyReadwriteArray2<'_, f32> =
        state.getattr("purchased_this_period")?.extract()?;
    let mut recent_inventory_inflow: PyReadwriteArray2<'_, f32> =
        state.getattr("recent_inventory_inflow")?.extract()?;
    let mut recent_purchase_value: PyReadwriteArray2<'_, f32> =
        state.getattr("recent_purchase_value")?.extract()?;
    let mut recent_sales_value: PyReadwriteArray2<'_, f32> =
        state.getattr("recent_sales_value")?.extract()?;
    let mut recent_inventory_inflow_value: PyReadwriteArray2<'_, f32> =
        state.getattr("recent_inventory_inflow_value")?.extract()?;
    let mut purchase_times: PyReadwriteArray2<'_, i32> =
        state.getattr("purchase_times")?.extract()?;
    let mut sales_times: PyReadwriteArray2<'_, i32> = state.getattr("sales_times")?.extract()?;
    let mut sum_period_purchase_value: PyReadwriteArray2<'_, f32> =
        state.getattr("sum_period_purchase_value")?.extract()?;
    let mut sum_period_sales_value: PyReadwriteArray2<'_, f32> =
        state.getattr("sum_period_sales_value")?.extract()?;
    let mut friend_activity: PyReadwriteArray2<'_, f32> =
        state.getattr("friend_activity")?.extract()?;
    let mut friend_purchased: PyReadwriteArray3<'_, f32> =
        state.getattr("friend_purchased")?.extract()?;
    let mut friend_sold: PyReadwriteArray3<'_, f32> = state.getattr("friend_sold")?.extract()?;
    let trade = state.getattr("trade")?;
    let mut proposal_friend_slot: PyReadwriteArray1<'_, i32> =
        trade.getattr("proposal_friend_slot")?.extract()?;
    let mut proposal_target_agent: PyReadwriteArray1<'_, i32> =
        trade.getattr("proposal_target_agent")?.extract()?;
    let mut proposal_need_good: PyReadwriteArray1<'_, i32> =
        trade.getattr("proposal_need_good")?.extract()?;
    let mut proposal_offer_good: PyReadwriteArray1<'_, i32> =
        trade.getattr("proposal_offer_good")?.extract()?;
    let mut proposal_quantity: PyReadwriteArray1<'_, f32> =
        trade.getattr("proposal_quantity")?.extract()?;
    let mut proposal_score: PyReadwriteArray1<'_, f32> =
        trade.getattr("proposal_score")?.extract()?;
    let mut accepted_mask: PyReadwriteArray1<'_, bool> =
        trade.getattr("accepted_mask")?.extract()?;
    let mut accepted_quantity: PyReadwriteArray1<'_, f32> =
        trade.getattr("accepted_quantity")?.extract()?;

    let mut market_periodic_tce_cost = market_periodic_tce_cost.as_array_mut();
    let mut stock = stock.as_array_mut();
    let purchase_price = purchase_price.as_array();
    let sales_price = sales_price.as_array();
    let mut transparency = transparency.as_array_mut();
    let mut friend_id = friend_id.as_array_mut();
    let mut need = need.as_array_mut();
    let mut recent_sales = recent_sales.as_array_mut();
    let mut recent_purchases = recent_purchases.as_array_mut();
    let mut sold_this_period = sold_this_period.as_array_mut();
    let mut purchased_this_period = purchased_this_period.as_array_mut();
    let mut recent_inventory_inflow = recent_inventory_inflow.as_array_mut();
    let mut recent_purchase_value = recent_purchase_value.as_array_mut();
    let mut recent_sales_value = recent_sales_value.as_array_mut();
    let mut recent_inventory_inflow_value = recent_inventory_inflow_value.as_array_mut();
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

    let mut changed_agents: HashSet<usize> = HashSet::new();
    let mut executed_count = 0_i32;
    let mut scheduled_quantity_total = 0.0_f64;
    let mut executed_quantity_total = 0.0_f64;
    let mut inventory_trade_volume = 0.0_f64;

    for index in 0..count {
        if agent_ids[index] < 0
            || need_goods[index] < 0
            || friend_slots[index] < 0
            || friend_ids_input[index] < 0
            || offer_goods[index] < 0
        {
            continue;
        }
        let agent_id = agent_ids[index] as usize;
        let need_good = need_goods[index] as usize;
        let friend_slot = friend_slots[index] as usize;
        let friend_idx = friend_ids_input[index] as usize;
        let offer_good = offer_goods[index] as usize;
        if agent_id >= stock.nrows()
            || friend_idx >= stock.nrows()
            || need_good >= goods
            || offer_good >= goods
            || friend_slot >= acquaintances
            || friend_id[[agent_id, friend_slot]] != friend_ids_input[index]
        {
            continue;
        }

        let max_exchange = max_exchanges[index];
        if max_exchange < min_trade_quantity {
            continue;
        }
        scheduled_quantity_total += max_exchange;
        let max_need = max_needs[index];
        let switch_average = switch_averages[index];
        let need_transparency = need_transparencies[index];
        let receiving_transparency = receiving_transparencies[index];
        let reciprocal_slot = if reciprocal_slots_input[index] >= 0 {
            reciprocal_slots_input[index] as usize
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

        let my_gift_out =
            (max_exchange * switch_average) / receiving_transparency.max(EPSILON as f64);
        let friend_need_out = max_exchange / need_transparency.max(EPSILON as f64);
        let friend_gift_in = max_exchange * switch_average;

        if deal_type == SURPLUS_DEAL {
            stock[[agent_id, need_good]] += max_exchange as f32;
            recent_inventory_inflow[[agent_id, need_good]] += max_exchange as f32;
            inventory_trade_volume += max_exchange;
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
        inventory_trade_volume += friend_gift_in;

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

        let my_need_purchase_price = purchase_price[[agent_id, need_good]];
        let my_offer_sales_price = sales_price[[agent_id, offer_good]];
        let friend_need_sales_price = sales_price[[friend_idx, need_good]];
        let friend_offer_purchase_price = purchase_price[[friend_idx, offer_good]];
        let exchange_value_to_me = exchange_value_to_me_like_numpy_float32(
            my_need_purchase_price,
            my_offer_sales_price,
            switch_average,
            receiving_transparency,
        );
        let exchange_value_to_friend = exchange_value_to_friend_like_numpy_float32(
            friend_offer_purchase_price,
            friend_need_sales_price,
            switch_average,
            need_transparency,
        );
        let my_value_correction = exchange_value_to_me.max(EPSILON as f64).sqrt();
        let friend_value_correction = exchange_value_to_friend.max(EPSILON as f64).sqrt();
        let my_purchase_unit_value =
            div_like_numpy_float32(my_need_purchase_price, my_value_correction);
        let my_sales_unit_value = mul_like_numpy_float32(my_offer_sales_price, my_value_correction);
        let friend_purchase_unit_value =
            div_like_numpy_float32(friend_offer_purchase_price, friend_value_correction);
        let friend_sales_unit_value =
            mul_like_numpy_float32(friend_need_sales_price, friend_value_correction);

        add_assign_like_numpy_float32(
            &mut recent_purchase_value[[agent_id, need_good]],
            max_exchange * my_purchase_unit_value,
        );
        add_assign_like_numpy_float32(
            &mut recent_sales_value[[agent_id, offer_good]],
            friend_gift_in * my_sales_unit_value,
        );
        add_assign_like_numpy_float32(
            &mut recent_purchase_value[[friend_idx, offer_good]],
            friend_gift_in * friend_purchase_unit_value,
        );
        add_assign_like_numpy_float32(
            &mut recent_sales_value[[friend_idx, need_good]],
            friend_need_out * friend_sales_unit_value,
        );
        if deal_type == SURPLUS_DEAL {
            add_assign_like_numpy_float32(
                &mut recent_inventory_inflow_value[[agent_id, need_good]],
                max_exchange * my_purchase_unit_value,
            );
        }
        add_assign_like_numpy_float32(
            &mut recent_inventory_inflow_value[[friend_idx, offer_good]],
            friend_gift_in * friend_purchase_unit_value,
        );

        if my_value_correction > 1.0 && friend_value_correction > 1.0 {
            add_assign_like_numpy_float32(
                &mut sum_period_purchase_value[[agent_id, need_good]],
                my_purchase_unit_value,
            );
            add_assign_like_numpy_float32(
                &mut sum_period_sales_value[[agent_id, offer_good]],
                my_sales_unit_value,
            );
            add_assign_like_numpy_float32(
                &mut sum_period_purchase_value[[friend_idx, offer_good]],
                friend_purchase_unit_value,
            );
            add_assign_like_numpy_float32(
                &mut sum_period_sales_value[[friend_idx, need_good]],
                friend_sales_unit_value,
            );
        }

        proposal_friend_slot[agent_id] = friend_slot as i32;
        proposal_target_agent[agent_id] = friend_idx as i32;
        proposal_need_good[agent_id] = need_good as i32;
        proposal_offer_good[agent_id] = offer_good as i32;
        proposal_quantity[agent_id] = max_exchange as f32;
        proposal_score[agent_id] = scores[index];
        accepted_mask[agent_id] = true;
        accepted_quantity[agent_id] = max_exchange as f32;

        let _remaining_need_after_trade = max_need - max_exchange;
        executed_count += 1;
        executed_quantity_total += max_exchange;
    }

    if !changed_agents.is_empty() {
        let mut changed_list: Vec<usize> = changed_agents.into_iter().collect();
        changed_list.sort_unstable();
        runner.call_method1("_sync_friend_slot_maps", (changed_list,))?;
    }

    Ok((
        executed_count,
        scheduled_quantity_total,
        executed_quantity_total,
        inventory_trade_volume,
    ))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn run_parallel_phenomenon_agent_tail(
    py: Python<'_>,
    runner: PyObject,
    agent_id: usize,
    deal_type: i32,
    attempts_remaining: usize,
) -> PyResult<(i32, f64, f64, f64, i32)> {
    if deal_type != CONSUMPTION_DEAL && deal_type != SURPLUS_DEAL {
        return Err(PyValueError::new_err(
            "deal_type must be 1 (surplus) or 2 (consumption)",
        ));
    }
    if attempts_remaining == 0 {
        return Ok((0, 0.0, 0.0, 0.0, 0));
    }

    let runner = runner.bind(py);
    let state = runner.getattr("state")?;
    let market = runner.getattr("market")?;
    let config = runner.getattr("config")?;

    let goods: usize = config.getattr("goods")?.extract()?;
    let acquaintances: usize = config.getattr("acquaintances")?.extract()?;
    let initial_transparency_for_execution: f64 =
        config.getattr("initial_transparency")?.extract()?;
    let initial_transparency = initial_transparency_for_execution as f32;
    let history: i32 = config.getattr("history")?.extract()?;
    let local_liquidity_stock_bias: f64 = config
        .getattr("experimental_local_liquidity_stock_bias")?
        .extract()?;
    let local_liquidity_min_sales: f64 = config
        .getattr("experimental_local_liquidity_min_sales")?
        .extract()?;
    let aspirational_stock_target: f64 = config
        .getattr("experimental_aspirational_stock_target")?
        .extract()?;
    let min_trade_quantity: f64 = config.getattr("min_trade_quantity")?.extract()?;
    let trade_rounding_buffer: f64 = config.getattr("trade_rounding_buffer")?.extract()?;
    let initial_transactions = 2.0_f32;

    let market_elastic_need: PyReadonlyArray1<'_, f32> =
        market.getattr("elastic_need")?.extract()?;
    let mut market_periodic_tce_cost: PyReadwriteArray1<'_, f32> =
        market.getattr("periodic_tce_cost")?.extract()?;
    let mut stock: PyReadwriteArray2<'_, f32> = state.getattr("stock")?.extract()?;
    let role: PyReadonlyArray2<'_, i32> = state.getattr("role")?.extract()?;
    let stock_limit: PyReadonlyArray2<'_, f32> = state.getattr("stock_limit")?.extract()?;
    let purchase_price: PyReadonlyArray2<'_, f32> = state.getattr("purchase_price")?.extract()?;
    let sales_price: PyReadonlyArray2<'_, f32> = state.getattr("sales_price")?.extract()?;
    let needs_level: PyReadonlyArray1<'_, f32> = state.getattr("needs_level")?.extract()?;
    let mut transparency: PyReadwriteArray3<'_, f32> = state.getattr("transparency")?.extract()?;
    let mut friend_id: PyReadwriteArray2<'_, i32> = state.getattr("friend_id")?.extract()?;
    let mut need: PyReadwriteArray2<'_, f32> = state.getattr("need")?.extract()?;
    let recent_production: PyReadonlyArray2<'_, f32> =
        state.getattr("recent_production")?.extract()?;
    let mut recent_sales: PyReadwriteArray2<'_, f32> = state.getattr("recent_sales")?.extract()?;
    let mut recent_purchases: PyReadwriteArray2<'_, f32> =
        state.getattr("recent_purchases")?.extract()?;
    let mut sold_this_period: PyReadwriteArray2<'_, f32> =
        state.getattr("sold_this_period")?.extract()?;
    let mut purchased_this_period: PyReadwriteArray2<'_, f32> =
        state.getattr("purchased_this_period")?.extract()?;
    let mut recent_inventory_inflow: PyReadwriteArray2<'_, f32> =
        state.getattr("recent_inventory_inflow")?.extract()?;
    let mut recent_purchase_value: PyReadwriteArray2<'_, f32> =
        state.getattr("recent_purchase_value")?.extract()?;
    let mut recent_sales_value: PyReadwriteArray2<'_, f32> =
        state.getattr("recent_sales_value")?.extract()?;
    let mut recent_inventory_inflow_value: PyReadwriteArray2<'_, f32> =
        state.getattr("recent_inventory_inflow_value")?.extract()?;
    let mut purchase_times: PyReadwriteArray2<'_, i32> =
        state.getattr("purchase_times")?.extract()?;
    let mut sales_times: PyReadwriteArray2<'_, i32> = state.getattr("sales_times")?.extract()?;
    let mut sum_period_purchase_value: PyReadwriteArray2<'_, f32> =
        state.getattr("sum_period_purchase_value")?.extract()?;
    let mut sum_period_sales_value: PyReadwriteArray2<'_, f32> =
        state.getattr("sum_period_sales_value")?.extract()?;
    let mut friend_activity: PyReadwriteArray2<'_, f32> =
        state.getattr("friend_activity")?.extract()?;
    let mut friend_purchased: PyReadwriteArray3<'_, f32> =
        state.getattr("friend_purchased")?.extract()?;
    let mut friend_sold: PyReadwriteArray3<'_, f32> = state.getattr("friend_sold")?.extract()?;
    let trade = state.getattr("trade")?;
    let mut proposal_friend_slot: PyReadwriteArray1<'_, i32> =
        trade.getattr("proposal_friend_slot")?.extract()?;
    let mut proposal_target_agent: PyReadwriteArray1<'_, i32> =
        trade.getattr("proposal_target_agent")?.extract()?;
    let mut proposal_need_good: PyReadwriteArray1<'_, i32> =
        trade.getattr("proposal_need_good")?.extract()?;
    let mut proposal_offer_good: PyReadwriteArray1<'_, i32> =
        trade.getattr("proposal_offer_good")?.extract()?;
    let mut proposal_quantity: PyReadwriteArray1<'_, f32> =
        trade.getattr("proposal_quantity")?.extract()?;
    let mut proposal_score: PyReadwriteArray1<'_, f32> =
        trade.getattr("proposal_score")?.extract()?;
    let mut accepted_mask: PyReadwriteArray1<'_, bool> =
        trade.getattr("accepted_mask")?.extract()?;
    let mut accepted_quantity: PyReadwriteArray1<'_, f32> =
        trade.getattr("accepted_quantity")?.extract()?;

    let elastic_need = market_elastic_need.as_array();
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
    let mut recent_purchase_value = recent_purchase_value.as_array_mut();
    let mut recent_sales_value = recent_sales_value.as_array_mut();
    let mut recent_inventory_inflow_value = recent_inventory_inflow_value.as_array_mut();
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

    if agent_id >= stock.nrows() {
        return Ok((0, 0.0, 0.0, 0.0, 0));
    }

    let blocked_partner_mask = vec![false; stock.nrows()];
    let mut changed_agents: HashSet<usize> = HashSet::new();
    let mut executed_count = 0_i32;
    let mut scheduled_quantity_total = 0.0_f64;
    let mut executed_quantity_total = 0.0_f64;
    let mut inventory_trade_volume = 0.0_f64;
    let mut attempts_used = 0_i32;

    for _ in 0..attempts_remaining {
        let candidates = plan_agent_parallel_phenomenon_candidates(
            0,
            agent_id,
            deal_type,
            goods,
            acquaintances,
            initial_transparency,
            initial_transparency_for_execution,
            history,
            local_liquidity_stock_bias,
            local_liquidity_min_sales,
            aspirational_stock_target,
            min_trade_quantity,
            trade_rounding_buffer,
            &blocked_partner_mask,
            elastic_need,
            stock.view(),
            role,
            stock_limit,
            purchase_price,
            sales_price,
            needs_level,
            transparency.view(),
            friend_id.view(),
            need.view(),
            recent_sales.view(),
            recent_purchases.view(),
            recent_inventory_inflow.view(),
            recent_production,
            friend_sold.view(),
            true,
        );
        let Some(candidate) = candidates.first().copied() else {
            break;
        };
        if candidate.need_good < 0 || candidate.offer_good < 0 || candidate.friend_id < 0 {
            break;
        }
        let need_good = candidate.need_good as usize;
        let offer_good = candidate.offer_good as usize;
        let friend_idx = candidate.friend_id as usize;
        let friend_slot = candidate.friend_slot as usize;
        if need_good >= goods
            || offer_good >= goods
            || friend_idx >= stock.nrows()
            || friend_slot >= acquaintances
        {
            break;
        }

        attempts_used += 1;
        let max_exchange = candidate.max_exchange;
        scheduled_quantity_total += max_exchange;
        let reciprocal_slot = if candidate.reciprocal_slot >= 0 {
            candidate.reciprocal_slot as usize
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

        let my_gift_out = (max_exchange * candidate.switch_average)
            / candidate.receiving_transparency.max(EPSILON as f64);
        let friend_need_out = max_exchange / candidate.need_transparency.max(EPSILON as f64);
        let friend_gift_in = max_exchange * candidate.switch_average;

        if deal_type == SURPLUS_DEAL {
            stock[[agent_id, need_good]] += max_exchange as f32;
            recent_inventory_inflow[[agent_id, need_good]] += max_exchange as f32;
            inventory_trade_volume += max_exchange;
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
        inventory_trade_volume += friend_gift_in;

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

        let my_need_purchase_price = purchase_price[[agent_id, need_good]];
        let my_offer_sales_price = sales_price[[agent_id, offer_good]];
        let friend_need_sales_price = sales_price[[friend_idx, need_good]];
        let friend_offer_purchase_price = purchase_price[[friend_idx, offer_good]];
        let exchange_value_to_me = exchange_value_to_me_like_numpy_float32(
            my_need_purchase_price,
            my_offer_sales_price,
            candidate.switch_average,
            candidate.receiving_transparency,
        );
        let exchange_value_to_friend = exchange_value_to_friend_like_numpy_float32(
            friend_offer_purchase_price,
            friend_need_sales_price,
            candidate.switch_average,
            candidate.need_transparency,
        );
        let my_value_correction = exchange_value_to_me.max(EPSILON as f64).sqrt();
        let friend_value_correction = exchange_value_to_friend.max(EPSILON as f64).sqrt();
        let my_purchase_unit_value =
            div_like_numpy_float32(my_need_purchase_price, my_value_correction);
        let my_sales_unit_value = mul_like_numpy_float32(my_offer_sales_price, my_value_correction);
        let friend_purchase_unit_value =
            div_like_numpy_float32(friend_offer_purchase_price, friend_value_correction);
        let friend_sales_unit_value =
            mul_like_numpy_float32(friend_need_sales_price, friend_value_correction);

        add_assign_like_numpy_float32(
            &mut recent_purchase_value[[agent_id, need_good]],
            max_exchange * my_purchase_unit_value,
        );
        add_assign_like_numpy_float32(
            &mut recent_sales_value[[agent_id, offer_good]],
            friend_gift_in * my_sales_unit_value,
        );
        add_assign_like_numpy_float32(
            &mut recent_purchase_value[[friend_idx, offer_good]],
            friend_gift_in * friend_purchase_unit_value,
        );
        add_assign_like_numpy_float32(
            &mut recent_sales_value[[friend_idx, need_good]],
            friend_need_out * friend_sales_unit_value,
        );
        if deal_type == SURPLUS_DEAL {
            add_assign_like_numpy_float32(
                &mut recent_inventory_inflow_value[[agent_id, need_good]],
                max_exchange * my_purchase_unit_value,
            );
        }
        add_assign_like_numpy_float32(
            &mut recent_inventory_inflow_value[[friend_idx, offer_good]],
            friend_gift_in * friend_purchase_unit_value,
        );

        if my_value_correction > 1.0 && friend_value_correction > 1.0 {
            add_assign_like_numpy_float32(
                &mut sum_period_purchase_value[[agent_id, need_good]],
                my_purchase_unit_value,
            );
            add_assign_like_numpy_float32(
                &mut sum_period_sales_value[[agent_id, offer_good]],
                my_sales_unit_value,
            );
            add_assign_like_numpy_float32(
                &mut sum_period_purchase_value[[friend_idx, offer_good]],
                friend_purchase_unit_value,
            );
            add_assign_like_numpy_float32(
                &mut sum_period_sales_value[[friend_idx, need_good]],
                friend_sales_unit_value,
            );
        }

        proposal_friend_slot[agent_id] = friend_slot as i32;
        proposal_target_agent[agent_id] = friend_idx as i32;
        proposal_need_good[agent_id] = need_good as i32;
        proposal_offer_good[agent_id] = offer_good as i32;
        proposal_quantity[agent_id] = max_exchange as f32;
        proposal_score[agent_id] = candidate.score;
        accepted_mask[agent_id] = true;
        accepted_quantity[agent_id] = max_exchange as f32;

        executed_count += 1;
        executed_quantity_total += max_exchange;
    }

    if !changed_agents.is_empty() {
        let mut changed_list: Vec<usize> = changed_agents.into_iter().collect();
        changed_list.sort_unstable();
        runner.call_method1("_sync_friend_slot_maps", (changed_list,))?;
    }

    Ok((
        executed_count,
        scheduled_quantity_total,
        executed_quantity_total,
        inventory_trade_volume,
        attempts_used,
    ))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn run_parallel_phenomenon_stage(
    py: Python<'_>,
    runner: PyObject,
    deal_type: i32,
    proposer_ids: PyReadonlyArray1<'_, i32>,
    max_attempts: usize,
) -> PyResult<(i32, i32, i32, i32, f64, f64, f64, i32, i32)> {
    if deal_type != CONSUMPTION_DEAL && deal_type != SURPLUS_DEAL {
        return Err(PyValueError::new_err(
            "deal_type must be 1 (surplus) or 2 (consumption)",
        ));
    }
    let runner = runner.bind(py);
    let state = runner.getattr("state")?;
    let market = runner.getattr("market")?;
    let config = runner.getattr("config")?;

    let goods: usize = config.getattr("goods")?.extract()?;
    let acquaintances: usize = config.getattr("acquaintances")?.extract()?;
    let initial_transparency_for_execution: f64 =
        config.getattr("initial_transparency")?.extract()?;
    let initial_transparency = initial_transparency_for_execution as f32;
    let history: i32 = config.getattr("history")?.extract()?;
    let local_liquidity_stock_bias: f64 = config
        .getattr("experimental_local_liquidity_stock_bias")?
        .extract()?;
    let local_liquidity_min_sales: f64 = config
        .getattr("experimental_local_liquidity_min_sales")?
        .extract()?;
    let aspirational_stock_target: f64 = config
        .getattr("experimental_aspirational_stock_target")?
        .extract()?;
    let min_trade_quantity: f64 = config.getattr("min_trade_quantity")?.extract()?;
    let trade_rounding_buffer: f64 = config.getattr("trade_rounding_buffer")?.extract()?;
    let initial_transactions = 2.0_f32;

    let mut active_agent_ids: Vec<usize> = proposer_ids
        .as_array()
        .iter()
        .filter_map(|raw_agent_id| {
            if *raw_agent_id < 0 {
                None
            } else {
                Some(*raw_agent_id as usize)
            }
        })
        .collect();
    if active_agent_ids.is_empty() || max_attempts == 0 {
        return Ok((0, 0, 0, 0, 0.0, 0.0, 0.0, 0, 1));
    }

    let market_elastic_need: PyReadonlyArray1<'_, f32> =
        market.getattr("elastic_need")?.extract()?;
    let mut market_periodic_tce_cost: PyReadwriteArray1<'_, f32> =
        market.getattr("periodic_tce_cost")?.extract()?;
    let mut stock: PyReadwriteArray2<'_, f32> = state.getattr("stock")?.extract()?;
    let role: PyReadonlyArray2<'_, i32> = state.getattr("role")?.extract()?;
    let stock_limit: PyReadonlyArray2<'_, f32> = state.getattr("stock_limit")?.extract()?;
    let purchase_price: PyReadonlyArray2<'_, f32> = state.getattr("purchase_price")?.extract()?;
    let sales_price: PyReadonlyArray2<'_, f32> = state.getattr("sales_price")?.extract()?;
    let needs_level: PyReadonlyArray1<'_, f32> = state.getattr("needs_level")?.extract()?;
    let mut transparency: PyReadwriteArray3<'_, f32> = state.getattr("transparency")?.extract()?;
    let mut friend_id: PyReadwriteArray2<'_, i32> = state.getattr("friend_id")?.extract()?;
    let mut need: PyReadwriteArray2<'_, f32> = state.getattr("need")?.extract()?;
    let recent_production: PyReadonlyArray2<'_, f32> =
        state.getattr("recent_production")?.extract()?;
    let mut recent_sales: PyReadwriteArray2<'_, f32> = state.getattr("recent_sales")?.extract()?;
    let mut recent_purchases: PyReadwriteArray2<'_, f32> =
        state.getattr("recent_purchases")?.extract()?;
    let mut sold_this_period: PyReadwriteArray2<'_, f32> =
        state.getattr("sold_this_period")?.extract()?;
    let mut purchased_this_period: PyReadwriteArray2<'_, f32> =
        state.getattr("purchased_this_period")?.extract()?;
    let mut recent_inventory_inflow: PyReadwriteArray2<'_, f32> =
        state.getattr("recent_inventory_inflow")?.extract()?;
    let mut recent_purchase_value: PyReadwriteArray2<'_, f32> =
        state.getattr("recent_purchase_value")?.extract()?;
    let mut recent_sales_value: PyReadwriteArray2<'_, f32> =
        state.getattr("recent_sales_value")?.extract()?;
    let mut recent_inventory_inflow_value: PyReadwriteArray2<'_, f32> =
        state.getattr("recent_inventory_inflow_value")?.extract()?;
    let mut purchase_times: PyReadwriteArray2<'_, i32> =
        state.getattr("purchase_times")?.extract()?;
    let mut sales_times: PyReadwriteArray2<'_, i32> = state.getattr("sales_times")?.extract()?;
    let mut sum_period_purchase_value: PyReadwriteArray2<'_, f32> =
        state.getattr("sum_period_purchase_value")?.extract()?;
    let mut sum_period_sales_value: PyReadwriteArray2<'_, f32> =
        state.getattr("sum_period_sales_value")?.extract()?;
    let mut friend_activity: PyReadwriteArray2<'_, f32> =
        state.getattr("friend_activity")?.extract()?;
    let mut friend_purchased: PyReadwriteArray3<'_, f32> =
        state.getattr("friend_purchased")?.extract()?;
    let mut friend_sold: PyReadwriteArray3<'_, f32> = state.getattr("friend_sold")?.extract()?;
    let trade = state.getattr("trade")?;
    let mut proposal_friend_slot: PyReadwriteArray1<'_, i32> =
        trade.getattr("proposal_friend_slot")?.extract()?;
    let mut proposal_target_agent: PyReadwriteArray1<'_, i32> =
        trade.getattr("proposal_target_agent")?.extract()?;
    let mut proposal_need_good: PyReadwriteArray1<'_, i32> =
        trade.getattr("proposal_need_good")?.extract()?;
    let mut proposal_offer_good: PyReadwriteArray1<'_, i32> =
        trade.getattr("proposal_offer_good")?.extract()?;
    let mut proposal_quantity: PyReadwriteArray1<'_, f32> =
        trade.getattr("proposal_quantity")?.extract()?;
    let mut proposal_score: PyReadwriteArray1<'_, f32> =
        trade.getattr("proposal_score")?.extract()?;
    let mut accepted_mask: PyReadwriteArray1<'_, bool> =
        trade.getattr("accepted_mask")?.extract()?;
    let mut accepted_quantity: PyReadwriteArray1<'_, f32> =
        trade.getattr("accepted_quantity")?.extract()?;

    let elastic_need = market_elastic_need.as_array();
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
    let mut recent_purchase_value = recent_purchase_value.as_array_mut();
    let mut recent_sales_value = recent_sales_value.as_array_mut();
    let mut recent_inventory_inflow_value = recent_inventory_inflow_value.as_array_mut();
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

    let population = stock.nrows();
    active_agent_ids.retain(|agent_id| *agent_id < population);
    let mut attempts_remaining = vec![0usize; population];
    for agent_id in &active_agent_ids {
        attempts_remaining[*agent_id] = max_attempts;
    }
    let mut changed_agents: HashSet<usize> = HashSet::new();
    let mut reciprocal_slots_cache = vec![-1_i32; population.saturating_mul(acquaintances)];
    for agent_id in 0..population {
        for friend_slot in 0..acquaintances {
            let friend_raw = friend_id[[agent_id, friend_slot]];
            if friend_raw >= 0 {
                reciprocal_slots_cache[(agent_id * acquaintances) + friend_slot] =
                    find_friend_slot_scan_internal(
                        friend_id.view(),
                        friend_raw as usize,
                        agent_id as i32,
                    );
            }
        }
    }
    let mut wave_count = 0_i32;
    let mut candidate_agents_total = 0_i32;
    let mut scheduled_exchanges_total = 0_i32;
    let mut dropped_exchanges_total = 0_i32;
    let mut scheduled_quantity_total = 0.0_f64;
    let mut executed_quantity_total = 0.0_f64;
    let mut inventory_trade_volume = 0.0_f64;
    let mut exhausted_agents_total = 0_i32;
    let mut stalled_waves_total = 0_i32;

    while !active_agent_ids.is_empty() {
        wave_count += 1;
        let mut candidates: Vec<ParallelPhenomenonCandidate> = active_agent_ids
            .par_iter()
            .enumerate()
            .flat_map_iter(|(proposer_order, agent_id)| {
                let reciprocal_start = agent_id.saturating_mul(acquaintances);
                let reciprocal_end = reciprocal_start.saturating_add(acquaintances);
                plan_agent_parallel_phenomenon_candidates_cached_links(
                    proposer_order,
                    *agent_id,
                    deal_type,
                    goods,
                    acquaintances,
                    initial_transparency,
                    initial_transparency_for_execution,
                    history,
                    local_liquidity_stock_bias,
                    local_liquidity_min_sales,
                    aspirational_stock_target,
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
                    friend_id.view(),
                    need.view(),
                    recent_sales.view(),
                    recent_purchases.view(),
                    recent_inventory_inflow.view(),
                    recent_production,
                    friend_sold.view(),
                    friend_id.row(*agent_id),
                    ArrayView1::from(&reciprocal_slots_cache[reciprocal_start..reciprocal_end]),
                    true,
                )
            })
            .collect();
        if candidates.is_empty() {
            stalled_waves_total += 1;
            break;
        }
        candidate_agents_total += candidates.len() as i32;
        let mut candidate_seen = vec![false; population];
        for candidate in &candidates {
            if candidate.agent_id >= 0 {
                let agent_idx = candidate.agent_id as usize;
                if agent_idx < population {
                    candidate_seen[agent_idx] = true;
                    attempts_remaining[agent_idx] = attempts_remaining[agent_idx].saturating_sub(1);
                }
            }
        }

        candidates.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.order.cmp(&right.order))
        });
        let mut reserved_agents = vec![false; population];
        let mut scheduled: Vec<ParallelPhenomenonCandidate> = Vec::new();
        for candidate in candidates {
            if candidate.agent_id < 0 || candidate.friend_id < 0 {
                dropped_exchanges_total += 1;
                continue;
            }
            let proposer_id = candidate.agent_id as usize;
            let partner_id = candidate.friend_id as usize;
            if proposer_id >= population || partner_id >= population {
                dropped_exchanges_total += 1;
                continue;
            }
            if reserved_agents[proposer_id] || reserved_agents[partner_id] {
                dropped_exchanges_total += 1;
                continue;
            }
            reserved_agents[proposer_id] = true;
            reserved_agents[partner_id] = true;
            scheduled.push(candidate);
        }
        if scheduled.is_empty() {
            stalled_waves_total += 1;
            break;
        }

        for candidate in scheduled {
            let agent_id = candidate.agent_id as usize;
            let need_good = candidate.need_good as usize;
            let offer_good = candidate.offer_good as usize;
            let friend_idx = candidate.friend_id as usize;
            let friend_slot = candidate.friend_slot as usize;
            if need_good >= goods
                || offer_good >= goods
                || friend_idx >= population
                || friend_slot >= acquaintances
            {
                continue;
            }
            scheduled_exchanges_total += 1;
            scheduled_quantity_total += candidate.max_exchange;
            let reciprocal_slot = if candidate.reciprocal_slot >= 0 {
                candidate.reciprocal_slot as usize
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
                reciprocal_slots_cache[(agent_id * acquaintances) + friend_slot] =
                    ensured_slot as i32;
                reciprocal_slots_cache[(friend_idx * acquaintances) + ensured_slot] =
                    friend_slot as i32;
                ensured_slot
            };
            let max_exchange = candidate.max_exchange;
            let my_gift_out = (max_exchange * candidate.switch_average)
                / candidate.receiving_transparency.max(EPSILON as f64);
            let friend_need_out = max_exchange / candidate.need_transparency.max(EPSILON as f64);
            let friend_gift_in = max_exchange * candidate.switch_average;

            if deal_type == SURPLUS_DEAL {
                stock[[agent_id, need_good]] += max_exchange as f32;
                recent_inventory_inflow[[agent_id, need_good]] += max_exchange as f32;
                inventory_trade_volume += max_exchange;
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
            inventory_trade_volume += friend_gift_in;
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

            let my_need_purchase_price = purchase_price[[agent_id, need_good]];
            let my_offer_sales_price = sales_price[[agent_id, offer_good]];
            let friend_need_sales_price = sales_price[[friend_idx, need_good]];
            let friend_offer_purchase_price = purchase_price[[friend_idx, offer_good]];
            let exchange_value_to_me = exchange_value_to_me_like_numpy_float32(
                my_need_purchase_price,
                my_offer_sales_price,
                candidate.switch_average,
                candidate.receiving_transparency,
            );
            let exchange_value_to_friend = exchange_value_to_friend_like_numpy_float32(
                friend_offer_purchase_price,
                friend_need_sales_price,
                candidate.switch_average,
                candidate.need_transparency,
            );
            let my_value_correction = exchange_value_to_me.max(EPSILON as f64).sqrt();
            let friend_value_correction = exchange_value_to_friend.max(EPSILON as f64).sqrt();
            let my_purchase_unit_value =
                div_like_numpy_float32(my_need_purchase_price, my_value_correction);
            let my_sales_unit_value =
                mul_like_numpy_float32(my_offer_sales_price, my_value_correction);
            let friend_purchase_unit_value =
                div_like_numpy_float32(friend_offer_purchase_price, friend_value_correction);
            let friend_sales_unit_value =
                mul_like_numpy_float32(friend_need_sales_price, friend_value_correction);
            add_assign_like_numpy_float32(
                &mut recent_purchase_value[[agent_id, need_good]],
                max_exchange * my_purchase_unit_value,
            );
            add_assign_like_numpy_float32(
                &mut recent_sales_value[[agent_id, offer_good]],
                friend_gift_in * my_sales_unit_value,
            );
            add_assign_like_numpy_float32(
                &mut recent_purchase_value[[friend_idx, offer_good]],
                friend_gift_in * friend_purchase_unit_value,
            );
            add_assign_like_numpy_float32(
                &mut recent_sales_value[[friend_idx, need_good]],
                friend_need_out * friend_sales_unit_value,
            );
            if deal_type == SURPLUS_DEAL {
                add_assign_like_numpy_float32(
                    &mut recent_inventory_inflow_value[[agent_id, need_good]],
                    max_exchange * my_purchase_unit_value,
                );
            }
            add_assign_like_numpy_float32(
                &mut recent_inventory_inflow_value[[friend_idx, offer_good]],
                friend_gift_in * friend_purchase_unit_value,
            );
            if my_value_correction > 1.0 && friend_value_correction > 1.0 {
                add_assign_like_numpy_float32(
                    &mut sum_period_purchase_value[[agent_id, need_good]],
                    my_purchase_unit_value,
                );
                add_assign_like_numpy_float32(
                    &mut sum_period_sales_value[[agent_id, offer_good]],
                    my_sales_unit_value,
                );
                add_assign_like_numpy_float32(
                    &mut sum_period_purchase_value[[friend_idx, offer_good]],
                    friend_purchase_unit_value,
                );
                add_assign_like_numpy_float32(
                    &mut sum_period_sales_value[[friend_idx, need_good]],
                    friend_sales_unit_value,
                );
            }
            proposal_friend_slot[agent_id] = friend_slot as i32;
            proposal_target_agent[agent_id] = friend_idx as i32;
            proposal_need_good[agent_id] = need_good as i32;
            proposal_offer_good[agent_id] = offer_good as i32;
            proposal_quantity[agent_id] = max_exchange as f32;
            proposal_score[agent_id] = candidate.score;
            accepted_mask[agent_id] = true;
            accepted_quantity[agent_id] = max_exchange as f32;
            executed_quantity_total += max_exchange;
        }

        let mut next_active_agent_ids: Vec<usize> = Vec::new();
        for agent_id in active_agent_ids {
            if candidate_seen[agent_id] && attempts_remaining[agent_id] > 0 {
                next_active_agent_ids.push(agent_id);
            } else if candidate_seen[agent_id] && attempts_remaining[agent_id] == 0 {
                exhausted_agents_total += 1;
            }
        }
        active_agent_ids = next_active_agent_ids;
    }

    if !changed_agents.is_empty() {
        let mut changed_list: Vec<usize> = changed_agents.into_iter().collect();
        changed_list.sort_unstable();
        runner.call_method1("_sync_friend_slot_maps", (changed_list,))?;
    }

    Ok((
        wave_count,
        candidate_agents_total,
        scheduled_exchanges_total,
        dropped_exchanges_total,
        scheduled_quantity_total,
        executed_quantity_total,
        inventory_trade_volume,
        exhausted_agents_total,
        stalled_waves_total,
    ))
}

#[allow(clippy::too_many_arguments)]
fn run_basket_session_internal(
    agent_id: usize,
    deal_type: i32,
    goods: usize,
    acquaintances: usize,
    initial_transparency_for_execution: f64,
    initial_transparency: f32,
    history: i32,
    local_liquidity_stock_bias: f64,
    local_liquidity_min_sales: f64,
    aspirational_stock_target: f64,
    replan_after_each_trade: bool,
    disable_replan_cache: bool,
    disable_offer_prefilter: bool,
    pairwise_offer_exhaustion: bool,
    candidate_depth: usize,
    min_trade_quantity: f64,
    trade_rounding_buffer: f64,
    max_attempts: usize,
    initial_transactions: f32,
    elastic_need: ArrayView1<'_, f32>,
    market_periodic_tce_cost: &mut ArrayViewMut1<'_, f32>,
    stock: &mut ArrayViewMut2<'_, f32>,
    role: ArrayView2<'_, i32>,
    stock_limit: ArrayView2<'_, f32>,
    purchase_price: ArrayView2<'_, f32>,
    sales_price: ArrayView2<'_, f32>,
    needs_level: ArrayView1<'_, f32>,
    transparency: &mut ArrayViewMut3<'_, f32>,
    friend_id: &mut ArrayViewMut2<'_, i32>,
    need: &mut ArrayViewMut2<'_, f32>,
    recent_production: ArrayView2<'_, f32>,
    recent_sales: &mut ArrayViewMut2<'_, f32>,
    recent_purchases: &mut ArrayViewMut2<'_, f32>,
    sold_this_period: &mut ArrayViewMut2<'_, f32>,
    purchased_this_period: &mut ArrayViewMut2<'_, f32>,
    recent_inventory_inflow: &mut ArrayViewMut2<'_, f32>,
    recent_purchase_value: &mut ArrayViewMut2<'_, f32>,
    recent_sales_value: &mut ArrayViewMut2<'_, f32>,
    recent_inventory_inflow_value: &mut ArrayViewMut2<'_, f32>,
    purchase_times: &mut ArrayViewMut2<'_, i32>,
    sales_times: &mut ArrayViewMut2<'_, i32>,
    sum_period_purchase_value: &mut ArrayViewMut2<'_, f32>,
    sum_period_sales_value: &mut ArrayViewMut2<'_, f32>,
    friend_activity: &mut ArrayViewMut2<'_, f32>,
    friend_purchased: &mut ArrayViewMut3<'_, f32>,
    friend_sold: &mut ArrayViewMut3<'_, f32>,
    proposal_friend_slot: &mut ArrayViewMut1<'_, i32>,
    proposal_target_agent: &mut ArrayViewMut1<'_, i32>,
    proposal_need_good: &mut ArrayViewMut1<'_, i32>,
    proposal_offer_good: &mut ArrayViewMut1<'_, i32>,
    proposal_quantity: &mut ArrayViewMut1<'_, f32>,
    proposal_score: &mut ArrayViewMut1<'_, f32>,
    accepted_mask: &mut ArrayViewMut1<'_, bool>,
    accepted_quantity: &mut ArrayViewMut1<'_, f32>,
    changed_agents: &mut HashSet<usize>,
    totals: &mut ExchangeStageTotals,
    mut profile: Option<&mut BasketProfile>,
) -> bool {
    if agent_id >= stock.nrows() || goods == 0 || acquaintances == 0 {
        return false;
    }
    if let Some(profile) = profile.as_deref_mut() {
        profile.sessions += 1;
    }

    let proposed_before = totals.proposed_count;
    let mut forbidden_offer_by_need = vec![false; goods * goods];
    let agent_friend_ids: Vec<i32> = (0..acquaintances)
        .map(|friend_slot| friend_id[[agent_id, friend_slot]])
        .collect();
    let mut reciprocal_slots = vec![-1_i32; acquaintances];
    let mut has_known_partner = false;
    let reciprocal_scan_start = profile.as_ref().map(|_| Instant::now());
    for friend_slot in 0..acquaintances {
        let fid = agent_friend_ids[friend_slot];
        if fid >= 0 {
            has_known_partner = true;
            reciprocal_slots[friend_slot] =
                find_friend_slot_scan_internal(friend_id.view(), fid as usize, agent_id as i32);
        }
    }
    if let (Some(profile), Some(start)) = (profile.as_deref_mut(), reciprocal_scan_start) {
        profile.reciprocal_scan_ns += start.elapsed().as_nanos();
    }
    if !has_known_partner {
        return false;
    }
    if let Some(profile) = profile.as_deref_mut() {
        profile.active_sessions += 1;
    }

    let use_replan_cache = replan_after_each_trade && candidate_depth == 1 && !disable_replan_cache;
    let mut cached_candidates: Vec<Option<BasketCandidate>> = vec![None; goods];
    let mut dirty_needs: Vec<bool> = vec![true; goods];
    let mut static_candidate_cursors: Vec<usize> = vec![0; goods];
    let mut static_candidate_lists: Option<Vec<Vec<StaticBasketCandidate>>> = None;
    let mut static_offer_to_needs: Option<Vec<u128>> = None;
    let mut static_offer_friend_to_needs: Option<Vec<u128>> = None;
    let mut own_offer_available: Option<Vec<bool>> = None;
    let mut friend_supply_available: Option<Vec<bool>> = None;
    let mut friend_accept_available: Option<Vec<bool>> = None;
    let mut attempts = 0usize;
    while attempts < max_attempts {
        let candidates = if use_replan_cache {
            let dirty_count = dirty_needs.iter().filter(|is_dirty| **is_dirty).count();
            if attempts > 0 && static_candidate_lists.is_none() {
                let static_build_start = profile.as_ref().map(|_| Instant::now());
                static_candidate_lists = Some(build_static_basket_candidate_lists(
                    agent_id,
                    goods,
                    acquaintances,
                    initial_transparency,
                    role,
                    purchase_price,
                    sales_price,
                    transparency.view(),
                    ArrayView1::from(&agent_friend_ids[..]),
                    ArrayView1::from(&reciprocal_slots[..]),
                ));
                if let (Some(profile), Some(start)) = (profile.as_deref_mut(), static_build_start) {
                    profile.static_builds += 1;
                    profile.static_build_ns += start.elapsed().as_nanos();
                }
                let static_candidate_lists_ref = static_candidate_lists.as_ref().unwrap();
                let static_index_start = profile.as_ref().map(|_| Instant::now());
                let (offer_to_needs, offer_friend_to_needs) = build_static_basket_candidate_indexes(
                    static_candidate_lists_ref,
                    goods,
                    acquaintances,
                );
                if let (Some(profile), Some(start)) = (profile.as_deref_mut(), static_index_start) {
                    profile.static_index_builds += 1;
                    profile.static_index_ns += start.elapsed().as_nanos();
                }
                static_offer_to_needs = Some(offer_to_needs);
                static_offer_friend_to_needs = Some(offer_friend_to_needs);
                if own_offer_available.is_none() {
                    let (own_available, supply_available, accept_available) =
                        build_session_availability_cache(
                            agent_id,
                            goods,
                            acquaintances,
                            &agent_friend_ids,
                            elastic_need,
                            stock.view(),
                            role,
                            stock_limit,
                            needs_level,
                        );
                    own_offer_available = Some(own_available);
                    friend_supply_available = Some(supply_available);
                    friend_accept_available = Some(accept_available);
                }
            }
            if let Some(static_candidate_lists_ref) = static_candidate_lists.as_ref() {
                for need_good in 0..goods {
                    if !dirty_needs[need_good] {
                        continue;
                    }
                    let max_need_start = profile.as_ref().map(|_| Instant::now());
                    let max_need = basket_stage_max_need(
                        agent_id,
                        need_good,
                        deal_type,
                        history,
                        local_liquidity_stock_bias,
                        local_liquidity_min_sales,
                        aspirational_stock_target,
                        elastic_need,
                        stock.view(),
                        stock_limit,
                        needs_level,
                        need.view(),
                        recent_sales.view(),
                        recent_purchases.view(),
                        recent_inventory_inflow.view(),
                        recent_production,
                        friend_id.view(),
                        friend_sold.view(),
                        transparency.view(),
                    );
                    if let (Some(profile), Some(start)) = (profile.as_deref_mut(), max_need_start) {
                        profile.max_need_calls += 1;
                        profile.max_need_ns += start.elapsed().as_nanos();
                    }
                    cached_candidates[need_good] = if max_need <= 0.0 {
                        None
                    } else {
                        let start_index = static_candidate_cursors[need_good];
                        let forbidden_row_start = need_good * goods;
                        let forbidden_offer_row = &forbidden_offer_by_need
                            [forbidden_row_start..forbidden_row_start + goods];
                        let first_valid_start = profile.as_ref().map(|_| Instant::now());
                        let result = first_valid_static_basket_candidate(
                            need_good,
                            goods,
                            start_index,
                            forbidden_offer_row,
                            &static_candidate_lists_ref[need_good],
                            &agent_friend_ids,
                            own_offer_available.as_deref().unwrap_or(&[]),
                            friend_supply_available.as_deref().unwrap_or(&[]),
                            friend_accept_available.as_deref().unwrap_or(&[]),
                        );
                        if let (Some(profile), Some(start)) =
                            (profile.as_deref_mut(), first_valid_start)
                        {
                            profile.first_valid_calls += 1;
                            profile.first_valid_ns += start.elapsed().as_nanos();
                        }
                        match result {
                            Some((candidate, candidate_index)) => {
                                static_candidate_cursors[need_good] = candidate_index;
                                Some(candidate)
                            }
                            None => {
                                static_candidate_cursors[need_good] =
                                    static_candidate_lists_ref[need_good].len();
                                None
                            }
                        }
                    };
                    dirty_needs[need_good] = false;
                }
            } else if dirty_count > goods.saturating_div(4).max(1) {
                cached_candidates.fill(None);
                let full_plan_start = profile.as_ref().map(|_| Instant::now());
                let full_candidates = plan_agent_basket_candidates_from_cached_links(
                    agent_id,
                    deal_type,
                    goods,
                    acquaintances,
                    initial_transparency,
                    history,
                    local_liquidity_stock_bias,
                    local_liquidity_min_sales,
                    aspirational_stock_target,
                    &forbidden_offer_by_need,
                    elastic_need,
                    stock.view(),
                    role,
                    stock_limit,
                    purchase_price,
                    sales_price,
                    needs_level,
                    transparency.view(),
                    ArrayView1::from(&agent_friend_ids[..]),
                    ArrayView1::from(&reciprocal_slots[..]),
                    need.view(),
                    recent_sales.view(),
                    recent_purchases.view(),
                    recent_inventory_inflow.view(),
                    recent_production,
                    friend_id.view(),
                    friend_sold.view(),
                    candidate_depth,
                    disable_offer_prefilter,
                );
                if let (Some(profile), Some(start)) = (profile.as_deref_mut(), full_plan_start) {
                    profile.full_plan_calls += 1;
                    profile.full_plan_ns += start.elapsed().as_nanos();
                }
                for candidate in full_candidates {
                    if candidate.need_good >= 0 {
                        let need_good = candidate.need_good as usize;
                        if need_good < goods {
                            cached_candidates[need_good] = Some(candidate);
                        }
                    }
                }
                dirty_needs.fill(false);
            } else {
                let static_build_start = profile.as_ref().map(|_| Instant::now());
                static_candidate_lists = Some(build_static_basket_candidate_lists(
                    agent_id,
                    goods,
                    acquaintances,
                    initial_transparency,
                    role,
                    purchase_price,
                    sales_price,
                    transparency.view(),
                    ArrayView1::from(&agent_friend_ids[..]),
                    ArrayView1::from(&reciprocal_slots[..]),
                ));
                if let (Some(profile), Some(start)) = (profile.as_deref_mut(), static_build_start) {
                    profile.static_builds += 1;
                    profile.static_build_ns += start.elapsed().as_nanos();
                }
                let static_candidate_lists_ref = static_candidate_lists.as_ref().unwrap();
                let static_index_start = profile.as_ref().map(|_| Instant::now());
                let (offer_to_needs, offer_friend_to_needs) = build_static_basket_candidate_indexes(
                    static_candidate_lists_ref,
                    goods,
                    acquaintances,
                );
                if let (Some(profile), Some(start)) = (profile.as_deref_mut(), static_index_start) {
                    profile.static_index_builds += 1;
                    profile.static_index_ns += start.elapsed().as_nanos();
                }
                static_offer_to_needs = Some(offer_to_needs);
                static_offer_friend_to_needs = Some(offer_friend_to_needs);
                if own_offer_available.is_none() {
                    let (own_available, supply_available, accept_available) =
                        build_session_availability_cache(
                            agent_id,
                            goods,
                            acquaintances,
                            &agent_friend_ids,
                            elastic_need,
                            stock.view(),
                            role,
                            stock_limit,
                            needs_level,
                        );
                    own_offer_available = Some(own_available);
                    friend_supply_available = Some(supply_available);
                    friend_accept_available = Some(accept_available);
                }
                for need_good in 0..goods {
                    if !dirty_needs[need_good] {
                        continue;
                    }
                    let max_need_start = profile.as_ref().map(|_| Instant::now());
                    let max_need = basket_stage_max_need(
                        agent_id,
                        need_good,
                        deal_type,
                        history,
                        local_liquidity_stock_bias,
                        local_liquidity_min_sales,
                        aspirational_stock_target,
                        elastic_need,
                        stock.view(),
                        stock_limit,
                        needs_level,
                        need.view(),
                        recent_sales.view(),
                        recent_purchases.view(),
                        recent_inventory_inflow.view(),
                        recent_production,
                        friend_id.view(),
                        friend_sold.view(),
                        transparency.view(),
                    );
                    if let (Some(profile), Some(start)) = (profile.as_deref_mut(), max_need_start) {
                        profile.max_need_calls += 1;
                        profile.max_need_ns += start.elapsed().as_nanos();
                    }
                    cached_candidates[need_good] = if max_need <= 0.0 {
                        None
                    } else {
                        let start_index = static_candidate_cursors[need_good];
                        let forbidden_row_start = need_good * goods;
                        let forbidden_offer_row = &forbidden_offer_by_need
                            [forbidden_row_start..forbidden_row_start + goods];
                        let first_valid_start = profile.as_ref().map(|_| Instant::now());
                        let result = first_valid_static_basket_candidate(
                            need_good,
                            goods,
                            start_index,
                            forbidden_offer_row,
                            &static_candidate_lists_ref[need_good],
                            &agent_friend_ids,
                            own_offer_available.as_deref().unwrap_or(&[]),
                            friend_supply_available.as_deref().unwrap_or(&[]),
                            friend_accept_available.as_deref().unwrap_or(&[]),
                        );
                        if let (Some(profile), Some(start)) =
                            (profile.as_deref_mut(), first_valid_start)
                        {
                            profile.first_valid_calls += 1;
                            profile.first_valid_ns += start.elapsed().as_nanos();
                        }
                        match result {
                            Some((candidate, candidate_index)) => {
                                static_candidate_cursors[need_good] = candidate_index;
                                Some(candidate)
                            }
                            None => {
                                static_candidate_cursors[need_good] =
                                    static_candidate_lists_ref[need_good].len();
                                None
                            }
                        }
                    };
                    dirty_needs[need_good] = false;
                }
            }
            let mut best_candidate: Option<BasketCandidate> = None;
            for candidate in cached_candidates.iter().flatten().copied() {
                match best_candidate {
                    Some(best)
                        if best.score > candidate.score
                            || (best.score == candidate.score && best.order < candidate.order) => {}
                    _ => best_candidate = Some(candidate),
                }
            }
            best_candidate.into_iter().collect()
        } else {
            let full_plan_start = profile.as_ref().map(|_| Instant::now());
            let planned_candidates = plan_agent_basket_candidates_from_cached_links(
                agent_id,
                deal_type,
                goods,
                acquaintances,
                initial_transparency,
                history,
                local_liquidity_stock_bias,
                local_liquidity_min_sales,
                aspirational_stock_target,
                &forbidden_offer_by_need,
                elastic_need,
                stock.view(),
                role,
                stock_limit,
                purchase_price,
                sales_price,
                needs_level,
                transparency.view(),
                ArrayView1::from(&agent_friend_ids[..]),
                ArrayView1::from(&reciprocal_slots[..]),
                need.view(),
                recent_sales.view(),
                recent_purchases.view(),
                recent_inventory_inflow.view(),
                recent_production,
                friend_id.view(),
                friend_sold.view(),
                candidate_depth,
                disable_offer_prefilter,
            );
            if let (Some(profile), Some(start)) = (profile.as_deref_mut(), full_plan_start) {
                profile.full_plan_calls += 1;
                profile.full_plan_ns += start.elapsed().as_nanos();
            }
            planned_candidates
        };
        if candidates.is_empty() {
            break;
        }

        let mut processed_candidate = false;
        let mut executed_any = false;
        let mut changed_forbidden = false;
        for candidate in candidates {
            if attempts >= max_attempts {
                break;
            }
            if candidate.need_good < 0 || candidate.offer_good < 0 {
                continue;
            }
            let need_good = candidate.need_good as usize;
            let offer_good = candidate.offer_good as usize;
            if need_good >= goods || offer_good >= goods {
                continue;
            }
            let forbidden_index = (need_good * goods) + offer_good;
            if forbidden_offer_by_need[forbidden_index] {
                continue;
            }

            let max_need_start = profile.as_ref().map(|_| Instant::now());
            let max_need = basket_stage_max_need(
                agent_id,
                need_good,
                deal_type,
                history,
                local_liquidity_stock_bias,
                local_liquidity_min_sales,
                aspirational_stock_target,
                elastic_need,
                stock.view(),
                stock_limit,
                needs_level,
                need.view(),
                recent_sales.view(),
                recent_purchases.view(),
                recent_inventory_inflow.view(),
                recent_production,
                friend_id.view(),
                friend_sold.view(),
                transparency.view(),
            );
            if let (Some(profile), Some(start)) = (profile.as_deref_mut(), max_need_start) {
                profile.max_need_calls += 1;
                profile.max_need_ns += start.elapsed().as_nanos();
            }
            if max_need <= 0.0 {
                if use_replan_cache {
                    cached_candidates[need_good] = None;
                    dirty_needs[need_good] = true;
                    changed_forbidden = true;
                }
                continue;
            }

            processed_candidate = true;
            totals.proposed_count += 1;
            if let Some(profile) = profile.as_deref_mut() {
                profile.proposed += 1;
            }
            let specific_plan_start = profile.as_ref().map(|_| Instant::now());
            let plan = plan_specific_exchange_candidate(
                agent_id,
                need_good,
                initial_transparency_for_execution,
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
                friend_id.view(),
                candidate,
            );
            attempts += 1;
            if let (Some(profile), Some(start)) = (profile.as_deref_mut(), specific_plan_start) {
                profile.attempts += 1;
                profile.specific_plan_calls += 1;
                profile.specific_plan_ns += start.elapsed().as_nanos();
            }

            let Some(plan) = plan else {
                if !forbidden_offer_by_need[forbidden_index] {
                    forbidden_offer_by_need[forbidden_index] = true;
                    changed_forbidden = true;
                }
                if use_replan_cache {
                    cached_candidates[need_good] = None;
                    dirty_needs[need_good] = true;
                }
                continue;
            };
            if plan.reason_code != PLAN_OK {
                if !forbidden_offer_by_need[forbidden_index] {
                    forbidden_offer_by_need[forbidden_index] = true;
                    changed_forbidden = true;
                }
                if use_replan_cache {
                    cached_candidates[need_good] = None;
                    dirty_needs[need_good] = true;
                }
                continue;
            }

            let execute_start = profile.as_ref().map(|_| Instant::now());
            let friend_idx = plan.candidate.friend_id as usize;
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
                    friend_id,
                    friend_activity,
                    friend_purchased,
                    friend_sold,
                    transparency,
                ) as usize;
                reciprocal_slots[friend_slot] = ensured_slot as i32;
                changed_agents.insert(friend_idx);
                ensured_slot
            };

            let max_exchange = plan.max_exchange;
            let remaining_need_after_trade = max_need - max_exchange;
            let my_gift_out = (max_exchange * plan.switch_average)
                / plan.receiving_transparency.max(EPSILON as f64);
            let friend_need_out = max_exchange / plan.need_transparency.max(EPSILON as f64);
            let friend_gift_in = max_exchange * plan.switch_average;
            let my_need_stock_before = stock[[agent_id, need_good]];
            let friend_need_stock_before = stock[[friend_idx, need_good]];
            let friend_offer_stock_before = stock[[friend_idx, offer_good]];

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

            let my_need_purchase_price = purchase_price[[agent_id, need_good]];
            let my_offer_sales_price = sales_price[[agent_id, offer_good]];
            let friend_need_sales_price = sales_price[[friend_idx, need_good]];
            let friend_offer_purchase_price = purchase_price[[friend_idx, offer_good]];
            let exchange_value_to_me = exchange_value_to_me_like_numpy_float32(
                my_need_purchase_price,
                my_offer_sales_price,
                plan.switch_average,
                plan.receiving_transparency,
            );
            let exchange_value_to_friend = exchange_value_to_friend_like_numpy_float32(
                friend_offer_purchase_price,
                friend_need_sales_price,
                plan.switch_average,
                plan.need_transparency,
            );
            let my_value_correction = exchange_value_to_me.max(EPSILON as f64).sqrt();
            let friend_value_correction = exchange_value_to_friend.max(EPSILON as f64).sqrt();
            let my_purchase_unit_value =
                div_like_numpy_float32(my_need_purchase_price, my_value_correction);
            let my_sales_unit_value =
                mul_like_numpy_float32(my_offer_sales_price, my_value_correction);
            let friend_purchase_unit_value =
                div_like_numpy_float32(friend_offer_purchase_price, friend_value_correction);
            let friend_sales_unit_value =
                mul_like_numpy_float32(friend_need_sales_price, friend_value_correction);

            add_assign_like_numpy_float32(
                &mut recent_purchase_value[[agent_id, need_good]],
                max_exchange * my_purchase_unit_value,
            );
            add_assign_like_numpy_float32(
                &mut recent_sales_value[[agent_id, offer_good]],
                friend_gift_in * my_sales_unit_value,
            );
            add_assign_like_numpy_float32(
                &mut recent_purchase_value[[friend_idx, offer_good]],
                friend_gift_in * friend_purchase_unit_value,
            );
            add_assign_like_numpy_float32(
                &mut recent_sales_value[[friend_idx, need_good]],
                friend_need_out * friend_sales_unit_value,
            );
            if deal_type == SURPLUS_DEAL {
                add_assign_like_numpy_float32(
                    &mut recent_inventory_inflow_value[[agent_id, need_good]],
                    max_exchange * my_purchase_unit_value,
                );
            }
            add_assign_like_numpy_float32(
                &mut recent_inventory_inflow_value[[friend_idx, offer_good]],
                friend_gift_in * friend_purchase_unit_value,
            );

            if my_value_correction > 1.0 && friend_value_correction > 1.0 {
                add_assign_like_numpy_float32(
                    &mut sum_period_purchase_value[[agent_id, need_good]],
                    my_purchase_unit_value,
                );
                add_assign_like_numpy_float32(
                    &mut sum_period_sales_value[[agent_id, offer_good]],
                    my_sales_unit_value,
                );
                add_assign_like_numpy_float32(
                    &mut sum_period_purchase_value[[friend_idx, offer_good]],
                    friend_purchase_unit_value,
                );
                add_assign_like_numpy_float32(
                    &mut sum_period_sales_value[[friend_idx, need_good]],
                    friend_sales_unit_value,
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
            if let Some(profile) = profile.as_deref_mut() {
                profile.accepted += 1;
            }
            executed_any = true;

            if let (Some(own_available), Some(supply_available), Some(accept_available)) = (
                own_offer_available.as_mut(),
                friend_supply_available.as_mut(),
                friend_accept_available.as_mut(),
            ) {
                refresh_session_availability_good(
                    own_available,
                    supply_available,
                    accept_available,
                    agent_id,
                    friend_slot,
                    friend_idx,
                    offer_good,
                    goods,
                    elastic_need,
                    stock.view(),
                    role,
                    stock_limit,
                    needs_level,
                );
                refresh_session_availability_good(
                    own_available,
                    supply_available,
                    accept_available,
                    agent_id,
                    friend_slot,
                    friend_idx,
                    need_good,
                    goods,
                    elastic_need,
                    stock.view(),
                    role,
                    stock_limit,
                    needs_level,
                );
            }

            let exhausted_gift =
                remaining_need_after_trade >= 1.0 && stock[[agent_id, offer_good]] < 1.0;
            let exhausted_offer_good = offer_good_exhausted_for_agent(
                agent_id,
                offer_good,
                elastic_need,
                stock.view(),
                needs_level,
            );
            if exhausted_offer_good && !pairwise_offer_exhaustion {
                if forbid_offer_good_for_all_needs(&mut forbidden_offer_by_need, goods, offer_good)
                {
                    changed_forbidden = true;
                    if use_replan_cache {
                        if let Some(offer_to_needs) = static_offer_to_needs.as_ref() {
                            let mask = offer_to_needs[offer_good];
                            mark_dirty_need_mask(
                                mask,
                                goods,
                                &mut dirty_needs,
                                &mut static_candidate_cursors,
                            );
                            clear_cached_need_mask(mask, goods, &mut cached_candidates);
                        } else {
                            dirty_needs.fill(true);
                            static_candidate_cursors.fill(0);
                            cached_candidates.fill(None);
                        }
                    }
                }
            } else if exhausted_gift && !forbidden_offer_by_need[forbidden_index] {
                forbidden_offer_by_need[forbidden_index] = true;
                changed_forbidden = true;
            }
            if use_replan_cache {
                let mut dirty_all = false;
                if deal_type == SURPLUS_DEAL {
                    let own_offer_threshold =
                        (elastic_need[need_good] * needs_level[agent_id]) + 1.0;
                    if my_need_stock_before <= own_offer_threshold
                        && stock[[agent_id, need_good]] > own_offer_threshold
                    {
                        if let Some(offer_to_needs) = static_offer_to_needs.as_ref() {
                            mark_dirty_need_mask(
                                offer_to_needs[need_good],
                                goods,
                                &mut dirty_needs,
                                &mut static_candidate_cursors,
                            );
                        } else {
                            dirty_all = true;
                        }
                    }
                }
                let friend_accept_threshold = if role[[friend_idx, need_good]] == ROLE_RETAILER {
                    stock_limit[[friend_idx, need_good]] - 1.0
                } else {
                    elastic_need[need_good] * needs_level[friend_idx] - 1.0
                };
                if friend_need_stock_before >= friend_accept_threshold
                    && stock[[friend_idx, need_good]] < friend_accept_threshold
                {
                    if let Some(offer_friend_to_needs) = static_offer_friend_to_needs.as_ref() {
                        let index = (need_good * acquaintances) + friend_slot;
                        mark_dirty_need_mask(
                            offer_friend_to_needs[index],
                            goods,
                            &mut dirty_needs,
                            &mut static_candidate_cursors,
                        );
                    } else {
                        dirty_all = true;
                    }
                }
                if dirty_all {
                    dirty_needs.fill(true);
                    static_candidate_cursors.fill(0);
                } else {
                    dirty_needs[need_good] = true;
                    dirty_needs[offer_good] = true;
                    for cached in cached_candidates.iter().flatten().copied() {
                        let cached_need = cached.need_good as usize;
                        let cached_offer = cached.offer_good as usize;
                        let cached_friend_slot = cached.friend_slot as usize;
                        if cached_offer == offer_good
                            || (cached_friend_slot == friend_slot && cached_offer == offer_good)
                            || cached_need == need_good
                            || cached_need == offer_good
                        {
                            dirty_needs[cached_need] = true;
                        }
                    }
                }
                let friend_offer_supply_threshold =
                    (elastic_need[offer_good] * needs_level[friend_idx]) + 1.0;
                if friend_offer_supply_threshold.is_finite()
                    && friend_offer_stock_before <= friend_offer_supply_threshold
                    && stock[[friend_idx, offer_good]] > friend_offer_supply_threshold
                {
                    dirty_needs[offer_good] = true;
                    static_candidate_cursors[offer_good] = 0;
                }
                cached_candidates[need_good] = None;
                cached_candidates[offer_good] = None;
            }
            if let (Some(profile), Some(start)) = (profile.as_deref_mut(), execute_start) {
                profile.execute_ns += start.elapsed().as_nanos();
            }
            if replan_after_each_trade {
                break;
            }
        }

        if replan_after_each_trade {
            if !processed_candidate || (!executed_any && !changed_forbidden) {
                break;
            }
        } else {
            break;
        }
    }

    totals.proposed_count > proposed_before
}

#[pyfunction]
fn run_phenomenon_session_stage(
    py: Python<'_>,
    runner: PyObject,
    deal_type: i32,
    proposer_ids: PyReadonlyArray1<'_, i32>,
    max_attempts: usize,
) -> PyResult<(i32, i32, f64, f64, i32)> {
    if deal_type != CONSUMPTION_DEAL && deal_type != SURPLUS_DEAL {
        return Err(PyValueError::new_err(
            "deal_type must be 1 (surplus) or 2 (consumption)",
        ));
    }

    let runner = runner.bind(py);
    let state = runner.getattr("state")?;
    let market = runner.getattr("market")?;
    let config = runner.getattr("config")?;

    let goods: usize = config.getattr("goods")?.extract()?;
    let acquaintances: usize = config.getattr("acquaintances")?.extract()?;
    let initial_transparency_for_execution: f64 =
        config.getattr("initial_transparency")?.extract()?;
    let initial_transparency = initial_transparency_for_execution as f32;
    let history: i32 = config.getattr("history")?.extract()?;
    let local_liquidity_stock_bias: f64 = config
        .getattr("experimental_local_liquidity_stock_bias")?
        .extract()?;
    let local_liquidity_min_sales: f64 = config
        .getattr("experimental_local_liquidity_min_sales")?
        .extract()?;
    let aspirational_stock_target: f64 = config
        .getattr("experimental_aspirational_stock_target")?
        .extract()?;
    let session_replan_passes: usize = config
        .getattr("experimental_session_replan_passes")?
        .extract()?;
    let session_replan_after_trade: bool = config
        .getattr("experimental_session_replan_after_trade")?
        .extract()?;
    let session_disable_replan_cache: bool = config
        .getattr("experimental_session_disable_replan_cache")?
        .extract()?;
    let session_disable_offer_prefilter: bool = config
        .getattr("experimental_session_disable_offer_prefilter")?
        .extract()?;
    let session_pairwise_offer_exhaustion: bool = config
        .getattr("experimental_session_pairwise_offer_exhaustion")?
        .extract()?;
    let session_candidate_depth: usize = config
        .getattr("experimental_session_candidate_depth")?
        .extract()?;
    let min_trade_quantity: f64 = config.getattr("min_trade_quantity")?.extract()?;
    let trade_rounding_buffer: f64 = config.getattr("trade_rounding_buffer")?.extract()?;
    let initial_transactions = 2.0_f32;

    let market_elastic_need: PyReadonlyArray1<'_, f32> =
        market.getattr("elastic_need")?.extract()?;
    let mut market_periodic_tce_cost: PyReadwriteArray1<'_, f32> =
        market.getattr("periodic_tce_cost")?.extract()?;
    let mut stock: PyReadwriteArray2<'_, f32> = state.getattr("stock")?.extract()?;
    let role: PyReadonlyArray2<'_, i32> = state.getattr("role")?.extract()?;
    let stock_limit: PyReadonlyArray2<'_, f32> = state.getattr("stock_limit")?.extract()?;
    let purchase_price: PyReadonlyArray2<'_, f32> = state.getattr("purchase_price")?.extract()?;
    let sales_price: PyReadonlyArray2<'_, f32> = state.getattr("sales_price")?.extract()?;
    let needs_level: PyReadonlyArray1<'_, f32> = state.getattr("needs_level")?.extract()?;
    let mut transparency: PyReadwriteArray3<'_, f32> = state.getattr("transparency")?.extract()?;
    let mut friend_id: PyReadwriteArray2<'_, i32> = state.getattr("friend_id")?.extract()?;
    let mut need: PyReadwriteArray2<'_, f32> = state.getattr("need")?.extract()?;
    let recent_production: PyReadonlyArray2<'_, f32> =
        state.getattr("recent_production")?.extract()?;
    let mut recent_sales: PyReadwriteArray2<'_, f32> = state.getattr("recent_sales")?.extract()?;
    let mut recent_purchases: PyReadwriteArray2<'_, f32> =
        state.getattr("recent_purchases")?.extract()?;
    let mut sold_this_period: PyReadwriteArray2<'_, f32> =
        state.getattr("sold_this_period")?.extract()?;
    let mut purchased_this_period: PyReadwriteArray2<'_, f32> =
        state.getattr("purchased_this_period")?.extract()?;
    let mut recent_inventory_inflow: PyReadwriteArray2<'_, f32> =
        state.getattr("recent_inventory_inflow")?.extract()?;
    let mut recent_purchase_value: PyReadwriteArray2<'_, f32> =
        state.getattr("recent_purchase_value")?.extract()?;
    let mut recent_sales_value: PyReadwriteArray2<'_, f32> =
        state.getattr("recent_sales_value")?.extract()?;
    let mut recent_inventory_inflow_value: PyReadwriteArray2<'_, f32> =
        state.getattr("recent_inventory_inflow_value")?.extract()?;
    let mut purchase_times: PyReadwriteArray2<'_, i32> =
        state.getattr("purchase_times")?.extract()?;
    let mut sales_times: PyReadwriteArray2<'_, i32> = state.getattr("sales_times")?.extract()?;
    let mut sum_period_purchase_value: PyReadwriteArray2<'_, f32> =
        state.getattr("sum_period_purchase_value")?.extract()?;
    let mut sum_period_sales_value: PyReadwriteArray2<'_, f32> =
        state.getattr("sum_period_sales_value")?.extract()?;
    let mut friend_activity: PyReadwriteArray2<'_, f32> =
        state.getattr("friend_activity")?.extract()?;
    let mut friend_purchased: PyReadwriteArray3<'_, f32> =
        state.getattr("friend_purchased")?.extract()?;
    let mut friend_sold: PyReadwriteArray3<'_, f32> = state.getattr("friend_sold")?.extract()?;
    let trade = state.getattr("trade")?;
    let mut proposal_friend_slot: PyReadwriteArray1<'_, i32> =
        trade.getattr("proposal_friend_slot")?.extract()?;
    let mut proposal_target_agent: PyReadwriteArray1<'_, i32> =
        trade.getattr("proposal_target_agent")?.extract()?;
    let mut proposal_need_good: PyReadwriteArray1<'_, i32> =
        trade.getattr("proposal_need_good")?.extract()?;
    let mut proposal_offer_good: PyReadwriteArray1<'_, i32> =
        trade.getattr("proposal_offer_good")?.extract()?;
    let mut proposal_quantity: PyReadwriteArray1<'_, f32> =
        trade.getattr("proposal_quantity")?.extract()?;
    let mut proposal_score: PyReadwriteArray1<'_, f32> =
        trade.getattr("proposal_score")?.extract()?;
    let mut accepted_mask: PyReadwriteArray1<'_, bool> =
        trade.getattr("accepted_mask")?.extract()?;
    let mut accepted_quantity: PyReadwriteArray1<'_, f32> =
        trade.getattr("accepted_quantity")?.extract()?;

    let elastic_need = market_elastic_need.as_array();
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
    let mut recent_purchase_value = recent_purchase_value.as_array_mut();
    let mut recent_sales_value = recent_sales_value.as_array_mut();
    let mut recent_inventory_inflow_value = recent_inventory_inflow_value.as_array_mut();
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

    let mut totals = ExchangeStageTotals::default();
    let mut changed_agents: HashSet<usize> = HashSet::new();
    let mut active_session_agents = 0_i32;
    let population = stock.nrows();
    for raw_agent_id in proposer_ids.as_array().iter() {
        if *raw_agent_id < 0 {
            continue;
        }
        let agent_id = *raw_agent_id as usize;
        if agent_id >= population {
            continue;
        }
        let mut agent_had_session = false;
        for _pass in 0..session_replan_passes.max(1) {
            let accepted_before = totals.accepted_count;
            if run_basket_session_internal(
                agent_id,
                deal_type,
                goods,
                acquaintances,
                initial_transparency_for_execution,
                initial_transparency,
                history,
                local_liquidity_stock_bias,
                local_liquidity_min_sales,
                aspirational_stock_target,
                session_replan_after_trade,
                session_disable_replan_cache,
                session_disable_offer_prefilter,
                session_pairwise_offer_exhaustion,
                session_candidate_depth,
                min_trade_quantity,
                trade_rounding_buffer,
                max_attempts,
                initial_transactions,
                elastic_need,
                &mut market_periodic_tce_cost,
                &mut stock,
                role,
                stock_limit,
                purchase_price,
                sales_price,
                needs_level,
                &mut transparency,
                &mut friend_id,
                &mut need,
                recent_production,
                &mut recent_sales,
                &mut recent_purchases,
                &mut sold_this_period,
                &mut purchased_this_period,
                &mut recent_inventory_inflow,
                &mut recent_purchase_value,
                &mut recent_sales_value,
                &mut recent_inventory_inflow_value,
                &mut purchase_times,
                &mut sales_times,
                &mut sum_period_purchase_value,
                &mut sum_period_sales_value,
                &mut friend_activity,
                &mut friend_purchased,
                &mut friend_sold,
                &mut proposal_friend_slot,
                &mut proposal_target_agent,
                &mut proposal_need_good,
                &mut proposal_offer_good,
                &mut proposal_quantity,
                &mut proposal_score,
                &mut accepted_mask,
                &mut accepted_quantity,
                &mut changed_agents,
                &mut totals,
                None,
            ) {
                agent_had_session = true;
            }
            if totals.accepted_count == accepted_before {
                break;
            }
        }
        if agent_had_session {
            active_session_agents += 1;
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
        active_session_agents,
    ))
}

#[pyfunction]
fn run_agent_basket_exchange_stage(
    py: Python<'_>,
    runner: PyObject,
    agent_id: usize,
    deal_type: i32,
) -> PyResult<(i32, i32, f64, f64)> {
    let runner = runner.bind(py);
    let state = runner.getattr("state")?;
    let market = runner.getattr("market")?;
    let config = runner.getattr("config")?;

    let goods: usize = config.getattr("goods")?.extract()?;
    let acquaintances: usize = config.getattr("acquaintances")?.extract()?;
    let initial_transparency_for_execution: f64 =
        config.getattr("initial_transparency")?.extract()?;
    let initial_transparency = initial_transparency_for_execution as f32;
    let history: i32 = config.getattr("history")?.extract()?;
    let local_liquidity_stock_bias: f64 = config
        .getattr("experimental_local_liquidity_stock_bias")?
        .extract()?;
    let local_liquidity_min_sales: f64 = config
        .getattr("experimental_local_liquidity_min_sales")?
        .extract()?;
    let aspirational_stock_target: f64 = config
        .getattr("experimental_aspirational_stock_target")?
        .extract()?;
    let session_candidate_depth: usize = config
        .getattr("experimental_session_candidate_depth")?
        .extract()?;
    let session_disable_offer_prefilter: bool = config
        .getattr("experimental_session_disable_offer_prefilter")?
        .extract()?;
    let session_pairwise_offer_exhaustion: bool = config
        .getattr("experimental_session_pairwise_offer_exhaustion")?
        .extract()?;
    let min_trade_quantity: f64 = config.getattr("min_trade_quantity")?.extract()?;
    let trade_rounding_buffer: f64 = config.getattr("trade_rounding_buffer")?.extract()?;
    let max_attempts = goods * acquaintances;
    let initial_transactions = 2.0_f32;

    if deal_type != CONSUMPTION_DEAL && deal_type != SURPLUS_DEAL {
        return Err(PyValueError::new_err(
            "deal_type must be 1 (surplus) or 2 (consumption)",
        ));
    }

    let mut totals = ExchangeStageTotals::default();
    let mut changed_agents: HashSet<usize> = HashSet::new();

    {
        let market_elastic_need: PyReadonlyArray1<'_, f32> =
            market.getattr("elastic_need")?.extract()?;
        let mut market_periodic_tce_cost: PyReadwriteArray1<'_, f32> =
            market.getattr("periodic_tce_cost")?.extract()?;
        let mut stock: PyReadwriteArray2<'_, f32> = state.getattr("stock")?.extract()?;
        let role: PyReadonlyArray2<'_, i32> = state.getattr("role")?.extract()?;
        let stock_limit: PyReadonlyArray2<'_, f32> = state.getattr("stock_limit")?.extract()?;
        let purchase_price: PyReadonlyArray2<'_, f32> =
            state.getattr("purchase_price")?.extract()?;
        let sales_price: PyReadonlyArray2<'_, f32> = state.getattr("sales_price")?.extract()?;
        let needs_level: PyReadonlyArray1<'_, f32> = state.getattr("needs_level")?.extract()?;
        let mut transparency: PyReadwriteArray3<'_, f32> =
            state.getattr("transparency")?.extract()?;
        let mut friend_id: PyReadwriteArray2<'_, i32> = state.getattr("friend_id")?.extract()?;
        let mut need: PyReadwriteArray2<'_, f32> = state.getattr("need")?.extract()?;
        let recent_production: PyReadonlyArray2<'_, f32> =
            state.getattr("recent_production")?.extract()?;
        let mut recent_sales: PyReadwriteArray2<'_, f32> =
            state.getattr("recent_sales")?.extract()?;
        let mut recent_purchases: PyReadwriteArray2<'_, f32> =
            state.getattr("recent_purchases")?.extract()?;
        let mut sold_this_period: PyReadwriteArray2<'_, f32> =
            state.getattr("sold_this_period")?.extract()?;
        let mut purchased_this_period: PyReadwriteArray2<'_, f32> =
            state.getattr("purchased_this_period")?.extract()?;
        let mut recent_inventory_inflow: PyReadwriteArray2<'_, f32> =
            state.getattr("recent_inventory_inflow")?.extract()?;
        let mut recent_purchase_value: PyReadwriteArray2<'_, f32> =
            state.getattr("recent_purchase_value")?.extract()?;
        let mut recent_sales_value: PyReadwriteArray2<'_, f32> =
            state.getattr("recent_sales_value")?.extract()?;
        let mut recent_inventory_inflow_value: PyReadwriteArray2<'_, f32> =
            state.getattr("recent_inventory_inflow_value")?.extract()?;
        let mut purchase_times: PyReadwriteArray2<'_, i32> =
            state.getattr("purchase_times")?.extract()?;
        let mut sales_times: PyReadwriteArray2<'_, i32> =
            state.getattr("sales_times")?.extract()?;
        let mut sum_period_purchase_value: PyReadwriteArray2<'_, f32> =
            state.getattr("sum_period_purchase_value")?.extract()?;
        let mut sum_period_sales_value: PyReadwriteArray2<'_, f32> =
            state.getattr("sum_period_sales_value")?.extract()?;
        let mut friend_activity: PyReadwriteArray2<'_, f32> =
            state.getattr("friend_activity")?.extract()?;
        let mut friend_purchased: PyReadwriteArray3<'_, f32> =
            state.getattr("friend_purchased")?.extract()?;
        let mut friend_sold: PyReadwriteArray3<'_, f32> =
            state.getattr("friend_sold")?.extract()?;
        let trade = state.getattr("trade")?;
        let mut proposal_friend_slot: PyReadwriteArray1<'_, i32> =
            trade.getattr("proposal_friend_slot")?.extract()?;
        let mut proposal_target_agent: PyReadwriteArray1<'_, i32> =
            trade.getattr("proposal_target_agent")?.extract()?;
        let mut proposal_need_good: PyReadwriteArray1<'_, i32> =
            trade.getattr("proposal_need_good")?.extract()?;
        let mut proposal_offer_good: PyReadwriteArray1<'_, i32> =
            trade.getattr("proposal_offer_good")?.extract()?;
        let mut proposal_quantity: PyReadwriteArray1<'_, f32> =
            trade.getattr("proposal_quantity")?.extract()?;
        let mut proposal_score: PyReadwriteArray1<'_, f32> =
            trade.getattr("proposal_score")?.extract()?;
        let mut accepted_mask: PyReadwriteArray1<'_, bool> =
            trade.getattr("accepted_mask")?.extract()?;
        let mut accepted_quantity: PyReadwriteArray1<'_, f32> =
            trade.getattr("accepted_quantity")?.extract()?;

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
        let mut recent_purchase_value = recent_purchase_value.as_array_mut();
        let mut recent_sales_value = recent_sales_value.as_array_mut();
        let mut recent_inventory_inflow_value = recent_inventory_inflow_value.as_array_mut();
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
        let mut forbidden_offer_by_need = vec![false; goods * goods];
        let agent_friend_ids: Vec<i32> = (0..acquaintances)
            .map(|friend_slot| friend_id[[agent_id, friend_slot]])
            .collect();
        let mut reciprocal_slots = vec![-1_i32; acquaintances];
        let mut has_known_partner = false;
        for friend_slot in 0..acquaintances {
            let fid = agent_friend_ids[friend_slot];
            if fid >= 0 {
                has_known_partner = true;
                reciprocal_slots[friend_slot] =
                    find_friend_slot_scan_internal(friend_id.view(), fid as usize, agent_id as i32);
            }
        }
        if !has_known_partner {
            return Ok((0, 0, 0.0, 0.0));
        }
        let mut attempts = 0usize;

        while attempts < max_attempts {
            let candidates = plan_agent_basket_candidates_from_cached_links(
                agent_id,
                deal_type,
                goods,
                acquaintances,
                initial_transparency,
                history,
                local_liquidity_stock_bias,
                local_liquidity_min_sales,
                aspirational_stock_target,
                &forbidden_offer_by_need,
                elastic_need,
                stock.view(),
                role,
                stock_limit,
                purchase_price,
                sales_price,
                needs_level,
                transparency.view(),
                ArrayView1::from(&agent_friend_ids[..]),
                ArrayView1::from(&reciprocal_slots[..]),
                need.view(),
                recent_sales.view(),
                recent_purchases.view(),
                recent_inventory_inflow.view(),
                recent_production,
                friend_id.view(),
                friend_sold.view(),
                session_candidate_depth,
                session_disable_offer_prefilter,
            );
            if candidates.is_empty() {
                break;
            }

            let mut executed_any = false;
            let mut changed_forbidden = false;
            let mut processed_candidate = false;
            for candidate in candidates {
                if attempts >= max_attempts {
                    break;
                }
                if candidate.need_good < 0 || candidate.offer_good < 0 {
                    continue;
                }
                let need_good = candidate.need_good as usize;
                let offer_good = candidate.offer_good as usize;
                if need_good >= goods || offer_good >= goods {
                    continue;
                }
                let forbidden_index = (need_good * goods) + offer_good;
                if forbidden_offer_by_need[forbidden_index] {
                    continue;
                }

                let max_need = basket_stage_max_need(
                    agent_id,
                    need_good,
                    deal_type,
                    history,
                    local_liquidity_stock_bias,
                    local_liquidity_min_sales,
                    aspirational_stock_target,
                    elastic_need,
                    stock.view(),
                    stock_limit,
                    needs_level,
                    need.view(),
                    recent_sales.view(),
                    recent_purchases.view(),
                    recent_inventory_inflow.view(),
                    recent_production,
                    friend_id.view(),
                    friend_sold.view(),
                    transparency.view(),
                );
                if max_need <= 0.0 {
                    continue;
                }

                processed_candidate = true;
                totals.proposed_count += 1;
                let plan = plan_specific_exchange_candidate(
                    agent_id,
                    need_good,
                    initial_transparency_for_execution,
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
                    friend_id.view(),
                    candidate,
                );
                attempts += 1;

                let Some(plan) = plan else {
                    if !forbidden_offer_by_need[forbidden_index] {
                        forbidden_offer_by_need[forbidden_index] = true;
                        changed_forbidden = true;
                    }
                    if session_candidate_depth <= 1 {
                        break;
                    }
                    continue;
                };
                if plan.reason_code != PLAN_OK {
                    if !forbidden_offer_by_need[forbidden_index] {
                        forbidden_offer_by_need[forbidden_index] = true;
                        changed_forbidden = true;
                    }
                    if session_candidate_depth <= 1 {
                        break;
                    }
                    continue;
                }

                let friend_idx = plan.candidate.friend_id as usize;
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
                    reciprocal_slots[friend_slot] = ensured_slot as i32;
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

                let my_need_purchase_price = purchase_price[[agent_id, need_good]];
                let my_offer_sales_price = sales_price[[agent_id, offer_good]];
                let friend_need_sales_price = sales_price[[friend_idx, need_good]];
                let friend_offer_purchase_price = purchase_price[[friend_idx, offer_good]];
                let exchange_value_to_me = exchange_value_to_me_like_numpy_float32(
                    my_need_purchase_price,
                    my_offer_sales_price,
                    plan.switch_average,
                    plan.receiving_transparency,
                );
                let exchange_value_to_friend = exchange_value_to_friend_like_numpy_float32(
                    friend_offer_purchase_price,
                    friend_need_sales_price,
                    plan.switch_average,
                    plan.need_transparency,
                );
                let my_value_correction = exchange_value_to_me.max(EPSILON as f64).sqrt();
                let friend_value_correction = exchange_value_to_friend.max(EPSILON as f64).sqrt();
                let my_purchase_unit_value =
                    div_like_numpy_float32(my_need_purchase_price, my_value_correction);
                let my_sales_unit_value =
                    mul_like_numpy_float32(my_offer_sales_price, my_value_correction);
                let friend_purchase_unit_value =
                    div_like_numpy_float32(friend_offer_purchase_price, friend_value_correction);
                let friend_sales_unit_value =
                    mul_like_numpy_float32(friend_need_sales_price, friend_value_correction);

                add_assign_like_numpy_float32(
                    &mut recent_purchase_value[[agent_id, need_good]],
                    max_exchange * my_purchase_unit_value,
                );
                add_assign_like_numpy_float32(
                    &mut recent_sales_value[[agent_id, offer_good]],
                    friend_gift_in * my_sales_unit_value,
                );
                add_assign_like_numpy_float32(
                    &mut recent_purchase_value[[friend_idx, offer_good]],
                    friend_gift_in * friend_purchase_unit_value,
                );
                add_assign_like_numpy_float32(
                    &mut recent_sales_value[[friend_idx, need_good]],
                    friend_need_out * friend_sales_unit_value,
                );
                if deal_type == SURPLUS_DEAL {
                    add_assign_like_numpy_float32(
                        &mut recent_inventory_inflow_value[[agent_id, need_good]],
                        max_exchange * my_purchase_unit_value,
                    );
                }
                add_assign_like_numpy_float32(
                    &mut recent_inventory_inflow_value[[friend_idx, offer_good]],
                    friend_gift_in * friend_purchase_unit_value,
                );

                if my_value_correction > 1.0 && friend_value_correction > 1.0 {
                    add_assign_like_numpy_float32(
                        &mut sum_period_purchase_value[[agent_id, need_good]],
                        my_purchase_unit_value,
                    );
                    add_assign_like_numpy_float32(
                        &mut sum_period_sales_value[[agent_id, offer_good]],
                        my_sales_unit_value,
                    );
                    add_assign_like_numpy_float32(
                        &mut sum_period_purchase_value[[friend_idx, offer_good]],
                        friend_purchase_unit_value,
                    );
                    add_assign_like_numpy_float32(
                        &mut sum_period_sales_value[[friend_idx, need_good]],
                        friend_sales_unit_value,
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
                executed_any = true;

                let exhausted_gift =
                    remaining_need_after_trade >= 1.0 && stock[[agent_id, offer_good]] < 1.0;
                if !session_pairwise_offer_exhaustion
                    && offer_good_exhausted_for_agent(
                        agent_id,
                        offer_good,
                        elastic_need,
                        stock.view(),
                        needs_level,
                    )
                {
                    if forbid_offer_good_for_all_needs(
                        &mut forbidden_offer_by_need,
                        goods,
                        offer_good,
                    ) {
                        changed_forbidden = true;
                    }
                } else if exhausted_gift && !forbidden_offer_by_need[forbidden_index] {
                    forbidden_offer_by_need[forbidden_index] = true;
                    changed_forbidden = true;
                }

                // Planning may score the whole local basket, but committing a
                // trade changes inventory and partner capacity. Replan after
                // one decision so the basket path remains a search
                // optimization, not a simultaneous multi-trade institution.
                break;
            }

            if !processed_candidate || (!executed_any && !changed_forbidden) {
                break;
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

fn run_full_native_agent_basket_cycle(
    engine: &Bound<'_, PyAny>,
    state: &Bound<'_, PyAny>,
    market: &Bound<'_, PyAny>,
    config: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let population: usize = config.getattr("population")?.extract()?;
    let goods: usize = config.getattr("goods")?.extract()?;
    let acquaintances: usize = config.getattr("acquaintances")?.extract()?;
    let cycle: usize = engine.getattr("cycle")?.extract()?;
    let seed_raw: i64 = config.getattr("seed")?.extract()?;
    let seed = seed_raw.max(0) as u64;
    let period_length: f32 = config.getattr("cycle_time_budget")?.extract()?;
    let history: i32 = config.getattr("history")?.extract()?;
    let basic_round_elastic: bool = config.getattr("basic_round_elastic")?.extract()?;
    let stock_limit_multiplier: f32 = config.getattr("stock_limit_multiplier")?.extract()?;
    let max_needs_increase: f32 = config.getattr("max_needs_increase")?.extract()?;
    let max_needs_reduction: f32 = config.getattr("max_needs_reduction")?.extract()?;
    let small_needs_increase: f32 = config.getattr("small_needs_increase")?.extract()?;
    let lifestyle_promotion_threshold: f32 =
        config.getattr("lifestyle_promotion_threshold")?.extract()?;
    let small_needs_reduction: f32 = config.getattr("small_needs_reduction")?.extract()?;
    let switch_time: f32 = config.getattr("switch_time")?.extract()?;
    let initial_efficiency: f64 = config.getattr("initial_efficiency")?.extract()?;
    let gifted_efficiency_floor: f64 = config.getattr("gifted_efficiency_floor")?.extract()?;
    let initial_transparency_for_execution: f64 =
        config.getattr("initial_transparency")?.extract()?;
    let initial_transparency = initial_transparency_for_execution as f32;
    let activity_discount: f64 = config.getattr("activity_discount")?.extract()?;
    let spoilage_rate: f64 = config.getattr("spoilage_rate")?.extract()?;
    let stock_spoil_threshold: f64 = config.getattr("stock_spoil_threshold")?.extract()?;
    let price_reduction: f64 = config.getattr("price_reduction")?.extract()?;
    let price_hike_f32: f32 = config.getattr("price_hike")?.extract()?;
    let price_hike = price_hike_f32 as f64;
    let price_leap: f64 = config.getattr("price_leap")?.extract()?;
    let min_trade_quantity: f64 = config.getattr("min_trade_quantity")?.extract()?;
    let trade_rounding_buffer: f64 = config.getattr("trade_rounding_buffer")?.extract()?;
    let max_stocklimit_decrease: f64 = config.getattr("max_stocklimit_decrease")?.extract()?;
    let max_stocklimit_increase: f64 = config.getattr("max_stocklimit_increase")?.extract()?;
    let max_efficiency_downgrade: f64 = config.getattr("max_efficiency_downgrade")?.extract()?;
    let max_efficiency_upgrade: f64 = config.getattr("max_efficiency_upgrade")?.extract()?;
    let local_liquidity_stock_bias: f64 = config
        .getattr("experimental_local_liquidity_stock_bias")?
        .extract()?;
    let local_liquidity_min_sales: f64 = config
        .getattr("experimental_local_liquidity_min_sales")?
        .extract()?;
    let aspirational_stock_target: f64 = config
        .getattr("experimental_aspirational_stock_target")?
        .extract()?;
    let session_candidate_depth: usize = config
        .getattr("experimental_session_candidate_depth")?
        .extract()?;
    // Keep the direct per-agent basket branch selectable: long phenomenon
    // runs use the config flag to decide whether a local basket is replayed
    // after every trade or only revalidated from the existing shopping list.
    let session_replan_after_trade: bool = config
        .getattr("experimental_session_replan_after_trade")?
        .extract()?;
    let session_disable_replan_cache: bool = config
        .getattr("experimental_session_disable_replan_cache")?
        .extract()?;
    let session_disable_offer_prefilter: bool = config
        .getattr("experimental_session_disable_offer_prefilter")?
        .extract()?;
    let session_pairwise_offer_exhaustion: bool = config
        .getattr("experimental_session_pairwise_offer_exhaustion")?
        .extract()?;
    let use_value_price_floor_fraction: f64 = config
        .getattr("use_value_price_floor_fraction")?
        .extract()?;
    let legacy_price_floor: Option<f64> = config.getattr("legacy_price_floor")?.extract()?;
    let max_attempts = goods.saturating_mul(acquaintances);
    let initial_transactions = 2.0_f32;

    let market_elastic_need: PyReadonlyArray1<'_, f32> =
        market.getattr("elastic_need")?.extract()?;
    let mut market_periodic_tce_cost: PyReadwriteArray1<'_, f32> =
        market.getattr("periodic_tce_cost")?.extract()?;
    let mut market_periodic_spoilage: PyReadwriteArray1<'_, f32> =
        market.getattr("periodic_spoilage")?.extract()?;
    let base_need: PyReadonlyArray2<'_, f32> = state.getattr("base_need")?.extract()?;
    let mut need: PyReadwriteArray2<'_, f32> = state.getattr("need")?.extract()?;
    let mut stock: PyReadwriteArray2<'_, f32> = state.getattr("stock")?.extract()?;
    let mut stock_limit: PyReadwriteArray2<'_, f32> = state.getattr("stock_limit")?.extract()?;
    let mut previous_stock_limit: PyReadwriteArray2<'_, f32> =
        state.getattr("previous_stock_limit")?.extract()?;
    let mut efficiency: PyReadwriteArray2<'_, f32> = state.getattr("efficiency")?.extract()?;
    let mut learned_efficiency: PyReadwriteArray2<'_, f32> =
        state.getattr("learned_efficiency")?.extract()?;
    let mut purchase_price: PyReadwriteArray2<'_, f32> =
        state.getattr("purchase_price")?.extract()?;
    let mut sales_price: PyReadwriteArray2<'_, f32> = state.getattr("sales_price")?.extract()?;
    let mut purchase_times: PyReadwriteArray2<'_, i32> =
        state.getattr("purchase_times")?.extract()?;
    let mut sales_times: PyReadwriteArray2<'_, i32> = state.getattr("sales_times")?.extract()?;
    let mut sum_period_purchase_value: PyReadwriteArray2<'_, f32> =
        state.getattr("sum_period_purchase_value")?.extract()?;
    let mut sum_period_sales_value: PyReadwriteArray2<'_, f32> =
        state.getattr("sum_period_sales_value")?.extract()?;
    let mut recent_production: PyReadwriteArray2<'_, f32> =
        state.getattr("recent_production")?.extract()?;
    let mut recent_sales: PyReadwriteArray2<'_, f32> = state.getattr("recent_sales")?.extract()?;
    let mut recent_purchases: PyReadwriteArray2<'_, f32> =
        state.getattr("recent_purchases")?.extract()?;
    let mut recent_inventory_inflow: PyReadwriteArray2<'_, f32> =
        state.getattr("recent_inventory_inflow")?.extract()?;
    let mut recent_purchase_value: PyReadwriteArray2<'_, f32> =
        state.getattr("recent_purchase_value")?.extract()?;
    let mut recent_sales_value: PyReadwriteArray2<'_, f32> =
        state.getattr("recent_sales_value")?.extract()?;
    let mut recent_inventory_inflow_value: PyReadwriteArray2<'_, f32> =
        state.getattr("recent_inventory_inflow_value")?.extract()?;
    let mut produced_this_period: PyReadwriteArray2<'_, f32> =
        state.getattr("produced_this_period")?.extract()?;
    let mut produced_last_period: PyReadwriteArray2<'_, f32> =
        state.getattr("produced_last_period")?.extract()?;
    let mut sold_this_period: PyReadwriteArray2<'_, f32> =
        state.getattr("sold_this_period")?.extract()?;
    let mut sold_last_period: PyReadwriteArray2<'_, f32> =
        state.getattr("sold_last_period")?.extract()?;
    let mut purchased_this_period: PyReadwriteArray2<'_, f32> =
        state.getattr("purchased_this_period")?.extract()?;
    let mut purchased_last_period: PyReadwriteArray2<'_, f32> =
        state.getattr("purchased_last_period")?.extract()?;
    let mut spoilage: PyReadwriteArray2<'_, f32> = state.getattr("spoilage")?.extract()?;
    let mut periodic_spoilage: PyReadwriteArray1<'_, f32> =
        state.getattr("periodic_spoilage")?.extract()?;
    let talent_mask: PyReadonlyArray2<'_, f32> = state.getattr("talent_mask")?.extract()?;
    let mut role: PyReadwriteArray2<'_, i32> = state.getattr("role")?.extract()?;
    let mut time_remaining: PyReadwriteArray1<'_, f32> =
        state.getattr("time_remaining")?.extract()?;
    let mut timeout: PyReadwriteArray1<'_, i32> = state.getattr("timeout")?.extract()?;
    let mut period_failure: PyReadwriteArray1<'_, bool> =
        state.getattr("period_failure")?.extract()?;
    let mut period_time_debt: PyReadwriteArray1<'_, f32> =
        state.getattr("period_time_debt")?.extract()?;
    let mut needs_level: PyReadwriteArray1<'_, f32> = state.getattr("needs_level")?.extract()?;
    let mut recent_needs_increment: PyReadwriteArray1<'_, f32> =
        state.getattr("recent_needs_increment")?.extract()?;
    let mut transparency: PyReadwriteArray3<'_, f32> = state.getattr("transparency")?.extract()?;
    let mut friend_id: PyReadwriteArray2<'_, i32> = state.getattr("friend_id")?.extract()?;
    let mut friend_activity: PyReadwriteArray2<'_, f32> =
        state.getattr("friend_activity")?.extract()?;
    let mut friend_purchased: PyReadwriteArray3<'_, f32> =
        state.getattr("friend_purchased")?.extract()?;
    let mut friend_sold: PyReadwriteArray3<'_, f32> = state.getattr("friend_sold")?.extract()?;
    let trade = state.getattr("trade")?;
    let mut proposal_friend_slot: PyReadwriteArray1<'_, i32> =
        trade.getattr("proposal_friend_slot")?.extract()?;
    let mut proposal_target_agent: PyReadwriteArray1<'_, i32> =
        trade.getattr("proposal_target_agent")?.extract()?;
    let mut proposal_need_good: PyReadwriteArray1<'_, i32> =
        trade.getattr("proposal_need_good")?.extract()?;
    let mut proposal_offer_good: PyReadwriteArray1<'_, i32> =
        trade.getattr("proposal_offer_good")?.extract()?;
    let mut proposal_quantity: PyReadwriteArray1<'_, f32> =
        trade.getattr("proposal_quantity")?.extract()?;
    let mut proposal_score: PyReadwriteArray1<'_, f32> =
        trade.getattr("proposal_score")?.extract()?;
    let mut accepted_mask: PyReadwriteArray1<'_, bool> =
        trade.getattr("accepted_mask")?.extract()?;
    let mut accepted_quantity: PyReadwriteArray1<'_, f32> =
        trade.getattr("accepted_quantity")?.extract()?;

    let elastic_need = market_elastic_need.as_array();
    let mut market_periodic_tce_cost = market_periodic_tce_cost.as_array_mut();
    let mut market_periodic_spoilage = market_periodic_spoilage.as_array_mut();
    let base_need = base_need.as_array();
    let mut need = need.as_array_mut();
    let mut stock = stock.as_array_mut();
    let mut stock_limit = stock_limit.as_array_mut();
    let mut previous_stock_limit = previous_stock_limit.as_array_mut();
    let mut efficiency = efficiency.as_array_mut();
    let mut learned_efficiency = learned_efficiency.as_array_mut();
    let mut purchase_price = purchase_price.as_array_mut();
    let mut sales_price = sales_price.as_array_mut();
    let mut purchase_times = purchase_times.as_array_mut();
    let mut sales_times = sales_times.as_array_mut();
    let mut sum_period_purchase_value = sum_period_purchase_value.as_array_mut();
    let mut sum_period_sales_value = sum_period_sales_value.as_array_mut();
    let mut recent_production = recent_production.as_array_mut();
    let mut recent_sales = recent_sales.as_array_mut();
    let mut recent_purchases = recent_purchases.as_array_mut();
    let mut recent_inventory_inflow = recent_inventory_inflow.as_array_mut();
    let mut recent_purchase_value = recent_purchase_value.as_array_mut();
    let mut recent_sales_value = recent_sales_value.as_array_mut();
    let mut recent_inventory_inflow_value = recent_inventory_inflow_value.as_array_mut();
    let mut produced_this_period = produced_this_period.as_array_mut();
    let mut produced_last_period = produced_last_period.as_array_mut();
    let mut sold_this_period = sold_this_period.as_array_mut();
    let mut sold_last_period = sold_last_period.as_array_mut();
    let mut purchased_this_period = purchased_this_period.as_array_mut();
    let mut purchased_last_period = purchased_last_period.as_array_mut();
    let mut spoilage = spoilage.as_array_mut();
    let mut periodic_spoilage = periodic_spoilage.as_array_mut();
    let talent_mask = talent_mask.as_array();
    let mut role = role.as_array_mut();
    let mut time_remaining = time_remaining.as_array_mut();
    let mut timeout = timeout.as_array_mut();
    let mut period_failure = period_failure.as_array_mut();
    let mut period_time_debt = period_time_debt.as_array_mut();
    let mut needs_level = needs_level.as_array_mut();
    let mut recent_needs_increment = recent_needs_increment.as_array_mut();
    let mut transparency = transparency.as_array_mut();
    let mut friend_id = friend_id.as_array_mut();
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

    let mut totals = NativeStageTotals::default();
    let mut exchange_totals = ExchangeStageTotals::default();
    let mut changed_agents: HashSet<usize> = HashSet::new();
    let mut basket_profile = BasketProfile::from_env();

    for agent_id in 0..population {
        let (cycle_need_total, stock_consumed_total) = prepare_agent_for_consumption_internal(
            agent_id,
            goods,
            period_length,
            history,
            basic_round_elastic,
            stock_limit_multiplier,
            max_needs_increase,
            max_needs_reduction,
            small_needs_increase,
            lifestyle_promotion_threshold,
            small_needs_reduction,
            base_need,
            need.view_mut(),
            stock.view_mut(),
            purchase_price.view(),
            sales_price.view(),
            purchased_last_period.view(),
            recent_sales.view(),
            sold_this_period.view(),
            sold_last_period.view(),
            recent_purchases.view(),
            efficiency.view(),
            period_failure.view(),
            period_time_debt.view(),
            needs_level.view_mut(),
            recent_needs_increment.view_mut(),
            elastic_need,
        );
        totals.cycle_need_total += cycle_need_total;
        totals.stock_consumption_total += stock_consumed_total;

        run_basket_session_internal(
            agent_id,
            CONSUMPTION_DEAL,
            goods,
            acquaintances,
            initial_transparency_for_execution,
            initial_transparency,
            history,
            local_liquidity_stock_bias,
            local_liquidity_min_sales,
            aspirational_stock_target,
            session_replan_after_trade,
            session_disable_replan_cache,
            session_disable_offer_prefilter,
            session_pairwise_offer_exhaustion,
            session_candidate_depth,
            min_trade_quantity,
            trade_rounding_buffer,
            max_attempts,
            initial_transactions,
            elastic_need,
            &mut market_periodic_tce_cost,
            &mut stock,
            role.view(),
            stock_limit.view(),
            purchase_price.view(),
            sales_price.view(),
            needs_level.view(),
            &mut transparency,
            &mut friend_id,
            &mut need,
            recent_production.view(),
            &mut recent_sales,
            &mut recent_purchases,
            &mut sold_this_period,
            &mut purchased_this_period,
            &mut recent_inventory_inflow,
            &mut recent_purchase_value,
            &mut recent_sales_value,
            &mut recent_inventory_inflow_value,
            &mut purchase_times,
            &mut sales_times,
            &mut sum_period_purchase_value,
            &mut sum_period_sales_value,
            &mut friend_activity,
            &mut friend_purchased,
            &mut friend_sold,
            &mut proposal_friend_slot,
            &mut proposal_target_agent,
            &mut proposal_need_good,
            &mut proposal_offer_good,
            &mut proposal_quantity,
            &mut proposal_score,
            &mut accepted_mask,
            &mut accepted_quantity,
            &mut changed_agents,
            &mut exchange_totals,
            basket_profile
                .as_mut()
                .map(|profile| profile as &mut BasketProfile),
        );

        let produced_need_total = produce_need_internal(
            agent_id,
            goods,
            efficiency.view(),
            need.view_mut(),
            time_remaining.view_mut(),
            recent_production.view_mut(),
            produced_this_period.view_mut(),
            timeout.view_mut(),
        );
        totals.production_total += produced_need_total;

        let mut pending_need = 0.0_f32;
        for good_id in 0..goods {
            pending_need += need[[agent_id, good_id]];
        }
        period_failure[agent_id] = time_remaining[agent_id] < 0.0 || pending_need > EPSILON;

        add_random_friend_internal(
            agent_id,
            population,
            acquaintances,
            goods,
            seed,
            cycle,
            initial_transparency,
            initial_transactions,
            &mut friend_id,
            &mut friend_activity,
            &mut friend_purchased,
            &mut friend_sold,
            &mut transparency,
        );

        if period_time_debt[agent_id] < 0.0 {
            let half_debt = period_time_debt[agent_id] / 2.0;
            time_remaining[agent_id] += half_debt;
            period_time_debt[agent_id] = half_debt;
        }

        let produced_surplus_total = surplus_production_internal(
            agent_id,
            goods,
            switch_time,
            stock_limit_multiplier,
            price_hike_f32,
            base_need,
            stock.view_mut(),
            stock_limit.view(),
            talent_mask,
            purchase_times.view(),
            efficiency.view(),
            sales_price.view(),
            time_remaining.view_mut(),
            recent_production.view_mut(),
            produced_this_period.view_mut(),
        );
        totals.production_total += produced_surplus_total;
        totals.surplus_output_total += produced_surplus_total;

        run_basket_session_internal(
            agent_id,
            SURPLUS_DEAL,
            goods,
            acquaintances,
            initial_transparency_for_execution,
            initial_transparency,
            history,
            local_liquidity_stock_bias,
            local_liquidity_min_sales,
            aspirational_stock_target,
            session_replan_after_trade,
            session_disable_replan_cache,
            session_disable_offer_prefilter,
            session_pairwise_offer_exhaustion,
            session_candidate_depth,
            min_trade_quantity,
            trade_rounding_buffer,
            max_attempts,
            initial_transactions,
            elastic_need,
            &mut market_periodic_tce_cost,
            &mut stock,
            role.view(),
            stock_limit.view(),
            purchase_price.view(),
            sales_price.view(),
            needs_level.view(),
            &mut transparency,
            &mut friend_id,
            &mut need,
            recent_production.view(),
            &mut recent_sales,
            &mut recent_purchases,
            &mut sold_this_period,
            &mut purchased_this_period,
            &mut recent_inventory_inflow,
            &mut recent_purchase_value,
            &mut recent_sales_value,
            &mut recent_inventory_inflow_value,
            &mut purchase_times,
            &mut sales_times,
            &mut sum_period_purchase_value,
            &mut sum_period_sales_value,
            &mut friend_activity,
            &mut friend_purchased,
            &mut friend_sold,
            &mut proposal_friend_slot,
            &mut proposal_target_agent,
            &mut proposal_need_good,
            &mut proposal_offer_good,
            &mut proposal_quantity,
            &mut proposal_score,
            &mut accepted_mask,
            &mut accepted_quantity,
            &mut changed_agents,
            &mut exchange_totals,
            basket_profile
                .as_mut()
                .map(|profile| profile as &mut BasketProfile),
        );

        period_time_debt[agent_id] += time_remaining[agent_id];
        if period_time_debt[agent_id] > 0.0 {
            period_time_debt[agent_id] = 0.0;
        }

        let produced_leisure_total = leisure_production_internal(
            agent_id,
            goods,
            stock.view_mut(),
            stock_limit.view(),
            talent_mask,
            purchase_price.view(),
            time_remaining.view_mut(),
            recent_production.view_mut(),
            produced_this_period.view_mut(),
        );
        totals.production_total += produced_leisure_total;
        totals.surplus_output_total += produced_leisure_total;

        end_agent_period_internal(
            agent_id,
            cycle,
            goods,
            acquaintances,
            history,
            initial_efficiency,
            gifted_efficiency_floor,
            initial_transparency_for_execution,
            stock_limit_multiplier as f64,
            activity_discount,
            spoilage_rate,
            stock_spoil_threshold,
            price_reduction,
            price_hike,
            price_leap,
            min_trade_quantity,
            max_stocklimit_decrease,
            max_stocklimit_increase,
            max_efficiency_downgrade,
            max_efficiency_upgrade,
            base_need,
            stock.view_mut(),
            stock_limit.view_mut(),
            previous_stock_limit.view_mut(),
            efficiency.view_mut(),
            learned_efficiency.view_mut(),
            recent_production.view_mut(),
            recent_sales.view_mut(),
            recent_purchases.view_mut(),
            recent_inventory_inflow.view_mut(),
            recent_purchase_value.view_mut(),
            recent_sales_value.view_mut(),
            recent_inventory_inflow_value.view_mut(),
            produced_this_period.view_mut(),
            produced_last_period.view_mut(),
            sold_this_period.view_mut(),
            sold_last_period.view_mut(),
            purchased_this_period.view_mut(),
            purchased_last_period.view_mut(),
            purchase_times.view_mut(),
            sales_times.view_mut(),
            sum_period_purchase_value.view_mut(),
            sum_period_sales_value.view_mut(),
            spoilage.view_mut(),
            periodic_spoilage.view_mut(),
            talent_mask,
            role.view_mut(),
            purchase_price.view_mut(),
            sales_price.view_mut(),
            friend_activity.view_mut(),
            friend_purchased.view(),
            transparency.view_mut(),
            needs_level.view(),
            elastic_need,
            market_periodic_spoilage.view_mut(),
            use_value_price_floor_fraction,
            legacy_price_floor,
        );
    }

    add_engine_metric(engine, "_cycle_need_total", totals.cycle_need_total)?;
    add_engine_metric(engine, "_production_total", totals.production_total)?;
    add_engine_metric(engine, "_surplus_output_total", totals.surplus_output_total)?;
    add_engine_metric(
        engine,
        "_stock_consumption_total",
        totals.stock_consumption_total,
    )?;
    add_engine_metric(
        engine,
        "_leisure_extra_need_total",
        totals.leisure_extra_need_total,
    )?;
    add_trade_stage_metrics(
        engine,
        exchange_totals.proposed_count,
        exchange_totals.accepted_count,
        exchange_totals.accepted_volume,
        exchange_totals.inventory_trade_volume,
    )?;
    if let Some(profile) = basket_profile.as_ref() {
        profile.print(cycle, population, goods, acquaintances);
    }
    Ok(())
}

#[pyfunction]
fn run_exact_cycle(py: Python<'_>, engine: PyObject) -> PyResult<()> {
    let engine = engine.bind(py);
    let legacy_cycle = PyModule::import_bound(py, "emergent_money.legacy_cycle")?;
    let runner_type = legacy_cycle.getattr("LegacyCycleRunner")?;
    let runner = runner_type.call1((engine.clone(),))?;

    let uses_hybrid_exchange: bool = runner
        .call_method0("_uses_experimental_hybrid_exchange")?
        .extract()?;
    if uses_hybrid_exchange {
        runner.call_method0("run")?;
        return Ok(());
    }
    let uses_native_exchange_stage: bool = runner
        .call_method0("_uses_experimental_native_exchange_stage")?
        .extract()?;
    let uses_agent_basket_planning: bool = runner
        .call_method0("_uses_experimental_agent_basket_planning")?
        .extract()?;

    runner.call_method0("_reset_cycle_state")?;
    let config = runner.getattr("config")?;
    let state = runner.getattr("state")?;
    let market = runner.getattr("market")?;
    let uses_native_stage_math: bool = config
        .getattr("experimental_native_stage_math")?
        .extract()?;
    if uses_agent_basket_planning && uses_native_stage_math {
        run_full_native_agent_basket_cycle(engine, &state, &market, &config)?;
        runner.call_method0("_finalize_cycle_after_agent_loop")?;
        return Ok(());
    }
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
    let max_leisure_extra_multiplier: f32 =
        config.getattr("max_leisure_extra_multiplier")?.extract()?;

    let mut totals = NativeStageTotals::default();
    for agent_id in 0..population {
        run_agent_cycle_owned(
            py,
            engine,
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
            uses_native_exchange_stage,
            uses_agent_basket_planning,
            &mut totals,
        )?;
    }

    add_engine_metric(engine, "_cycle_need_total", totals.cycle_need_total)?;
    add_engine_metric(engine, "_production_total", totals.production_total)?;
    add_engine_metric(engine, "_surplus_output_total", totals.surplus_output_total)?;
    add_engine_metric(
        engine,
        "_stock_consumption_total",
        totals.stock_consumption_total,
    )?;
    add_engine_metric(
        engine,
        "_leisure_extra_need_total",
        totals.leisure_extra_need_total,
    )?;
    runner.call_method0("_finalize_cycle_after_agent_loop")?;
    Ok(())
}

#[pymodule]
fn _legacy_native_search(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(find_best_exchange, module)?)?;
    module.add_function(wrap_pyfunction!(plan_best_exchange, module)?)?;
    module.add_function(wrap_pyfunction!(plan_exchange_basket, module)?)?;
    module.add_function(wrap_pyfunction!(
        plan_parallel_phenomenon_candidates,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(prepare_agent_for_consumption, module)?)?;
    module.add_function(wrap_pyfunction!(produce_need, module)?)?;
    module.add_function(wrap_pyfunction!(prepare_leisure_round, module)?)?;
    module.add_function(wrap_pyfunction!(run_exchange_stage, module)?)?;
    module.add_function(wrap_pyfunction!(
        execute_planned_parallel_phenomenon_batch,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        run_parallel_phenomenon_agent_tail,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(run_parallel_phenomenon_stage, module)?)?;
    module.add_function(wrap_pyfunction!(run_phenomenon_session_stage, module)?)?;
    module.add_function(wrap_pyfunction!(run_agent_basket_exchange_stage, module)?)?;
    module.add_function(wrap_pyfunction!(surplus_production, module)?)?;
    module.add_function(wrap_pyfunction!(leisure_production, module)?)?;
    module.add_function(wrap_pyfunction!(end_agent_period, module)?)?;
    module.add_function(wrap_pyfunction!(run_exact_cycle, module)?)?;
    Ok(())
}
