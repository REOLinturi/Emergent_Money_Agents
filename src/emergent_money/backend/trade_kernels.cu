extern "C" __global__
void resolve_trade_proposals_kernel(
    const float* stock,
    const float* need,
    const float* stock_limit,
    const int* sorted_proposers,
    const int* target_agent,
    const int* need_good,
    const int* offer_good,
    const float* quantity,
    const float* score,
    const int proposal_count,
    const int goods,
    unsigned char* accepted_mask,
    float* accepted_quantity,
    float* proposer_need_satisfied,
    float* proposer_stock_added,
    float* target_need_satisfied,
    float* target_stock_added,
    float* working_stock,
    float* working_need
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    for (int order_idx = 0; order_idx < proposal_count; ++order_idx) {
        const int proposer = sorted_proposers[order_idx];
        const float current_score = score[proposer];
        if (current_score <= 0.0f) {
            break;
        }

        const int target = target_agent[proposer];
        const int wanted_good = need_good[proposer];
        const int offered_good = offer_good[proposer];
        const float proposed_quantity = quantity[proposer];

        if (target < 0 || wanted_good < 0 || offered_good < 0 || proposer == target) {
            continue;
        }
        if (proposed_quantity <= 0.0f || wanted_good == offered_good) {
            continue;
        }

        const int proposer_offer_idx = proposer * goods + offered_good;
        const int target_need_idx = target * goods + wanted_good;
        const int proposer_wanted_idx = proposer * goods + wanted_good;
        const int target_offer_idx = target * goods + offered_good;

        const float proposer_supply = working_stock[proposer_offer_idx];
        const float target_supply = working_stock[target_need_idx];
        const float proposer_need = working_need[proposer_wanted_idx];
        const float proposer_stock_room = fmaxf(stock_limit[proposer_wanted_idx] - working_stock[proposer_wanted_idx], 0.0f);
        const float proposer_interest = proposer_need + proposer_stock_room;
        const float target_need_value = working_need[target_offer_idx];
        const float target_stock_room = fmaxf(stock_limit[target_offer_idx] - working_stock[target_offer_idx], 0.0f);
        const float target_interest = target_need_value + target_stock_room;

        float executable_quantity = proposed_quantity;
        executable_quantity = fminf(executable_quantity, proposer_supply);
        executable_quantity = fminf(executable_quantity, target_supply);
        executable_quantity = fminf(executable_quantity, proposer_interest);
        executable_quantity = fminf(executable_quantity, target_interest);

        if (executable_quantity <= 0.0f) {
            continue;
        }

        accepted_mask[proposer] = 1;
        accepted_quantity[proposer] = executable_quantity;

        working_stock[proposer_offer_idx] -= executable_quantity;
        working_stock[target_need_idx] -= executable_quantity;

        const float proposer_consumed = fminf(working_need[proposer_wanted_idx], executable_quantity);
        working_need[proposer_wanted_idx] -= proposer_consumed;
        const float proposer_leftover = executable_quantity - proposer_consumed;
        if (proposer_leftover > 0.0f) {
            working_stock[proposer_wanted_idx] += proposer_leftover;
        }

        const float target_consumed = fminf(working_need[target_offer_idx], executable_quantity);
        working_need[target_offer_idx] -= target_consumed;
        const float target_leftover = executable_quantity - target_consumed;
        if (target_leftover > 0.0f) {
            working_stock[target_offer_idx] += target_leftover;
        }

        proposer_need_satisfied[proposer] = proposer_consumed;
        proposer_stock_added[proposer] = proposer_leftover;
        target_need_satisfied[proposer] = target_consumed;
        target_stock_added[proposer] = target_leftover;
    }
}

__device__ __forceinline__ unsigned int mix_u32(const unsigned int value) {
    unsigned int x = value;
    x ^= x >> 16;
    x *= 0x7FEB352Du;
    x ^= x >> 15;
    x *= 0x846CA68Bu;
    x ^= x >> 16;
    return x;
}

__device__ __forceinline__ int sample_contact_candidate_device(
    const unsigned int seed,
    const unsigned int cycle,
    const int agent_id,
    const int attempt,
    const int population
) {
    if (population <= 1) {
        return -1;
    }

    unsigned int mixed = seed;
    mixed ^= (cycle + 1u) * 0x9E3779B9u;
    mixed ^= ((unsigned int)agent_id + 1u) * 0x85EBCA6Bu;
    mixed ^= ((unsigned int)attempt + 1u) * 0xC2B2AE35u;
    mixed = mix_u32(mixed);

    int candidate = (int)(mixed % (unsigned int)(population - 1));
    if (candidate >= agent_id) {
        candidate += 1;
    }
    return candidate;
}

__device__ __forceinline__ bool friend_row_contains(
    const int* friend_row,
    const int acquaintances,
    const int candidate_id
) {
    for (int slot = 0; slot < acquaintances; ++slot) {
        if (friend_row[slot] == candidate_id) {
            return true;
        }
    }
    return false;
}

__device__ __forceinline__ int find_friend_slot_device(const int* friend_row, const int acquaintances, const int agent_id) {
    for (int slot = 0; slot < acquaintances; ++slot) {
        if (friend_row[slot] == agent_id) {
            return slot;
        }
    }
    return -1;
}

__device__ __forceinline__ int select_friend_slot_device(
    const int* friend_row,
    const float* activity_row,
    const int acquaintances
) {
    for (int slot = 0; slot < acquaintances; ++slot) {
        if (friend_row[slot] < 0) {
            return slot;
        }
    }

    int best_slot = 0;
    float best_activity = activity_row[0];
    for (int slot = 1; slot < acquaintances; ++slot) {
        if (activity_row[slot] < best_activity) {
            best_activity = activity_row[slot];
            best_slot = slot;
        }
    }
    return best_slot;
}

extern "C" __global__
void commit_resolved_trades_kernel(
    const int population,
    const int goods,
    const int acquaintances,
    const int* proposal_friend_slot,
    const int* proposal_target_agent,
    const int* proposal_need_good,
    const int* proposal_offer_good,
    const unsigned char* accepted_mask,
    const float* accepted_quantity,
    const float* proposer_stock_added,
    const float* target_stock_added,
    const float initial_transparency,
    float* updated_recent_sales,
    float* updated_recent_purchases,
    float* updated_recent_inventory_inflow,
    int* updated_friend_id,
    float* updated_friend_activity,
    float* updated_transparency
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    for (int proposer = 0; proposer < population; ++proposer) {
        if (accepted_mask[proposer] == 0) {
            continue;
        }

        const int target = proposal_target_agent[proposer];
        const int friend_slot = proposal_friend_slot[proposer];
        const int need_good = proposal_need_good[proposer];
        const int offer_good = proposal_offer_good[proposer];
        const float quantity = accepted_quantity[proposer];

        if (quantity <= 0.0f) {
            continue;
        }

        const int proposer_offer_idx = proposer * goods + offer_good;
        const int target_need_idx = target * goods + need_good;
        const int proposer_need_idx = proposer * goods + need_good;
        const int target_offer_idx = target * goods + offer_good;

        updated_recent_sales[proposer_offer_idx] += quantity;
        updated_recent_purchases[proposer_need_idx] += quantity;
        updated_recent_sales[target_need_idx] += quantity;
        updated_recent_purchases[target_offer_idx] += quantity;
        updated_recent_inventory_inflow[proposer_need_idx] += proposer_stock_added[proposer];
        updated_recent_inventory_inflow[target_offer_idx] += target_stock_added[proposer];

        const float transparency_gain = fminf(0.05f, 0.01f * log1pf(quantity));

        if (friend_slot >= 0 && friend_slot < acquaintances) {
            const int proposer_friend_idx = proposer * acquaintances + friend_slot;
            updated_friend_activity[proposer_friend_idx] += quantity;
            const int proposer_transparency_idx = (proposer_friend_idx * goods) + need_good;
            updated_transparency[proposer_transparency_idx] = fminf(
                1.0f,
                updated_transparency[proposer_transparency_idx] + transparency_gain
            );
        }

        int* target_friend_row = updated_friend_id + (target * acquaintances);
        float* target_activity_row = updated_friend_activity + (target * acquaintances);
        int reciprocal_slot = find_friend_slot_device(target_friend_row, acquaintances, proposer);
        if (reciprocal_slot < 0) {
            reciprocal_slot = select_friend_slot_device(target_friend_row, target_activity_row, acquaintances);
            target_friend_row[reciprocal_slot] = proposer;
            target_activity_row[reciprocal_slot] = 2.0f + quantity;
            float* target_transparency_row = updated_transparency + (((target * acquaintances) + reciprocal_slot) * goods);
            for (int good = 0; good < goods; ++good) {
                target_transparency_row[good] = initial_transparency;
            }
        }

        const int reciprocal_idx = target * acquaintances + reciprocal_slot;
        updated_friend_activity[reciprocal_idx] += quantity;
        const int reciprocal_transparency_idx = (reciprocal_idx * goods) + offer_good;
        updated_transparency[reciprocal_transparency_idx] = fminf(
            1.0f,
            updated_transparency[reciprocal_transparency_idx] + transparency_gain
        );
    }
}

extern "C" __global__
void plan_contact_candidates_kernel(
    const int population,
    const int acquaintances,
    const unsigned int seed,
    const unsigned int cycle,
    const int* friend_id,
    int* candidate_ids
) {
    const int agent_id = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (agent_id >= population) {
        return;
    }

    candidate_ids[agent_id] = -1;
    if (population <= 1) {
        return;
    }

    const int* friend_row = friend_id + (agent_id * acquaintances);
    const int max_attempts = max(2 * acquaintances, 8);

    for (int attempt = 0; attempt < max_attempts; ++attempt) {
        const int candidate_id = sample_contact_candidate_device(seed, cycle, agent_id, attempt, population);
        if (!friend_row_contains(friend_row, acquaintances, candidate_id)) {
            candidate_ids[agent_id] = candidate_id;
            return;
        }
    }

    for (int candidate_id = 0; candidate_id < population; ++candidate_id) {
        if (candidate_id == agent_id) {
            continue;
        }
        if (!friend_row_contains(friend_row, acquaintances, candidate_id)) {
            candidate_ids[agent_id] = candidate_id;
            return;
        }
    }
}

extern "C" __global__
void apply_contact_candidates_kernel(
    const int population,
    const int acquaintances,
    const int goods,
    const int* candidate_ids,
    const float initial_activity,
    const float initial_transparency,
    int* friend_id,
    float* friend_activity,
    float* transparency
) {
    const int agent_id = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (agent_id >= population) {
        return;
    }

    const int candidate_id = candidate_ids[agent_id];
    if (candidate_id < 0) {
        return;
    }

    const int row_offset = agent_id * acquaintances;
    int existing_slot = -1;
    for (int slot = 0; slot < acquaintances; ++slot) {
        if (friend_id[row_offset + slot] == candidate_id) {
            existing_slot = slot;
            break;
        }
    }

    if (existing_slot >= 0) {
        const int activity_idx = row_offset + existing_slot;
        friend_activity[activity_idx] = fmaxf(friend_activity[activity_idx], initial_activity);
        return;
    }

    int target_slot = -1;
    for (int slot = 0; slot < acquaintances; ++slot) {
        if (friend_id[row_offset + slot] < 0) {
            target_slot = slot;
            break;
        }
    }

    if (target_slot < 0) {
        target_slot = 0;
        float lowest_activity = friend_activity[row_offset];
        for (int slot = 1; slot < acquaintances; ++slot) {
            const float current_activity = friend_activity[row_offset + slot];
            if (current_activity < lowest_activity) {
                lowest_activity = current_activity;
                target_slot = slot;
            }
        }
    }

    friend_id[row_offset + target_slot] = candidate_id;
    friend_activity[row_offset + target_slot] = initial_activity;

    const int transparency_offset = ((agent_id * acquaintances) + target_slot) * goods;
    for (int good = 0; good < goods; ++good) {
        transparency[transparency_offset + good] = initial_transparency;
    }
}

extern "C" __global__
void score_trade_block_kernel(
    const int population,
    const int friend_start,
    const int local_friend_count,
    const int need_start,
    const int need_width,
    const int offer_start,
    const int offer_width,
    const float initial_transparency,
    const int* friend_index_block,
    const float* self_interest_need,
    const float* self_stock_offer,
    const float* self_purchase_need,
    const float* self_sales_offer,
    const float* friend_stock_need,
    const float* friend_interest_offer,
    const float* friend_purchase_offer,
    const float* friend_sales_need,
    const float* transparency_need,
    float* best_score,
    int* best_friend_slot,
    int* best_target_agent,
    int* best_need_good,
    int* best_offer_good,
    float* best_quantity
) {
    const int agent_id = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (agent_id >= population) {
        return;
    }

    float local_best_score = best_score[agent_id];
    int local_best_friend_slot = best_friend_slot[agent_id];
    int local_best_target_agent = best_target_agent[agent_id];
    int local_best_need_good = best_need_good[agent_id];
    int local_best_offer_good = best_offer_good[agent_id];
    float local_best_quantity = best_quantity[agent_id];

    for (int local_friend = 0; local_friend < local_friend_count; ++local_friend) {
        const int friend_idx_offset = agent_id * local_friend_count + local_friend;
        const int target_agent_id = friend_index_block[friend_idx_offset];
        if (target_agent_id < 0) {
            continue;
        }

        for (int local_need = 0; local_need < need_width; ++local_need) {
            const int self_need_offset = agent_id * need_width + local_need;
            const float self_interest = self_interest_need[self_need_offset];
            if (self_interest <= 0.0f) {
                continue;
            }

            const int friend_need_offset = (friend_idx_offset * need_width) + local_need;
            const float friend_stock_for_need = friend_stock_need[friend_need_offset];
            if (friend_stock_for_need <= 0.0f) {
                continue;
            }

            const float self_purchase = self_purchase_need[self_need_offset];
            const float friend_sales_for_need = friend_sales_need[friend_need_offset];
            const float transparency = transparency_need[friend_need_offset];

            for (int local_offer = 0; local_offer < offer_width; ++local_offer) {
                if ((need_start + local_need) == (offer_start + local_offer)) {
                    continue;
                }

                const int self_offer_offset = agent_id * offer_width + local_offer;
                const float self_stock = self_stock_offer[self_offer_offset];
                if (self_stock <= 0.0f) {
                    continue;
                }

                const int friend_offer_offset = (friend_idx_offset * offer_width) + local_offer;
                const float friend_interest = friend_interest_offer[friend_offer_offset];
                if (friend_interest <= 0.0f) {
                    continue;
                }

                float quantity = self_interest;
                quantity = fminf(quantity, self_stock);
                quantity = fminf(quantity, friend_stock_for_need);
                quantity = fminf(quantity, friend_interest);
                if (quantity <= 0.0f) {
                    continue;
                }

                const float friend_purchase = friend_purchase_offer[friend_offer_offset];
                const float self_sales = self_sales_offer[self_offer_offset];
                const float friend_term = (friend_purchase / fmaxf(friend_sales_for_need, 1e-6f)) * transparency;
                const float self_term = self_sales / fmaxf(self_purchase * initial_transparency, 1e-6f);
                const float exchange_index = friend_term - self_term;
                if (exchange_index <= 0.0f) {
                    continue;
                }

                const float score = quantity * exchange_index;
                if (score > local_best_score) {
                    local_best_score = score;
                    local_best_friend_slot = friend_start + local_friend;
                    local_best_target_agent = target_agent_id;
                    local_best_need_good = need_start + local_need;
                    local_best_offer_good = offer_start + local_offer;
                    local_best_quantity = quantity;
                }
            }
        }
    }

    best_score[agent_id] = local_best_score;
    best_friend_slot[agent_id] = local_best_friend_slot;
    best_target_agent[agent_id] = local_best_target_agent;
    best_need_good[agent_id] = local_best_need_good;
    best_offer_good[agent_id] = local_best_offer_good;
    best_quantity[agent_id] = local_best_quantity;
}