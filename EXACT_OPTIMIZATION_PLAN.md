## Exact Legacy Optimization Plan

### Purpose

This document defines the stepwise optimization policy for the exact legacy execution path in `legacy_cycle.py`.

The goal is not merely to make the simulation faster. The goal is:

- preserve report-faithful economic mechanics
- preserve the agent-by-agent execution order
- preserve long-run emergent behavior
- reduce runtime only through validated execution improvements

This document is the controlling plan for future optimization work on the exact path.

### Non-Negotiable Constraint

For the exact path, the following may not change without an explicit behavioral revalidation phase:

- agent processing order
- trade opportunity set
- need, stock, and debt update order
- barter profitability rule
- negotiated exchange ratio rule
- transaction-cost waste rule
- stock-limit, learning, price, and transparency updates

Optimization may change how the same work is executed, but not what work is performed.

### Current Empirical Situation

Observed on the current exact CPU path at report scale `3000 / 30 / 100`:

- runtime is in the many-hours class for `1000+` cycles
- growth appears to emerge only after the early transitional phase
- trade volume, utility proxy, and production can rise sharply after the first tens of cycles
- cycle signals begin to appear before `50` cycles

This means the current path is behaviorally promising, but too slow for robust experiment turnover.

### Optimization Policy

Each optimization step must satisfy all of the following before the next step begins:

1. The change is isolated to a narrow part of the exact path.
2. A regression test exists for the optimized behavior.
3. Small exact scenarios remain identical to the pre-change reference behavior.
4. Long-run metrics are checked for obvious drift.
5. The optimization can be turned into a benchmarkable claim.

### Validation Ladder

Every optimization step is validated on three levels.

#### Level 1: local invariants

Examples:

- friend slot lookup returns the same slot as before
- stock never increases from spoilage
- transaction waste remains non-negative
- checkpoint/resume remains valid

#### Level 2: tiny exact scenarios

These must remain identical cycle-by-cycle:

- `2 agents / 2 goods / 1 acquaintance`
- `4-8 agents / small goods / sparse acquaintance graph`
- tests that force barter, spoilage, and leisure-round behavior

Acceptance rule:

- exact same trade counts
- exact same accepted quantities
- exact same market metrics for the scenario

#### Level 3: long-run guardrails

These are not bitwise-equality tests. They are behavioral checks.

Required tracked signals:

- production trend
- utility trend
- trade volume
- TCE waste in time
- spoilage in time
- rare-goods monetary share
- cycle strength and dominant cycle length

Acceptance rule:

- no obvious collapse of growth or cycle formation relative to the immediate pre-optimization baseline

### Phased Roadmap

#### Phase 0: baseline and benchmark harness

Goal:

- make exact-path changes measurable and resumable

Deliverables:

- checkpoint/resume long-run execution
- repeatable report-scale command lines
- stable summary and metric artifacts

Status:

- done

#### Phase 1: remove repeated linear friend-slot scans

Hypothesis:

- a large amount of exact-path time is spent repeatedly scanning the same acquaintance row to find reciprocal slots

Change type:

- purely local data-structure optimization

Allowed change:

- cache per-agent friend-id to slot mappings inside the exact cycle runner
- keep the cache synchronized when friendships are inserted or replaced

Forbidden change:

- no pruning of acquaintances
- no change to friend replacement policy
- no change to reciprocity semantics

Validation:

- unit test for slot lookup after slot replacement
- full regression suite

#### Phase 2: reduce repeated row access and scalar conversion overhead

Hypothesis:

- repeated `self.state[...]` indexing and `float(...)` conversion inside the innermost exact loops adds avoidable Python overhead

Change type:

- local variable caching and row aliasing only

Allowed change:

- bind frequently used per-agent and per-friend rows to local variables
- hoist invariant values out of innermost loops

Forbidden change:

- no vectorized pruning
- no altered tie-breaking

Validation:

- exact small-scenario equivalence
- report-scale one-cycle timing comparison

#### Phase 3: split search into exact-preserving prefilters

Hypothesis:

- many candidate friend/good combinations can be rejected using exact necessary conditions before entering the full profitability calculation

Change type:

- exact-preserving guard reordering

Allowed change:

- hoist strict rejection tests earlier
- reuse computed thresholds inside the same search

Forbidden change:

- no heuristic candidate pruning
- no top-k approximation

Validation:

- exact small-scenario equivalence
- compare trade proposals and accepted trades cycle-by-cycle on fixed seeds

#### Phase 4: native hot-loop port

Trigger:

- only after Phases 1-3 stop producing substantial gains

Candidate targets:

- Rust extension
- Cython / compiled helper
- Numba if exact control remains adequate

Policy:

- port only hot inner loops
- keep Python orchestration as the semantic reference shell

Validation:

- same fixed-seed outputs on small exact scenarios
- same checkpoint/resume semantics

#### Phase 5: conflict-safe hybrid parallelism

Trigger:

- only after the sequential exact path is both stable and well-profiled

Candidate approach:

- construct conflict-free exchange sets
- execute only non-overlapping negotiations in parallel
- preserve barriered state updates

Validation:

- no longer exact bitwise equivalence by default
- must be explicitly compared against the sequential reference over multiple seeds and long runs

### Decision Rule For Rust

Rust is justified only if:

- Phases 1-3 have already removed obvious Python overhead
- the remaining hotspot is still dominated by tight imperative loops
- the port target can be isolated without rewriting the whole model

Rust is not justified merely because the current path is slow.

### Current Status

Completed and accepted:

- Phase 1 is accepted. Per-agent friend-slot indexing preserved exact report-path outputs on the `38 -> 40` checkpoint comparison and reduced runtime by about `11 %`.

Completed and accepted with relaxed tolerance:

- Phase 2 is accepted under a relaxed tolerance for boundary-case differences. The current probe remains very close to the exact reference, with only minor cycle-39 trade-count drift in checkpoint comparison.
- Phase 3 is now accepted under the same tolerance. A strict candidate-offer prefilter in `_find_best_exchange` preserved the same small boundary-case drift while reducing the one-cycle `38 -> 39` checkpoint runtime from roughly `163 s` to roughly `96 s`.
- This tolerance applies only because the user explicitly accepted isolated, economically minor differences at threshold cases.

Phase 4 progress in this environment:

- The exchange-search hotspot is now isolated behind a backend seam with a Python fallback and an optional native attachment point.
- Current profiling result: `_find_best_exchange` dominates the cycle runtime, with `_make_surplus_deals` and `_satisfy_needs_by_exchange` next because they mostly call into that search.
- A compiled backend was not built here because the environment has no Rust, C, C++, CMake, Numba, or Cython toolchain installed.

Phase 5 progress in this environment:

- An off-by-default conflict-safe batching scaffold now exists in `hybrid_batching.py`.
- Exact-runner-side experimental planning and execution hooks now expose seeded consumption and surplus batch plans without changing the default sequential execution path.
- The current scheduler now reserves agents globally across the whole plan, so one experimental plan never schedules the same agent into multiple batches.
- Small validation fixtures compare both the experimental consumption planner and the experimental consumption executor against the sequential exact reference and confirm correct conflict dropping plus matching accepted-trade counts on the tested scenarios.
- The first naive opt-in exact-cycle hook was rejected because it fully prepared all agents before exchange and collapsed trade counts to zero.
- The current opt-in exact-cycle hook now runs in frontier waves: only a bounded proposer window is prepared at once, same-frontier agents are blocked from serving as one another's trade counterparts, and active proposers can re-enter successive micro-waves before they are completed.
- Frontier waves now use the same bounded retry budget as the sequential consumption search (`goods * acquaintances`), which eliminates the replan livelock seen in larger multi-seed drift probes.
- A repeatable comparison harness is now available behind `python -m emergent_money --compare-hybrid-consumption ...` and in `drift_compare.py`, so hybrid probes can be run and saved without ad hoc scripts.
- The exact runner now records lightweight wave-level diagnostics for the experimental consumption frontier: frontier count, wave count, candidate/scheduled/executed exchanges, retry exhaustion, and per-wave rows.
- The compare harness now also aggregates failure-reason counters: `no_candidate` reasons, scheduler conflicts, execution failures, retry exhaustion, and planned/executed exchange quantity.
- On the disjoint fixture, `frontier_size=1` now reproduces the sequential exact result at step level.
- On a small multi-seed drift check (`population=16`, `goods=4`, `acquaintances=4`, `4` cycles, seeds `2009/2011/2013`), `frontier_size=4`, `batch_count=2` matched the sequential result exactly.
- The first reason pass showed that execution failures in the consumption frontier were almost entirely `rounding_buffer_below_min`, while the dominant no-candidate reason was `no_offer_goods`.
- The planner now preflights candidate exchanges against the same threshold logic before scheduling them, so doomed `rounding_buffer_below_min` trades are filtered out instead of being retried as execution failures.
- After that preflight change, execution failures and retry exhaustion dropped to zero on the tested probes.
- The experimental execution path now also reuses the planner's `planned_quantity`, and the planner precomputes base offer goods once per agent instead of rescanning all goods for every need-good candidate.
- On the current `32/6/6/6` five-seed probes, the base-offer precompute preserved outputs exactly and reduced end-to-end compare runtime by about `36 %` relative to the immediately preceding cached-execution version.
- On a five-seed probe (`population=24`, `goods=5`, `acquaintances=5`, `4` cycles, seeds `2009/2011/2013/2015/2017`), `frontier_size=6`, `batch_count=2` still stayed close to sequential: mean deltas remained roughly `-0.4` accepted trades, `-0.64` trade volume, `-0.44` production, `0.0` utility, and `-0.007` rare-money share. The wave summary shifted to about `4.75` waves per cycle with zero execution failures.
- On a larger five-seed probe (`population=32`, `goods=6`, `acquaintances=6`, `6` cycles, seeds `2009/2011/2013/2015/2017`), `frontier_size=8` improved after preflight: mean deltas moved from roughly `-1.4 / -1.71` (trade count / trade volume) to roughly `-1.0 / -0.63`, while utility stayed at about `+0.0015`. Its executed quantity per exchange was about `2.22`.
- Reducing the frontier on that same `32/6/6/6` probe to `frontier_size=4` still produced the best trade-count agreement, now at about `0.0` accepted-trade delta, though trade volume and production still drifted upward (`+1.56` volume, `+3.47` production). Its executed quantity per exchange was about `2.16`, so the remaining positive volume drift is not explained by larger average hybrid trade size alone.
- In the current consumption-only frontier implementation, `batch_count=1` and `batch_count=2` produced identical summaries on the tested `32/6/6/6` probes, so frontier size is currently the dominant drift control knob.
- The latest diagnostic read suggests that residual drift is no longer driven by failed execution plumbing. It is now mostly a frontier-ordering and timing question: many agents simply have no tradable surplus (`no_offer_goods`), and the remaining difference comes from which feasible exchanges the frontier makes possible at the same time and in which cycles that volume gets realized.
- The next frontier experiment is to stop blocking same-frontier agents from serving as one another's partners, while still allowing each agent to participate in at most one committed negotiation per wave. This stays closer to the real-time interpretation of "one decision at a time" and may recover trade volume that is currently deferred or lost during cycles `3-5`.
- The follow-up frontier experiment is to let the scheduler preserve proposer order inside a wave instead of resolving conflicts purely by score. That is closer to the exact agent-by-agent reference and should reduce drift when two feasible negotiations compete for the same partner.
- Allowing same-frontier partners as an opt-in experimental mode reduced the worst negative trade-volume and production deltas at cycles `4-5` on the `32/6/6/6` five-seed probe, but it did not improve the final-cycle drift. Trade-count delta stayed at `-1.0 / 0.0` (`frontier=8 / 4`), while final production drift increased.
- Preserving proposer order inside the scheduler had no effect while same-frontier partners remained blocked, and it made the `allow-frontier-partners` probes materially worse by pushing the largest negative trade-volume and production deltas to cycle `5`. That option therefore remains experimental-only and is not promoted as the current best frontier policy.
- The next frontier-structure experiment is a rolling frontier: only the currently active agents remain in prepared consumption state, retired agents complete their full cycle immediately, and new agents are prepared only when slots free up. This should be closer to the sequential reference than the current fixed frontier blocks.
- The rolling-frontier experiment did not improve the `32/6/6/6` five-seed probe. With `frontier=8` it drifted to about `-1.6` trades and `-1.23` volume; with `frontier=4` it still showed larger negative mid-cycle deltas than the fixed frontier and moved utility/rare-money further away from the sequential reference.
- A smaller fixed frontier is currently the strongest result. On the same `32/6/6/6` five-seed probe, `frontier=2` reduced mean volume drift to about `+0.43` and peak negative volume drift to about `-2.53`, while `frontier=3` preserved zero trade-count drift, reduced mean volume drift to about `+0.21`, and cut the worst negative volume drift to about `-1.53`. Runtime stayed effectively unchanged in these compare-scale probes.
- Current best experimental balance: fixed frontier, blocked same-frontier partners, proposer-order off, and `frontier_size` in the `2-3` range, with `3` currently the best overall compromise between trade-count agreement and reduced mid-cycle drift.
- A dedicated frontier-sweep harness is now the preferred way to map that envelope. It compares multiple frontier sizes under the same seed set and reports both per-frontier summaries, the exact-reference recommendation, and a separate non-trivial recommendation that excludes `frontier=1` and now prioritizes mean/peak trade-volume stability over raw trade-count matching.
- On the first full `frontier=1..8` sweep for the `32/6/6/6` five-seed probe, the heuristic naturally selected `frontier=1` as the exact reference. Among the non-trivial hybrid frontiers at `6` cycles, `frontier=3` looked best: zero mean trade-count drift, about `+0.21` mean volume drift, and the smallest worst-case negative trade-volume drift (about `-1.53`) among the tested frontiers above `1`.
- A longer `12`-cycle follow-up sweep over `frontier=1..4` changed that picture. On the same `32/6/6/6` five-seed probe, `frontier=2` became the best non-trivial candidate: zero mean trade-count drift, essentially zero mean volume drift (`+0.0012`), about `+0.92` mean production drift, and the same peak negative volume drift as before (about `-2.53`). `frontier=3` degraded to about `-0.4` trades and `-1.41` volume over that longer horizon.
- A `24`-cycle follow-up sweep over `frontier=1..4` was noisier and showed that no non-trivial frontier is yet tightly locked to the sequential reference over longer horizons. However, once the non-trivial recommendation was changed to prioritize mean/peak trade-volume stability over raw trade-count matching, `frontier=2` remained the preferred hybrid baseline over `3` and `4`.
- The next hybrid extension is now in place behind an explicit opt-in flag: fixed-frontier exact cycles can route the surplus-deal stage through the same frontier-wave planner after each frontier has been advanced to the surplus phase. The default sequential exact path remains unchanged, and rolling-frontier runs still complete the surplus stage sequentially for now.
- New exact regressions now lock the surplus planner and executor against the sequential reference on both disjoint and conflicting fixtures, and a frontier-1 cycle-level regression confirms that the opt-in consumption+surplus path still reproduces the sequential exact result on a deterministic surplus scenario.
- The first general `32/6/6/6` drift probes with surplus enabled (`6` cycles over `5` seeds, and `12` cycles over `3` seeds) still selected `frontier=2` as the non-trivial recommendation, but they also showed that the surplus stage had not yet activated on those short horizons: the recorded wave diagnostics were still consumption-only.
- A dedicated micro-scenario with cross-frontier partners confirms that the new cycle hook is not dormant. In that engineered `frontier=2` case, the hybrid run recorded both `consumption` and `surplus` waves and matched the sequential reference exactly on accepted trades, trade volume, production, and utility.
- The compare harness now records explicit stage-activation summaries: per-stage cycle counts, per-stage wave counts, seeds with activation, and the first cycle where each stage appears. That makes it possible to distinguish "surplus never activated" from "surplus activated and drift changed" without post-processing logs.
- On a longer `36`-cycle `32/6/6/6` probe over seeds `2009/2011/2013`, `frontier=2` with surplus enabled activated surplus on all three seeds, first appearing at cycles `34`, `28`, and `31`. The same run still drifted materially from sequential (`+2.33` trades, `+34.64` volume, `+65.10` production), so `frontier=2` is not yet a safe long-horizon hybrid setting.
- That same `36`-cycle probe also showed that surplus hybridization is directionally helpful once it activates. The consumption-only `frontier=2` run drifted much further (`+5.33` trades, `+67.16` volume, `+115.79` production), so the surplus stage cut the long-horizon over-trading/over-production drift substantially instead of worsening it.
- Scaling the population did not rescue the frontier-batching idea. On `64/6/6/6` with `frontier=2`, two-seed `36`-cycle probes still drifted badly, but now in the opposite direction (`-7.5` trades, `-128.86` volume, `-204.40` production). Keeping the acquaintance ratio high (`64/12/12`) flipped the sign again and drifted strongly positive (`+10.0` trades, `+107.34` volume, `+127.70` production). Scheduler-conflict counts remained tiny (`3` and `6`), so dense small-network collisions are not the main residual problem.
- A new surplus-only hybrid mode now exists: consumption stays exact sequential while the surplus stage uses conflict-safe frontier waves. It is semantically safe at `frontier=1` and keeps scheduler conflicts at zero on the tested `32/6/6/6` probes, but it still drifts too much at `frontier=2` over `36` cycles (`+5.67` trades, `+28.26` volume, `+62.29` production).
- Runtime measurements on the same probe scale show why frontier batching is no longer the main optimization path. At `32/6/6/6`, `36` cycles, full hybrid was slightly slower than sequential (`0.2507 s` vs `0.2415 s`). At `64/6/6/6`, full hybrid was clearly slower (`0.4947 s` vs `0.4347 s`). Surplus-only hybrid was only modestly faster at `64/6/6/6` (`0.4045 s`), but not enough to justify the remaining drift.
- The Rust path has now moved from design to scaffold. An optional `native/legacy_search` PyO3 crate mirrors the current `_legacy_native_search` module contract, and the Python loader accepts both package-local and top-level native module installs. This is the selected optimization track for the next implementation phase.
- A dedicated native-search verification harness now exists. It captures real `find_best_exchange` calls from the sequential exact cycle, replays them against Python and native backends, reports mismatches, and benchmarks relative throughput. This gives the Rust path an explicit acceptance gate before any long-run use.
- The native path now also covers exchange planning (`plan_best_exchange`) and matches the Python exact cycle in installed-mode tests (`77 passed`). On the focused search/plan replay harness the native backend now reaches about `1.44x` throughput, but whole-cycle speedup is still only about `1.00x-1.03x` on the current `32/6/6` and `64/6/6` probes.
- Updated profiling shows the main hotspot has moved away from exchange search. The largest cumulative costs are now `_end_agent_period`, `_calibrate_friend_transparency`, `_complete_agent_period_after_surplus`, and the remaining request-building path. The next meaningful Rust step is therefore a larger per-agent/post-period port, not another smaller search micro-seam.
- A dedicated `compare-native-cycle` harness now exists. It runs the Python exact cycle and the explicit native exact-cycle entrypoint side by side over the same seeds, compares full engine/state parity after every cycle, and reports the first mismatching cycle plus state field.
- The attempted native `_end_agent_period` port is intentionally not on the default execution path. It produced strong microbench speedups, but exact multi-cycle parity broke around cycle `4` from ulp-scale float drift that later flipped branch-sensitive surplus-production choices.
- This leaves a genuine architecture decision for the next native phase: either port a larger contiguous block that contains both end-of-period updates and the immediately following branch-sensitive surplus logic, or port the entire sequential exact agent cycle. Continuing with smaller scalar-update micro-seams is no longer the preferred path.
- The native cycle backend now exposes three earlier-stage helpers on the accepted experimental path: `prepare_agent_for_consumption`, `produce_need`, and `prepare_leisure_round`. Their direct stage-parity tests now pass against the Python reference.
- An explicit experimental flag `experimental_native_stage_math` is wired into the exact runner. When enabled, those three helpers run natively while the rest of the exact legacy cycle remains on the Python reference path.
- The current acceptance probe is `runs/experimental_native_stage_math_probe_v9.json`: a `32/6/6` ten-cycle probe over seeds `2009/2011/2013` with zero mismatches, zero `recent_needs_increment` drift, exact observed cycle outputs (`production_total`, `accepted_trade_volume`, `utility_proxy_total`), and about `1.09x-1.14x` speedup versus Python on this machine.
- This is now the accepted experimental-native baseline for the exact runner. It remains experimental because the speedup envelope is still modest and only the early preparation / need-production / leisure-prepare stages are covered so far.
- A larger experimental native boundary now exists for exchange/deal execution. `run_exchange_stage` owns the full per-agent consumption or surplus barter loop in Rust, including search/planning, direct stock/need updates, trade bookkeeping, and reciprocal-link insertion with a post-stage friend-slot-map sync.
- This new boundary is intentionally not accepted as an exact slice. Short exact-reference comparisons against the pure Python legacy path stayed very close (`runs/native_exact_reference_exchange_stage_only.json`), but they still showed ulp-scale drift in `engine._inventory_trade_volume` and a rare `sum_period_purchase_value` cell. Longer behavior probes then amplified that into real divergences: `runs/native_behavior_probe_exchange_stage_only_32_6_6_100.json` first flipped accepted trades around cycles `26-32`, and `runs/native_macro_compare_exchange_stage_32_6_6_100.json` showed materially different final production and trade-volume levels by cycle `100`.
- The performance signal is still useful. When the experimental native exchange stage is enabled together with the native whole-cycle bridge, normal `engine.step()` runs now benchmark at about `1.29x` on `runs/native_engine_step_exchange_stage_wholecycle_32_6_6_40.json`. That is better than the accepted safe slice, but not yet good enough to justify the observed long-horizon drift.
- Direct Rust stage ports for `surplus_production` and `leisure_production` still exist and pass isolated parity tests, but wiring them into the experimental exact-cycle path remains rejected. The larger lesson from the rejected probes is unchanged: the next safe native boundary is not `surplus_production` alone, and the new evidence suggests the same for exchange-stage ownership unless the post-trade price/value bookkeeping can be made more exact. The next serious candidate remains a larger contiguous exact block that owns exchange/deal execution together with the immediately downstream price/value update logic, or else the whole sequential exact core.
- A dedicated Stage A isolation harness now exists in `native_exchange_stage_compare.py`. It mirrors the post-period harness pattern: the live exact run stays on the accepted path, each consumption/surplus exchange stage is cloned, the native exchange candidate runs on the clone, and the clone is compared back to the live Python state.
- That isolation harness now shows that Stage A is much closer than before. On `runs/native_exchange_stage_compare_32_6_6_10_v2.json` the first remaining mismatch is tiny and local: `engine._inventory_trade_volume` plus a couple of ulp-scale bookkeeping cells in `periodic_tce_cost` / period value arrays.
- Even with those remaining ulp-level commit differences, report-scale behavior is now surprisingly stable. `runs/native_behavior_compare_report_scale_exchange_stage_10cycles_s3_v1.json` shows very small mean final deltas over three seeds at `3000/30/100`, while `runs/native_behavior_compare_report_scale_exchange_stage_20cycles_seed2009_v1.json` and `runs/native_behavior_compare_report_scale_exchange_stage_40cycles_seed2009_v1.json` keep end-state drift below about `1%` on the main production/trade/utility metrics for seed `2009`.
- The performance payoff is correspondingly large. The same report-scale behavior probes show about `18x-24x` speedup, and the large-scale anchor `runs/estimate_exact_stage_math_exchange_stage_10000_100_150_3cycles_v1.json` averages about `3.95 s` per cycle on `10000/100/150`, which projects to roughly `2.2 h` for `2000` cycles.
- This creates a real fork in the plan. The fully exact baseline remains the accepted stage-math path, but there is now a plausible fast-path for large exploratory runs if the observed sub-1%-class drift on the key macro metrics remains acceptable after longer validation.
- The `run_exact_cycle` entrypoint first became a real Rust-side whole-cycle orchestrator, and that shell held exact parity on `runs/native_cycle_compare_32_6_6_whole_cycle_orchestrator.json`, but it was effectively speed-neutral (`~0.995x`).
- A dedicated Stage B isolation harness now exists in `native_post_period_compare.py`. It keeps the live exact run on the accepted Python path, clones each agent's pre-post-period state, runs a candidate native post-period block (`period_time_debt` carry, `leisure_production`, `end_agent_period`) on the clone, and compares the resulting state against the live Python `_complete_agent_period_after_surplus`. This is now the acceptance gate before any new Rust post-period integration attempt.
- After aligning the native `stock_limit` mixed-precision calculation with Python, the isolated post-period block held exactly on `runs/native_post_period_compare_32_6_6_100.json` for `100` cycles on seeds `2009/2011/2013` while measuring about `5.72x` speedup inside that isolated block.
- A direct integration attempt of that post-period block into `experimental_native_stage_math` was rejected. It looked excellent through `40` cycles, but `compare-native-cycle` and `compare-native-behavior` later showed first long-horizon drift around cycles `48-66`, starting from ulp-scale `purchase_price` / `sales_price` differences and then amplifying into materially different production and trade totals by cycle `100`.
- Because of that rejection, the execution path was rolled back: the exact runner still uses the older accepted stage-math slice, and the new post-period block remains harness-only until the remaining price-state drift is isolated.
- The accepted whole-cycle slice now owns the whole `_run_agent_cycle` orchestration in Rust while still delegating the branch-sensitive interior callbacks (`_satisfy_needs_by_exchange`, `_make_surplus_deals`, `_surplus_production`, `_leisure_production`, `_end_agent_period`, `_add_random_friend`) to Python. The safe native stages inside that agent cycle are called directly from Rust.
- Acceptance artifact: `runs/native_cycle_compare_32_6_6_safe_slice_final.json` with `32/6/6`, `10` cycles, seeds `2009/2011/2013`, and `0` mismatches.
- Longer parity probes also held exactly on this slice: `runs/native_cycle_compare_32_6_6_safe_slice_30.json` (`30` cycles) and `runs/native_cycle_compare_64_6_6_safe_slice_20.json` (`20` cycles), both with `0` mismatches on seeds `2009/2011/2013`.
- The current measured speedup envelope for this accepted slice is modest but real: about `1.12x-1.15x` on the accepted `32/6/6` and `64/6/6` probes in this environment.
- A dedicated `native_stage_math_trace_compare.py` harness now exists for the accepted stage-math path. It runs one or more exact cycles agent by agent, compares the Python-reference and native-stage paths after every substage (`prepare_consumption`, `produce_need`, leisure round, surplus, post-period), and reports the first mismatching agent/stage/field.
- That harness isolated the remaining report-scale first-cycle drift to the native stage-math wrappers' aggregate semantics, not to the state mutation itself. `prepare_leisure_round` and `produce_need` now use Python-reference aggregate/time calculations around the native row mutation, and `runs/native_stage_math_trace_report_scale_cycle1_seed2009_v5.json` now holds exactly on the first full `3000/30/100` cycle.
- After switching the Rust whole-cycle entrypoint back to the corrected Python stage wrappers, `runs/native_behavior_compare_stage_math_32_6_6_100_v3.json` now holds exact behavior for `100` cycles on seeds `2009/2011/2013` at about `1.55x` speedup.
- Report scale is now materially better. `runs/native_cycle_compare_report_scale_10cycles_seed2009_v1.json` holds full-state parity for `10` cycles at `3000/30/100`, and `runs/native_behavior_compare_report_scale_20cycles_seed2009_v1.json` holds zero snapshot-level drift for `20` cycles on the same scale. The observed speedup envelope on those report-scale probes is now about `4.2x-5.5x`.
- The dedicated stage-trace harness also now holds beyond the first report-scale cycle: `runs/native_stage_math_trace_report_scale_cycle1_seed2009_v5.json` is exact for cycle `1`, and `runs/native_stage_math_trace_report_scale_3cycles_seed2009_v2.json` is exact through cycle `3`.
- Current large-scale anchor: `runs/estimate_exact_stage_math_outerloop_10000_100_150_3cycles_v1.json` averages about `29.88 s` per cycle across the first `3` exact cycles on `10000/100/150` in this environment. This is still above true overnight feasibility for `2000` cycles, but it is now far below the earlier multi-day estimate and close enough that one more substantial optimization phase could plausibly cross the overnight threshold.
- An attempted follow-on cut that also inlined `_surplus_production`, `_leisure_production`, and then `_end_agent_period` into the Rust-owned agent cycle was rejected. It produced promising short-horizon speedups (`~1.84x-2.50x`) but drifted by cycles `11-14` in the `32/6/6` and `64/6/6` behavior probes, eventually flipping accepted trade counts on some seeds.
- This sharpens the next blocker further. The next worthwhile native port is not another isolated tail helper, but a larger contiguous exact block that owns exchange/deal execution together with the immediately adjacent state transitions, or else the full sequential exact core.

Next step:

- keep the frontier-wave hybrid path experimental only; it is no longer the main optimization track
- use the accepted safe native slice as the reference acceleration baseline
- continue toward a Rust-owned sequential exact core before revisiting any meaningful parallel execution

### Sequential Exact-Core Plan

Stage A: Rust owns the full trade path

- port `_satisfy_needs_by_exchange` and `_make_surplus_deals` together with the immediately adjacent trade bookkeeping
- include post-trade value, purchase, sales, inventory-trade, and TCE accounting in the same contiguous Rust block
- acceptance gate: no per-cycle mismatches on `32/6/6` for `100` cycles across seeds `2009/2011/2013`
- second gate: no macro drift outside floating-point noise on `64/6/6` for `50` cycles across the same seeds

Stage B: Rust owns the post-period state transition

- port `_surplus_production`, `_leisure_production`, `_end_agent_period`, `_complete_agent_period_after_surplus`, and `_calibrate_friend_transparency` as one block
- avoid reintroducing the previously rejected split where `_end_agent_period` drifted ahead of downstream branch-sensitive logic
- acceptance gate: no state-parity mismatches on `32/6/6` for `100` cycles and `64/6/6` for `50` cycles

Stage C: Rust owns the full sequential exact core

- move the remaining branch-sensitive callbacks into Rust so that one cycle runs without Python callbacks inside the agent loop
- keep Python responsible for configuration, checkpointing, CLI, dashboard, and analysis only
- acceptance gate: parity on the existing `compare-native-cycle` harness plus checkpoint/resume compatibility

Stage D: report-scale performance validation

- benchmark accepted exact-core builds on `3000/30/100`
- benchmark extrapolation anchors on `10000/100/150`
- target: move report-scale `1000-2000` cycle runs comfortably below the current `2.5-5.0 h` envelope and bring `10000/100/150/2000` toward overnight feasibility

### Parallelism Policy After Full Rust Port

The model should stay sequential at the trade-commit level until the full Rust exact core is complete and validated. The core risk is semantic, not implementation difficulty: a highly attractive seller or producer can be a feasible counterparty for several agents in the same notional instant, but only one trade can actually commit. If those competing trades are evaluated and committed concurrently, the order of success changes stocks, needs, prices, time budgets, and learning signals.

Because of that, the first parallel targets must be read-only or reduction-style work:

- friend and good candidate scoring
- search over feasible counterparties while reading a committed state snapshot
- market-level reductions and aggregate metric calculation
- batch recomputation of diagnostics that do not mutate agent state

The following remain off-limits until proven otherwise:

- overlapping trade commit for agents that share a possible counterparty
- concurrent mutation of inventories, needs, prices, or friendship state across live negotiations
- any GPU or thread-pool scheme that advances multiple agents past the same decision barrier before conflict resolution

### Collision-Resolution Note

In theory, a speculative parallel trade search with rollback is possible:

- build candidate trades from a read-only snapshot
- detect collisions where two or more trades touch the same agent
- choose one winning trade per conflicting agent set
- discard or retry the losers on the next micro-step

But this is still not semantically free. The loser trade was chosen under a stale state that no longer exists after the winner commits, so rollback alone is not enough; the loser must be replanned. That makes the method closer to an asynchronous matching algorithm than to true simultaneous execution, and it should be treated as an experimental post-port optimization, not as part of the exact reference path.

### Current Direction

- finish the Rust sequential exact core first
- accept only parity-preserving ports into the exact path
- defer serious CPU-thread or GPU parallelism until the full Rust core is stable
- evaluate parallelism only through explicit multi-seed comparison against the Rust sequential reference
- use the new `native_behavior_compare` harness as the behavioral gate for larger Rust ports, not only the strict per-cycle parity harness
- current reference artifact: `runs/native_behavior_compare_exchange_stage_32_6_6_40.json`, which shows the exchange-stage path is still faster-but-too-drifting to accept
- a new exchange-stage trace harness now exists in `native_exchange_stage_trace_compare.py`; it compares per-agent stage events between the Python exact reference and an experimental native path and reports the first mismatching stage event
- current trace artifacts: `runs/native_exchange_trace_compare_32_6_6_40.json` and the looser-tolerance `runs/native_exchange_trace_compare_32_6_6_40_loose.json`
- current Stage A diagnosis: before the first macro drift, the native exchange-stage path first shows only ulp-scale bookkeeping differences; when tolerances are widened, the first structural divergence appears around cycles `26-32` as an exchange-stage event-count / stage-order mismatch rather than an immediately different partner or goods choice
- this points the next Rust debugging step toward leisure/surplus stage-boundary timing and the surrounding transition logic, not toward the inner barter search itself
- operational correction: the rejected `experimental_native_exchange_stage` path no longer activates in normal exact runs; it now requires an explicit engine-level opt-in used only by the comparison harnesses
- consequence: normal exact runs stay on the parity-preserving path (native search/planning plus the accepted safe native stage-math slice), while the drift-prone exchange-stage commit path remains available only for diagnostics and future debugging

### Operational Baselines

Exact reference baseline:

- use `experimental_native_stage_math=True`
- keep `experimental_native_exchange_stage=False`
- this is the accepted report-faithful path for exact validation, checkpointed report-scale runs, and all final confirmation work
- current acceptance artifacts:
  - `runs/native_cycle_compare_report_scale_10cycles_seed2009_v1.json`
  - `runs/native_behavior_compare_report_scale_20cycles_seed2009_v1.json`
  - `runs/estimate_exact_stage_math_outerloop_10000_100_150_3cycles_v1.json`
- current large-scale anchor: about `29.88 s / cycle` on `10000 / 100 / 150`

Fast exploratory baseline:

- use `experimental_native_stage_math=True`
- use `experimental_native_exchange_stage=True`
- this is not the exact reference path
- it is acceptable only for exploratory large-run screening when small macro drift is tolerated and any important finding is later rechecked against the exact baseline
- current reference artifacts:
  - `runs/native_behavior_compare_report_scale_exchange_stage_10cycles_s3_v1.json`
  - `runs/native_behavior_compare_report_scale_exchange_stage_20cycles_seed2009_v1.json`
  - `runs/native_behavior_compare_report_scale_exchange_stage_40cycles_seed2009_v1.json`
  - `runs/estimate_exact_stage_math_exchange_stage_10000_100_150_3cycles_v1.json`
- current large-scale anchor: about `3.95 s / cycle` on `10000 / 100 / 150`, projecting to roughly `2.2 h` for `2000` cycles on this machine

Decision rule:

- if the question is "is the model still report-faithful," use the exact reference baseline
- if the question is "can we cheaply screen a very large scenario for growth, money emergence, and cycles," the fast exploratory baseline is acceptable as a first-pass instrument
- no exploratory result should be treated as final until it has been cross-checked on the exact baseline at smaller or report scale

### Immediate Next Steps

1. Run long report-scale exact validation on the accepted baseline.

- objective: confirm that the now-accepted stage-math slice still reproduces the report-style dynamics over the longer horizon where growth and cycles should emerge
- default target: `3000 / 30 / 100`, `1000-2000` cycles, checkpointed

2. Run large exploratory screening on the fast baseline.

- objective: make `10000 / 100 / 150 / 2000` practical enough to use as a search instrument for macro phenomena
- record production, utility, rare-money share, cycle metrics, spoilage, and TCE waste
- treat these runs as hypothesis generation, not final evidence

3. Cross-check any interesting large-run finding against the exact baseline.

- downscale to report size or another exact-feasible checkpointed size
- verify that the qualitative finding survives without the experimental exchange-stage shortcut

4. Continue Stage A exactness work only if promotion of the fast path becomes strategically necessary.

- the remaining Stage A drift is now small enough to be useful operationally
- but exact promotion still requires elimination of the residual exchange bookkeeping and stage-boundary drift
- do not spend time on this before the exact long-run and large exploratory workflows have both been exercised

5. Keep parallelism out of the mutable exact core until the full Rust sequential core is complete.

- near-term speed work should still prioritize larger contiguous Rust ownership of the sequential core
- any later CPU-thread or GPU parallelism must be evaluated only against the Rust sequential reference
