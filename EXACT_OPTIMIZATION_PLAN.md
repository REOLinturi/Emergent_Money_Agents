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

The phenomenon path is separate from this exact path. The current retained non-exact exploratory branch is the per-agent basket path with Rust-owned after-trade re-planning: `--experimental-agent-basket-planning` plus `--experimental-session-replan-after-trade`, with session clearing disabled. The active agent evaluates its locally visible basket opportunity set, commits one revalidated trade, rebuilds the local plan from the changed state, and only then continues. This keeps the primitive barter timing asymmetry while avoiding stale one-shot shopping lists.

Operational guardrail: keep only two paths as normal work targets. Use the exact path for reference validation and short correctness probes. Use the Rust replan-after-trade per-agent basket path for phenomenon-scale exploratory runs. Static no-replan basket lists, session-clearing, and wave-based variants are historical diagnostics/rollback branches; do not extend or optimize them unless explicitly investigating a named anomaly.

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

Execution classes are now kept explicitly separate:

- `exact-baseline`: parity-preserving reference-compatible execution. This path may use only accepted native slices and must keep `experimental_native_exchange_stage=False`.
- `exact-tolerant acceleration`: still uses the legacy decision model and sequential proposer order, but may accept boundary-case numerical differences and path-dependent individual-agent divergence when the differences are not a detectable systematic macro bias and the observed economic phenomena survive multi-seed checks.
- `realism path`: intentionally changes the decision model, for example basket-level planning by the active agent. This path must stay opt-in and must never be presented as a Legacy-C exact result.

The near-term optimization target is `exact-baseline` first. If a Rust slice cannot be made strictly exact after focused attempts, it may be proposed for `exact-tolerant acceleration`, but only after the comparison reports classify the mismatch as a plausible boundary/path-dependence effect and after report-scale macro phenomena remain robust.

Stage A: Rust owns the full trade path

- port `_satisfy_needs_by_exchange` and `_make_surplus_deals` together with the immediately adjacent trade bookkeeping
- include post-trade value, purchase, sales, inventory-trade, and TCE accounting in the same contiguous Rust block
- acceptance gate: no per-cycle mismatches on `32/6/6` for `100` cycles across seeds `2009/2011/2013`
- second gate: no macro drift outside floating-point noise on `64/6/6` for `50` cycles across the same seeds
- tolerant fallback gate: if strict parity still fails, mismatches must be plausible non-tendentious boundary/path-dependence effects. Acceptance is then phenomenon-level: growth, monetization, welfare plateau or cycle, friction/spoilage behavior, role concentration, and inequality dynamics must remain qualitatively similar over multi-seed report-scale probes even if individual agents and exact cycle timing diverge.

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

### GPU and Hardware Strategy

Current local hardware check, `2026-04-27`:

- local GPU: `NVIDIA GeForce RTX 4090 Laptop GPU`
- reported VRAM: `16376 MiB`
- current Python CUDA stack: CuPy is not installed, so `--backend cuda` is not currently available from the CLI on this machine

Memory is not the only issue. For `10000 / 100 / 150`, persistent model state is still modest if stored compactly: roughly below `1 GiB` for the main arrays, with the transparency tensor alone about `0.56 GiB`. The scaling problem comes from candidate evaluation. A fully materialized score tensor over `population * acquaintances * goods * goods` would be about `15 billion` candidate cells at `10000 / 100 / 150`; one `float32` score array alone would be about `60 GiB`, before validity masks and intermediate arrays. Therefore any viable GPU path must stream, tile, and reduce candidates instead of materializing the full search space.

Implications:

- the laptop GPU can be useful for small and medium GPU-kernel development, correctness checks, and tiled experiments
- the laptop's `16 GiB` VRAM is not the right target for dense or near-dense large exploratory runs
- a tower GPU with about `96 GiB` VRAM should be preferred for the maximally parallel phenomenon-preserving path, especially if candidate matrices, larger tiles, or larger populations are used
- even on `96 GiB`, avoid full `population * acquaintances * goods * goods` materialization; use tiled or streaming reductions that still examine the full economically available opportunity set
- moving large runs to the tower is sensible once the CUDA stack is installed and the GPU path is benchmarked there

Next GPU-oriented optimization targets:

- restore a working CUDA backend environment, probably through a CuPy build matching the installed NVIDIA driver/CUDA stack
- profile the existing CUDA proposal kernel, because the current scaling bottleneck is proposal/candidate volume, not checkpointing
- replace one-thread-per-agent inner candidate loops with a parallel candidate reduction where threads cover friend/good/offer combinations and reduce to the best proposal per agent
- keep trade commit sequential or conflict-resolved until phenomenon-level equivalence has been demonstrated
- use `3000 / 30 / 100` and `10000 / 30 / 150` as validation sizes before retrying `10000 / 100 / 150 / 2000`

### Opportunity-Set Policy

The current research direction rejects speedups that come from arbitrary candidate pruning.

Principle:

- do not reduce `demand_candidates`, `supply_candidates`, acquaintances, or good-pair scans merely to make the code faster
- any narrower opportunity set must have an explicit real-world interpretation, such as bounded attention, search fatigue, posted-price filters, specialist inventory constraints, or another modeled agent heuristic
- when no such heuristic is being studied, the phenomenon path should let the active agent evaluate the full set of relevant known counterparties and goods
- use parallel scoring, tiled reductions, cache reuse, and compiled loops to make that full search affordable
- keep exact/legacy controls available for comparison, but do not treat the old `4`-candidate buffers as the preferred realism target

Engineering consequence:

- exact-reference optimization may still preserve the historical implementation's opportunity set for validation
- phenomenon-model optimization should prefer `demand_candidates=goods` and `supply_candidates=goods` where the research question assumes a trader can inspect all known goods
- GPU and CPU-thread work should focus on full-search reductions rather than top-k approximation
- if a pruning heuristic is introduced later, it must be exposed as an explicit model variant, documented, and compared against the full-search baseline

### Basket Decision Semantics

The basket path is a realism-model variant, not an exact Legacy-C reference. It may let the active
agent inspect all locally visible needs, friends, and goods before choosing the best next exchange,
but it must still preserve the primitive-market assumption that one agent commits one decision at a
time.

Accepted rule:

- full-basket scoring is allowed, and should eventually be parallelized as a read-only reduction
- after one accepted trade, rejected candidate, or exhausted-offer update, the agent must replan from the new inventory and partner state
- executing multiple trades from one stale candidate slate is not acceptable as a default phenomenon path, because it creates a simultaneous multi-trade institution that the Legacy report does not assume
- if a later model intentionally studies posted prices, clearing houses, auctions, or other richer market institutions, multi-trade execution can be reintroduced as a separate documented model variant

Diagnosis recorded on `2026-04-28`:

- long `3000/30/100/2000` basket runs showed a pathology not seen in no-basket exact-reference style runs: median living standard stayed near `1.05`, aspiration shortfall was above half the population, accepted trade volume was about `2.5x` the no-basket reference, and private price tails reached about `1e20`
- the pathology was concentrated in the old basket implementation that planned a whole basket and then executed several trades before replanning
- comparable no-basket runs still showed high friction, inequality, rare-money emergence, and boom-to-decline behavior, but not the same extreme price explosion or persistent majority-at-baseline outcome
- the code has therefore been changed so both the Python fallback and Rust basket stage replan after each processed candidate
- a first `500/30/100/500` probe with the corrected replan-one basket semantics kept prices bounded and produced plausible welfare dispersion, but it was substantially slower than the old stale-slate basket path

Engineering consequence:

- the next basket-speed target is parallel or compiled full-opportunity scoring for the single next decision
- do not regain speed by committing many stale basket candidates in one pass
- exact-path speed work remains separate and must continue to preserve the historical sequential decision rule

### Report-Faithful Heuristic Corrections

`Exact` now means faithful to the report-level model assumptions, not blind preservation of every
Legacy-C implementation accident. Legacy-C remains an important reference, but when the C code
contains a clear heuristic bug that contradicts the report and real barter logic, the maintained
exact path should correct it and document the divergence.

Accepted correction on `2026-04-29`:

- retailer purchase-price thresholds must not end a period above the same retailer's sales-price threshold
- the correction is applied in both Python and Rust-native post-period logic
- the practical merchant interpretation is simple: a trader should not plan to buy inventory at a price above the price at which it is willing to resell it
- this is consistent with the report's wording that normal purchase prices should be checked against minimum sales prices, while allowing private money-good prices to remain high when supply-chain position justifies it
- a one-cycle probe from the pathological `3000/30/100/2000` basket checkpoint reduced retailer `purchase_price > sales_price` violations from `1652` to `0`
- the existing huge private-price tail does not vanish in one cycle because some sales-price thresholds had already ratcheted upward; new validation runs must check whether the corrected invariant prevents the tail from forming in the first place

Accepted correction on `2026-05-03`:

- agents must not satisfy aspirational need by forcing own production into negative available time
- if available time is insufficient, own production is capped proportionally, residual need remains visible in `state.need`, and the period is marked as a failure
- producer price heuristics follow the report-level logic: normal stock with weak sales anchors the sales threshold to production cost, and large unsold stock can discount to cost and then below cost if no demand appears
- the correction is applied in both the Python exact path and the Rust-native stage-math path
- dedicated unit tests now cover time-limited need production, remaining unmet need, producer cost anchoring, and producer discounting after persistent no-sales inventory

Accepted correction on `2026-05-03`, after the `3000/30/100/3000` rules-fix run:

- price thresholds must not decay numerically to zero for actively traded goods
- the maintained path now uses a use-value price floor rather than the old global hard floor
- the floor is anchored to the agent's focused production cost for that good and discounted by inventory overhang: if stock exceeds the visible consumption/sales horizon, adjusted for spoilage, the floor can fall below production cost
- this preserves the real-world interpretation that every good has some use value, while still allowing fire-sale pricing when visible inventories are too large to consume before spoilage
- retailer resale margin is preserved at the floor: purchase thresholds may not end at or above sales thresholds
- the correction is applied in both Python and Rust-native post-period price logic, with tests covering useful inventory, overhang discounting, and native parity

### Current Direction

- finish the Rust sequential exact core first
- accept only parity-preserving ports into the exact path
- defer serious CPU-thread or GPU parallelism until the full Rust core is stable
- evaluate parallelism only through explicit multi-seed comparison against the Rust sequential reference
- use the new `native_behavior_compare` harness as the behavioral gate for larger Rust ports, not only the strict per-cycle parity harness
- current strict/tolerant artifacts:
  - `runs/native_cycle_compare_exchange_stage_32_6_6_40_tolerant_gate.json`
  - `runs/native_cycle_compare_exchange_stage_32_6_6_100_tolerant_gate.json`
  - `runs/native_cycle_compare_exchange_stage_64_6_6_50_tolerant_gate.json`
  - `runs/native_behavior_compare_report_scale_exchange_stage_20cycles_s2_tolerant_gate.json`
- a new exchange-stage trace harness now exists in `native_exchange_stage_trace_compare.py`; it compares per-agent stage events between the Python exact reference and an experimental native path and reports the first mismatching stage event
- current trace artifacts: `runs/native_exchange_trace_compare_32_6_6_40.json` and the looser-tolerance `runs/native_exchange_trace_compare_32_6_6_40_loose.json`
- current Stage A diagnosis: the earlier long-horizon drift has been narrowed. The whole-cycle full-state harness now reports strict mismatches only in `engine._inventory_trade_volume`, at about `1e-14` to `1e-12`, with zero snapshot deltas and zero material mismatches through the current `32/6/6/100` and `64/6/6/50` gates.
- report-scale behavior is stable only on the short gates so far: `3000/30/100`, `20` cycles, seeds `2009/2011`, zero macro drift, about `23x` speedup; and `runs/native_behavior_compare_report_scale_exchange_stage_40cycles_s2_tolerant_gate.json`, `40` cycles, seeds `2009/2011`, zero macro drift, about `15.7x` speedup.
- The longer gated attempt `runs/native_behavior_compare_report_scale_exchange_stage_100cycles_s2_partial_gate.json` rejects strict behavioral parity but no longer automatically rejects phenomenon-level use. It stopped by the explicit runtime guard at cycle `88` of seed `2009`, but the first behavioral mismatch was already at cycle `64`: accepted trade count differed by `1`, with nonzero production, stock, and trade-volume deltas. By cycle `88`, the deltas had amplified materially, which is expected in a path-dependent nonlinear system. The open question is now whether the macro phenomena remain robust and non-tendentiously biased.
- The first concrete Stage A divergence is now localized. `runs/native_exchange_trace_report_scale_seed2009_65cycles_stageA_diag.json` shows the first event mismatch at cycle `64`, surplus stage, agent `461`, event index `923`: the Python reference records `224` proposed / `36` accepted proposals for that stage, while the native exchange-stage records `223` proposed / `35` accepted. The last stored proposal is still the same in both paths, so the root is an earlier marginal proposal inside the same surplus loop, not a different final candidate.
- `runs/native_cycle_compare_report_scale_exchange_stage_65_seed2009_material_diag_v3.json` confirms that, after tolerating only `engine._inventory_trade_volume` accumulator roundoff, the first material full-state mismatch is also cycle `64`: accepted trade count differs by `1`, accepted volume by about `2.20`, production by about `22.12`, and stock total by `256`. There are `61` tolerated inventory-volume accumulator mismatches before that, but no material snapshot drift.
- A focused cycle-64 state probe shows the first trade-allocation difference around goods `11/19` in the surplus network. The reference has an extra `461 <-> 2574` trade for that goods pair, while the native path later has an extra `875 <-> 2574` trade for the same goods pair; the reference also has an extra `2126 <-> 1798` surplus trade for goods `13/11`. This points to accumulated intra-stage floating-point execution differences around a marginal surplus-loop threshold, rather than a gross ordering or partner-selection bug.
- The first checkpointed fast long-run artifact is `runs/exact_tolerant_exchange_3000_30_500_seed2009`: `3000/30/100`, `500` cycles, seed `2009`, about `6.0 s/cycle`. It shows continued production/trade/stock growth at cycle `500`, rare-money share about `87%`, utility peaking earlier around cycle `355`, and friction rising to about `27%` of the time budget. Because the 100-cycle gate found material drift, this run must be treated as exploratory fast-path evidence only, not exact-tolerant validation.
- The first phenomenon-level multi-seed gate now exists for the fast exchange-stage path: `runs/exact_tolerant_exchange_3000_30_500_seed2009`, `runs/phenom_tolerant_exchange_3000_30_500_seed2011`, and `runs/phenom_tolerant_exchange_3000_30_500_seed2013`, summarized in `runs/phenom_tolerant_exchange_3000_30_500_summary.csv`. All three 500-cycle report-scale runs show the same phenomenon flags: production growth, rare-money emergence, utility peak before the end, rising friction, and high living-standard inequality. Individual levels and peak timings differ substantially, which is expected under path dependence.
- In the three-seed 500-cycle gate, utility peaks at cycles `355/395/350`, while living-standard mean peaks much earlier at cycles `125/110/115`. By cycle `500`, production and trade remain high or rising, rare-money share is about `0.83-0.87`, and living-standard mean has fallen to about `29-38%` of its peak. This is qualitatively consistent with the target phenomenon: GDP-like activity can keep growing while welfare no longer follows and transaction/friction burdens rise.
- The seed `2009` fast-path run has now been extended to `1000` cycles. The artifact summary is `runs/exact_tolerant_exchange_3000_30_500_seed2009/artifact_summary_1000.txt`. Production is still at its maximum at cycle `1000` (`1.77e10`), trade volume peaked around cycle `910`, rare-money share remains high at about `93%`, and stock remains at its maximum. Utility peaked much earlier at cycle `355` and is now about `39%` of that peak. Friction is high: about `85%` of the time budget, with TCE about `82%`. This strengthens the qualitative evidence that the fast path reproduces the important decoupling between output-like aggregates and welfare-like utility, but it has not yet demonstrated a full downturn/recovery cycle.
- The same seed `2009` run has now reached `1500` cycles. The artifact summary is `runs/exact_tolerant_exchange_3000_30_500_seed2009/artifact_summary_1500.txt`. Production peaked around cycle `1290` and trade volume around cycle `1265`; at cycle `1500` both are below those peaks, while utility has recovered modestly from the cycle-1000 value but remains only about `49%` of its cycle-355 peak. Rare-money share remains high at about `92%`, stock is below its cycle-1145 peak, and friction remains very high at about `99%` of the nominal time budget. This is now a plausible late-boom/turning-point pattern, but still not a complete second cycle.
- The seed `2009` fast-path run has now reached the planned `2000`-cycle screen. The artifact summary is `runs/exact_tolerant_exchange_3000_30_500_seed2009/artifact_summary_2000.txt`. Production peaked at cycle `1290` and ends at about `76%` of that peak; accepted trade volume peaked at cycle `1580` and ends at about `59%` of that peak; utility peaked at cycle `355` and ends at about `42%` of that peak. Rare-money share remains high at about `91%`. Stock peaked at cycle `1145` and ends at about `63%` of that peak. Friction and TCE remain high, with friction ending at about `96%` and TCE at about `93%` of the nominal time budget. This captures a strong boom/overhead/decline pattern, but not yet a clean recovery into a second boom.
- A smaller exact-baseline cross-check is now in progress at `1000/30/100`, seed `2009`, with `experimental_native_stage_math=True` and `experimental_native_exchange_stage=False`. The first completed artifact is `runs/exact_baseline_calibration_1000_30_100_20_seed2009_v1/artifact_summary_1000_dedup.txt`. It confirms the same key early phenomena without the native exchange stage: utility peaks at cycle `355`, rare-money share reaches about `96%` and ends at about `94%`, friction/TCE rise sharply, and living-standard inequality becomes high. At cycle `1000`, production and trade are still near their peaks, so this exact-baseline run must be extended before it can confirm the full boom-to-decline transition.
- The exact-baseline cross-check has now also reached `2000` cycles: `runs/exact_baseline_calibration_1000_30_100_20_seed2009_v1/artifact_summary_2000_dedup.txt`. It confirms the boom-to-decline transition without the native exchange stage. Production peaks at cycle `1465` and ends at about `72%` of peak; accepted trade volume peaks at cycle `1455` and ends at about `36%` of peak; utility peaks much earlier at cycle `355` and ends at about `47%` of peak. Rare-money share ends at about `90%`, friction at about `90%`, TCE at about `87%`, and stock ends at about `41%` of peak. A compact comparison with the fast-path `3000/30/100/2000` run is in `runs/phenomenon_gate_fast_vs_exact_summary.csv`.
- The artifact analyzer now deduplicates metric samples by cycle and keeps the latest row. This matters for interrupted and resumed runs because a crash or timeout can leave sampled metrics after the latest checkpoint, and a resumed run can replay those cycles.
- The accepted interpretation is now explicitly phenomenon-level for this path. Because the model is nonlinear and path-dependent, small branch or rounding differences may eventually shift individual trajectories and cycle timing. That is acceptable only if multi-seed runs continue to show the same macroeconomic phenomena without a detectable one-sided bias. Event-exact claims remain reserved for the exact reference baseline.
- consequence: normal exact final-validation runs must still use the parity-preserving baseline unless the user explicitly selects `experimental_native_exchange_stage=True`. The exchange-stage path preserves the legacy decision model and sequential proposer order, so it is not a `realism path`. It is now treated as a phenomenon-level exact-tolerant candidate pending multi-seed robustness checks, not as an event-exact candidate.

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
- until promoted by the gates above, classify it as `exact-tolerant acceleration` only for exploratory screening, not as final evidence
- if a run also enables `experimental_agent_basket_planning`, classify it as `realism path`, because the agent decision rule changes
- current reference artifacts:
  - `runs/native_behavior_compare_report_scale_exchange_stage_10cycles_s3_v1.json`
  - `runs/native_behavior_compare_report_scale_exchange_stage_20cycles_seed2009_v1.json`
  - `runs/native_behavior_compare_report_scale_exchange_stage_40cycles_seed2009_v1.json`
  - `runs/estimate_exact_stage_math_exchange_stage_10000_100_150_3cycles_v1.json`
  - `runs/native_cycle_compare_exchange_stage_32_6_6_100_tolerant_gate.json`
  - `runs/native_cycle_compare_exchange_stage_64_6_6_50_tolerant_gate.json`
  - `runs/native_behavior_compare_report_scale_exchange_stage_20cycles_s2_tolerant_gate.json`
  - `runs/native_behavior_compare_report_scale_exchange_stage_40cycles_s2_tolerant_gate.json`
  - rejected longer gate: `runs/native_behavior_compare_report_scale_exchange_stage_100cycles_s2_partial_gate.json`
  - divergence trace: `runs/native_exchange_trace_report_scale_seed2009_65cycles_stageA_diag.json`
  - divergence full-state check: `runs/native_cycle_compare_report_scale_exchange_stage_65_seed2009_material_diag_v3.json`
  - `runs/exact_tolerant_exchange_3000_30_500_seed2009`
  - `runs/exact_tolerant_exchange_3000_30_500_seed2009/artifact_summary_1000.txt`
  - `runs/exact_tolerant_exchange_3000_30_500_seed2009/artifact_summary_1500.txt`
  - `runs/exact_tolerant_exchange_3000_30_500_seed2009/artifact_summary_2000.txt`
  - `runs/phenom_tolerant_exchange_3000_30_500_seed2011`
  - `runs/phenom_tolerant_exchange_3000_30_500_seed2013`
  - `runs/phenom_tolerant_exchange_3000_30_500_summary.csv`
  - exact-baseline cross-check: `runs/exact_baseline_calibration_1000_30_100_20_seed2009_v1/artifact_summary_1000_dedup.txt`
  - exact-baseline cross-check: `runs/exact_baseline_calibration_1000_30_100_20_seed2009_v1/artifact_summary_1500_dedup.txt`
  - exact-baseline cross-check: `runs/exact_baseline_calibration_1000_30_100_20_seed2009_v1/artifact_summary_2000_dedup.txt`
  - compact gate comparison: `runs/phenomenon_gate_fast_vs_exact_summary.csv`
- current large-scale anchor: about `3.95 s / cycle` on `10000 / 100 / 150`, projecting to roughly `2.2 h` for `2000` cycles on this machine
- current promotion status: rejected for event-exact long-horizon use, but accepted as a phenomenon-level screening path for exploratory runs. It passes the first `3000/30/100/500` multi-seed fast-path gate, the `3000/30/100/2000` single-seed fast-path gate, and a `1000/30/100/2000` exact-baseline cross-check for rare-money emergence, output/welfare decoupling, rising friction/TCE, high inequality, and a boom-to-decline transition. It should still not be used as final exact evidence, and it still needs multi-seed exact-baseline confirmation before being promoted beyond screening.
- overnight large-screen attempt, `2026-04-27`: `runs/nightly_large_fast_20260427/large_fast_10000_100_150_2000_seed2009` reached only cycle `60` before the foreground tool timeout. This invalidates the naive early-cycle extrapolation above for long 100-good runs. Cycle cost grew from about `139 s/cycle` around cycles `15-20` to about `843 s/cycle` around cycles `55-60`. Proposed trade count reached about `17.1M` per sampled cycle by cycle `60`, while accepted count was still about `1.0M`. The run is therefore useful as a scaling diagnostic, not as a macro-phenomenon run.
- follow-up calibration, `runs/calibrate_fast_10000_30_150_25_seed2009`, shows that keeping population `10000` and acquaintances `150` but reducing goods to `30` is much more feasible, though still not comfortably overnight for `2000` cycles. It reached cycle `100`; the first `25` cycles averaged about `5.2 s/cycle`, while cycles `25-100` averaged about `47 s/cycle`. At cycle `100`, utility and production were still rising, rare-money share was about `21%`, and friction was about `16%`.
- checkpoint compression is now optional. Use `--uncompressed-checkpoint` for long local screening runs when disk space is available. On the `10000/30/150` cycle-100 checkpoint, compressed checkpoint writing took about `17.2 s` and produced about `174 MiB`; uncompressed writing took about `0.43 s` and produced about `560 MiB`. This is a safe operational speedup because it changes only artifact serialization, not model state or decisions.
- `RAYON_NUM_THREADS` does not materially improve the current installed native exchange-stage module. A one-cycle cycle-100 benchmark at `10000/30/150` took about `61.2 s` with `RAYON_NUM_THREADS=1` and about `63.6 s` with `RAYON_NUM_THREADS=8`. The current compiled path therefore does not expose enough of the expensive inner search to effective CPU-thread parallelism.
- the agent basket planning path remains a phenomenon-model candidate, not a speed win yet. On `3000/30/100/40`, it roughly halved the last sampled proposal count (`463k -> 239k`) while preserving accepted count/volume, but runtime was similar. On `10000/30/150/25`, it reduced later proposal count (`866k -> 575k` at cycle `25`) but was slower overall (`164 s` vs `129 s`). It may become useful after the native basket path is optimized or moved to GPU-style reductions.
- local Rust build tooling is now available. Rust `1.95.0` and Cargo are installed under the user profile, MSVC Build Tools 2019 are available through `VC\Auxiliary\Build\vcvars64.bat`, and the project `.venv` has a working maturin. The reliable build command is:

```powershell
$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"
cmd.exe /c '"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" && .venv\Scripts\python.exe -m maturin build --release -m native\legacy_search\Cargo.toml'
```

- first post-toolchain Rust optimization: `run_exchange_stage` now caches the active agent's reciprocal friend slots once per stage and updates the cache only when a missing reciprocal link is created. This preserves the decision model and removes repeated `acquaintances * acquaintances` scans inside the need/attempt loops. On the `10000/30/150` cycle-100 checkpoint, the old system module took about `57.2 s` for one cycle with uncompressed checkpointing; the rebuilt module in `.venv` took about `40.0 s`, roughly `1.43x` faster. Regression tests covering native exchange/cycle behavior pass in `.venv`.
- second post-toolchain Rust optimization, `2026-04-27`: the exact exchange stage now uses a dense per-good forbidden-offer bit vector instead of a per-need `HashSet`, reuses the candidate-offer buffer inside the attempt loop, and the basket/phenomenon exchange stage reuses cached reciprocal friend slots across basket replans. These are data-structure and cache changes only; trade search order and commit order are unchanged. On the same `10000/30/150` cycle-100 checkpoint, one uncompressed exact/native cycle measured `36.2 s`, about another `1.10x` over the reciprocal-slot cache build and about `1.58x` over the old system module. The `3000/30/100/40` basket benchmark with seed `2027` improved from about `93.6 s` to `86.8 s` while preserving the same macro totals for the benchmark run. Native regression gate after rebuild: `32 passed`.
- third post-toolchain Rust optimization, `2026-04-27`: the native search uses the Rayon parallel scan only for larger candidate grids (`>= 65,536` cells). The old `4,096` threshold caused report-scale and `100`-good searches to enter Rayon for many small repeated searches, where scheduling/reduction overhead dominated useful work. The same exact/native `10000/30/150` cycle-100 checkpoint measured `27.8 s` after this change, down from `36.2 s`, with identical benchmark metrics.
- fourth post-toolchain Rust optimization, `2026-04-27`: basket/phenomenon planning now forms per-need candidates sequentially for normal `30-100` good runs instead of launching a Rayon job for every basket replan. Candidate ordering is still determined by explicit score/order sorting before commit, so this is a scheduling change rather than a behavioral change. The `3000/30/100/40` basket benchmark with seed `2027` measured `29.0 s`, down from `80.4 s` after the search-threshold change and from `93.6 s` before this optimization round, with the same observed utility, production, and rare-money values in the benchmark output.
- fifth post-toolchain Rust optimization, `2026-04-27`: the sequential native search now holds the current friend's stock, role, limit, purchase-price, and sales-price rows as local views inside the friend loop, avoiding repeated two-dimensional `ndarray` indexing in the offer loop. This is exact-preserving. The `10000/30/150` cycle-100 exact/native checkpoint measured `26.9 s`, a small improvement from `27.8 s`; the basket benchmark was effectively neutral (`29.6 s` versus `29.0 s`, same macro output), so this should be kept only if longer repeated benchmarks confirm it remains neutral or positive.
- sixth post-toolchain Rust optimization, `2026-04-27`: the exact exchange stage now reuses the forbidden-offer, base-offer, and candidate-offer buffers across needs inside one stage call instead of allocating fresh vectors for every need. A generation-counter variant was rejected because it was not faster; the accepted version keeps a simple reusable bool buffer and clears it per need. This is exact-preserving and mostly removes allocation churn rather than changing the hot arithmetic.
- seventh post-toolchain Rust optimization, `2026-04-27`: the normal sequential search now has a contiguous-slice fast path for standard NumPy layouts, with fallback to the previous `ndarray` view path for non-contiguous views. The slice path preserves friend/offer scan order and first-best tie behavior. On the `10000/30/150` cycle-100 exact/native checkpoint, the benchmark improved to about `25.5 s`; the `3000/30/100/40` basket benchmark improved to about `28.0 s` with the same output metrics.
- eighth post-toolchain Rust optimization, `2026-04-27`: the Rust-owned exact agent loop calls the exact native exchange stage directly for the accepted exact/native exchange path, bypassing the Python `_satisfy_needs_by_exchange` and `_make_surplus_deals` wrappers and adding the same trade metrics from Rust. This removed the 20,000 Python wrapper calls from a `10000`-agent cycle. The per-cycle benchmark on the same `10000/30/150` checkpoint measured about `24.0 s` when run sequentially. The basket path is intentionally left on the Python wrapper route for now because the direct basket call was not faster in the current `3000/30/100/40` benchmark.
- benchmarking note: do not run CPU-heavy exact and basket benchmarks in parallel when comparing optimization variants. A parallel measurement of the two reference benchmarks produced misleadingly worse times because they contended for CPU and memory bandwidth. Use sequential benchmark runs for acceptance numbers.
- consequence: do not schedule `10000/100/150/2000` as a routine overnight run until the candidate/proposal-heavy exchange phase is further optimized or bounded. For immediate exploratory macro work, prefer `3000/30/100` for full 2000-cycle multi-seed runs, or `10000/30/150` for shorter scale diagnostics. Treat 100-good runs as performance probes unless a new optimization changes the scaling curve.

Decision rule:

- if the question is "is the model still report-faithful," use the exact reference baseline
- if the question is "can we cheaply screen a very large scenario for growth, money emergence, and cycles," the fast exploratory baseline is acceptable as a first-pass instrument
- no exploratory result should be treated as final until it has been cross-checked on the exact baseline at smaller or report scale

### Immediate Next Steps

1. Validate Stage A at phenomenon level before using it as the default fast path.

- objective: confirm that the native exchange-stage path reproduces the same macro phenomena despite path-dependent individual/event divergence
- first gate: run `3000 / 30 / 100` to at least `500` cycles on multiple seeds and compare production growth, utility peak/plateau, rare-money emergence, friction rise, stock accumulation, inequality, and cycle/boom-bust signals
- second gate: if the fast path passes, run one exact-baseline smaller or shorter cross-check for any important qualitative claim
- debugging hook: transaction-level tracing inside Rust remains useful if macro bias appears, but it is no longer the immediate blocker if the phenomenon-level gate passes
- status: first gate passed on three seeds through `500` cycles, seed `2009` has been extended to `2000` cycles, and a `1000/30/100/2000` exact-baseline cross-check has confirmed the same key qualitative phenomena including the boom-to-decline transition. The fast path can now be used for large exploratory screening, while exact-baseline runs remain required for final validation of any scientific claim.

2. Keep the accepted exact baseline available for final validation.

- objective: confirm that the now-accepted stage-math slice still reproduces the report-style dynamics over the longer horizon where growth and cycles should emerge
- default target: `3000 / 30 / 100`, `1000-2000` cycles, checkpointed

3. Run large exploratory screening on the fast baseline only after the extended gate.

- objective: make `10000 / 100 / 150 / 2000` practical enough to use as a search instrument for macro phenomena
- record production, utility, rare-money share, cycle metrics, spoilage, and TCE waste
- treat these runs as hypothesis generation, not final evidence
- current status: first overnight attempt showed that `10000 / 100 / 150` is dominated by proposal volume and is not yet feasible for `2000` cycles. The next engineering step is to profile and reduce the exchange proposal workload before retrying a 100-good long run.

4. Cross-check any interesting large-run finding against the exact baseline.

- downscale to report size or another exact-feasible checkpointed size
- verify that the qualitative finding survives without the experimental exchange-stage shortcut

5. Continue Stage A exactness work only if extended gates find material drift.

- the remaining strict Stage A difference is currently inventory-volume floating-point roundoff
- if no material drift appears, prefer documenting this as exact-tolerant rather than spending disproportionate effort on bit-for-bit accumulator parity

6. Keep parallelism out of the mutable exact core until the full Rust sequential core is complete.

- near-term speed work should still prioritize larger contiguous Rust ownership of the sequential core
- any later CPU-thread or GPU parallelism must be evaluated only against the Rust sequential reference

### Phenomenon Path Status

Status: keep only the exact path and the Rust replan-after-trade per-agent basket path as normal work targets.

- exact path remains the reference and is not replaced
- the active phenomenon path is `--experimental-agent-basket-planning` plus `--experimental-session-replan-after-trade`, with session clearing disabled
- the agent's opportunity set is still only its own state, prices, history, and known acquaintances; no global dashboard or market aggregates are exposed
- after each accepted barter decision, Rust rebuilds the active agent's local opportunity set from the changed state before the next decision
- stale candidates are skipped by revalidation against current stock, need, price, transparency, and partner capacity
- pairwise offer exhaustion is part of the active path: if one offered good is exhausted, only the active need/offer pair is invalidated before replanning
- the path may use native stage-math helpers for preparation, production, leisure, and period-end arithmetic where isolated validation exists
- selected checkpoints and short windows should still be checked with the exact path before drawing final conclusions

Retired or diagnostic-only branches:

- `--experimental-parallel-phenomenon-exchange` wave scheduling is deprecated. It remains only for rollback/comparison because it changes timing toward simultaneous snapshot decisions.
- `--experimental-session-clearing-phenomenon-exchange` is no longer the preferred realism path. It remains only for explicit local-clearing diagnostics because it can over-synchronize the market and weaken price dispersion and merchant margins.
- Static no-replan per-agent basket lists are no longer the preferred fast branch. The latest 3000/100/100 probe was quick but much weaker macro-economically than the promising 500-cycle per-agent basket artifact.
- Multi-pass and candidate-depth variants remain stale-list diagnostics unless explicitly promoted by a later validation gate.
- `--experimental-session-global-offer-exhaustion` is rejected except for named rollback diagnostics. It over-prunes the active basket by banning one exhausted offer good for all needs; the 2026-05-07 A/B probes showed that pairwise exhaustion restored the promising c50 profile while global exhaustion reproduced the weak-growth branch.

Next safe speed target:

- reduce the cost of Rust after-trade re-planning without changing the per-agent decision order
- reuse available-offer and pairwise exhausted-offer information inside the active agent's local planner
- keep shared accumulators, especially market TCE by good, deterministic
- postpone CUDA or conflict-free session batching until the Rust replan branch has a validated speed/phenomenon baseline

### Post-Current-Run Backend Metrics Backlog

The previous current long run has finished, so this backlog can now be implemented incrementally. These changes are useful for the dashboard and research workflow, but they must remain observational: they must not mutate run state or change checkpoint resume semantics.

1. Persist welfare and effort distributions as sampled history.

- add checkpoint/sample-time DTO fields for living-standard mean, median, p10, p90, p99, Gini, and top-decile share
- add the same time series for Smith-style need-basket effort cost
- keep `/api/inequality` as the latest snapshot, but add a history-capable endpoint or extend sampled metrics so charts can show welfare cycles, not only production/trade cycles
- validation: deterministic toy-state tests for percentile, Gini, and top-share calculations
- status: sampled `metrics.jsonl` rows and `summary.latest_market` now carry the inequality, living-standard, Smith-cost, and friction fields; dashboard chart wiring now includes a welfare path for living-standard median/mean/p10/p90; remaining work is synthetic fixture coverage for exact percentile expectations

2. Add flow accounting behind the service boundary.

- split gross production into direct need consumption, surplus stock growth, inventory-mediated exchange, spoilage, and TCE/time-equivalent loss
- record both physical totals and time/price-valued totals
- expose ratios needed to ask how much gross output is absorbed by transaction friction rather than welfare
- validation: conservation-style tests where produced + opening stock equals consumed + closing stock + spoilage + transferred stock within tolerance

3. Add purchasing-power and inflation proxies suited to barter.

- fixed need-basket cost at baseline prices
- current average-market need-basket cost
- rare-money-basket price index for goods currently serving monetary roles
- agent-specific private basket cost using each agent's own purchase-price estimates
- validation: fixed synthetic price states where each index has an obvious expected value
- status: an observational value-weighted rare-money metric is now emitted beside the original quantity-based money-role metric. New checkpoints use transaction-time observed purchase, sale, and inventory-inflow values recorded at executed trades; older checkpoints fall back to current private-price weighting for compatibility. The money-role score is now restricted to agents currently acting as retailers for that good, so it measures merchant-style intermediation rather than every purchase that enters stock. This reporting path does not change agent decisions.

4. Add income, wealth, and merchant-margin diagnostics.

- period production income valued by time cost and by current market prices
- period realized consumption value
- net inventory accumulation and inventory turnover
- retailer gross margin proxy: sales value minus purchase value, separated from unsold stock revaluation
- validation: fixtures that force one producer, one consumer, and one intermediary trade chain

5. Add specialization and role-stability diagnostics.

- agent specialization index by production-time concentration
- good-level producer concentration and top-producer time focus as historical series
- role-transition rates for producer, retailer, and consumer classifications
- "exclusion" candidates: agents with low living standard, high Smith cost, low trade success, or shrinking network activity
- validation: small states with known role assignments and production shares

6. Add network spillover and brokerage metrics.

- weighted realized-interaction graph from `friend_activity` or recent exchange intensity
- rich-core 1-hop and 2-hop living-standard comparisons
- Burt-style weighted constraint, effective size, partner concentration, and partner-specialization diversity
- matched-control scaffolding for degree, role count, and prior wealth controls
- validation: hand-built graph fixtures for constraint/effective-size calculations

7. Keep diagnostics backend-aware and sampling-controlled.

- compute dense reductions on the active backend where practical, then copy only compact aggregates to host
- compute heavier graph diagnostics from checkpoint artifacts or offline batches, not every dashboard poll
- expose all additions through stable DTOs and service endpoints; dashboard code must not read raw backend arrays
- benchmark overhead at report scale before enabling any new metric by default in long runs

Acceptance rule:

- these additions are observational only; they must not change cycle decisions, trade order, random streams, or checkpoint resume behavior
- every new metric gets a fixed-input regression test and a brief dashboard explanation before it is used for research interpretation

Agent information-boundary rule:

- any decision-changing heuristic must use only the active agent's own state, own history, and direct-acquaintance observations from attempted or executed trades
- global dashboard metrics, population aggregates, and ex post monetary-role scores are forbidden as decision inputs
- the current opt-in local-liquidity stock-buffer heuristic satisfies this by reading only `friend_sold`, direct-link transparency, and the active agent's own recent turnover; see `MODEL_INFORMATION_BOUNDARY.md`
