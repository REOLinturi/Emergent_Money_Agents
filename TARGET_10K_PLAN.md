# Target 10K GPU-First Plan for Emergent Money Agents

## Purpose

This document refines the modernization plan for the concrete target model:

- population: 10,000 agents
- goods / needs: at least 30
- acquaintances per agent: up to 100
- implementation priority: paper-faithful dynamics with CUDA and GPU parallelism as the primary execution target

## Source Priority and Reconciled Assumptions

The source material is not internally consistent on scale parameters.

- `EmergentMoney.pdf` describes test data with `n = 3000`, `m = 30`, and a network of up to 100 acquaintances per agent.
- `Working_Legacy_Code_Reference.pdf` shows a legacy C configuration with `POPULATION 300`, `MAXGROUP 10`, and `SKILLS 31`.

For the new implementation, the priorities should be:

1. Paper-level model intent is primary.
2. Legacy code is a behavioral and heuristic reference, not a target architecture.
3. Legacy compile-time limits are historical hardware concessions and should not shape the new kernel.

That means the working target is explicitly:

- `N = 10000`
- `G >= 30`
- `F = 100`

## Terminology Mapping

The requested targets map to the source terminology as follows:

- population -> agents
- production varieties / needs -> goods, utilities, needs, skills
- possible acquaintances -> social network capacity, friend slots, acquaintance graph

## Main Conclusions After GPU-First Review

1. The array-based direction is correct and should be kept.
2. GPU execution must be designed into the kernel from the start, not added after an object-heavy rewrite.
3. The CPU path is still required, but only as a deterministic reference and validation path.
4. The main technical risk is not raw memory size. It is irregular barter matching with many branches and write conflicts.
5. The previous plan needs three corrections: explicit device-resident state, a two-phase barter kernel, and safeguards against pruning-induced model bias.

## What Needed Correction in the Previous Plan

The earlier 10k plan correctly rejected an object-per-agent kernel, but it still had three weak points.

### 1. Pruning can change the economics if it is too aggressive

At the target size, a naive barter scan is too expensive:

- agents: 10,000
- acquaintances per agent: 100
- goods: 30
- naive barter search: `N * F * G * G`
- order of magnitude: `10,000 * 100 * 30 * 30 = 900,000,000` good-pair checks per barter phase

That search has to be pruned on the optimized GPU path. However, any pruning can distort the dynamics if it eliminates rare but economically important exchanges.

Required safeguard:

- the CPU reference path stays exhaustive within the currently known acquaintance network
- top-K pruning is allowed only in calibrated optimized paths and only after comparison against exhaustive small-scale reference runs
- the active acquaintance frontier must include exploration, not only exploitation
- periodic full rescoring must be run on a sample of agents or cycles to check for drift

### 2. GPU barter cannot be implemented as direct in-place trading from many threads

A CUDA kernel that both scores trades and writes final stock updates immediately will run into collisions and nondeterministic behavior.

Required correction:

- separate barter into `proposal -> resolve -> commit`
- proposal phase: parallel candidate scoring
- resolve phase: choose conflict-free accepted trades
- commit phase: apply accepted transfers in a controlled pass

This is the most important GPU-specific architectural requirement missing from many CPU-first agent simulations.

### 3. Device transfers must be treated as a first-class cost

If the state lives on the GPU but every cycle copies full tensors back to the CPU for logging or UI, most CUDA gains will disappear.

Required correction:

- keep full state resident on the active backend
- move only aggregated metrics or selected debug slices to the host
- treat the dashboard as a consumer of snapshots, not as a driver of core tensor movement

## Recommended Architecture

### 1. Device-resident structure-of-arrays kernel

Use contiguous arrays for the core state:

- per-agent, per-good tensors with shape `[N, G]`
- per-agent acquaintance arrays with shape `[N, F]`
- dyadic tensors only where the economics require them

Core tensors:

- `base_need[N, G]`
- `need[N, G]`
- `stock[N, G]`
- `stock_limit[N, G]`
- `efficiency[N, G]`
- `purchase_price[N, G]`
- `sales_price[N, G]`
- `recent_prod[N, G]`
- `recent_sales[N, G]`
- `recent_purchases[N, G]`
- `talent_mask[N, G]`
- `friend_id[N, F]`
- `friend_activity[N, F]`
- `transparency[N, F, G]`

Work buffers for GPU barter:

- `active_friend_id[N, F_active]`
- `candidate_need_good[N, K_need]`
- `candidate_offer_good[N, K_offer]`
- `proposal_friend_slot[N, P]`
- `proposal_need_good[N, P]`
- `proposal_offer_good[N, P]`
- `proposal_score[N, P]`
- `accepted_trade_mask[...]`

The important design rule is that the same shapes and tensor semantics are used by both the CPU reference backend and the CUDA backend.

### 2. Use backend abstraction from day one

The implementation should not be rewritten later from NumPy to CUDA. It should be written once against a thin backend layer.

Backend responsibilities:

- array allocation
- device transfer
- top-K operations
- random seeding
- synchronization where needed
- scalar extraction for logging

Planned backends:

- `numpy`: deterministic reference backend
- `cuda`: CuPy-based device backend, with room for custom kernels later

### 3. Keep the social graph sparse and explicit

Do not use a dense `N x N` relationship matrix.

At `10,000 * 100`, the directed acquaintance graph has at most 1,000,000 edges, which is manageable in sparse fixed-width adjacency form.

The storage cost is acceptable. The hard part is evaluating the graph efficiently during barter.

### 4. Separate acquaintance capacity from active trade frontier in optimized paths

The paper-level target of 100 acquaintances should be preserved. The CPU reference now evaluates every known acquaintance each cycle; only optimized paths may narrow the frontier.

Use two layers:

- `acquaintance_capacity = 100`
- `active_trade_frontier = top K acquaintances this cycle`, for example 16 to 32

Promotion into the active frontier should depend on:

- recent successful trade volume
- transparency / trust
- current complementarity of need and stock
- random exploration budget

Safeguard against bias:

- every `R` cycles, or for a random sample of agents, score a wider frontier and compare accepted trades to the pruned result

### 5. Reduce barter search from `G^2` to top-K candidate goods only in optimized paths

Do not compare every wanted good against every offered good for every acquaintance in the optimized GPU path. The CPU reference now keeps the exhaustive good-pair scan inside the known acquaintance network.

For each agent and cycle, precompute:

- top-K unmet or high-value demanded goods
- top-K surplus or high-margin offered goods

Then evaluate only the reduced cross-product:

- example: `K_need = 4`, `K_offer = 4`
- effective search cost becomes roughly `N * F_active * 16`, not `N * F * 900`

Safeguard against bias:

- compare the top-K policy against exhaustive search on smaller scenarios
- increase K selectively for agents with low fulfillment or unstable price adjustments

### 6. Use bitmasks to skip impossible barter cases cheaply

With 30 goods, one 32-bit mask can encode:

- goods with unmet need
- goods with positive stock
- goods with talent

Cheap mask intersections can reject many acquaintance pairs before float math starts.

This is especially important in CUDA kernels, where reducing useless branch paths matters as much as raw arithmetic throughput.

### 7. Barter must use a two-phase GPU execution model

Recommended barter pipeline:

1. Build candidate frontiers.
2. Score proposals in parallel.
3. Resolve conflicts so one good unit or stock slice is not committed twice.
4. Commit accepted trades.
5. Update trust, transparency, and recent flow tensors.

This structure is necessary for both correctness and reproducibility.

### 8. Keep a deterministic numeric mode for validation

The new implementation does not need to inherit legacy integer arithmetic as the primary runtime path. That was a 2009 performance constraint.

Recommended numeric policy:

- primary runtime: `float32`
- optional compact dyadic tensors: `float16` if validated safe
- optional debug mode later: scaled integer or stricter deterministic reductions only if threshold behavior proves too sensitive

The point is not to preserve historical arithmetic. The point is to preserve the model behavior.

## Practical Memory Budget at 10k / 30 / 100

Approximate tensor sizes:

- one dense `[N, G]` `float32` tensor: `10,000 * 30 * 4 bytes ~= 1.2 MB`
- one `[N, F]` `int32` tensor: `10,000 * 100 * 4 bytes ~= 4.0 MB`
- one `[N, F, G]` `float32` tensor: `10,000 * 100 * 30 * 4 bytes ~= 120 MB`
- one `[N, F, G]` `float16` tensor: `~= 60 MB`

Conclusion:

- the target model is comfortably within a 96 GB GPU memory budget
- the real performance challenge is kernel structure, not raw memory capacity

## GPU Strategy for RTX 6000 Class Hardware

For an RTX 6000 class device with 96 GB VRAM, the recommended approach is:

1. Design the state layout for device residency from the start.
2. Keep the validation path on CPU with identical tensor semantics.
3. Move dense phases to CUDA first:
   - reset cycle buffers
   - consume from stock
   - produce for needs
   - produce surplus
   - apply spoilage and stock limits
   - aggregate market metrics
4. Implement barter proposal scoring next.
5. Implement conflict resolution and commit after the proposal path is stable.
6. Keep UI and logging out of the hot path.

Important caution:

- CuPy alone is enough for dense tensor phases
- barter proposal scoring may still need custom kernels or Numba CUDA later
- multi-GPU support is unnecessary at this stage and should not be a design goal yet

## Recommended Stack

### Core engine

- Python 3.11
- NumPy for the reference backend
- CuPy for the CUDA backend
- optional Numba later for custom kernels or CPU micro-optimizations
- Pytest for regression testing

### UI

- Plotly Dash

Mesa should not be used for the main runtime path. It is useful for small pedagogical prototypes but works against the GPU-first array architecture.

### Data and experiment outputs

- CSV or Parquet for aggregate time series
- JSON or YAML for scenario configs
- optional checkpoint snapshots for long runs

## Revised Delivery Plan

### Phase 0: source reconciliation and invariant capture

Deliverables:

- parameter sheet for the baseline model
- mapping from paper concepts to tensor fields
- list of invariants taken from the paper and legacy heuristics

Acceptance criteria:

- explicit decision that paper-level model intent is primary
- explicit decision that legacy compile-time limits are not binding

### Phase 1: backend-aware scaffold

Deliverables:

- project skeleton
- backend abstraction
- config system
- device-compatible state container

Acceptance criteria:

- same state semantics for CPU and CUDA backends
- no object-per-agent dependency in the kernel

### Phase 2: deterministic reference kernel

Deliverables:

- initialization path
- cycle reset
- stock consumption
- need production
- surplus production
- spoilage and price-update placeholders

Acceptance criteria:

- reproducible small runs
- unit tests pass
- metrics can be produced without UI

### Phase 3: barter frontier and proposal kernel

Deliverables:

- active frontier selection
- top-K candidate goods
- proposal buffers
- proposal scoring path

Acceptance criteria:

- pruning can be inspected and benchmarked
- proposal tensors are ready for conflict resolution

### Phase 4: resolve and commit barter

Deliverables:

- conflict resolution stage
- accepted trade commit stage
- trust and transparency updates tied to accepted trades

Acceptance criteria:

- no double-commit conflicts
- conservation checks pass

### Phase 5: validation against paper-level behavior

Deliverables:

- small exhaustive reference scenarios
- pruning drift checks
- qualitative validation of specialization, longer supply chains, and exchange-medium concentration

Acceptance criteria:

- behavioral correspondence with the paper is acceptable
- pruning does not obviously suppress emergent money dynamics

### Phase 6: 10,000-agent performance run on CUDA

Deliverables:

- full-scale benchmark report
- memory profile
- hot-path profile

Acceptance criteria:

- stable multi-cycle execution at `10,000 / 30 / 100`
- CUDA backend clearly outperforms the reference backend on the target scenario

### Phase 7: dashboard and experiment controls

Deliverables:

- global metrics dashboard
- agent drill-down on sampled state
- intervention controls
- checkpoint support

Acceptance criteria:

- UI consumes snapshots, not full-state copies each cycle
- UI does not throttle the kernel

## Risks to Manage Early

1. Aggressive pruning can remove rare exchange paths that matter to emergent money formation.
2. Non-deterministic GPU reductions can create threshold drift in pricing and tie-breaking.
3. Proposal and commit phases can silently violate conservation unless explicitly tested.
4. Host-device copies can erase CUDA gains if logging and UI are naive.
5. A paper-faithful model may still require heuristic simplifications in the first CUDA version; those must be documented and tested against smaller exhaustive runs.

## Immediate Implementation Guidance

The next implementation step should be to create a backend-aware project skeleton with:

- one shared config model
- one shared state layout
- a NumPy reference backend
- a CuPy CUDA backend stub
- a simulation engine whose phases already match the future CUDA pipeline

That skeleton should be considered the first real code artifact of the new model, not merely scaffolding.

## Engineering Discipline and UI Boundary

The coding rules, backend decision, and UI/service interfaces are now fixed in `ARCHITECTURE.md`.

The implementation rules are:

- heuristics stay separate from backend mechanics
- UI reads snapshot DTOs through a service layer
- CPU and CUDA backends share the same tensor semantics
- pruning rules must remain testable against smaller exhaustive reference scenarios

## Current Reference Milestone

The current CPU reference implementation now covers these parts of the plan:

- backend-aware state allocation
- deterministic initialization with paper-first talent defaults
- sparse directed acquaintance slots with one explored contact update per cycle
- activity-ranked active acquaintance frontier for snapshots and debug views
- exhaustive barter proposal scoring across all known acquaintances and all good pairs
- first-round barter proposal, resolve, and commit
- second-round leisure demand extension and stock-room-aware barter
- cycle-level metrics aggregated across the whole two-round cycle

The next implementation target remains the same:

- move `resolve -> commit` toward backend-specific execution without changing the above semantics
- keep the CPU path as the deterministic calibration reference for CUDA
