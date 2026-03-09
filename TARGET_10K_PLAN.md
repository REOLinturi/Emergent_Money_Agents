# Target 10K GPU-First Plan for Emergent Money Agents

## Purpose

This document records the current execution plan for the practical target model:

- population: `10,000`
- goods / needs: `100`
- acquaintances per agent: `150`
- target run length: `1,000+` cycles
- implementation priority: paper-faithful dynamics with CUDA and GPU parallelism as the primary execution target
- operational target: complete a `10,000 / 100 / 150 / 1,000+` run in at most a few hours on a single high-end GPU

The paper's `30 / 100` test scale remains the behavioral lower bound, but it is no longer the practical performance target.

## Source Priority

1. The paper-level model intent is primary.
2. Legacy C is a behavioral and heuristic reference, not a target architecture.
3. Legacy compile-time limits and integer arithmetic are historical hardware concessions and do not constrain the new kernel.

## Fixed Modeling Constraint

The current optimization policy is strict:

- do not simplify barter decision rules
- do not prune acquaintances on the optimized path
- do not prune good pairs on the optimized path
- preserve the same barter semantics on CPU and CUDA

Performance work may change how the exhaustive search is executed, but not what is evaluated.

## Current Model Semantics

The current implementation preserves these report-aligned mechanics:

- sparse directed acquaintance slots start empty
- one explored random contact is evaluated per agent and cycle
- empty slots are filled before least-active replacement
- barter is split into `proposal -> resolve -> commit`
- the first market round serves immediate need
- the second market round serves leisure-driven extra demand and inventory-room trades
- learning can eventually dominate the initial talent advantage
- proposal scoring evaluates all known acquaintances and all good pairs

## Why the GPU Problem Is Still Hard

At the practical target, one exhaustive barter round implies:

- agents: `10,000`
- acquaintances: `150`
- goods: `100`
- naive good-pair checks: `N * F * G * G`
- order of magnitude: `10,000 * 150 * 100 * 100 = 15,000,000,000` pair checks per barter phase

The challenge is therefore not model definition but execution structure:

- irregular friend lookups
- large 4D intermediate tensors
- conflict-free commit
- keeping state resident on the active backend

## Current Execution Strategy

### 1. Array-based structure-of-arrays state

Core state remains device-compatible and shared across backends:

- dense per-agent per-good tensors such as `need[N, G]`, `stock[N, G]`, `efficiency[N, G]`, `purchase_price[N, G]`, `sales_price[N, G]`
- sparse fixed-width acquaintance tensors such as `friend_id[N, F]`, `friend_activity[N, F]`, `transparency[N, F, G]`
- trade work buffers for proposals, accepted quantities, and snapshots

### 2. Exact blocked CUDA proposal scan

The current CUDA path keeps exhaustive barter semantics by tiling the search instead of pruning it.

The proposal path now scans:

- friend slots in blocks of `cuda_friend_block`
- needed goods in blocks of `cuda_goods_block`
- offered goods in blocks of `cuda_goods_block`

For each block, CUDA computes the same quantity and exchange-index logic as the CPU reference path, then keeps the best profitable trade per agent with the same stable tie behavior.

This changes execution cost and memory locality, but not the decision rule.

### 3. Backend-owned conflict handling

Barter remains split into:

1. proposal
2. resolve
3. commit

Current backend status:

- proposal: NumPy reference loop plus CuPy blocked exhaustive implementation
- resolve: backend contract with NumPy-native execution and CUDA-native device kernel
- commit: backend contract with NumPy-native execution and CUDA-native device kernel that reuses resolve-stage goods state instead of replaying goods transfers

The next remaining gap is no longer correctness of barter execution but further fusion and batching of dense phases.

### 4. Device-resident state policy

Full state should remain on the active backend.

Only the following should move to the host during long runs:

- aggregate metrics
- sampled snapshots
- explicit debug slices
- checkpoint output when requested

UI, logging, and reporting must not pull full tensors back every cycle.

## Practical Memory Budget at 10k / 100 / 150

Approximate persistent tensor sizes:

- one dense `[N, G]` `float32` tensor: `10,000 * 100 * 4 bytes ~= 4 MB`
- one `[N, F]` `int32` tensor: `10,000 * 150 * 4 bytes ~= 6 MB`
- one `[N, F, G]` `float32` tensor: `10,000 * 150 * 100 * 4 bytes ~= 600 MB`

The blocked proposal path also creates temporary 4D work buffers. With the current tuned block sizes, this fits comfortably on the verified `RTX 4090` test machine and is far inside the expected budget of a `96 GB` class workstation GPU.

Conclusion:

- the practical target is memory-feasible
- the remaining performance work is about kernel structure, not capacity

## Recommended Stack

### Core engine

- Python `3.11`
- `NumPy` for the reference backend
- `CuPy` for the CUDA backend
- optional custom CUDA kernels later only for measured hot spots
- `Pytest` for regression testing

### UI

- Plotly Dash behind the `SimulationService` snapshot boundary

## Current Measured Performance

Measured in this workspace on an `RTX 4090` after warm-up:

- `3,000 / 30 / 100`
  - single-thread CPU: about `11.75 s` per cycle
  - CUDA: about `0.031 s` per cycle
- `10,000 / 100 / 150`
  - CUDA with `cuda_friend_block=12`, `cuda_goods_block=25`: about `0.157 s` per cycle
  - `1,000` cycles: about `2.6 min`
  - `3,000` cycles: about `7.9 min`

These measurements show that the current exact CUDA path is now comfortably beyond the practical runtime target.

## Delivery Plan From Here

### Phase 1: exact reference semantics

Status: done

Deliverables already in place:

- deterministic initialization
- paper-aligned talents and learning scaffold
- sparse directed acquaintance growth
- two-round cycle semantics
- exhaustive barter proposal scoring on CPU
- regression coverage for small exhaustive cases

### Phase 2: exact CUDA proposal acceleration

Status: done

Deliverables already in place:

- blocked exhaustive CUDA proposal path
- configurable friend and goods block sizes
- regression proving blocked proposal matches the reference loop on small scenarios
- practical throughput measurements on real hardware

### Phase 3: device-native resolve and commit

Status: done

Deliverables now in place:

- CUDA-native resolve kernel behind the existing backend contract
- CUDA-native commit kernel behind the existing backend contract
- conservation and no-double-spend regressions shared with the CPU path

Acceptance criteria met:

- no semantic drift from the current CPU reference behavior
- no host-side commit loop in normal CUDA execution

### Phase 4: dense-phase fusion and long-run stability

Deliverables:

- fuse or batch dense update phases where measurements justify it
- long-run checkpoint support
- monitoring for host-device copies during full-scale runs

Acceptance criteria:

- stable `10,000 / 100 / 150 / 1,000+` execution
- runtime remains in the few-hours class on a single high-end GPU

### Phase 5: dashboard and experiment control

Deliverables:

- global metrics dashboard
- sampled agent drill-down
- intervention controls
- checkpoint restore flow

Acceptance criteria:

- UI consumes snapshots only
- UI does not throttle kernel throughput

## Risks To Keep Visible

1. Host-device copies can still erase gains if logging or UI bypass the snapshot boundary.
2. Large temporary block tensors can create hidden memory pressure if block sizes are increased carelessly.
3. The current resolve and commit kernels are semantically exact but still intentionally simple; later fusion should be measurement-driven.
4. Any future pruning experiment must remain optional, explicit, and validated against the exact path.

## Immediate Next Step

The next implementation task is clear:

- profile the dense non-barter phases and fuse or batch the ones that still create avoidable CUDA launch overhead, while preserving the current exact `proposal -> resolve -> commit` contract.