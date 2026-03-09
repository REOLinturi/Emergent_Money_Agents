# Architecture and Engineering Rules

## Purpose

This document records the implementation rules that are fixed for the current Emergent Money rebuild.

They exist to preserve:

- paper-faithful behavior
- automatic testability
- maintainability under heuristic change
- compatibility between CPU and CUDA execution paths

## Non-Negotiable Rules

1. The paper-level model is primary. Legacy C is a heuristic reference, not an architectural template.
2. The runtime is array-based. There is no object-per-agent kernel.
3. Backend mechanics and economic rules stay separate.
4. UI code does not read full backend state directly. It uses service-layer snapshots.
5. CPU and CUDA paths must preserve the same state semantics.
6. Performance work may change execution strategy, but it may not simplify barter decision rules.
7. Every performance optimization must add a reference-preservation test against fixed inputs or a reference backend.
8. All known acquaintances and all good pairs remain part of the current barter semantics.

## Code Layers

### 1. Domain configuration and state

Files:

- `config.py`
- `state.py`
- `dto.py`

Rules:

- `SimulationConfig` is the single source of truth for scenario and kernel parameters.
- `SimulationState` defines the tensor layout shared by all backends.
- DTOs define the host-side service boundary.

### 2. Backend layer

Files:

- `backend/base.py`
- `backend/numpy_backend.py`
- `backend/cuda_backend.py`

Rules:

- Backends allocate arrays and expose primitive operations.
- Backends own the trade proposal-support, resolve, commit, and contact-planning contracts used by the engine.
- Backends do not encode independent economic policy.
- The reference backend is `NumPy`.
- The production backend is `CuPy`.
- Custom CUDA kernels may be added later, but only behind the same backend contract.

### 3. Simulation phases

Files:

- `initialization.py`
- `engine.py`
- future modules such as `phases/proposal.py`, `phases/resolve.py`, `phases/pricing.py`

Rules:

- The engine controls ordering.
- Each phase should stay small and testable.
- Barter remains split into `proposal -> resolve -> commit`.
- Proposal scoring must not mutate stock.
- Commit must preserve conservation and avoid double-spend.

### 4. Service and UI boundary

Files:

- `service.py`
- later Dash application files

Rules:

- The service layer is the only supported UI integration surface.
- UI consumers request snapshots and submit commands.
- Full tensors remain on the active backend unless a narrow snapshot path copies data to the host.

## Report-Aligned Mechanics Currently Fixed

The following behaviors are intentionally aligned with the original report and legacy annotations:

- initial production efficiency starts at `1.0`
- talent is a binary random assignment, not a continuously random talent strength
- a talented good gets a modest initial efficiency advantage of `+50%`
- the default talent density is `50%`, matching the paper's explicit test-calibration sentence
- learning efficiency grows as the square root of discounted recent production divided by discounted own need over the same window
- surplus production is restricted to talented goods in the current paper-aligned path
- surplus-good choice is ranked by current efficiency and private sales price
- learned efficiency may later exceed the initial talent advantage
- the cycle contains a second, leisure-driven demand round after the first basic-needs round
- the second round allows barter into inventory room, not only into immediate unmet need

Important note:

- the paper text is not fully self-consistent on talent density; the code therefore keeps density configurable, but the talent mechanism itself is fixed

## Current Cycle Semantics

The deterministic reference path currently uses this order:

1. reset cycle needs, time budget, discounted histories, and trade buffers
2. run the basic-needs market round:
   - consume from stock
   - build the active acquaintance frontier for snapshots from the agent's own activity memory
   - refresh candidate-good buffers for snapshots and later experiments
   - score, resolve, and commit barter across all known acquaintances and all good pairs for immediate need
   - produce for remaining need
   - introduce one explored random acquaintance candidate and replace an empty or least-active slot if needed
   - produce surplus to stock
3. if leisure time remains, add temporary extra demand using the paper's `needsincrement` idea with market-level price-elastic weighting
4. run a second market round with the same exhaustive known-network barter scan, now allowing trade either for immediate need or for inventory room
5. update learning-based efficiency, spoilage, and private prices

This is the baseline semantics that all backends must preserve.

## CUDA Execution Policy

The current CUDA path preserves the same barter semantics as the CPU reference path.

It does this by changing execution strategy, not economics:

- all known acquaintances are still evaluated
- all good pairs are still evaluated
- CUDA proposal scoring uses a blocked exhaustive scan over friend slots, needed goods, and offered goods
- the block sizes are configurable through `SimulationConfig.cuda_friend_block` and `SimulationConfig.cuda_goods_block`
- the current tuned defaults are `12` friend slots and `25` goods per block

This blocked scan exists to reduce Python-loop overhead and improve GPU occupancy. It is not a heuristic filter.

Current backend status:

- proposal: NumPy reference loop and CuPy blocked exhaustive implementation
- resolve: NumPy-native backend contract and CUDA-native device kernel
- commit: NumPy-native backend contract and CUDA-native device kernel that consumes resolve-stage goods state directly

## Preferred Programming Style

### Pure and isolated heuristics

Prefer:

- explicit inputs and outputs
- deterministic seeding
- stable tie-breaking where practical
- one well-defined place for each policy rule

Avoid:

- hidden mutation spread across many modules
- backend-specific policy rules in random files
- UI-dependent state changes in the kernel

### Phase-level invariants

Each simulation phase should have testable invariants.

Examples:

- consumption does not create or destroy stock
- proposal generation does not mutate stock
- commit does not double-spend stock
- spoilage only decreases stock
- prices stay within valid bounds
- cycle metrics aggregate all trade rounds, not only the final proposal buffer

### Small exhaustive reference cases

Any execution optimization must be validated against tiny exhaustive scenarios.

Required examples:

- 2 agents, 2 goods, 1 acquaintance
- 4 to 8 agents with small graphs and small good counts

These cases guard against silent model drift.

## Backend Decision

The chosen backend strategy is:

- `NumPy` for deterministic reference runs and tests
- `CuPy` for the main CUDA execution path
- custom CUDA kernels later only for measured hot paths

Not selected as the primary runtime approach:

- `Mesa`
- `PyTorch`
- `JAX`
- `FLAME GPU`

Reason:

- the project needs paper-faithful custom heuristics and explicit control over barter logic more than it needs a generic agent framework

## Service Interface for UI

The first-class UI boundary is `SimulationService`.

Current implemented methods:

- `create(config, backend_name)`
- `reset(config, backend_name)`
- `step(cycles)`
- `pause()`
- `resume()`
- `get_status()`
- `get_market_snapshot()`
- `get_agent_snapshot(agent_id)`
- `get_network_slice(agent_id, limit)`
- `get_trade_sample(agent_ids, limit)`

Current DTOs:

- `RunStatus`
- `MarketSnapshot`
- `AgentSnapshot`
- `NetworkSlice`
- `TradeProposalView`

Future service methods should follow the same pattern:

- input: explicit command or parameter object
- output: stable DTO or list of DTOs
- no raw backend arrays returned to UI callers

## Current Technical Scope

The current codebase includes:

- backend-aware state allocation
- deterministic initialization
- sparse directed acquaintance slots that start empty
- one explored contact update per agent and cycle with backend-owned candidate planning plus empty-slot fill or least-active replacement
- activity-ranked active acquaintance frontier for snapshots
- candidate-good buffers for snapshots and later experiments
- backend-owned `proposal -> resolve -> commit` contracts
- exhaustive barter scoring across all known acquaintances and all good pairs on the CPU reference path
- the same exhaustive barter scoring on CUDA through a blocked friend-and-goods scan
- stock-room-aware second-round barter resolution
- leisure-driven temporary demand extension
- service-layer snapshots
- cycle-level market metrics aggregated across all trade rounds

It does not yet include:

- policy intervention commands
- dashboard implementation

## Current Measured Throughput

Measured in this workspace on an `RTX 4090` after warm-up:

- `3,000 / 30 / 100`: CUDA path about `0.031 s` per cycle versus about `11.75 s` for single-thread CPU
- `10,000 / 100 / 150`: CUDA path about `0.157 s` per cycle with `cuda_friend_block=12` and `cuda_goods_block=25`

These numbers are implementation-specific, but they confirm that the current exact CUDA path is now well inside the intended runtime class.