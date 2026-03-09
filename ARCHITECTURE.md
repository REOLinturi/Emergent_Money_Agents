# Architecture and Engineering Rules

## Purpose

This document records the implementation rules that are now fixed for the new Emergent Money codebase.

They are intended to improve:

- automatic testability
- heuristic replaceability
- long-term maintainability
- compatibility between CPU and CUDA execution paths

## Non-Negotiable Engineering Rules

1. The paper-level model is primary. Legacy C is a heuristic reference, not an architectural template.
2. The simulation kernel is array-based. There is no object-per-agent runtime model.
3. Backend code and economic heuristics are separated.
4. UI code is not allowed to read full backend state directly. It must use service-layer snapshots.
5. Every optimization must preserve the same state semantics on CPU and CUDA backends.
6. Every pruning rule must remain calibratable against smaller exhaustive reference runs.

## Code Organization Rules

The codebase is split into four layers.

### 1. Domain configuration and state

Files:

- `config.py`
- `state.py`
- `dto.py`

Rules:

- `SimulationConfig` is the single source of truth for scenario parameters.
- `SimulationState` defines the tensor layout shared by all backends.
- DTO classes define host-side snapshots and service return values.

### 2. Backend layer

Files:

- `backend/base.py`
- `backend/numpy_backend.py`
- `backend/cuda_backend.py`

Rules:

- Backends allocate arrays and expose primitive operations.
- Backends own the trade resolve and commit contracts used by the engine.
- Backends do not encode economic rules beyond preserving the agreed resolve and commit semantics.
- The reference backend is `NumPy`.
- The production backend is `CuPy`.
- Custom CUDA kernels may be added later, but only behind the same backend contract.

### 3. Simulation phases

Files:

- `initialization.py`
- `engine.py`
- future phase modules such as `phases/proposal.py`, `phases/resolve.py`, `phases/pricing.py`

Rules:

- Each phase should be a small, testable unit.
- The preferred direction is to move phase logic out of `engine.py` into dedicated modules as behavior grows.
- The engine controls ordering; phase modules implement behavior.
- Barter remains split into `proposal -> resolve -> commit`.

### 4. Service and UI boundary

Files:

- `service.py`
- Dash app files later

Rules:

- The service layer is the only supported UI integration surface.
- UI consumers request snapshots and submit commands.
- Full tensors remain on the active backend unless a narrow debug or snapshot path explicitly copies data to the host.

## Report-Aligned Mechanics Currently Fixed

The following behaviors are intentionally aligned with the original report and legacy annotations:

- initial production efficiency starts at `1.0`
- talent is a binary random assignment, not a continuously random talent strength
- a talented good gets a modest initial efficiency advantage of `+50%`
- the default talent density is `50%`, matching the paper's explicit test-calibration sentence
- learning efficiency grows as the square root of discounted recent production divided by discounted own need over the same window
- surplus production is restricted to talented goods in the current paper-aligned reference path
- surplus-good choice is ranked by current efficiency and private sales price
- learned efficiency may later exceed the initial talent advantage
- the cycle now contains a second, leisure-driven demand round after the first basic-needs round
- the second round allows barter into inventory room, not only into immediate unmet need

Important note:

- the paper text is not fully self-consistent on talent density: one passage says `50% of all needs` as talents in the test simulation, while another says about `20% probability` per utility; the code keeps talent density configurable, but the default is now the paper's explicit test calibration and the talent mechanism itself is fixed
- the paper also mentions a slower inter-cycle adjustment of the leisure-driven needs increase; the current reference path uses a per-cycle cap instead of a separate persistent leisure state, and this should be revisited during calibration

## Current Reference Cycle

The deterministic reference path currently uses this order:

1. reset cycle needs, time budget, discounted histories, and trade buffers
2. run the basic-needs market round:
   - consume from stock
   - build the active acquaintance frontier for snapshots from the agent's own activity memory
   - refresh candidate-good buffers for snapshots and later pruning experiments
   - score and resolve barter proposals across all known acquaintances and all good pairs for immediate need
   - produce for remaining need
   - introduce one explored random acquaintance candidate and replace an empty or least-active slot if needed
   - produce surplus to stock
3. if leisure time remains, add temporary extra demand using the paper's `needsincrement` idea with market-level price-elastic weighting
4. run a second market round with the same exhaustive known-network barter scan, now allowing trade either for immediate need or for inventory room
5. update learning-based efficiency, spoilage, and private prices

This two-round structure is now the baseline CPU reference semantics that CUDA must preserve. Any later pruning belongs to calibrated optimized paths, not to the reference model.

## Preferred Programming Style

### Pure and isolated heuristics

When possible, heuristics should be written as small functions that operate on explicit arrays or slices and return derived scores or choices.

Avoid:

- implicit mutation spread across many modules
- hidden dependence on UI state
- backend-specific heuristic code in random modules

Prefer:

- explicit inputs and outputs
- deterministic seeding
- stable tie-breaking where practical
- one well-defined place for each policy rule

### Phase-level invariants

Each simulation phase should have testable invariants.

Examples:

- consumption does not create or destroy stock
- proposal generation does not mutate stock
- commit does not double-spend stock
- spoilage only decreases stock
- prices stay within valid bounds
- cycle metrics aggregate all trade rounds, not only the final overwritten proposal buffer

### Small exhaustive reference cases

Any heuristic that reduces search space should be validated against tiny exhaustive scenarios.

Required examples:

- 2 agents, 2 goods, 1 acquaintance
- 4 to 8 agents with small graphs and small good counts

These cases are the protection against silent model drift.

## Backend Decision

The chosen backend strategy is:

- `NumPy` for deterministic reference runs and tests
- `CuPy` for the main CUDA execution path
- custom CUDA kernels later for hot barter paths if CuPy array ops are not enough

Not selected as the primary backend approach:

- `Mesa`
- `PyTorch`
- `JAX`
- `FLAME GPU`

Reason:

- the project needs paper-faithful custom heuristics and explicit control over sparse barter logic more than it needs a generic agent framework

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
- one explored contact update per agent and cycle with empty-slot fill or least-active replacement
- activity-ranked active acquaintance frontier with deterministic tie-breaks for snapshots
- candidate-good buffers for snapshots and future pruning calibration
- backend-owned `proposal -> resolve -> commit` with NumPy-native and CUDA host-fallback implementations
- exhaustive barter scoring across all known acquaintances and all good pairs in the CPU reference path
- stock-room-aware second-round barter resolution in the CPU reference path
- leisure-driven temporary demand extension
- service-layer snapshots
- cycle-level market metrics aggregated across all trade rounds

It does not yet include:

- CUDA-native resolve and commit kernels beyond the current backend-level host fallback
- policy intervention commands
- dashboard implementation

Those belong to the next iterations, but the interface boundary is already fixed.
