# Market Order and Parallelism

## Purpose

This note explains why the current exact reference path keeps barter commit sequential, how that choice maps to real human exchange, and when more parallel execution may become economically plausible without changing the model into a different institution.

## Two Exchange Semantics

### 1. Sequential committed barter

In the sequential exact path, one agent's negotiation commits before the next relevant negotiation sees the world.

This means:

- inventories change immediately after a trade
- unmet need changes immediately after a trade
- time use, prices, transparency, and learning signals change immediately after a trade
- the next negotiation is evaluated against the already changed state

This is the closest computational analogue to a local bazaar or village barter process in which people act in parallel at society scale, but each actual agent can only close one negotiation at a time.

### 2. Speculative parallel barter

In a speculative parallel path, multiple agents can evaluate or even tentatively plan trades against the same pre-commit snapshot.

This means:

- several buyers may all see the same attractive seller as available
- several agents may all plan around the same stock before any one trade commits
- stale information survives until conflict resolution
- after one trade wins, the losing plans must be discarded or recomputed

This is not merely a faster implementation of the same institution. It is a different market microstructure: more like decentralized simultaneous search under delayed confirmation.

## Analogy to Early Human Exchange

For relatively primitive barter economies, the sequential committed interpretation is the better default model.

Reason:

- the same desirable counterparty cannot complete several negotiations at once
- each realized trade immediately affects what that person still has, still needs, and can still promise
- a later trader does not meet the same seller in the same state as an earlier trader did

So the exact sequential path should not be read as "the whole society literally acts in one global queue". It is better read as many local interactions whose state becomes binding as soon as a concrete trade occurs.

The speculative parallel interpretation is also realistic in a narrower sense, but for a different world:

- people search widely and simultaneously
- information is delayed or noisy
- many intentions fail at confirmation time
- some agents repeatedly lose access to high-quality counterparties because someone else got there first

That can happen in reality, but if it materially changes growth, trade volume, or money emergence, then the model has changed institutionally, not just computationally.

## Why the Exact Reference Stays Sequential

The project uses the report-faithful exact path as the reference because the target is to reproduce the report's economic mechanism, not merely to generate broadly similar-looking dynamics.

A faster path is acceptable only if it still preserves the relevant phenomena:

- aggregate growth
- utility growth
- emergence of media of exchange
- cyclical behavior
- failure and recovery dynamics

If a parallel path changes which trades happen, not only when they happen, then it changes the economic weighting of the process.

## Collision Handling and Rollback

A speculative parallel design with collision detection is possible in theory:

1. build tentative trades from a read-only snapshot
2. detect collisions where proposed trades share an agent
3. choose one winner in each conflicting set
4. discard or retry the losers

But rollback alone is not enough.

Once the winner commits, the losers are no longer valid decisions from the same world state. They must be replanned. Because of that, speculative rollback is best understood as an asynchronous matching algorithm, not as exact simultaneous barter.

## When More Parallelism Becomes More Realistic

Greater parallelism becomes more defensible when the economy itself develops institutions that reduce the importance of immediate bilateral sequencing.

Examples:

- posted prices or public quotations
- merchants or retailers holding inventory for repeated sale
- explicit market days or matching windows
- specialist intermediaries with standing offers
- richer transparency and price reporting

In that world, many agents can perform read-only work in parallel without changing semantics very much:

- scanning posted prices
- scoring candidate suppliers
- ranking opportunities
- building provisional demand or supply schedules

Even then, scarce inventory still has to be committed in some conflict-resolving order. But the amount of safe pre-commit parallel work becomes much larger.

## Opportunity Sets and Realistic Search

The project should not treat arbitrary candidate pruning as a neutral optimization.

For a primitive barter setting, it is plausible that a trader cannot know the whole society. It is also plausible, once a trader is interacting with known acquaintances, that the trader can inspect what those acquaintances hold and compare all relevant goods before making one decision. That latter assumption supports a full local search over known counterparties and goods.

Therefore:

- reducing the number of goods or counterparties considered is a model assumption, not just an implementation detail
- such reduction is acceptable only when it represents a named behavioral mechanism, for example bounded attention, incomplete display of stock, search cost, reputation filters, or institutional market rules
- if the purpose is to preserve the broad real-world barter analogy, speedups should come first from parallel evaluation of the same opportunity set
- full-search basket evaluation is a plausible realism path even though it differs from the exact legacy implementation

This makes the engineering target clear: keep mutable trade commit controlled, but parallelize read-only opportunity evaluation and best-offer reductions as aggressively as the hardware allows.

## Information Boundary

Parallel or richer heuristic paths must not give agents a global market view.

An agent may use its own stock, needs, prices, and direct trading history, plus observations from attempted or completed trades with known acquaintances. It may infer local acceptability if its own direct friends have accepted a good often enough. It may not read population-level money scores, global role counts, market-wide prices, or dashboard aggregates.

This boundary is part of the realism constraint. It allows a trader to inspect and compare locally visible opportunities, but prevents a hidden auctioneer or statistical office from entering the primitive barter model.

## Practical Engineering Rule

Current rule:

- keep trade commit sequential in the exact reference path
- finish the Rust sequential exact core first
- explore parallelism around read-only search, scoring, and reductions before considering any behavior-changing pruning

Future rule once richer market institutions are modeled:

- allow more parallel quote lookup and candidate construction
- consider batch matching only where the market mechanism itself justifies delayed commit
- validate every such change against the Rust sequential reference over multiple seeds and long horizons

## Current Interpretation for This Repository

The accepted interpretation is therefore:

- the sequential exact path is the best current model of primitive barter dynamics
- faster speculative paths are useful as experiments, not as automatic replacements
- more aggressive parallelism may become appropriate later if the modeled economy itself acquires posted-price and intermediary-like behavior

## Implemented Phenomenon-Screening Paths

The repository now has two phenomenon-screening paths. The old wave-based path remains available for rollback and comparison, but is marked deprecated. New exploratory phenomenon runs should use `--experimental-session-clearing-phenomenon-exchange`.

Neither path is the exact reference. Both keep the information boundary local but change the timing model.

### Deprecated wave path

`--experimental-parallel-phenomenon-exchange` is the deprecated wave path:

- all active agents in the wave evaluate locally visible exchange opportunities from the same read-only snapshot
- candidate scoring and best-offer reductions are performed in Rust with Rayon across agents
- the scheduler keeps at most one committed decision per agent per wave
- Rust materializes only each active agent's best next proposal for the scheduler; it still searches the full local opportunity set before that reduction
- after each wave, active-agent continuation is determined by the next Rust planning pass rather than by a Python full-goods surplus scan. This keeps the same local information boundary while avoiding a redundant `agents * goods * waves` Python bottleneck.
- candidates sharing a proposer or partner are treated as conflicts and discarded or retried in the next wave
- the actual state mutation still uses the same exchange execution semantics for the scheduled conflict-free trades
- preparation, need production, surplus production, leisure, and period-end arithmetic use existing native helpers where they have isolated parity coverage. The exact path remains separately gated before any such helper is promoted to strict reference use.
- the entire phenomenon exchange stage can now be executed inside Rust, so Python no longer loops over every conflict-resolution wave. This is still not enough for the largest `100`-good probes when the market creates thousands of sequential surplus waves. The next large speed gain is therefore a model-level timing choice, such as bilateral session clearing, not another Python-overhead cleanup.

This path is kept only as a validation baseline while the session-clearing path is tested. It should not accumulate new production features unless needed for comparison.

### Session-clearing path

`--experimental-session-clearing-phenomenon-exchange` is the new preferred realism path:

- all active agents first enter the relevant decision stage
- each agent then runs a local trading session over its known acquaintances and goods
- the agent may compare the locally visible opportunity set rather than repeatedly seeing one fixed friend/good pair at a time
- every candidate trade is revalidated against current stock, need, price, transparency, and capacity before commit
- stale or no-longer-profitable candidates are skipped rather than committed from an obsolete snapshot
- no agent receives global money scores, market-wide prices, role counts, or dashboard aggregates

The current implementation builds one ranked local shopping list per session stage, commits feasible candidates in that order, and does not rebuild the whole local basket after every accepted trade. This is the main speed distinction from the deprecated wave path. It is a semantic replacement candidate, not an exact replay path. If no anomalies appear, the wave path should be removed rather than maintained indefinitely.

With no explicit hybrid parameters, both phenomenon paths use a population-wide frontier, disable frontier-partner blocking, and run both the consumption and surplus exchange stages. This makes them suitable for large exploratory runs where the goal is phenomenon-level screening rather than event-exact replay.

### Basket completion and re-planning diagnostics

The 100-good probes exposed a failure mode that was mostly hidden at 30 goods: agents can accumulate large aggregate inventories while many individual consumption baskets remain incomplete. The practical symptom is rising physical production without a corresponding rise in living-standard metrics.

Two mechanisms were identified:

- Surplus exchange needs an explicit, local, own-consumption motive. The optional `--experimental-aspirational-stock-target N` gives an agent a reason to trade surplus goods for missing goods in its own future consumption buffer, up to `N` times current own need. This does not give the agent any global information.
- A single precomputed shopping list can become too stale. If the list is not rebuilt often enough, high-scoring early trades can leave later basket gaps without usable fallback trades.

Current implementation status:

- `--experimental-session-replan-passes N` rebuilds the local shopping list for each agent session up to `N` times. This is the current practical speed/quality control.
- `--experimental-session-replan-after-trade` rebuilds the local shopping list after each accepted barter decision inside Rust. This is semantically closest to one decision at a time, but current probes show it is much slower and not clearly better than a bounded multi-pass session for large screening runs.
- `--experimental-session-candidate-depth N` keeps up to `N` locally ranked alternatives for each need-good in the one-agent basket shopping list. It is still local-information only: the active agent ranks opportunities visible through its own acquaintance links, then commits feasible trades from that local list. `1` preserves the original fast session-clearing behavior.
- Failed revalidation does not immediately force a rebuild; the agent can continue the same local shopping list because no inventory state changed.

Empirical probe notes:

- `100 agents / 100 goods / 50 acquaintances / 80 cycles`, aspirational target `2.0`: one session pass reached utility about `2.42`; eight session passes reached about `4.80`; after-trade replan reached about `4.83` but was much slower; full serial basket planning with native stage math reached about `5.59`.
- `300 agents / 100 goods / 50 acquaintances / 200 cycles`, aspirational target `2.0`: one pass still left the median agent with many zero-stock goods and utility about `2.35`; eight passes reached utility about `9.04`, median zero-stock goods `0`, and median current-basket completion about `0.99`.
- `3000 agents / 100 goods / 100 acquaintances`, aspirational target `2.0`, eight session passes: the first 100 cycles took about 519 seconds and reached utility about `5.59`; a continued run reached cycle `1025` with utility about `5.36`, rare-goods monetary share about `0.91`, value-weighted rare-goods monetary share about `0.61`, median zero-stock goods `0`, and median current-basket completion about `0.99`. The 525-1025 interval was mildly declining, so this path shows a plausible post-growth/adjustment phase rather than monotone early growth.
- Candidate-depth probes, aspirational target `2.0`: at `100/100/50` for `80` cycles, depth `4` completed median baskets while running about `1.8x` faster than eight replan passes; depth `8` was still faster but did not clearly beat depth `4` in basket completion. At `3000/100/100` for `40` cycles, depth `4` took about `2.28` seconds/cycle versus about `3.69` seconds/cycle for eight replan passes, with median zero-stock goods `0` and median current-basket completion about `0.997`. This makes depth `4` the current first candidate for one-agent basket optimization probes.

Practical recommendation for phenomenon-screening runs:

- use `--experimental-session-clearing-phenomenon-exchange`
- use `--experimental-aspirational-stock-target 2.0`
- start with `--experimental-session-candidate-depth 4 --experimental-session-replan-passes 1`
- keep `--experimental-session-replan-passes 8` as the conservative comparison baseline
- reserve `--experimental-session-replan-after-trade` for semantic diagnostics, not default large runs

This keeps the information boundary local while avoiding the unrealistic result where agents hold large undifferentiated inventories but fail to complete the baskets needed to raise living standard.

Validation rule:

- use the exact path for selected short cross-checks and final reference runs
- use the parallel phenomenon path for long searches over parameters and seeds
- accept path-dependent timing differences only if growth, monetization, welfare, friction, inequality, and cycle phenomena remain robust and non-tendentiously biased

The next safe optimization step is native execution of an already scheduled conflict-free batch. That must accumulate shared market deltas, such as TCE by good, in per-thread buffers and apply them deterministically after the batch. Direct parallel writes to shared market arrays are not acceptable.
