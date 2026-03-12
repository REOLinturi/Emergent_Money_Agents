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

## Practical Engineering Rule

Current rule:

- keep trade commit sequential in the exact reference path
- finish the Rust sequential exact core first
- only then explore parallelism around read-only search, scoring, and reductions

Future rule once richer market institutions are modeled:

- allow more parallel quote lookup and candidate construction
- consider batch matching only where the market mechanism itself justifies delayed commit
- validate every such change against the Rust sequential reference over multiple seeds and long horizons

## Current Interpretation for This Repository

The accepted interpretation is therefore:

- the sequential exact path is the best current model of primitive barter dynamics
- faster speculative paths are useful as experiments, not as automatic replacements
- more aggressive parallelism may become appropriate later if the modeled economy itself acquires posted-price and intermediary-like behavior
