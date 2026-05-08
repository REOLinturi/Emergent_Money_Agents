# Phenomenon Baseline

This file records the current fallback baseline for the non-exact phenomenon path. If a later speed or parallelism experiment produces weaker macro behavior, unexpected welfare stagnation, price explosions, or loss of cycle dynamics, return to this baseline before continuing.

## Active Baseline

Run artifact:

- `runs/agentbasket_replan_pairwise_b1_ast2_3000_100_100_500_seed2009_20260507_night`

Commit:

- `a213152 Fix per-agent basket offer exhaustion semantics`

Command shape:

```powershell
$env:PYTHONPATH = 'src'
python -m emergent_money --backend numpy --population 3000 --goods 100 --acquaintances 100 --active-acquaintances 100 --demand-candidates 100 --supply-candidates 100 --cycles 500 --seed 2009 --experimental-native-stage-math --experimental-agent-basket-planning --experimental-session-replan-after-trade --experimental-session-candidate-depth 1 --experimental-local-liquidity-stock-bias 1.0 --experimental-aspirational-stock-target 2.0 --checkpoint-dir runs\agentbasket_replan_pairwise_b1_ast2_3000_100_100_500_seed2009_YYYYMMDD --checkpoint-every 5 --sample-every 5
```

The critical semantics are:

- per-agent basket planning
- after-trade replan inside Rust
- pairwise offer exhaustion: invalidate only the active `need_good / offer_good` pair
- no session-clearing or wave-based synchronized market path
- no global market aggregates in agent decisions

## 500-Cycle Reference Result

The baseline completed 500 cycles in about `28620 s` (`7 h 57 min`, about `57 s/cycle` average).

## Current Hot-Path Speed Anchor

After the 2026-05-08 Rust hot-path pass, the same accepted per-agent basket
semantics are retained but the mature c500 continuation is faster:

- c500 -> c501 before this pass: about `39.7 s`
- c500 -> c501 after this pass: `26.5-28.6 s` in repeated measurements
- c500 -> c505 in-memory continuation after this pass: `144.1 s`, about
  `28.8 s/cycle`
- conservative mature-state speedup: about `1.38x` versus the same c500
  one-cycle checkpoint measurement

The implemented speedups are semantics-preserving data-structure changes:

- static dirty indexes use per-good bitmasks instead of nested vectors
- static-candidate validity uses a per-session availability cache refreshed
  after each accepted barter
- basket candidate tie-order uses a 32-bit field, which is sufficient for the
  retained 3000/100/100 scale and keeps the same ordering there

Set `EM_PROFILE_BASKET=1` before running a checkpoint continuation to print
Rust-side timing counters for the per-agent basket path.

Final cycle 500:

- living-standard mean: `45.9`
- living-standard median: `39.9`
- living-standard p10/p90: `17.8 / 81.5`
- living-standard Gini: `0.317`
- aspiration-shortfall share: `0.87 %`
- production total: `55.3B`
- accepted trade volume: `60.8B`
- inventory trade volume: `104.7B`
- TCE share of output value: `17.5 %`
- spoilage share of output value: `5.7 %`
- rare-goods monetary share: `2.6 %`
- value-weighted rare-goods monetary share: `1.9 %`
- rare-goods exchange-media share: `14.0 %`

Compared with the previous promising 500-cycle run (`agentbasket_seq_direct_3000_100_100_500_seed2009_20260506`), this baseline is not weaker:

- runtime improved from about `40346 s` to `28620 s`
- final mean living standard increased from about `41.5` to `45.9`
- final median living standard increased from about `34.3` to `39.9`
- final production increased from about `49.1B` to `55.3B`
- final TCE and spoilage shares were slightly lower
- final rare-money share was lower, but the earlier run also ended with low rare-money share after an early rare-money peak

## Observed Dynamics

The run shows the desired macro phenomena:

- early growth from near-baseline subsistence
- a sharp first crisis around cycles `35-40`
- recovery to a high peak around cycle `150`
- a major downturn around cycle `170`
- later wave-like high-level fluctuations rather than monotone collapse
- persistent but not pathological inequality
- high trade, inventory circulation, friction, and spoilage in mature phases

Checkpoint anomaly checks did not show the previously rejected pathologies:

- no NaN or infinity in stock, needs, prices, stock limits, efficiencies, or living-standard state
- private purchase prices remained bounded; max purchase price was about `1.41`
- private sale prices remained bounded; max sale price was about `6.38`
- retailers had no `purchase_price > sales_price` violations
- the median agent had zero open need-goods above the minimum threshold at cycle 500
- the tiny negative period-time debt observed was floating-point roundoff (`~5e-5`), not economic debt

## Open Money Question

Rare goods do become important early in the run. Rare-money share peaks around cycle `60` at about `20 %`, and value-weighted rare-money peaks around `16 %`. By cycle `500`, however, the top money and exchange-media goods are mostly high-demand common goods (`g79+`, especially `g86`, `g89`, `g92`, `g94`, `g97`).

This is not currently classified as an anomaly because:

- the same broad pattern appeared in the previous promising 500-cycle run
- the economy is still high-utility and cyclic, not degrading
- rare exchange-media share remains nonzero but no longer dominant

Next research task:

- continue or repeat the baseline to longer horizons, for example `1000-2000` cycles, to test whether rare goods regain stronger monetary roles in a more mature economy
- separate true exchange media from high-volume supply-chain goods more sharply in the reporting metrics
- inspect whether rare-good producers become more concentrated and whether their local networks support later monetary re-emergence

## Rejection Rule for Future Experiments

Treat a speed or parallelism experiment as weaker than this baseline if it shows one or more of:

- median living standard remains near baseline after the early transition
- physical production rises but basket completion and living standard do not
- private prices develop extreme tails without plausible scarcity interpretation
- retailers buy above their own selling price
- cycle/crisis/recovery behavior disappears into monotone stagnation or monotone collapse
- rare-money and exchange-media metrics vanish without a clear replacement mechanism
- arbitrary candidate pruning replaces full local opportunity evaluation without a named behavioral interpretation

If any of these appear, revert to this baseline semantics before optimizing further.
