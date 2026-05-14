# Money Emergence Diagnostic Plan

This note records the working plan for explaining when rare goods become
exchange media and when high-volume common goods are sufficient. It is a
diagnostic plan, not a decision to force rare goods to become money.

## Current Observation

The current `3000/100/100` per-agent basket run does not show a slow migration
toward rare-money dominance. It shows the opposite sequence:

- Rare exchange-media share is very high early, peaking near cycle `60` at
  roughly `85%`.
- Rare money share also peaks early, near cycle `65` at roughly `20%`.
- By cycle `1225`, rare exchange-media share is roughly `2%`, rare money share
  roughly `2%`, and value-weighted rare money below `1%`.
- The top exchange-media goods are high-volume common goods, while rare goods
  remain visible but low in flow, transaction-cost loss, and reserve score.

This means "not enough cycles" is not the leading explanation. A later crisis
could still rotate the money role, but the long-run direction in this run is
away from rare goods after the early phase.

The latest stopped sample of
`runs/agentbasket_floor_b05_welfare_3000_100_100_1500_seed2009_20260510`
reaches cycle `1255`. The pattern remains the same:

- rare exchange-media share peaked near cycle `60` at about `84.6%`;
- rare money share peaked near cycle `65` at about `19.6%`;
- value-weighted rare money peaked near cycle `60` at about `15.5%`;
- by cycle `1255`, rare exchange-media share is about `2.1%`, rare money
  about `1.6%`, and value-weighted rare money about `0.9%`;
- aggregate production and living standard are high, so the rare-money decline
  is not simply macroeconomic collapse.

## Working Interpretation

The current model gives ordinary goods too much of the same monetary advantage
that standardized money should have.

In real exchange, a standardized grain sack, silver coin, or copper daler
requires less quality inspection, negotiation, and weighing than a generic
consumption good. That advantage is not yet represented strongly enough. In
the current run, average dyadic transparency is already around `0.925` for both
rare and common goods, and the circulation-breadth diagnostic is saturated for
nearly all goods. Once this happens, common goods win by volume:

- they are naturally demanded by many agents;
- they generate more observed local acceptance;
- they create more round-trip turnover;
- they accumulate more transaction-flow and transaction-cost-loss signal;
- dense 100-friend neighborhoods make direct barter and common-good resale
  sufficiently effective that a rare settlement medium is not necessary.

This is a plausible economic result under the current assumptions, not
automatically a bug. But it may be missing a historically important mechanism:
standardization and low verification friction.

## Theoretical Rationale

The central theoretical distinction is not "rare versus common" by itself. The
central distinction is saleability: how reliably a good can be accepted by many
counterparties, in many circumstances, across time, without a large price
discount or verification cost.

Important modeling principle: rarity is not the same thing as durability.
There is no general real-world reason why low-demand goods should always be
better preserved than high-demand goods. The historically grounded commodity
money claim is narrower: goods such as gold, silver, shells, salt, or other
recognized commodity monies combine several properties at once. They are
scarce or value-dense enough to settle large exchanges in small quantities,
durable enough to hold across time, recognizable and standardizable enough to
avoid repeated quality inspection, and saleable enough that agents expect
future counterparties to accept them.

Therefore `rare-good` storage is not a default realism assumption. It is a
named precious-metal-bundle experiment: rarity plus durability plus lower
verification friction. It must be compared against controls where storage is
independent of rarity (`none`, `mod3`, and later randomized storage classes).
If rare goods become money only in `rare-good`, the correct interpretation is
not "rarity causes money", but "rarity can support commodity money when it is
bundled with durability, recognizability, and low verification cost."

Carl Menger's origin-of-money argument is exactly this type of mechanism:
money emerges when agents learn that some commodities are more saleable than
others and therefore rationally accept those commodities even when they do not
want to consume them directly. Menger explicitly treats almost unlimited
saleability as an extreme case of ordinary commodity saleability and emphasizes
personal, quantitative, spatial, and temporal limits of saleability:
https://publicpolicy.pepperdine.edu/academics/research/faculty-research/intellectual-foundations/austrian-school/cmorgmon.htm

William Stanley Jevons gives the complementary material-properties list:
utility and value, portability, durability, homogeneity, divisibility, value
stability, and recognizability. These map directly onto model mechanisms:
storage cost, transaction cost, quality-verification cost, divisibility, price
volatility, and transparency:
https://oll.libertyfund.org/titles/jevons-money-and-the-mechanism-of-exchange?html=true

Gold and silver historically fit both accounts:

- They had persistent nonmonetary demand for ornament, status, hoarding, and
  later industrial or craft use, so their saleability did not depend on one
  narrow consumption chain.
- They were durable and easily preserved, so their temporal saleability was
  high.
- They had high value density, so portability was high relative to grain,
  cattle, copper, or bulky goods.
- They were divisible, meltable, and recoinable while retaining value.
- Their quality could be standardized by weighing, assay, and coinage; once
  coined or otherwise certified, verification cost fell sharply.
- State coinage and taxation later strengthened and stabilized the role, but
  the commodity-money mechanism can exist before the state layer.

Menger specifically points to precious metals' costliness, durability, and easy
preservation as reasons they became favored for hoarding and commerce:
https://monadnock.net/menger/money.html

### Why Rare Goods Should Become Money In This Model

Rare goods should become money-like if rarity is a proxy for all of the
following:

- high value density: small quantities settle large exchanges;
- strong specialization advantage: a few producers can supply many agents;
- low spoilage or stock cost;
- fast transparency gain or low verification friction;
- stable enough demand that agents trust future resale;
- broad local acceptance observed through direct acquaintance trade;
- positive feedback: more circulation raises transparency, and higher
  transparency lowers transaction cost, which causes more circulation.

Under those conditions, a rare good behaves like a precious metal: not because
everyone consumes much of it, but because it is compact, durable, recognizable,
and predictably resaleable. Agents then rationally hold it as working capital.

### Why Rare Goods Need Not Become Money

Rare goods do not have to become money if one or more of the following is true:

- common goods already have almost the same transparency and verification cost;
- common goods have much broader direct demand and therefore higher observed
  acceptance;
- the acquaintance network is dense enough that direct barter and ordinary
  resale solve most double-coincidence problems;
- rare goods are too scarce locally for agents to observe reliable acceptance;
- reserve heuristics do not reward holding compact exchange media strongly
  enough;
- price-elastic discretionary demand makes high-volume goods attractive enough
  to circulate as commodity money;
- money diagnostics classify high-volume round-trip commodity circulation as
  money-like, which may be valid in some cases but must be separated from
  producer-to-consumer supply-chain flow.

This means a "grain sack money" result can be theoretically valid. If grain is
standardized, storable, universally demanded, and cheap to verify, it can serve
as money in a local economy. Rare metal-like money should dominate only when
its value density, durability, recognizability, and resaleability overcome the
volume advantage of common goods.

## First Diagnostic Runs

A reusable diagnostic runner was added:

```powershell
python scripts\diagnose_money_emergence.py --population 300 --goods 40 --cycles 140 --sample-every 10
```

The first two probes are intentionally small and should not be treated as final
3000/100/100 evidence. They are useful for directional diagnosis.

### Small 300/40/60 Matrix

Output:

- `tmp/money_emergence_diagnostics_20260510_220509/results.json`
- `tmp/money_emergence_diagnostics_20260510_220509/results.csv`

Main observation: at cycle `60`, rare exchange-media share is still high in
most healthy variants. This reproduces the early rare-money signal and shows
that the small model is not incapable of rare-money emergence.

### Selected 300/40/140 Matrix

Output:

- `tmp/money_emergence_diagnostics_20260510_220936/results.json`
- `tmp/money_emergence_diagnostics_20260510_220936/results.csv`

Selected results:

| variant | LS mean | rare money last | rare exchange-media last | rare exchange-media peak |
| --- | ---: | ---: | ---: | ---: |
| baseline | 9.59 | 5.9% | 29.7% | 77.7% at c50 |
| friends_12 | 7.18 | 4.5% | 39.7% | 66.7% at c50 |
| friends_80 | 11.02 | 9.7% | 31.0% | 70.4% at c50 |
| transparency_05 | 6.51 | 2.6% | 25.1% | 65.8% at c70 |
| elasticity_0 | 2.81 | 3.7% | 45.2% | 88.6% at c70 |
| reserve_0 | 4.63 | 3.5% | 58.6% | 81.0% at c60 |

Interpretation:

- The early rare signal is robust in the small probes.
- The signal starts declining by cycle `140`, matching the larger-run pattern
  qualitatively.
- Lower initial transparency delays or weakens growth; at `0.30` the economy
  barely starts within 60 cycles.
- Lower demand elasticity preserves rare exchange-media share better, but at
  much lower welfare in this small probe.
- Removing reserve bias preserved rare exchange-media share better than the
  baseline in the 140-cycle probe, but also lowered welfare. This suggests the
  current reserve mechanism may be helping mature circulation while not
  specifically favoring rare media.
- Common-goods transparency becomes at least as high as rare-goods
  transparency in the healthy variants. This supports the hypothesis that
  ordinary goods currently receive too much of the low-verification-friction
  advantage that historically belonged to standardized money.

## Main Hypotheses To Test

## 2026-05-12 Main Control Pair

The first full-size control pair should isolate two mechanisms observed in the
`agentbasket_transparency_base_need_stor_mod3_raregrad_s05_3000_100_30_1500_seed2009_20260511_night`
run:

- Storage heterogeneity control: repeat the same `3000/100/30` run with
  `--experimental-storage-class-mode none` while keeping rare-gradient
  standardization. This tests whether the `mod3` storage assignment itself
  pushed money-like roles toward durable common goods.
- Rare durable control: repeat the run with
  `--experimental-storage-class-mode rare-good` and rare-gradient
  standardization. This makes the lowest-demand quartile well storable and
  leaves the other goods medium. It tests the historically motivated
  metal-money package: rarity plus lower verification friction plus durability.
- Transparency realism control: run the control pair with
  `--experimental-transparency-learning-mode recent-count`. In this mode, a
  large trade is only one observation; product transparency comes from repeated
  recent purchases or sales with the observed partner, and old observations
  decay through `activity_discount`. This better matches the idea that high
  transparency is a maintained market convention, not a permanent cumulative
  memory created by a few old or very large trades.

These are still phenomenon-path experiments, not exact Legacy-C controls. They
must be judged by macro phenomena and by whether rare goods become exchange
media without lowering welfare artificially.

### H1. The Run Is Too Short

Current evidence: weak. The rare signal is not trending upward; it peaked early
and then decayed.

Test:

- Continue one or two representative runs through later crises and measure
  whether rare goods regain exchange-media share after common-money stress.
- Track not just final rare share, but phase rotation after peaks and crashes.

Falsifies:

- If rare share remains low through multiple cycles, cycle length is not the
  main cause.

### H2. Transparency Becomes Too Easy For All Goods

Current evidence: strong candidate. Rare and common goods have very similar
average transparency, and all goods are broadly visible in the network.

Tests:

- Lower `initial_transparency`, for example `0.3`, `0.5`, `0.7`.
- Slow transparency learning separately from other memory effects.
- Add a product-level standardization or inspection-friction parameter.
- Compare cases where the low-friction standardized goods are rare versus
  cases where they are common or randomly assigned.

Expected result:

- If rare goods become money only when they have lower verification friction,
  the missing mechanism is standardization, not rarity itself.

### H3. The Network Is Too Dense Relative To Population And Goods

Current evidence: plausible. With `3000` agents, `100` goods, and `100`
acquaintances, each agent directly observes a large local market. Dense local
access reduces the need for a compact settlement medium.

Tests:

- Keep `3000/100` but vary acquaintances: `20`, `30`, `50`, `100`.
- Keep `100` acquaintances but scale population: `3000`, `10000`.
- Compare active acquaintance counts only if inactive acquaintances are not
  used in the current basket path.

Expected result:

- Rare money should be more likely when the local neighborhood is too small to
  supply all desired goods directly, but still large enough for repeated
  exchange-media circulation.

### H4. Current Demand Elasticity Makes Common Goods Sufficient

Current evidence: plausible. The corrected model keeps baseline needs as a
floor and allocates discretionary demand with price elasticity. Common goods
can still dominate exchange-media scores if they carry most transaction flow.

Tests:

- Compare `price_demand_elasticity = 0`, `1`, and `2`.
- Add a discretionary-demand concentration cap for diagnostic runs only.
- Compare baseline-floor semantics against Legacy-C's `elasticneed *
  needslevel` semantics as a diagnostic, not as the preferred model.

Expected result:

- If rare money appears mainly in the old full-elastic-need semantics, the
  Legacy-C signal may partly reflect a demand-specification artifact.

### H5. Reserve Formation Is Too Weak Or Too Strict

Current evidence: strong candidate. The current `exchange_media_reserve_score`
is effectively zero for most goods; only `g0` and `g1` show tiny positive
values in the latest checkpoint. Agents are not deliberately building working
capital reserves for future payment at meaningful scale.

Tests:

- Add reserve diagnostics with reason codes: no local acceptance, spread gate
  failed, volume threshold failed, stock-room failed, valuation failed.
- Log top local reserve candidates per sample agent without changing behavior.
- Test moderate normalized reserve thresholds using only local observations.

Expected result:

- If reserve candidates exist but trades do not execute, the issue is trade
  scoring or valuation.
- If reserve candidates do not exist, the issue is local acceptance detection
  or threshold calibration.

### H6. The Money Metric Is Mixing Money With Supply-Chain Throughput

Current evidence: still possible, but less likely than before. Current top
common goods have low consumer-flow share and high round-trip turnover, so
some of them may genuinely be used as exchange media rather than merely as
milk-like consumption goods passing through a chain.

Tests:

- Split money diagnostics into:
  - merchant round-trip flow;
  - non-consumption inflow;
  - producer-to-final-consumer chain flow;
  - stock held beyond own expected consumption;
  - flow diversity across partners;
  - flow-to-own-use ratio;
  - transaction-cost loss per own-use need.
- A good should be called money-like only if many agents acquire it and later
  pass it on, not just if producers sell it and consumers consume it.

Expected result:

- If common goods still dominate after this split, common goods are genuinely
  acting as commodity money under current assumptions.

### H7. Legacy-C Created A Real Signal Or An Artifact

Legacy-C robustly produced a rare-money signal in the original work. There are
several possible explanations:

- Real mechanism: early rare-good use raised transparency and acceptance,
  causing a positive feedback loop toward lower-friction exchange.
- Demand semantics: Legacy-C used `elasticneed * needslevel`, so the whole
  consumption basket, not only discretionary demand, could shift toward cheap
  goods.
- Integer arithmetic and thresholds: fixed integer units and overflow/rounding
  may have created hard discontinuities that favored rare goods.
- Stock and spoilage thresholds: integer stock limits may have punished common
  high-volume goods differently.
- Transparency and transaction-cost compounding may have differed subtly from
  the current float implementation.

Tests:

- Reproduce a small Legacy-C-like semantic mode in the current code only for
  diagnostics: old need formula, integer-like rounding, and old thresholds.
- Run paired small experiments against the corrected model.
- If rare money appears only under old semantics or rounding, the old result
  may be partly artifact.
- If rare money appears under corrected semantics when transparency or
  standardization is adjusted, the old result likely captured a real mechanism
  that the current path has diluted.

## Conditions For Grain-Sack Money Versus Rare Money

High-volume common goods can rationally serve as money when:

- they are widely needed;
- they are sufficiently standardized;
- they are storable enough;
- local networks are dense;
- transaction friction for ordinary goods is not much higher than for rare
  goods;
- direct barter and resale already work well.

Rare goods should become stronger money candidates when:

- they have high value density;
- they spoil little or not at all;
- their quality is easy to verify;
- transaction friction is lower than for ordinary goods;
- their producers are highly specialized;
- local networks are too sparse for direct barter to solve double coincidence
  cheaply;
- agents can observe local acceptance and hold them as working capital.

Taxation is a later institutional layer. A state can stabilize a chosen money
good by requiring taxes in it. That should be tested separately after the
pre-state barter mechanism is understood.

## Proposed Experiment Order

1. Finish observational diagnostics on the current run:
   - rare/common transparency;
   - reserve reason codes;
   - money metric decomposition;
   - phase-by-phase money rotation.
2. Run a small network-size matrix:
   - `3000/100/20`, `3000/100/30`, `3000/100/50`, `3000/100/100`;
   - same seed first, then multiple seeds.
3. Run a transparency matrix:
   - initial transparency and transparency-learning speed.
4. Add product standardization as an explicit experimental parameter:
   - no behavioral global knowledge;
   - only lower transaction friction or faster dyadic transparency for
     standardized goods.
5. Compare corrected demand semantics with Legacy-C-like demand semantics in
   controlled short runs.
6. Only after these, test taxation as an institutionally imposed acceptance
   mechanism.

## Next Overnight Test

Recommended next live run:

- `3000/100/30`
- `1000` cycles
- seed `2009`
- current accepted per-agent basket semantics
- dashboard on a new port, for example `8058`

Purpose:

- isolate the network-density hypothesis without adding a new economic
  mechanism;
- test whether `100` acquaintances makes the local market so rich that a rare
  settlement medium is unnecessary;
- keep welfare, crises, trade volume, rare exchange-media share, and top
  exchange-media goods comparable with the stopped `3000/100/100` run.

Command shape:

```powershell
.\scripts\start_agentbasket_overnight.ps1 `
  -RunName agentbasket_money_probe_3000_100_30_1000_seed2009_YYYYMMDD `
  -Population 3000 `
  -Goods 100 `
  -Acquaintances 30 `
  -Cycles 1000 `
  -Port 8058 `
  -Seed 2009 `
  -ExchangeMediaReserveBias 0.5 `
  -ExchangeMediaReserveMinAcceptance 2.0 `
  -ExchangeMediaReserveBootstrapFloor 1.0
```

Decision rule:

- If rare exchange-media share remains materially higher than in the
  `3000/100/100` run while welfare stays healthy, dense acquaintance networks
  are suppressing rare-money dominance.
- If rare share again peaks early and decays, the next likely test is an
  explicit standardization or verification-friction asymmetry.

## Overnight 3000/100/30 Result

Run:

- `runs/agentbasket_money_probe_3000_100_30_1000_seed2009_20260510`
- started `2026-05-10T22:56:17+03:00`
- completed `2026-05-11T02:26:22+03:00`
- runtime about `12603 s`, or about `3.5 h` for `1000` cycles

Main comparison against the previous `3000/100/100` run:

| metric at c1000 | 30 acquaintances | 100 acquaintances |
| --- | ---: | ---: |
| living-standard mean | `13.54` | `60.95` |
| fixed-basket living-standard mean | `2.27` | `2.96` |
| production total | `25.2B` | `71.6B` |
| accepted trade volume | `22.5B` | `79.3B` |
| inventory trade volume | `43.1B` | `127.1B` |
| rare exchange-media share | `35.7%` | `2.6%` |
| rare money share | `3.3%` | `2.3%` |
| value-weighted rare money share | `2.7%` | about `1-2%` range |
| friction share of output value | `9.4%` | `26.4%` |
| living-standard Gini | `0.376` | `0.340` |

Interpretation:

- The network-density hypothesis receives support: reducing acquaintances from
  `100` to `30` keeps rare goods materially more visible in the
  exchange-media metric.
- The result is not a clean victory for rare money. The merchant-money scores
  remain dominated by high-demand common goods such as `g97`, `g98`, `g99`,
  `g92`, and nearby common goods.
- The 30-acquaintance economy has much lower welfare and production than the
  100-acquaintance economy. Rare exchange-media persistence may therefore be
  partly a symptom of a more fragmented local market rather than a superior
  monetary equilibrium.
- The final state is in a downturn from a cycle-`790` welfare/production peak.
  Mean living standard peaked at `28.57` and ended at `13.54`; production
  peaked at `32.9B` and ended at `25.2B`.
- Rare exchange-media share peaked at `79.3%` around cycle `60`, fell to a
  minimum of `17.2%` around cycle `520`, and recovered to `35.7%` by cycle
  `1000`. This differs from the 100-acquaintance run, where rare share decayed
  toward about `2%`.

Top exchange-media goods at cycle `1000` in the 30-acquaintance run:

- common: `g97`, `g99`, `g92`, `g91`
- rare: `g2`, `g1`, `g5`, `g12`, `g4`, `g9`, `g3`, `g10`, `g14`

Notable detail:

- `g0` is not the leading rare medium. At cycle `1000`, `g0` has low
  exchange-media score despite a positive reserve diagnostic. The stronger
  rare candidates are `g1-g5` and some low-rank rare goods such as `g12`.
  This suggests that extreme rarity can be too locally scarce for broad
  circulation; slightly less rare goods may be more saleable.

Working conclusion:

- `30` acquaintances makes rare goods more money-like, but at a large welfare
  cost.
- `100` acquaintances gives a much richer economy but makes high-volume common
  goods sufficient as commodity money under current friction/transparency
  assumptions.
- The next decisive test should not merely lower network density further. It
  should add or vary product-specific standardization or verification friction
  while preserving a healthy network size, because historical gold/silver money
  is better explained by saleability and low verification cost than by rarity
  alone.

## Realistic Standardization / Verification-Friction Design

The `rare-gradient` and related product-level standardization modes are now
classified as diagnostic proxies, not as the preferred final realism mechanism.
They are useful because they show what happens if some goods are cheaper to
verify, assay, measure, or certify. They should not be treated as evidence that
the model may simply grant rare goods a permanent transparency advantage.

The preferred research direction is endogenous local standardization:
transparency should improve because agents and their direct acquaintances
repeatedly use, sell, observe, and accept a good. A good may then become
money-like because its use creates a local convention that lowers verification
and negotiation cost. That mechanism may select rare goods, common goods, or a
mixture. The model should not predetermine the winner.

## Transparency-Learning Scale

The report-level rule is that dyadic transparency rises with experience:
repeatedly receiving a good from a direct acquaintance makes that good-friend
relation less costly to verify. The operative scale should be the product's
original need, not its current elastic need.

This distinction matters for money emergence:

- If price elasticity lowers a good's current demand and traded volume, that
  should not by itself make the good more transparent.
- If a good circulates in multiples of its original need, transparency should
  rise because agents have repeated product-specific and dyadic experience.
- A large-demand good should not gain an artificial transparency advantage
  merely because its unit volume is large; its volume must be large relative to
  its original need.

The corrected implementation therefore normalizes both dyadic purchased volume
and recent purchased volume by `base_need`, with a time-window multiplier. It
does not normalize by `elastic_need`. This intentionally departs from the
Legacy-C formula where the denominator appears effectively constant for part of
the update; that C behavior is treated as a likely scaling artifact unless the
paper text later shows otherwise.

The next mechanism must preserve the model's information boundary: agents may
use only their own history and direct-acquaintance observations. They must not
receive a global signal that "good X is money."

The realistic intervention is therefore not a money preference. It is a
physical or institutional property of goods that changes transaction friction:
some goods are easier to identify, measure, divide, assay, or certify. This is
analogous to weighed metal, stamped coin, standardized grain sacks, or other
recognized commodity standards.

### Endogenous Local Standardization Principle

The next behavioral mechanism should preserve the model's information
boundary. Agents may use only:

- their own dyadic trade history;
- observations and claims from direct acquaintances;
- the observed breadth of a direct acquaintance's recent selling or accepting
  activity;
- decayed recent experience, not a permanent global memory.

Agents must not receive a global signal that "good X is money", and rare goods
must not receive an automatic transparency premium merely because they are
rare. Instead, lower transaction friction should arise from local evidence that
a good is recognizable, repeatedly accepted, and resold by many known agents.

Two candidate mechanisms are especially realistic:

- Friend-experience spillover: if a direct acquaintance has recently traded a
  good many times, that experience can partially reduce the verification cost
  perceived by the agent. This represents local reputation, conversation,
  observation, and repeated community use.
- Seller-breadth reputation: if the same acquaintance sells or accepts the same
  good with many different partners, that seller-good pair becomes more trusted
  and easier to verify. This represents a merchant or producer becoming known
  as a reliable source of a standardized good.

Both mechanisms should be capped, decayed over time, and normalized by the
good's original need or expected consumption scale. Otherwise high-volume
common goods would gain transparency merely because their unit volume is large,
not because they circulate unusually broadly relative to need.

Implementation status:

- Passive diagnostics have been added to the good snapshot and dashboard:
  `local_product_experience_score`, `seller_breadth_reputation_score`,
  `top_seller_breadth_share`, and `endogenous_standardization_score`.
- The passive diagnostics are trade-count based: one successful transaction is
  one observation, regardless of size. This matches the interpretation that
  transparency grows through repeated encounters, not through a single large
  shipment.
- Additional purity diagnostics separate broad consumption-good throughput
  from money-like intermediation: `seller_specialization_score`,
  `top_seller_specialization_share`, `merchant_round_trip_breadth`,
  `non_consumption_flow_share`, and `intermediation_purity_score`.
- These diagnostics do not affect trade scoring, transparency, reserve
  targets, prices, or any agent decision.
- They are intended to answer the next question before behavior changes:
  whether future money-like goods already show local friend-experience and
  seller-breadth signals in the data.
- A behavior-changing endogenous standardization path is now implemented but
  defaults to zero effect. It uses the same local evidence boundary: own
  dyadic transparency plus reputation signals refreshed from direct
  acquaintances' observed product experience, seller activity, and seller
  breadth. No global money ranking, global popularity, or rare-good label is
  exposed to agents.
- The effective boost is capped by the remaining transparency gap:
  `effective = raw + (1 - raw) * strength * local_signal`. The local signal is
  computed from direct-friend product experience and seller-breadth reputation,
  normalized by the good's original need scale. This allows low-direct-need
  goods to become transparent if they circulate repeatedly, without granting
  them an exogenous rare-good advantage.
- Enable the path with
  `--experimental-endogenous-standardization-strength <0..1>` and tune the
  need normalization with
  `--experimental-endogenous-standardization-need-power <nonnegative>`.
  The current first comparison value is strength `0.5`, need power `0.5`.

### Diagnostic Product-Level Mechanism

Add a product-level verification factor used when converting dyadic
transparency into transaction friction.

Current simplified logic:

- dyadic transparency below `1.0` makes trade lossy;
- the less transparent the good/friend relation is, the more extra quantity is
  required and the more TCE is recorded.

Proposed experimental logic:

```text
effective_transparency(agent, friend, good)
  = dyadic_transparency(agent, friend, good)
    + (1 - dyadic_transparency(agent, friend, good)) * standardization_score(good)
```

where `standardization_score(good)` is between `0` and `1`.

Interpretation:

- `0.0`: ordinary good; only learned dyadic familiarity reduces verification
  cost.
- `0.5`: partly standardized; even a less familiar counterparty can verify it
  more easily.
- `1.0`: fully standardized; verification friction is almost eliminated.

This does not tell the agent that the good is widely accepted. It only makes
the good cheaper to verify when encountered. If agents then observe that it is
locally accepted and useful, normal local learning and reserve heuristics can
make it money-like.

### Why This Is Realistic

Historically, gold and silver were not just rare. They were unusually good
objects for reducing verification and transport costs:

- high value density made them portable;
- durability made them temporally saleable;
- divisibility made them usable in different transaction sizes;
- assay, weighing, and coinage reduced quality uncertainty;
- repeated use made local acceptance observable.

The same logic can also produce non-metal commodity money such as standardized
grain sacks if grain is sufficiently measurable, durable, and broadly accepted.
Thus the model should not assume that rare goods must win. It should test when
their lower verification friction and value density overcome common goods'
volume and direct-use advantages.

### Experimental Variants

Run these as phenomenon-path variants, not exact-reference variants:

1. No standardization:
   - current baseline.
2. Rare-standardized:
   - rare quartile receives higher `standardization_score`.
3. Common-standardized:
   - high-demand quartile receives the same score.
4. Random-standardized:
   - same number of goods standardized, selected by seed.
5. Gradient by demand rank:
   - rare goods get gradually higher value-density / verification advantage,
     rather than a hard quartile cutoff.

Decision rule:

- If only rare-standardized goods become durable money while welfare remains
  high, the missing mechanism is plausibly precious-metal-like verification
  advantage.
- If common-standardized goods still dominate, then "grain sacks are enough"
  under these network and demand conditions.
- If random-standardized goods can become money, the result is primarily a
  standardization/network-effect mechanism rather than a rarity mechanism.

### Implementation Boundary

Do not change:

- global agent information;
- local acceptance observations;
- price heuristics directly;
- reserve heuristics to prefer rare goods by name;
- exact-reference path semantics.

Change only:

- effective transparency used in trade scoring and trade execution;
- optional diagnostics showing raw dyadic transparency versus standardized
  effective transparency;
- configuration flags selecting which product set receives a standardization
  factor.

This keeps the intervention useful as a controlled diagnostic: agents still
discover money through local exchange, but the experiment can ask what happens
if some goods are objectively cheaper to verify. It is not the preferred final
model of money emergence unless the product-level verification advantage can be
justified independently, as with weighed metal, coined metal, measured grain,
or another explicit standard.

### Implementation Status

Implemented as an explicit phenomenon-path experiment, defaulting to no effect.

Configuration flags:

- `--experimental-standardization-mode none|rare|common|rare-gradient|common-gradient|random`
- `--experimental-standardization-strength <0..1>`
- `--experimental-standardization-random-seed <int>`

The standardization vector is generated by
`SimulationConfig.exchange_standardization_scores()`. The Rust per-agent basket
path and Python fallback both use the resulting product-level score when
converting dyadic transparency into effective trade transparency:

```text
effective = raw + (1 - raw) * standardization_score(good)
```

The score is used in both candidate scoring and trade execution. It does not
alter agent observations, local acceptance history, reserve heuristics, prices
directly, or global information. Omitted or zero standardization reproduces the
previous behavior.

Current startup helper:

```powershell
.\scripts\start_agentbasket_overnight.ps1 `
  -Cycles 1000 `
  -Population 3000 `
  -Goods 100 `
  -Acquaintances 30 `
  -StandardizationMode rare `
  -StandardizationStrength 0.5 `
  -Port 8058
```

## Product Storability / Spoilage Design

Money emergence should also test durability. A good may be widely demanded and
easy to verify, but still be a poor medium of exchange if agents cannot hold it
as a reserve without large expected spoilage. Conversely, a durable good can be
worth holding even before the next exact counterparty is known.

The first implementation is deliberately simple and run-local:

## 2026-05-14 Endogenous Local Standardization Result

The first full `3000/100/30` run with endogenous local standardization was:

```text
runs/agentbasket_endog_std_s05_3000_100_30_1500_seed2009_20260514
```

Configuration highlights:

- no exogenous product-level standardization:
  `--experimental-standardization-mode none`;
- local repeated-trade transparency:
  `--experimental-transparency-learning-mode recent-count`;
- endogenous local standardization enabled:
  `--experimental-endogenous-standardization-strength 0.5`;
- need normalization:
  `--experimental-endogenous-standardization-need-power 0.5`;
- no storability advantage by rarity:
  `--experimental-storage-class-mode none`.

The run completed `1500` cycles with healthy macro behavior:

- mean living standard: `22.95`;
- median living standard: `20.15`;
- fixed-basket living standard mean: `4.11`;
- production: `29.46B`;
- TCE/output: `20.5%`;
- spoilage/output: `5.3%`.

The money-emergence signal changed materially relative to the earlier
`3000/100/30` no-standardization control:

- rare exchange-media share: `61.0%`;
- rare intermediation-purity share: `39.1%`;
- rare monetary share: `10.6%`;
- value-weighted rare monetary share: `14.9%`.

For comparison, the no-standardization `3000/100/30` control at cycle `1500`
had rare exchange-media share about `27.7%`, rare monetary share about `4.2%`,
value-weighted rare monetary share about `4.1%`, and rare
intermediation-purity share about `12.3%`.

This is important because the mechanism did not tell agents which goods are
money and did not grant rare goods an exogenous transparency advantage. The
effect came from local product experience, direct-acquaintance seller activity,
and seller-breadth reputation.

### Supply-Chain Versus Exchange-Media Check

The run was post-analyzed with:

```powershell
.\.venv\Scripts\python.exe scripts\analyze_exchange_media_purity.py `
  runs\agentbasket_endog_std_s05_3000_100_30_1500_seed2009_20260514
```

Outputs:

- `exchange_media_purity_report.json`;
- `exchange_media_purity_report.csv`.

The top exchange-media candidates were not ordinary consumer supply-chain
flows. Their consumer-flow shares were low:

- `g4`: `0.6%`;
- `g13`: `0.9%`;
- `g21`: `1.5%`;
- `g93`: `0.9%`;
- `g14`: `1.7%`;
- `g97`: `0.9%`.

Thus the current evidence supports a real non-consumption intermediation signal,
not just "milk flowing from farm to consumer through wholesalers." However, the
diagnostics also show two distinct money-like channels:

- rare settlement-like exchange media: high exchange-media score, high
  flow/TCE relative to own need, low consumer-flow share;
- high-demand merchant money: strong retailer round trips, high
  value-weighted monetary score, high intermediation-purity score.

This distinction should be kept in future reporting. Rare goods are becoming
important as exchange media, while some high-demand common goods remain strong
merchant working-capital goods.

The same purity report on the no-standardization control was dominated by
common high-demand goods (`g92`, `g93`, `g94`, `g97`, `g99`). The comparison
therefore suggests that endogenous local standardization did not merely raise a
dashboard metric; it shifted a meaningful part of non-consumption intermediation
toward rare goods.

- `good_id % 3 == 0`: poorly storable;
- `good_id % 3 == 1`: medium storable;
- `good_id % 3 == 2`: well storable.

Here `good_id` means the dense internal product index in the current run, not
the external report label. Therefore a spaced run such as `g0,g3,g6,...` still
receives all three storage classes.

The storage class affects two mechanisms:

- product-specific spoilage rate used when stock exceeds its limit;
- product-specific stock target multiplier, interpreted as the rough number of
  periods of inventory an agent is willing to hold.

Default experimental multipliers:

- poor: spoilage `2.0x`, target stock `0.5x`;
- medium: spoilage `1.0x`, target stock `1.0x`;
- good: spoilage `0.25x`, target stock `2.0x`.

This is not a money preference and does not reveal global information. It is an
objective physical property analogous to perishability, durability, bulk, and
storage cost. The hypothesis is that durable and standardized goods should be
more likely to become settlement media, while poorly storable goods should tend
to remain consumption-chain goods even if they have large direct demand.

The `rare-good` mode is intentionally different from `mod3`: it assigns good
storage to the lowest-demand quartile and leaves the other goods medium. This
does not assert that rare goods are generally more durable. It tests a
historical commodity-money bundle in which the rare or low-demand goods are
also metal-like: durable, compact, and relatively cheap to verify when
standardized. Any positive result from this mode must be reported as a bundle
effect, not as a pure rarity effect.

Configuration flag:

- `--experimental-storage-class-mode none|mod3|rare-good`

The helper script exposes this as:

```powershell
.\scripts\start_agentbasket_overnight.ps1 -StorageClassMode mod3
```

## Decision Criteria

- If rare money emerges when network density is lower, the current `100`
  acquaintance setting is too close to a rich local market for rare settlement
  media to matter.
- If rare money emerges only when standardized goods have lower friction, the
  missing historical mechanism is verification cost.
- If rare money emerges only when rare goods are also durable, the missing
  historical mechanism is not rarity alone but the precious-metal bundle:
  durability, verifiability, divisibility, and value density.
- If rare money emerges only in Legacy-C-like integer or demand semantics, the
  old signal may be partly artifact.
- If common goods remain money-like under all corrected diagnostics, the model
  is telling us that under these parameters "grain sacks are enough."
