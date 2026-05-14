# Research Mechanism and Hypotheses

This note captures a possible research framing for the current exact-reference simulation line. It is meant to be a paper-oriented document, not just a development memo.

## Proposed Contribution Paragraph

This project does not claim to discover that proximity to affluent or high-status actors is associated with better outcomes. That empirical pattern already has support in the social-capital and firm-network literature. The contribution here is narrower and more mechanistic: we build a decentralized exchange model in which local welfare advantages around rich agents emerge endogenously from barter, inventory holding, specialization, and the emergence of money-like goods. In the current simulation, the highest-living-standard agents are not primarily the largest producers. They are disproportionately retailer-like intermediaries located at valuable exchange bottlenecks. Their immediate partners perform better than agents farther away from the richest core, yet the same system can simultaneously generate extreme inequality and aggregate welfare stagnation or decline. The model therefore offers one explicit micro-mechanism for how local prosperity near wealth can coexist with economy-wide inefficiency rather than diffusing uniformly through the whole network.

## Relation To Existing Emergent-Money Simulations

The current model should be positioned as part of the agent-based and search-theoretic literature on money, not as an isolated invention. There are important predecessors:

- Kiyotaki and Wright model the endogenous emergence of commodity money and fiat money in decentralized bilateral exchange.
- Marimon, McGrattan, and Sargent simulate Kiyotaki-Wright-style exchange economies with artificially intelligent agents using classifier systems.
- Howitt and Clower simulate the emergence of economic organization from autarky, specialist trading firms, and a universal medium of exchange.
- Yang, Kwon, Jung, and Kim study commodity-money emergence in an agent-based commodity network, emphasizing storage fee and salability.
- Later econophysics and network papers study competition among commodities for money status and network effects in exchange-media emergence.
- Tesfatsion's Agent-Based Computational Economics framing is the broader methodological home for this type of bottom-up economic simulation.

The distinctive claim for this project is not simply that money can emerge. That is already established in several model families. The stronger and more specific contribution is the combined mechanism:

- locally observed barter rather than global market clearing
- private purchase and sales price thresholds
- learned production efficiency and specialization
- stock limits, spoilage, and transaction-cost loss
- transparency/trust improving through repeated dyadic exchange
- endogenous lifestyle or living-standard growth
- price-elastic discretionary demand
- money-like goods, inequality, local spillovers, and business-cycle-like dynamics emerging in the same economy

Based on the current literature check, closely related models exist, but no clearly identical combination has yet been found. This should be written carefully: the claim is "we have not found a close equivalent yet," not "no such model exists." A systematic literature review is still needed before making a novelty claim in a paper.

## Current Quality Assessment

The simulation is now beyond a proof-of-concept demo. The corrected per-agent basket path can generate several qualitatively relevant macro phenomena at the same time:

- specialization and large productivity growth
- high aggregate living-standard measures
- inventory-heavy booms and downturns
- transaction-cost and spoilage losses
- inequality in living standards and stock values
- local advantages around intermediary-rich agents
- temporary or partial emergence of exchange-media candidates

The main remaining scientific risks are not that the model is inert, but that key interpretations must be made precise:

- The living-standard metric separates baseline consumption from discretionary, price-elastic substitution. This must be documented explicitly whenever results are reported.
- Money metrics must distinguish genuine medium-of-exchange use from ordinary supply-chain throughput of consumption goods.
- The rare-goods money question must be treated as an empirical model question, not as a desired output. The current evidence shows early rare-money emergence followed by common-good dominance in mature phases.
- The model must pass sensitivity tests across population size, number of goods, network size, spoilage, transparency, stock limits, and elasticity settings.
- The exact-reference path and the realism/phenomenon path must be kept separate, with the exact path used as a validation instrument rather than as the only production-scale runner.
- Apparent business cycles must be tested across seeds and parameter regimes, not inferred from a single long run.

Current assessment: this is a publishable research-program candidate if robustness and measurement issues are handled rigorously. It is not yet a finished paper result. The next paper-quality step is a controlled experiment matrix plus a transparent measurement appendix.

## Rare Goods, Saleability, And Commodity Money

This is now a separate research thread and should not be lost inside speed-work notes. The working document is `MONEY_EMERGENCE_DIAGNOSTIC_PLAN.md`; the reusable short-run diagnostic runner is `scripts/diagnose_money_emergence.py`.

Current observation:

- In the stopped `3000/100/100` per-agent basket run `runs/agentbasket_floor_b05_welfare_3000_100_100_1500_seed2009_20260510`, rare exchange-media share peaked near cycle `60` at about `84.6%`.
- Rare money share peaked near cycle `65` at about `19.6%`.
- By cycle `1255`, rare exchange-media share had fallen to about `2.1%`, rare money to about `1.6%`, and value-weighted rare money to about `0.9%`.
- This happened while production and living standard remained high. The decline of rare-money share is therefore not just a macroeconomic collapse artifact.
- In small `300/40` diagnostic probes, the same qualitative pattern appears: rare exchange-media share is strong early and then begins to decay as common goods become mature exchange media.

Theoretical framing:

- Harvinainen tuote ei rahaistu siksi, että se on harvinainen. It becomes money-like if it becomes highly saleable: widely accepted, durable, recognizable, low-friction, and easy to resell.
- Carl Menger's origin-of-money argument explains money as the endogenous selection of the most saleable commodities. Agents accept these commodities not only for direct use, but because they expect others to accept them later.
- William Stanley Jevons' material-properties account translates naturally into model terms: portability maps to value density, durability to spoilage, homogeneity and recognizability to verification friction and transparency, divisibility to trade granularity, and value stability to private-price volatility.
- Gold and silver historically fit both accounts because they combined persistent nonmonetary demand, high value density, durability, divisibility, and standardizability through weighing, assay, and coinage.
- Taxation and state coinage are important later stabilizers, but should be modeled as a separate institutional layer after the pre-state barter mechanism is understood.

Modeling principle:

- Rarity, durability, standardization, transparency, divisibility, and value density must be treated as separate product properties. There is no general real-world rule that rare goods are automatically more durable. The historically grounded claim is narrower: gold, silver, shells, salt, and other commodity monies combined scarcity or value density with preservation, recognizability, and low verification cost.
- Therefore `rare-good` storability is not a neutral realism default. It is an explicit precious-metal-bundle experiment: "what happens if the low-demand goods are also the more durable, more standardizable goods?" The control case must keep storability independent of rarity, for example `none`, `mod3`, or later randomized storage classes.
- Results should be interpreted accordingly. If rare goods become money only under `rare-good`, the finding is not "rarity causes money." It is "rarity plus durability and low verification friction can create a commodity-money role."
- Product-level `rare-gradient` standardization is also a diagnostic proxy, not the preferred final explanation. It tests the effect of lower verification friction, but it bakes in which goods are easier to verify. The preferred final mechanism is endogenous local standardization: transparency improves because agents and their direct acquaintances repeatedly use, sell, accept, and discuss a good.
- Endogenous standardization must respect the local-information boundary. Agents may use their own dyadic history, direct acquaintances' recent product experience, and the observed breadth of a direct acquaintance's selling or accepting activity. They must not use global popularity, global money status, or an externally assigned "rare goods are transparent" rule.
- The implemented endogenous path follows this boundary. Each agent can benefit only from reputation-like signals carried by direct acquaintances: product experience, seller activity, and seller breadth. These signals are refreshed from that acquaintance's own local neighborhood and are scaled by the product's original need, so repeated circulation of a low-direct-need good can lower friction without telling the agent that the good is globally monetary.

Current interpretation:

- The present model may give ordinary common goods too much of the same low-verification advantage that historically belonged to standardized money. Average dyadic transparency becomes high for both rare and common goods.
- With a dense local network, high-volume common goods can rationally act as commodity money because direct demand and observed acceptance are broad.
- A "grain sack money" result is not automatically wrong. It becomes wrong only if the model implicitly assumes that ordinary goods are as standardized and cheap to verify as coins or measured grain, without representing that assumption.
- The next realism step is therefore not to force rare money. It is to test whether friend-experience spillover and seller-breadth reputation can make verification friction fall endogenously for goods that genuinely circulate as exchange media.

Immediate research question:

- Under what conditions do rare goods become the dominant exchange medium, and under what conditions are high-volume common goods sufficient?

Important experimental axes:

- acquaintance-network density: `3000/100/20`, `3000/100/30`, `3000/100/50`, `3000/100/100`;
- transparency and verification: initial transparency, transparency-learning speed, and eventually product-specific standardization friction;
- endogenous standardization: direct-friend experience spillover, seller-breadth reputation, local product-convention breadth, and their decay rates;
- reserve formation: whether agents locally observe and deliberately hold working-capital goods;
- demand elasticity: whether discretionary demand makes common goods sufficiently saleable;
- money metric decomposition: separate genuine exchange-media circulation from ordinary producer-to-consumer supply-chain throughput.

Next overnight recommendation:

- Run one live `3000/100/30` per-agent basket experiment for `1000` cycles with the current accepted phenomenon semantics.
- Do not run a wide matrix at the same time; CPU contention would make results harder to interpret.
- Reason: this isolates the strongest non-invasive hypothesis, namely whether `100` acquaintances makes the local market so rich that rare settlement media are unnecessary.
- Expected interpretation: if rare exchange-media share stays materially higher than in `3000/100/100` while welfare remains healthy, network density is suppressing rare-money dominance. If rare share still decays, the next likely cause is missing standardization or verification-friction asymmetry.

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

## Research Hypotheses

### H1. Local Rich-Neighbor Premium

Agents in the immediate neighborhood of the richest core have higher living standards than otherwise similar agents farther away in the exchange network.

Current support from the ongoing exact run:

- Around cycle `1469`, the median living-standard proxy of 1-hop neighbors of the richest `1%` was about `2.60`, versus about `2.04` for 2-hop neighbors.
- For the richest 10 individual agents, the corresponding medians were about `3.04` versus `2.19`.
- The effect is local, not global. The top `10%` is already so large in this dense network that essentially the whole population is within one hop of someone in that group.

### H2. Broker-Not-Producer Hypothesis

The agents with the highest living standards are not primarily dominant producers. They earn their position mainly by intermediation, inventory turnover, and advantageous network position.

Current support from the ongoing exact run:

- At cycle `1469`, the top `10%` by living standard held on average about `26.9` retailer roles but only about `1.0` producer roles.
- The remaining `90%` held on average about `13.9` retailer roles and `1.6` producer roles.
- The top `10%` accounted for only about `13.2%` of recent production, so their welfare position is not explained by producing most of the economy's output.

### H3. Local Spillovers Without Aggregate Welfare Gain

Local prosperity around rich intermediaries can coexist with aggregate welfare flatlining or declining when the economy becomes increasingly transaction-heavy and inventory-heavy.

Current support from the ongoing exact run:

- In the second major upswing, production, trade, and inventories rose far above the first-cycle highs.
- Utility and the living-standard distribution did not recover proportionally.
- Inequality became extreme at the same time that transaction-cost measures remained very high.

This is the key nontrivial implication. The model does not predict simple universal "trickle down." It predicts a local premium around the broker-rich core together with macro-level fragility.

### H4. Brokerage Should Appear More Clearly In The Realized Interaction Network Than In The Raw Acquaintance Graph

Because the acquaintance graph becomes dense and saturated over long runs, Burt-style brokerage should be tested primarily on weighted realized interaction links rather than on the raw binary contact graph alone.

Current support from the ongoing exact run:

- In the symmetrized raw acquaintance graph near cycle `1474`, unweighted clustering for the richest `1%` was almost the same as for the rest of the population (`0.0470` versus `0.0474`), so the static graph alone does not show a dramatic structural-hole signature.
- In the weighted interaction graph using normalized `friend_activity`, the richest `10%` had lower Burt-style constraint than the rest (`0.0267` versus `0.0321`, about `17%` lower).
- The richest `1%` also had lower weighted constraint than the rest (`0.0290` versus `0.0316`, about `8%` lower).
- The richest groups also had lower partner-weight concentration and slightly more diverse partner specialization mixes.

That combination is consistent with brokerage in the interaction layer even when the underlying acquaintance graph is already dense.

## Relation To Burt's Structural Holes Theory

Ronald Burt's core idea is that advantage accrues to actors who bridge otherwise weakly connected groups. The broker sees more variation, arbitrages between groups, and can coordinate flows that others cannot. That is a useful lens for this model, but it must be used carefully.

What fits Burt well:

- The richest agents in the simulation are strongly intermediary-like rather than dominant producers.
- Their immediate partners do better than agents farther from the richest core.
- In the weighted interaction network, they appear less constrained and less dependent on a narrow set of redundant ties.
- Their partner set is slightly more diverse by specialization than that of the rest of the population.

What does not yet justify a strong claim:

- The unweighted acquaintance graph is already very dense by long-run stages, so raw graph topology alone does not show a dramatic brokerage signature.
- We have not yet shown with a dedicated test that the richest agents bridge clearly separable communities in the strict Burt sense.
- We have not yet compared brokerage scores against alternative explanations such as simple popularity, degree, prior wealth, or good-specific scarcity.

Best current interpretation:

Burt's theory is useful here as a mechanism hypothesis, not yet as a concluded result. The present run suggests that successful traders function more like brokers in the weighted exchange layer than in the raw acquaintance graph. In other words, they may connect partially distinct specialization clusters through realized trade intensity even when the static social graph is already saturated.

## Why This May Be Scientifically Useful

The empirical literature already supports the broad claim that connections to affluent or high-status actors are associated with better outcomes. What this model can add is a causal-looking mechanism story:

1. Production becomes specialized.
2. Some goods become monetarily central.
3. A subset of agents accumulate intermediary roles across many goods.
4. Their direct partners gain a local welfare premium.
5. That premium does not generalize to society as a whole.
6. The same process can amplify inequality and transaction costs while aggregate welfare stalls.

If this mechanism survives stronger tests, the contribution is not "rich neighbors help." The contribution is "here is one decentralized exchange mechanism by which that pattern can emerge, and here is why it can remain sharply local."

## Current Evidence Snapshot From This Workspace

These diagnostics were computed from the current exact-reference artifact directory:

- `runs/report_exact_cfaithful_nofloor_from406_seed2009`
- checkpoint around cycle `1474`

Indicative findings:

- Top `10%` by living standard: retailer-heavy, not producer-heavy.
- Top `10%` production share: only about `13.2%`.
- Top `1%` and top-richest-agent neighborhoods show a clear local living-standard premium at 1 hop relative to 2 hops.
- Raw acquaintance-graph clustering is nearly the same for rich and non-rich agents.
- Weighted interaction-network constraint is materially lower for rich groups.

This combination is exactly why a Burt-style interpretation is promising but not yet complete.

## Research Plan

### 1. Freeze Regime Snapshots

Select checkpoints from four stages:

- early expansion
- first downturn
- recovery / second expansion
- later high-inequality plateau or downturn

The purpose is to test whether the local spillover and brokerage signals are stable across phases rather than artifacts of one cycle.

### 2. Add Explicit Brokerage Metrics

Add offline and eventually dashboard-visible diagnostics for:

- weighted Burt constraint
- effective size / nonredundant contacts
- weighted betweenness or a lighter brokerage proxy
- partner specialization diversity
- ego-neighbor redundancy

These should be computed on both:

- the raw acquaintance graph
- the realized weighted interaction graph

The second network is likely the more informative one.

### 3. Estimate The Local Premium More Strictly

For each checkpoint, compare living standard and Smith-style need cost across:

- richest-core 1-hop neighbors
- richest-core 2-hop neighbors
- matched controls with similar degree and role counts

The goal is to separate a brokerage-neighborhood effect from a mere popularity or centrality effect.

### 4. Test The Broker-Not-Producer Mechanism

Quantify whether high-living-standard agents differ from others by:

- production share
- role composition
- inventory turnover
- partner breadth
- weighted brokerage metrics

If the effect survives controls, that is the clearest mechanism result.

### 5. Compare Across Cycle Phases

Test whether the rich-neighbor premium behaves differently in:

- innovation-heavy expansions
- inventory-heavy overheating phases
- contractions

This is important because the mechanism may be state-dependent rather than constant.

### 6. Empirical Triangulation

Use public or partly public data to compare qualitative predictions:

- local exposure to high-SES people improves outcomes
- high-status connections matter more than generic density
- broker positions inside production or supply networks create spillovers

The model should be presented as a mechanism complement to the empirical literature, not as a substitute for it.

## Key Sources

### Structural Holes And Brokerage

- Ronald S. Burt, _Structural Holes: The Social Structure of Competition_ (1992). DOI: [10.4159/9780674029095](https://cir.nii.ac.jp/crid/1361981470578148736)
- Ronald S. Burt, ["Structural Holes and Good Ideas"](https://cir.nii.ac.jp/crid/1363670320787563648?lang=en), _American Journal of Sociology_ 110(2), 2004. DOI `10.1086/421787`
- Ronald S. Burt, ["Reinforced structural holes"](https://www.sciencedirect.com/science/article/pii/S0378873315000428), _Social Networks_ 43, 2015. DOI `10.1016/j.socnet.2015.04.008`

### Social Capital And Rich-Neighbor Effects

- Raj Chetty et al., ["Social Capital I: Measurement and Associations with Economic Mobility"](https://www.nber.org/papers/w30313), NBER Working Paper 30313, 2022
- [Social Capital Atlas](https://www.socialcapital.org/) for publicly released economic-connectedness data
- Brad Cannon, David Hirshleifer, and Joshua Thornton, ["Friends with Benefits: Social Capital and Household Financial Behavior"](https://www.nber.org/papers/w32186), NBER Working Paper 32186, 2024

### Firm Spillovers And Network Position

- Mary Amiti, Cedric Duprez, Jozef Konings, and John Van Reenen, ["FDI and superstar spillovers: Evidence from firm-to-firm transactions"](https://www.sciencedirect.com/science/article/pii/S0022199624000990), _Journal of International Economics_ 152, 2024
- Nathaniel Baum-Snow, Nicolas Gendron-Carrier, and Ronni Pavan, ["Local Productivity Spillovers"](https://www.aeaweb.org/articles?id=10.1257%2Faer.20211589), _American Economic Review_ 114(4), 2024

### Emergent Money And Agent-Based Computational Economics

- Carl Menger, ["On the Origin of Money"](https://publicpolicy.pepperdine.edu/academics/research/faculty-research/intellectual-foundations/austrian-school/cmorgmon.htm), _Economic Journal_ 2(6), 1892.
- William Stanley Jevons, [_Money and the Mechanism of Exchange_](https://oll.libertyfund.org/titles/jevons-money-and-the-mechanism-of-exchange?html=true), 1875.
- Nobuhiro Kiyotaki and Randall Wright, ["On Money as a Medium of Exchange"](https://collaborate.princeton.edu/en/publications/on-money-as-a-medium-of-exchange), _Journal of Political Economy_ 97(4), 1989. DOI `10.1086/261634`
- Nobuhiro Kiyotaki and Randall Wright, ["A Search-Theoretic Approach to Monetary Economics"](https://ideas.repec.org/a/aea/aecrev/v83y1993i1p63-77.html), _American Economic Review_ 83(1), 1993.
- Ramon Marimon, Ellen McGrattan, and Thomas J. Sargent, ["Money as a medium of exchange in an economy with artificially intelligent agents"](https://nyuscholars.nyu.edu/en/publications/money-as-a-medium-of-exchange-in-an-economy-with-artificially-int), _Journal of Economic Dynamics and Control_ 14(2), 1990. DOI `10.1016/0165-1889(90)90025-C`
- Peter Howitt and Robert Clower, ["The Emergence of Economic Organization"](https://www.sciencedirect.com/science/article/abs/pii/S0167268199000876), _Journal of Economic Behavior & Organization_ 41(1), 2000. DOI `10.1016/S0167-2681(99)00087-6`
- Jae-Suk Yang, Okyu Kwon, Woo-Sung Jung, and In-mook Kim, ["Agent-based approach for generation of a money-centered star network"](https://econpapers.repec.org/RePEc:eee:phsmap:v:387:y:2008:i:22:p:5498-5502), _Physica A_ 387(22), 2008. DOI `10.1016/j.physa.2008.05.025`
- Jae Hyoung Kim, ["Emergence of a Good as a Medium of Exchange in Different Types of Networks"](https://research.hhs.se/esploro/outputs/workingPaper/Emergence-of-a-Good-as-a/991001526898306056), Swedish House of Finance Research Paper 22-11, 2022. DOI `10.2139/ssrn.4037872`
- Leigh Tesfatsion, ["Agent-based computational economics"](https://www.scholarpedia.org/article/Agent-based_computational_economics), _Scholarpedia_ 2(2), 2007. DOI `10.4249/scholarpedia.1970`

## Data And Materials

### Already Available In This Workspace

- exact-reference checkpoints and metric histories in `runs/`
- current long-run artifact directory `runs/report_exact_cfaithful_nofloor_from406_seed2009`
- dashboard and analytics code under `src/emergent_money/`

### Easily Accessible External Data

- Social Capital Atlas for economic connectedness and related aggregate outcomes
- AEA replication materials linked from the AER article page for local productivity spillovers

### Harder But High-Value Data

- firm-to-firm transaction microdata used in superstar-supplier papers
- matched employer-employee or invoice data
- platform-scale social graph plus income outcomes

These are typically restricted. For a first empirical triangulation, the public social-capital datasets are the easiest route.

## Recommended Claim Discipline

Use the following hierarchy in writing:

- Established in the literature: rich or high-status connections are often associated with better outcomes.
- Supported in the current simulation: a local premium exists around the richest core, and the richest agents are intermediary-heavy rather than output-dominant.
- Suggested but not yet proven: the premium is specifically generated by Burt-style brokerage across differentiated groups.

That phrasing is strong enough to be interesting and careful enough to remain defensible.
