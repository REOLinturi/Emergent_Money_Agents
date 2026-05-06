# Research Mechanism and Hypotheses

This note captures a possible research framing for the current exact-reference simulation line. It is meant to be a paper-oriented document, not just a development memo.

## Proposed Contribution Paragraph

This project does not claim to discover that proximity to affluent or high-status actors is associated with better outcomes. That empirical pattern already has support in the social-capital and firm-network literature. The contribution here is narrower and more mechanistic: we build a decentralized exchange model in which local welfare advantages around rich agents emerge endogenously from barter, inventory holding, specialization, and the emergence of money-like goods. In the current simulation, the highest-living-standard agents are not primarily the largest producers. They are disproportionately retailer-like intermediaries located at valuable exchange bottlenecks. Their immediate partners perform better than agents farther away from the richest core, yet the same system can simultaneously generate extreme inequality and aggregate welfare stagnation or decline. The model therefore offers one explicit micro-mechanism for how local prosperity near wealth can coexist with economy-wide inefficiency rather than diffusing uniformly through the whole network.

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
