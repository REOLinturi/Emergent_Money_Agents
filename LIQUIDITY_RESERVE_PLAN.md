# Local Exchange-Media Reserve Plan

Status: base-use-value-normalized reserve iteration prepared in Python and
Rust. The diagnostics are always observational. The exchange-media reserve
itself is opt-in and remains off when
`experimental_exchange_media_reserve_bias` is `0.0`.

## Goal

The retained phenomenon path already lets the active agent evaluate its full
local barter basket and replan after each committed trade. The open money
question is narrower: agents should be able to hold a deliberate exchange-media
reserve when their own local experience suggests that a good is useful as a
payment or resale medium.

The mechanism must not give an agent global market knowledge. A good may become
attractive as an exchange medium only through information visible to that
agent:

- own stock, needs, prices, role, production, purchases, sales, and inventory
  inflow
- direct acquaintance slots currently known to the agent
- attempted or executed trades with those acquaintances
- dyadic transparency learned for that acquaintance and good
- per-friend local acceptance evidence such as friend-level sold/purchased
  history

It must not depend on global money scores, global rarity rankings, population
role counts, or dashboard aggregates.

## Legacy-Report Compatibility

The proposed structure is compatible with the report-level model, even though it
is not meant to reproduce the Legacy-C implementation line by line.

Relevant report claims, paraphrased from `EmergentMoney.pdf` PDF pages:

- Pages 12-13: private values and stock limits are affected by expected
  usefulness in exchange; agents have different heuristics when satisfying own
  needs versus investing in utilities for exchange profit.
- Page 13: surplus utilities are produced for future needs or for exchange, and
  recent experience that a utility is useful in exchange increases its stock
  limit.
- Pages 15 and 18: stock can satisfy own needs or be used in exchange; maximum
  stock is an agent-private estimate based on consumption and what has recently
  been handed out in exchange.
- Pages 15-17: agents have imperfect information and know only their network;
  the model is explicitly not a perfect-information market.
- Pages 21-22: rare goods can become exchange media in later stages because
  specialization and longer supply chains make them useful in exchange, but
  instability and boom-bust behavior can follow when stocks grow on mistaken
  demand expectations.

The only possible tension is timing: the report emphasizes the agent's own
experience from exchange, while the proposed bootstrap may also use directly
observed acceptance by acquaintances before the agent itself has large turnover.
This is acceptable if the evidence is strictly local and dyadic. It would become
a report conflict only if we used global exchange-media rankings or ex post
dashboard metrics inside the decision path.

## Current Gap

The current retained phenomenon path has two related stock-target mechanisms:

- `experimental_aspirational_stock_target` raises target stock from the agent's
  own consumption need and living-standard aspiration.
- `experimental_local_liquidity_stock_bias` raises surplus stock target from a
  local-liquidity score based on direct acquaintance acceptance, transparency,
  and own turnover.

The c1500 run shows that this is not sufficient to keep rare goods near the top
of the exchange-media list. Rare goods remain traded and stocked, but common
high-volume goods dominate the current exchange-media diagnostic. One plausible
mechanism-level explanation is that the present reserve target still behaves
too much like extra consumption or ordinary resale stock, and not enough like
working capital deliberately held because it is locally acceptable in future
barter.

This must be diagnosed before tuning. If rare goods fail because reserve targets
are too low, the fix belongs in target formation. If targets are already high
but trades do not execute, the issue is in valuation, scoring, stock-room
constraints, or price-spread logic.

## Proposed Structure

Introduce a named local exchange-media reserve in the phenomenon path only.
Keep the exact path unchanged.

The reserve should be computed per `agent, good` and added to the surplus
stock target only when local evidence passes gates.

Candidate inputs:

- `visible_acceptance`: discounted amount of this good accepted by current
  acquaintances in direct local history.
- `acceptance_breadth`: share of known acquaintances for whom the agent has
  observed acceptance of the good.
- `transparency_mean`: mean dyadic transparency for accepting acquaintances and
  this good.
- `own_turnover`: `min(recent_sales, recent_purchases + recent_inventory_inflow)`
  for this agent and good.
- `spread_gate`: a rationality gate that prevents buying for reserve when the
  agent's own maximum purchase price is above its own minimum sales price.
- `bootstrap_floor`: a small floor that allows a reserve to start from local
  acceptance evidence before large own turnover exists.

The first implementation used raw observed acceptance in both the score and the
reserve scale. The c900 `3000/100/100` run showed that this keeps macro
dynamics healthy but still lets high-consumption common goods dominate the
exchange-media list. The next iteration therefore makes the exchange-media
reserve explicitly use-value-normalized.

Important: this normalization uses the original use-value need,
`base_need[agent, good] * needs_level[agent]`, not the current market
`elastic_need`. `elastic_need` is endogenous and can already be distorted by
price and previous money-use dynamics. Using it inside the money-reserve
heuristic would let the model hide a good's baseline use value after that good
has become cheap, expensive, scarce, or money-like.

```text
local_need_window = own_need_scale * known_acquaintance_count * history
need_normalized_acceptance = visible_acceptance / local_need_window
acceptance_score = sqrt(acceptance_breadth)
volume_score = min(1, need_normalized_acceptance / min_acceptance)
turnover_score = max(0.25, own_turnover / max(own_sources, own_sales, eps))
liquidity_score = transparency_mean * acceptance_score * turnover_score * volume_score
reserve_scale = max(
  own_need_scale * bootstrap_floor,
  max(visible_acceptance, own_turnover) / known_acquaintance_count,
  own_exchange_budget_per_good * acceptance_breadth
)
exchange_media_reserve = reserve_bias * reserve_scale * liquidity_score
target_stock = max(base_stock_limit, own_aspiration_target, base_stock_limit + exchange_media_reserve)
```

The exact coefficients are experimental. The important semantics are:

- The good is not selected because it is globally rare.
- The good is not selected merely because it has a large absolute consumption
  volume. A common good must show circulation beyond the local consumption
  window before it looks like an exchange medium.
- The reserve grows from local acceptance, local transparency, and the agent's
  own observed ability or plausible ability to pass the good onward.
- The reserve can bootstrap before the agent has become a large retailer, but
  only after local evidence exists.
- The agent does not buy above its own resale threshold merely because a global
  metric labels the good as money.

Calibration note from the c970 `3000/100/100` reserve run: with the original
raw-volume reserve, living standard reached a high plateau and then turned down
from a peak near c930, while rare exchange-media share fell from an early peak
to roughly 7-10%. Replaying the latest checkpoint through the
base-use-value-normalized diagnostic shows that high thresholds select mainly
g0-g8 as reserve candidates, while very low thresholds again let high-volume
common goods dominate. The next test should therefore use a moderate normalized
threshold, initially around `0.01` to `0.001`, rather than the old raw-volume
default `2.0`.

## Parameter Realism

These parameters are not neutral tuning constants. They represent real-world
constraints and should be changed only with an explicit interpretation.

- `stock_limit_multiplier`: finite storage and working-capital capacity. Too
  low prevents both merchant inventories and money reserves; too high permits
  unrealistic hoarding and large mistaken demand bubbles. A value near `2.0`
  remains a reasonable default, but reserve stock should be allowed to lift the
  target only when local acceptance evidence exists.
- `spoilage_rate` and `stock_spoil_threshold`: storage cost, deterioration,
  obsolescence, and administrative carrying cost. The metric must remain
  distinct from transaction friction. A high value punishes money-like
  circulating stocks; a near-zero value can make hoarding too safe. The current
  `0.10` after a threshold of `2.0` is plausible as a combined carrying-cost
  proxy, but sensitivity tests are needed.
- `initial_transparency` and learned dyadic transparency: local market
  knowledge. Transparency must remain local and friend-good specific. If it
  rises too fast, the model approaches a perfect local market and collapses
  merchant margins; if it rises too slowly, inefficient exchange media can
  persist. The reserve heuristic may use only the agent's own dyadic
  observations.
- `activity_discount` and `history`: memory length. Too short a memory makes
  reserves chase noise; too long a memory keeps obsolete money goods alive.
  The current effective memory of roughly five periods should be treated as a
  behavioral assumption, not only a numerical smoothing parameter.
- `price_demand_elasticity`: demand substitution. It is central to welfare
  interpretation. The current living-standard metric sums fulfilled current
  elastic needs against the original baseline quantity; it is therefore a
  welfare proxy with substitution, not a strict Leontief "same basket at higher
  level" measure.

## Population Size and Theoretical Welfare Bound

The current evidence does not show population size `3000` as the main blocker
for money emergence. With 100 goods and 100 acquaintances, every good has
hundreds of potential talented producers in the full population, and the
network is dense enough for multi-step distribution. The present bottlenecks
look more like local information, working-capital formation, stock/carrying
cost, and exchange heuristics.

However, the theoretical optimum depends on how welfare is defined:

- Fixed original basket: every good must be produced in proportion to
  `base_need`. This is a hard upper-bound problem because high-demand common
  goods require very large quantities. In the latest c970 checkpoint, a crude
  current-capacity check makes the highest-demand goods the limiting factor.
- Elastic welfare basket: agents may substitute toward goods that become cheap
  or locally efficient. This is closer to the current model and explains how
  living standard can reach much higher values even when fixed-basket capacity
  for some common goods is low.
- Money-and-exchange optimum: production should specialize by comparative
  advantage, then use a small number of highly liquid goods as working capital
  to bridge local mismatches. This optimum is not a perfect-information rule
  for agents; it is an offline diagnostic for detecting whether the emergent
  system is leaving obvious gains unrealized.

The proposed estimator is a two-level diagnostic:

1. Compute a frictionless planner upper bound from current efficiencies and
   time budgets. This is an offline LP-style allocation problem: maximize
   living standard subject to each agent's time budget and per-good demand.
2. Compute a local-network lower bound using only feasible acquaintance paths
   and a chosen exchange medium, then subtract estimated transaction friction
   and spoilage. This gives a realistic target range, not a decision heuristic.

If the model's achieved living standard is far below the local-network lower
bound, the problem is likely in heuristics or trading logistics. If achieved
living standard is close to the local bound but below the global planner bound,
the gap is an expected cost of local information, finite networks, and
transaction friction.

## Two-Stage Implementation

### Stage 0: Diagnostics First

Implemented no-behavior diagnostics for selected checkpoints:

- current local-liquidity score by good
- current local-liquidity target increment by good
- proposed exchange-media reserve target by good
- reserve gap: current stock versus reserve-augmented target
- spread gate share: the share of agent-good cells where the agent's own
  maximum purchase threshold does not exceed its own minimum sales threshold

Still open:

- candidate rejection reason: no target, no offer stock, no partner stock, price
  score, transparency, or max-need cap

This answers whether rare goods are blocked by insufficient desire to hold them
or by trade execution/scoring after desire already exists.

### Stage 1: Reserve Target, Feature-Gated

Implemented a new opt-in parameter set:

- `experimental_exchange_media_reserve_bias`
- `experimental_exchange_media_reserve_min_acceptance`
- `experimental_exchange_media_reserve_bootstrap_floor`

Default all new behavior to zero/off. With defaults, current baseline metrics
must remain unchanged.

The target is implemented as a separate Rust helper called from
`basket_stage_max_need` and the equivalent direct basket-session target path.
The helper receives only local state and direct acquaintance history.

### Stage 2: Valuation Check Only If Needed

If Stage 0 shows that target desire exists but rare goods still never become
accepted candidates, add a bounded exchange-use valuation component.

The valuation component must obey the retailer rationality rule:

- buying for reserve is allowed only if the expected local resale or payment use
  is at least as good as the purchase threshold after transaction friction
- the agent must not raise its buy willingness above its own sell threshold
  simply because stock room exists
- high local acceptance can make the good more attractive, but cannot override
  the agent's observed margin constraints

This keeps the heuristic close to a barter trader's working-capital logic.

### Stage 3: Acceptance Experiments

Run in this order:

- small deterministic smoke test with new flags off: exact metric parity with
  current phenomenon baseline for the same seed and cycle count
- local-information test: global-only money evidence must not activate reserve
- 3000/100/100 short run, 50-100 cycles, checking early growth and no price
  pathology
- 3000/100/100 medium run, 500 cycles, comparing against
  `PHENOMENON_BASELINE.md`
- continuation to 1500+ cycles only if the 500-cycle run preserves high living
  standard and cycle dynamics

Primary acceptance metrics:

- living-standard mean, median, p10/p90, and Gini
- aspiration-shortfall share
- production, accepted trade, inventory trade
- friction/output and spoilage/output
- rare exchange-media share and value-weighted rare money share
- top exchange-media goods and their producer concentration
- evidence that top exchange-media goods are not merely high-volume supply-chain
  goods

## Rejection Rules

Reject the reserve mechanism if it creates any of these patterns:

- monotone stagnation where production rises but living standard does not
- retailer buy prices exceeding own sell prices in non-borderline cases
- large stock growth without turnover or local acceptance
- rare-good dominance caused by exogenous rarity rather than local trade
  evidence
- disappearance of cycle/crisis/recovery dynamics
- a synchronized clearing-market effect similar to the previously rejected
  session-clearing and wave paths

## Working Hypothesis

The best realism-preserving fix is not to force rare goods into money roles.
It is to let each agent form a local working-capital reserve for goods that its
own trading neighborhood has shown to be liquid. If rare goods are genuinely
the most efficient exchange media at the report scale, this mechanism should
make them re-emerge without violating the model's information boundary. If
common high-volume goods still dominate after this change, that becomes a
substantive model result rather than a reporting artifact.
