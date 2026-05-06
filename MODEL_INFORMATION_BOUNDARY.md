# Model Information Boundary

## Rule

Agent decisions must use only information available to that agent through its own local history.

Allowed decision inputs:

- the agent's own stock, needs, role, prices, efficiency, and recent production/trade history
- direct acquaintance slots currently known to that agent
- attempted or executed trades with those direct acquaintances
- dyadic transparency learned for that direct acquaintance and good
- per-friend acceptance evidence such as `friend_purchased` and `friend_sold`

Forbidden decision inputs:

- global market averages or totals
- whole-population role counts, concentration, or money-score rankings
- checkpoint-level dashboard metrics
- another agent's hidden history unless it is visible through a direct trade relationship
- ex post analytics such as Gini, rare-money share, or value-weighted monetary scores

## Implementation Implication

Dashboard and analytics metrics may aggregate globally, but those metrics are observational only. They must not feed back into cycle decisions, candidate selection, role choice, price adjustment, stock limits, or acquaintance renewal.

The opt-in local-liquidity heuristic follows this boundary. It can increase a surplus inventory target only when the active agent has observed direct friends accepting the good, and combines that with the agent's own recent turnover. It does not read global acceptance, global monetary scores, or population-level prices.

The transaction-value accounting added for monetary diagnostics is also observational. It records the value implied at the time of executed trades so that reporting does not have to infer value later from current private prices. It does not alter trade acceptance or price updates.

The monetary-role score is an ex post merchant-intermediation diagnostic. It counts recent purchase, sale, inventory-inflow, and observed value flows only for agent-good cells currently in the retailer role. This separates goods acquired for resale or later exchange from ordinary consumption stock inflow.

## Validation Rule

Every new heuristic that changes decisions must state its information source explicitly and have a regression test proving that it does not activate from global-only evidence. Every new dashboard metric must be documented as observational and kept outside the simulation decision path.
