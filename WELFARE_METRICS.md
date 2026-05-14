# Welfare Metrics And Capacity Diagnostics

This note records how dashboard welfare diagnostics should be interpreted. The
metrics are reporting-only diagnostics. They must not be fed back into agent
heuristics, because agents are allowed to know only their own history and direct
acquaintance observations.

## Current Living-Standard Measures

The dashboard now separates three related questions:

- `Living standard`: achieved consumption relative to the original baseline
  quantity. The original baseline basket is always a floor; price elasticity can
  only redirect the extra consumption above that floor.
- `Fixed-basket LS`: a strict original-basket multiplier. For each agent it
  asks how many complete baseline baskets are present in the currently fulfilled
  goods and then takes the limiting good. This is intentionally harsher than
  the elastic measure.
- `Substitution lift`: the ratio between total living standard and strict
  fixed-basket living standard. A high value means the extra welfare above the
  baseline depends strongly on substitution rather than receiving extra goods in
  the original proportions.

The fixed-basket metric is useful when gross production and the elastic living
standard diverge. If production is high but fixed-basket welfare is low, the
economy may be producing the wrong mix, routing goods through inventories, or
failing to distribute bottleneck goods.

## Offline Capacity Bounds

The dashboard also reports offline capacity diagnostics:

- `Fixed capacity`: a planner-style estimate for the original basket.
- `Elastic capacity`: the same estimate for the current elastic demand mix.

Each capacity card shows:

- `greedy lower`: a feasible specialization estimate from a simple offline
  allocator.
- `upper`: an optimistic no-friction bound that lets each good use its best
  producers without fully enforcing cross-good conflicts.

These numbers are not a proposed social planner and not a market mechanism. They
are an error-finding instrument. They help answer whether the current population
and productivity matrix could theoretically support a higher welfare level than
the decentralized exchange process is realizing.

## How To Use The Diagnostics

Use these comparisons:

- If `Living standard` is high but `Fixed-basket LS` is low, the baseline is
  being met but extra welfare comes from demand substitution. That may be
  realistic, but it should not be confused with everyone receiving the original
  basket at a high multiplier.
- If `Fixed capacity` is much higher than realized `Fixed-basket LS`, the
  bottleneck is not raw production potential; it is exchange, distribution,
  inventories, prices, or learning.
- If `Elastic capacity` is much higher than `Fixed capacity`, the discretionary
  demand mix is easier to satisfy than the original basket. This is useful for
  interpreting where additional welfare can arise after baseline needs are met.
- If both capacity bounds are near realized welfare, the economy may already be
  close to what its local productivity matrix can support under the current
  basket definition.

The diagnostics are deliberately approximate. They are meant to guide anomaly
search and experiment design, not to define correct behavior.
