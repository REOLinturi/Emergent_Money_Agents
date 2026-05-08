# Emergent Money Agents

This repository contains the historical Emergent Money documents together with a new GPU-first Python scaffold for rebuilding the simulation.

## Historical Sources

- `EmergentMoney.pdf` - original paper and model description
- `Working_Legacy_Code_Reference.pdf` - annotated legacy C reference
- `Modernizing the Emergent Money Simulation - Comprehensive Plan.pdf` - earlier modernization draft
- `TARGET_10K_PLAN.md` - updated implementation plan for the current `10,000 / 100 / 150` scaling target
- `EXACT_OPTIMIZATION_PLAN.md` - exact-path optimization plan, validation gates, and post-current-run backend metrics backlog
- `DEVELOPMENT_ENVIRONMENT.md` - required tools, setup commands, Rust/CUDA notes, and machine-transfer checklist
- `ARCHITECTURE.md` - engineering rules, backend choice, and UI/service interfaces
- `MODEL_INFORMATION_BOUNDARY.md` - agent decision information boundary: only local own-history and direct-acquaintance observations may affect heuristics
- `MARKET_ORDER_AND_PARALLELISM.md` - why exact barter commit stays sequential, how that maps to primitive exchange, and when richer market institutions can justify more parallel execution
- `PHENOMENON_BASELINE.md` - current fallback baseline for the Rust per-agent basket phenomenon path and rejection rules for weaker optimization experiments
- `RESEARCH_MECHANISM_AND_HYPOTHESES.md` - paper-oriented contribution framing, research hypotheses, Burt structural-holes note, and source/data plan for the network-spillover line

## Current Direction

The new implementation treats the paper-level model as primary and the legacy code as a behavioral reference.

The target architecture is:

- array-based
- backend-aware from day one
- GPU-first for dense tensor phases
- validated against a deterministic CPU reference path
- exposed to UI code only through a service and snapshot boundary

The current implementation already includes:

- binary talents with a paper-aligned `+50%` starting advantage
- a sparse directed acquaintance network that starts empty and grows one explored contact per cycle
- first-round barter and production for basic needs
- legacy-faithful leisure production after surplus deals; the older extra-demand leisure round remains opt-in for diagnostics only
- exhaustive barter scoring across all known acquaintances and all good pairs in the CPU reference path
- the same exhaustive barter semantics on CUDA through a blocked friend-and-goods scan, not heuristic pruning
- backend-owned contact planning and contact updates, with the CUDA path keeping acquaintance selection on-device
- stock-room-aware barter resolution for profitable inventory trades through a backend-owned `proposal -> resolve -> commit` contract with CUDA-native resolve and commit kernels that reuse resolve-stage goods state directly
- cycle-level metrics aggregated across both rounds

The active-friend and candidate-good buffers remain in the state for snapshots, debugging, and later experiments, but they do not define the current barter semantics.

The repository now keeps two active execution paths. The `exact` path is the reference/validation path. The current non-exact phenomenon path is `--experimental-agent-basket-planning` with Rust-owned `--experimental-session-replan-after-trade`: one active agent evaluates its locally visible basket opportunity set across known acquaintances, commits one validated barter decision, replans from the changed local state, and then eventually hands control to the next agent. Offer-good exhaustion is handled pairwise by need/offer pair; globally banning an exhausted offer good inside the active basket is retained only as a rejected diagnostic because it weakens the 3000/100/100 per-agent basket dynamics. Static no-replan basket lists, synchronized session-clearing, and wave-based variants are retained only as historical comparison/diagnostic branches unless explicitly selected.

## Scaffold Layout

- `src/emergent_money/config.py` - scenario and kernel parameters
- `src/emergent_money/backend/` - backend abstraction plus NumPy and CUDA backends
- `src/emergent_money/state.py` - device-compatible state containers and barter work buffers
- `src/emergent_money/initialization.py` - initial tensor creation
- `src/emergent_money/engine.py` - cycle pipeline scaffold
- `src/emergent_money/service.py` - in-process service boundary for UI and automation callers
- `src/emergent_money/dto.py` - host-side snapshot DTOs
- `tests/` - regression tests

## Quick Start

Create a virtual environment, install the package in editable mode, and run a small CPU smoke test.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .[dev]
python -m emergent_money --cycles 3 --population 128 --goods 12 --acquaintances 24
python -m pytest
```

For preparing another development or compute machine, use `DEVELOPMENT_ENVIRONMENT.md`. It records the required Python, Rust, MSVC, maturin, and optional CUDA/CuPy setup, plus the smoke tests that should pass before long runs.

## Dashboard Modes

Live dashboard against an in-process simulation service:

```powershell
$env:PYTHONPATH = 'src'
python -m emergent_money --dashboard --backend numpy --experimental-native-stage-math --population 3000 --goods 30 --acquaintances 100 --active-acquaintances 100
```

Read-only dashboard attached to a checkpointed long-run artifact directory:

```powershell
$env:PYTHONPATH = 'src'
python -m emergent_money --dashboard --dashboard-run-dir runs\report_exact_stage_math_1000_seed2009
```

The observer mode reads `metrics.jsonl`, `checkpoint_latest.json/.npz`, and `summary.json` when available. It never mutates the running simulation and is the recommended way to watch long exact-validation runs.

## CUDA Note

The scaffold includes a CUDA backend that expects a working CuPy installation, but CuPy is not pinned in `pyproject.toml` because the correct package depends on the target CUDA stack.

Install the matching CuPy package for the target machine before running `--backend cuda`.

### Verified Local CUDA Setup

The current workspace has been verified with this Windows setup:

- `cupy-cuda12x`
- `nvidia-cuda-nvrtc-cu12`
- `nvidia-cuda-runtime-cu12`

Example local startup flow:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .[dev]
python -m pip install cupy-cuda12x nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12
New-Item -ItemType Directory -Force -Path .cupy_cache
$env:CUPY_CACHE_DIR = "$PWD\.cupy_cache"
$env:PYTHONPATH = "src"
python -m emergent_money --backend cuda --cycles 2 --population 64 --goods 8 --acquaintances 12 --active-acquaintances 4
```

`CUPY_CACHE_DIR` is still useful in restricted environments, but the CUDA backend now defaults to a workspace-local `.cupy_cache` directory if the variable is unset.

## Optional Native Search Backend

The exact legacy CPU path isolates barter partner search behind an optional native module seam. The Python implementation remains the reference, but an optional Rust extension can replace only the `find_best_exchange` hot loop.

Scaffold files for that extension live under `native/legacy_search`. The Python loader accepts either:

- `emergent_money._legacy_native_search`
- top-level `_legacy_native_search`

Recommended local build flow once Rust is installed:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e .[dev,native]
maturin develop --manifest-path native\legacy_search\Cargo.toml
python -m pytest tests\test_legacy_mechanics.py -q
```

This keeps the legacy exact cycle in Python while moving only the isolated exchange-search backend into native code.

Once the native module builds successfully, validate it against the Python reference before using it in longer runs:

```powershell
$env:PYTHONPATH = 'src'
python -m emergent_money --compare-native-search --cycles 5 --population 32 --goods 6 --acquaintances 6 --active-acquaintances 3 --demand-candidates 2 --supply-candidates 2 --compare-seeds 2009 --compare-native-sample-limit 64 --compare-native-benchmark-iterations 10
```

## Practical Target Run

For the current practical target, use explicit scaling parameters and the tuned blocked CUDA scan:

```powershell
$env:CUPY_CACHE_DIR = "$PWD\.cupy_cache"
$env:PYTHONPATH = "src"
python -m emergent_money --backend cuda --cycles 1000 --population 10000 --goods 100 --acquaintances 150 --active-acquaintances 24 --cuda-friend-block 12 --cuda-goods-block 25
```

Measured in this workspace on an `RTX 4090` after warm-up:

- `10,000 / 100 / 150` with `friend_block=12` and `goods_block=25`: about `0.157 s` per cycle
- `1,000` cycles: about `2.6 min`
- `3,000` cycles: about `7.9 min`

These numbers are environment-specific, but they show that the current exact CUDA path is now comfortably inside the practical runtime target for the target scenario.
A whole-cycle parity harness is also available for the optional native exact-cycle entrypoint. It is intended as the acceptance gate for any larger Rust port than exchange search alone:

```powershell
$env:PYTHONPATH = 'src'
python -m emergent_money --compare-native-cycle --cycles 3 --population 32 --goods 6 --acquaintances 6 --active-acquaintances 3 --demand-candidates 2 --supply-candidates 2 --compare-seeds 2009 2011
```

This does not enable the native cycle in normal runs. It compares the explicit native whole-cycle entrypoint against the Python exact reference and reports the first mismatching cycle and state field if parity breaks. The current accepted whole-cycle slice is `runs/native_cycle_compare_32_6_6_safe_slice_final.json`: the native entrypoint now owns `_run_agent_cycle` orchestration, still delegates the branch-sensitive surplus/exchange/end-of-period callbacks to Python, and held parity on seeds `2009/2011/2013`. Longer acceptance probes in `runs/native_cycle_compare_32_6_6_safe_slice_30.json` and `runs/native_cycle_compare_64_6_6_safe_slice_20.json` also stayed exact, with about `1.12x-1.15x` speedup on this machine.

A smaller opt-in native stage path is also available for the exact CPU runner. It moves `prepare_agent_for_consumption`, `produce_need`, `prepare_leisure_round`, and the accepted native post-period helpers under the existing exact Python stage wrappers, while the Rust whole-cycle entrypoint owns the outer agent loop:

```powershell
$env:PYTHONPATH = 'src'
python -m emergent_money --experimental-native-stage-math --cycles 10 --population 32 --goods 6 --acquaintances 6 --active-acquaintances 3 --demand-candidates 2 --supply-candidates 2
```

This path is still experimental and remains off by default, but it is now the main accepted acceleration candidate. Current acceptance artifacts:

- `runs/native_behavior_compare_stage_math_32_6_6_100_v3.json`: exact behavior match on `32/6/6`, `100` cycles, seeds `2009/2011/2013`, with about `1.55x` speedup
- `runs/native_cycle_compare_report_scale_10cycles_seed2009_v1.json`: full-state parity on `3000/30/100`, `10` cycles, seed `2009`, with about `5.55x` speedup
- `runs/native_behavior_compare_report_scale_20cycles_seed2009_v1.json`: exact report-scale snapshot behavior on `3000/30/100`, `20` cycles, seed `2009`, with about `4.22x` speedup
- `runs/estimate_exact_stage_math_outerloop_10000_100_150_3cycles_v1.json`: current large-scale anchor, averaging about `29.88 s` per cycle across the first `3` exact cycles on `10000/100/150` in this environment

The exact CPU runner also has an opt-in native exchange-stage path. It keeps proposer order sequential, but moves the full per-agent exchange loop into Rust after the accepted stage-math slice:

```powershell
$env:PYTHONPATH = 'src'
python -m emergent_money --experimental-native-stage-math --experimental-native-exchange-stage --cycles 10 --population 256 --goods 30 --acquaintances 80 --active-acquaintances 24 --demand-candidates 4 --supply-candidates 4
```

Current acceptance artifacts after the price-floor parity fix:

- `runs/native_exchange_trace_compare_32_6_6_100_after_floorfix.json`: exact per-agent exchange trace match on `32/6/6`, `100` cycles, seeds `2009/2011/2013`
- `runs/native_behavior_compare_exchange_stage_32_6_6_100_after_floorfix.json`: exact snapshot behavior match on `32/6/6`, `100` cycles, seeds `2009/2011/2013`, with about `2.75x` speedup
- `runs/native_stage_math_trace_256_30_80_70_after_floorfix.json`: exact stage-math trace match on `256/30/80`, `70` cycles, seed `2009`
- `runs/native_behavior_compare_exchange_stage_256_30_80_100_after_floorfix.json`: exact snapshot behavior match on `256/30/80`, `100` cycles, seed `2009`, with about `7.56x` speedup

The accepted exact CPU acceleration stack is therefore:

- native search/planning behind the standard backend seam
- accepted native stage-math slice
- opt-in native exchange-stage loop, still preserving sequential proposer order

Short hot-path benchmarks in this workspace after the same fix measured about `24.7x` speedup on `256/30/80` for `5` cycles and about `35.8x` on `128/100/150` for `3` cycles versus the Python exact path. These are warm-up-sensitive microbenchmarks; the longer `256/30/80` behavioral acceptance run is the more conservative speed anchor.

A dedicated behavioral comparison harness is also available for larger Rust ports:

```powershell
$env:PYTHONPATH = 'src'
python -m emergent_money --compare-native-behavior --experimental-native-exchange-stage --cycles 40 --population 32 --goods 6 --acquaintances 6 --active-acquaintances 3 --demand-candidates 2 --supply-candidates 2 --compare-seeds 2009 2011 2013
```

This compares an accelerated native path against the pure Python exact reference at the snapshot-behavior level. It reports the first behavioral mismatch, mean final deltas, and relative speed.

Numerical tolerance rule: bit-for-bit parity remains useful for finding porting bugs, but borderline choices whose score gap is far below the TCE signal are not economically meaningful exactness failures by themselves. For explicitly experimental or realism-oriented paths, arithmetic differences that are at least an order of magnitude smaller than the relevant TCE signal are treated as behaviorally immaterial, as long as they do not alter non-borderline trade choices, role transitions, or the reproduced macro phenomena. See `ARCHITECTURE.md` for the validation distinction between the exact/parity gate, borderline tie classification, and the phenomenon gate.

The separate realism-oriented basket path is enabled with `--experimental-agent-basket-planning`. It lets the active agent evaluate the visible opportunity set across its need basket rather than reproducing the exact Legacy-C one-need-at-a-time loop. It is not an exact-reference path. The currently retained phenomenon variant is the Rust-owned per-agent basket path: `--experimental-agent-basket-planning`, `--experimental-session-replan-after-trade`, `--experimental-session-clearing-phenomenon-exchange` left off, and usually `--experimental-session-candidate-depth 1`. The active agent plans from its own observed acquaintance data, commits one revalidated trade, rebuilds its local plan from the changed state, and only then continues. This differs from exact Legacy-C sequencing, but avoids both stale one-shot shopping lists and synchronized local clearing markets.

Offer-good exhaustion rule: the active path forbids only the exhausted `need_good/offer_good` pair before replanning. A 2026-05-07 diagnostic found that the rejected global rule, which banned the exhausted offer good for every need in the active basket, reproduced the weak-growth anomaly; pairwise exhaustion restored the earlier promising 3000/100/100 c50 profile (`living_standard_mean` about `3.03`, rare-money about `15.6 %`). The global rule remains available only through `--experimental-session-global-offer-exhaustion` for named rollback diagnostics.

Current 3000/100/100 phenomenon smoke-test template:

```powershell
$env:PYTHONPATH = 'src'
python -m emergent_money --cycles 50 --population 3000 --goods 100 --acquaintances 100 --active-acquaintances 100 --demand-candidates 100 --supply-candidates 100 --seed 2009 --experimental-agent-basket-planning --experimental-native-stage-math --experimental-session-replan-after-trade --experimental-session-candidate-depth 1 --experimental-local-liquidity-stock-bias 1.0 --experimental-aspirational-stock-target 2.0 --checkpoint-dir runs\agentbasket_replan_offeravail_b1_ast2_3000_100_100_50_seed2009_YYYYMMDD --checkpoint-every 5 --sample-every 5
```

Path pruning rule: static no-replan basket lists are no longer the recommended phenomenon path because the 3000/100/100 probe completed quickly but produced much weaker macro growth than the earlier promising basket run. Session-clearing and wave-based variants are also no longer recommended because they can behave like an overly synchronized local clearing market, erasing price dispersion and merchant margin. Keep them only for explicitly named diagnostics or rollback comparisons.

Current phenomenon-run follow-up after the 500-cycle `agentbasket_seq_direct_3000_100_100_500_seed2009_20260506` artifact: treat the artifact as evidence for per-agent basket timing, not as proof that a static no-replan shopping list is sufficient. The next speed work should target the Rust `replan_after_trade` per-agent basket path and verify whether the recent offer-good invalidation and available-offer prefilter preserve the same growth, monetization, welfare, friction, inequality, and cycle phenomena with lower runtime.

Operational convention for phenomenon runs: shorthand such as `3000/100/100` means population/goods/acquaintances, and the full acquaintance count is active unless a different active count is explicitly stated. In commands this means `--active-acquaintances` should normally equal `--acquaintances`; reducing the active friend set is a separate heuristic experiment, not the default interpretation.

For Stage A debugging there is also an exchange-stage trace comparator:

```powershell
$env:PYTHONPATH = 'src'
python -m emergent_money --compare-native-exchange-trace --experimental-native-exchange-stage --cycles 40 --population 32 --goods 6 --acquaintances 6 --active-acquaintances 3 --demand-candidates 2 --supply-candidates 2 --compare-seeds 2009 2011 2013
```

This compares per-agent exchange-stage events between the Python exact reference and the native exchange-stage path.

For Stage B isolation there is now also a post-period comparator:

```powershell
$env:PYTHONPATH = 'src'
python -m emergent_money --compare-native-post-period --experimental-native-stage-math --cycles 40 --population 32 --goods 6 --acquaintances 6 --active-acquaintances 3 --demand-candidates 2 --supply-candidates 2 --compare-seeds 2009 2011 2013
```

There is now also a dedicated stage-math trace comparator for the first native drift inside the exact cycle:

```powershell
$env:PYTHONPATH = 'src'
python -m emergent_money --compare-native-stage-math-trace --experimental-native-stage-math --cycles 1 --population 3000 --goods 30 --acquaintances 100 --active-acquaintances 100 --demand-candidates 30 --supply-candidates 30 --seed 2009
```

This runs one or more exact cycles agent by agent, compares the Python-reference and native-stage paths after every substage, and reports the first mismatching agent/stage/field. The current reference artifacts `runs/native_stage_math_trace_report_scale_cycle1_seed2009_v5.json` and `runs/native_stage_math_trace_report_scale_3cycles_seed2009_v2.json` now hold exactly for the first `1` and `3` full report-scale cycles.

This keeps the live exact run on the accepted Python path, clones the pre-post-period state for each agent, and compares Python `_complete_agent_period_after_surplus` against a candidate native post-period implementation on the clone. It is the acceptance gate for any future Rust ownership of leisure-production plus end-of-period updates. The current reference artifact is `runs/native_post_period_compare_32_6_6_100.json`, which stayed exact for `100` cycles on seeds `2009/2011/2013` and measured about `5.72x` speedup for the isolated post-period block.

That block is not enabled in the execution path. A direct integration attempt still produced long-horizon drift once combined with the broader native stage-math path, so the accepted exact runner remains on the earlier stage-math slice until the next source of price-state drift is isolated.

