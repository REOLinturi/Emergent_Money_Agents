# Emergent Money Agents

This repository contains the historical Emergent Money documents together with a new GPU-first Python scaffold for rebuilding the simulation.

## Historical Sources

- `EmergentMoney.pdf` - original paper and model description
- `Working_Legacy_Code_Reference.pdf` - annotated legacy C reference
- `Modernizing the Emergent Money Simulation - Comprehensive Plan.pdf` - earlier modernization draft
- `TARGET_10K_PLAN.md` - updated implementation plan for the current `10,000 / 100 / 150` scaling target
- `ARCHITECTURE.md` - engineering rules, backend choice, and UI/service interfaces
- `MARKET_ORDER_AND_PARALLELISM.md` - why exact barter commit stays sequential, how that maps to primitive exchange, and when richer market institutions can justify more parallel execution

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
- second-round leisure-driven barter with temporary extra demand
- exhaustive barter scoring across all known acquaintances and all good pairs in the CPU reference path
- the same exhaustive barter semantics on CUDA through a blocked friend-and-goods scan, not heuristic pruning
- backend-owned contact planning and contact updates, with the CUDA path keeping acquaintance selection on-device
- stock-room-aware barter resolution for profitable inventory trades through a backend-owned `proposal -> resolve -> commit` contract with CUDA-native resolve and commit kernels that reuse resolve-stage goods state directly
- cycle-level metrics aggregated across both rounds

The active-friend and candidate-good buffers remain in the state for snapshots, debugging, and later experiments, but they do not define the current barter semantics.

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

A larger experimental exchange-stage path still exists for diagnostics, but it is now treated as a rejected optimization candidate rather than a normal exact-run option.

It remains available only through the dedicated comparison harnesses because it is faster but not report-faithful. Exact-reference compares stayed very close over `10` cycles (`runs/native_exact_reference_exchange_stage_only.json`), but longer behavior probes already show occasional trade-count drift and noticeably different macro totals (`runs/native_behavior_probe_exchange_stage_only_32_6_6_100.json`, `runs/native_macro_compare_exchange_stage_32_6_6_100.json`).

The accepted exact path therefore remains:

- native search/planning behind the standard backend seam
- accepted safe native stage-math slice
- no normal-run activation of the rejected native exchange-stage commit path

A dedicated behavioral comparison harness is also available for larger Rust ports:

```powershell
$env:PYTHONPATH = 'src'
python -m emergent_money --compare-native-behavior --experimental-native-exchange-stage --cycles 40 --population 32 --goods 6 --acquaintances 6 --active-acquaintances 3 --demand-candidates 2 --supply-candidates 2 --compare-seeds 2009 2011 2013
```

This compares an experimental native path against the pure Python exact reference at the snapshot-behavior level rather than full state parity. It reports the first behavioral mismatch, mean final deltas, and relative speed. The older small-scale artifact `runs/native_behavior_compare_exchange_stage_32_6_6_40.json` still shows clear drift and remains the reason this path is not part of the exact baseline.

However, after the accepted stage-math fixes, the same exchange-stage candidate now looks materially better at report scale. Current reference artifacts:

- `runs/native_exchange_stage_compare_32_6_6_10_v2.json`: Stage A isolation still shows tiny ulp-scale drift in consumption-stage bookkeeping, mainly `engine._inventory_trade_volume` and a few value/TCE cells
- `runs/native_behavior_compare_report_scale_exchange_stage_10cycles_s3_v1.json`: `3000/30/100`, `10` cycles, seeds `2009/2011/2013`, about `24.37x` speedup, with very small mean final deltas
- `runs/native_behavior_compare_report_scale_exchange_stage_20cycles_seed2009_v1.json`: `3000/30/100`, `20` cycles, seed `2009`, about `22.52x` speedup; end-state drift stayed around `0.06%-0.14%` on production/trade/utility and `~1.2%` relative on rare-money share
- `runs/native_behavior_compare_report_scale_exchange_stage_40cycles_seed2009_v1.json`: `3000/30/100`, `40` cycles, seed `2009`, about `18.10x` speedup; end-state drift was still below `1%` on trade count, trade volume, and utility, about `0.11%` on production, and about `2.24%` relative on rare-money share
- `runs/estimate_exact_stage_math_exchange_stage_10000_100_150_3cycles_v1.json`: `10000/100/150`, first `3` cycles averaged about `3.95 s` per cycle, which projects to roughly `2.2 h` for `2000` cycles on this machine

This means the exchange-stage path is no longer a candidate for the exact baseline, but it has become a credible fast-path candidate for large exploratory runs where small behavioral drift is acceptable.

For Stage A debugging there is also an exchange-stage trace comparator:

```powershell
$env:PYTHONPATH = 'src'
python -m emergent_money --compare-native-exchange-trace --experimental-native-exchange-stage --cycles 40 --population 32 --goods 6 --acquaintances 6 --active-acquaintances 3 --demand-candidates 2 --supply-candidates 2 --compare-seeds 2009 2011 2013
```

This compares per-agent exchange-stage events between the Python exact reference and the experimental native path. The current artifact `runs/native_exchange_trace_compare_32_6_6_40.json` shows that the first significant Stage A drift is not an immediately different barter partner choice; after tolerating ulp-scale float differences, the first structural divergence appears as an exchange-stage ordering/count mismatch around cycles `26-32`.

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

