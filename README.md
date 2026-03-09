# Emergent Money Agents

This repository contains the historical Emergent Money documents together with a new GPU-first Python scaffold for rebuilding the simulation.

## Historical Sources

- `EmergentMoney.pdf` - original paper and model description
- `Working_Legacy_Code_Reference.pdf` - annotated legacy C reference
- `Modernizing the Emergent Money Simulation - Comprehensive Plan.pdf` - earlier modernization draft
- `TARGET_10K_PLAN.md` - updated implementation plan for the `10,000 / 30 / 100` target model
- `ARCHITECTURE.md` - engineering rules, backend choice, and UI/service interfaces

## Current Direction

The new implementation treats the paper-level model as primary and the legacy code as a behavioral reference.

The target architecture is:

- array-based
- backend-aware from day one
- GPU-first for dense tensor phases
- validated against a deterministic CPU reference path
- exposed to UI code only through a service and snapshot boundary

The current CPU reference path already includes:

- binary talents with a paper-aligned `+50%` starting advantage
- a sparse directed acquaintance network that starts empty and grows one explored contact per cycle
- first-round barter and production for basic needs
- second-round leisure-driven barter with temporary extra demand
- exhaustive barter scoring across all known acquaintances and all good pairs in the reference path
- stock-room-aware barter resolution for profitable inventory trades through a backend-owned resolve and commit contract
- cycle-level metrics aggregated across both rounds

The active-friend and candidate-good buffers remain in the state for snapshots, debugging, and later CUDA pruning experiments, but they do not define the CPU reference barter semantics.

## Scaffold Layout

- `src/emergent_money/config.py` - scenario and kernel parameters
- `src/emergent_money/backend/` - backend abstraction plus NumPy and CUDA backends
- `src/emergent_money/state.py` - device-compatible state containers and barter work buffers
- `src/emergent_money/initialization.py` - initial tensor creation
- `src/emergent_money/engine.py` - cycle pipeline scaffold
- `src/emergent_money/service.py` - in-process service boundary for UI and automation callers
- `src/emergent_money/dto.py` - host-side snapshot DTOs
- `tests/` - initial regression tests

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

The scaffold includes a CUDA backend stub that expects a working CuPy installation, but CuPy is not pinned in `pyproject.toml` because the correct package depends on the target CUDA stack.

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
$env:CUPY_CACHE_DIR = "$PWD\\.cupy_cache"
$env:PYTHONPATH = "src"
python -m emergent_money --backend cuda --cycles 2 --population 64 --goods 8 --acquaintances 12 --active-acquaintances 4
```

`CUPY_CACHE_DIR` is especially useful in restricted environments where CuPy cannot write to the default user-profile cache directory.