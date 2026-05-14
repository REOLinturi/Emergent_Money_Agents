# Development Environment And Machine Transfer

This document lists the tools needed to move Emergent Money development and computation to another Windows machine. It separates the required CPU/Rust exact-path toolchain from the optional CUDA/GPU toolchain.

## Recommended Baseline

Use a project-local Python virtual environment for all development, benchmarks, dashboards, and long runs. Do not rely on globally installed Python packages when comparing native Rust builds, because the global interpreter may load an older `_legacy_native_search` module.

Required baseline tools:

- Git
- 64-bit Python `3.11`
- PowerShell
- Visual Studio Build Tools `2019` or newer, with the C++ build tools and a Windows SDK
- Rust stable toolchain through `rustup`
- Rust `rustfmt` component for formatting native code
- Project Python dependencies from `pyproject.toml`
- `maturin`, installed through the `native` optional dependency

Optional GPU tools:

- Recent NVIDIA driver visible through `nvidia-smi`
- CUDA-compatible CuPy package matching the target CUDA stack
- Sufficient GPU memory for the chosen scenario

The current laptop baseline is enough for CPU/Rust optimization work. Large GPU experiments should preferably run on the tower machine with the larger VRAM budget.

## Fresh Windows Setup

Install Python `3.11` and Git normally. Then install Visual Studio Build Tools with the C++ workload. On this project, the verified build environment used:

```text
C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat
```

Install Rust with `rustup` and make sure Cargo is on `PATH`:

```powershell
rustup default stable
rustup component add rustfmt
rustc --version
cargo --version
cargo fmt --version
```

If `rustc` and `cargo` are installed but not visible in a new PowerShell window, add this directory to the user `PATH`:

```text
%USERPROFILE%\.cargo\bin
```

## Project Setup

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .[dev,native]
```

Use the virtual environment interpreter explicitly in scripts and benchmarks when there is any doubt:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m emergent_money --help
```

Set `PYTHONPATH` for direct module execution from a checkout:

```powershell
$env:PYTHONPATH = "src"
```

## Rust Native Extension

The optional Rust/PyO3 extension lives in `native/legacy_search` and provides the `_legacy_native_search` module. It is part of the current exact-path acceleration work.

Build the wheel from the repository root:

```powershell
$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"
cmd.exe /c '"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" && .venv\Scripts\python.exe -m maturin build --release -m native\legacy_search\Cargo.toml'
```

Install the freshly built wheel into the project virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install --force-reinstall native\legacy_search\target\wheels\legacy_native_search-0.1.0-cp311-abi3-win_amd64.whl
```

If a running dashboard or long-run process locks the installed native module,
do not overwrite `site-packages` mid-run. Build to a temporary wheel directory
and put the package on the project `src` path for the next run:

```powershell
$env:CARGO_TARGET_DIR = "C:\tmp\em_legacy_search_target"
cmd.exe /c '"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" && .venv\Scripts\python.exe -m maturin build --release -m native\legacy_search\Cargo.toml --out C:\tmp\em_legacy_search_wheels'
New-Item -ItemType Directory -Force -Path src\_legacy_native_search
tar -xf C:\tmp\em_legacy_search_wheels\legacy_native_search-0.1.0-cp311-abi3-win_amd64.whl -C src _legacy_native_search/__init__.py _legacy_native_search/_legacy_native_search.pyd
```

The local `src\_legacy_native_search` directory is ignored by Git. With
`PYTHONPATH=src`, the run uses that freshly built extension before any older
global install.

Check that the virtual environment loads the native module:

```powershell
.\.venv\Scripts\python.exe -c "import _legacy_native_search; print(_legacy_native_search.__file__)"
```

Run the native-path regression gate:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m pytest tests\test_legacy_cycle_native.py tests\test_native_cycle_compare.py tests\test_native_behavior_compare.py tests\test_long_run.py -q
```

For a broader local check:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m pytest tests\test_long_run.py tests\test_artifact_analysis.py tests\test_native_cycle_compare.py tests\test_native_behavior_compare.py tests\test_dashboard.py tests\test_service.py tests\test_legacy_cycle_native.py -q
```

## CUDA / CuPy Setup

CUDA is optional for current exact CPU/Rust work, but needed for GPU backend experiments. First verify the driver:

```powershell
nvidia-smi
```

Install a CuPy package that matches the target CUDA stack. The current project has used CUDA 12-series wheels:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install cupy-cuda12x nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12
```

Use a workspace-local CuPy kernel cache:

```powershell
New-Item -ItemType Directory -Force -Path .cupy_cache
$env:CUPY_CACHE_DIR = "$PWD\.cupy_cache"
```

Run a small CUDA smoke test before any long GPU experiment:

```powershell
$env:PYTHONPATH = "src"
python -m emergent_money --backend cuda --cycles 2 --population 64 --goods 8 --acquaintances 12 --active-acquaintances 4
```

If CuPy reports that `CUDA_PATH` cannot be detected, first run the smoke test. The pip-provided runtime packages may still be sufficient. Set `CUDA_PATH` only if CuPy actually fails to compile or launch kernels on the target machine.

## Hardware Notes

CPU/Rust exact-path work benefits from high single-thread speed and fast memory. The exact trade-commit core is intentionally sequential until a Rust sequential reference is complete and validated, so simply adding many CPU threads does not automatically speed up the accepted exact path.

GPU memory matters for large parallel experiments. Avoid materializing dense arrays of shape `population * acquaintances * goods * goods` at report scale. For example, `10000 * 150 * 100 * 100` float32 scores would be about `60 GB` for one score tensor before auxiliary buffers. The laptop RTX 4090 with 16 GB VRAM is useful for development and tiled kernels; the 96 GB tower GPU is the better target for large GPU validation runs.

## Operational Commands

Read-only dashboard attached to a run directory:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m emergent_money --dashboard --dashboard-run-dir runs\some_run_dir
```

Long exact/native run with uncompressed checkpoints:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m emergent_money --backend numpy --experimental-native-stage-math --experimental-native-exchange-stage --cycles 2000 --population 3000 --goods 30 --acquaintances 100 --active-acquaintances 100 --checkpoint-dir runs\run_name --checkpoint-every 5 --sample-every 5 --uncompressed-checkpoint
```

`--uncompressed-checkpoint` is recommended for long local runs when disk space is sufficient. It changes artifact size, not model behavior, and avoids the large CPU cost of compressing `.npz` checkpoints.

## Reliable Long-Run Launch From Codex

Preferred helper for the current per-agent basket phenomenon path:

```powershell
.\scripts\start_agentbasket_overnight.ps1 -Cycles 1500 -Port 8057
```

The helper:

- creates a timestamped `runs\...` artifact directory unless `-RunName` is given
- writes `run.ps1` and `dashboard.ps1` into that run directory for repeatability
- starts both the long run and read-only dashboard in hidden PowerShell windows
- uses the accepted per-agent basket path with Rust stage math, after-trade replan, full `3000 / 100 / 100` local opportunity evaluation, and exchange-media reserve diagnostics
- sets `$ErrorActionPreference = "Continue"` only around the Python command so CuPy `CUDA_PATH` warnings written to stderr are logged but do not abort the wrapper
- prints the dashboard URL and a checkpoint-cycle check command

Example with an explicit run name:

```powershell
.\scripts\start_agentbasket_overnight.ps1 -RunName agentbasket_reserve_b05_welfare_3000_100_100_1500_seed2009_night -Cycles 1500 -Port 8057
```

When launching overnight runs from the Codex desktop sandbox, a plain PowerShell background launch can fail in recurring ways:

- inherited sandbox permissions may prevent writing logs or checkpoints
- child processes can disappear if the launch was not actually outside the sandbox
- PowerShell can treat native-process stderr as an error when `$ErrorActionPreference = "Stop"`, so harmless CuPy warnings can stop the wrapper
- an already-used dashboard port can leave the browser attached to an old run

If the helper returns but no checkpoint appears within about a minute, relaunch it out-of-sandbox/elevated through Codex. If a port is busy, choose a new `-Port`.

Canonical wrapper rule for future run scripts:

```powershell
$ErrorActionPreference = "Stop"
"run started $(Get-Date -Format o)" | Out-File -FilePath $WrapperLog -Encoding utf8

# Native Python stderr is not authoritative on this project: CPU-only CuPy
# imports can emit a CUDA_PATH warning even when the simulation is healthy.
$PreviousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& ".\.venv\Scripts\python.exe" @Args 2>&1 | Tee-Object -FilePath $TerminalLog
$ExitCode = $LASTEXITCODE
$ErrorActionPreference = $PreviousErrorActionPreference

"run ended $(Get-Date -Format o) exit=$ExitCode" | Out-File -FilePath $WrapperLog -Append -Encoding utf8
exit $ExitCode
```

Do not put the Python command under `$ErrorActionPreference = "Stop"` while redirecting `2>&1`. PowerShell can convert harmless native stderr, especially the CuPy `CUDA_PATH` warning, into `NativeCommandError` and stop the wrapper before it records the true Python exit code. Also prefer launching the generated run/dashboard scripts outside the Codex sandbox when they must write under `runs\...`; otherwise the symptom can be an empty run directory, a wrapper log with only `run started`, or no fresh checkpoint.

Older `.cmd`-based launch scripts remain useful for fixed historical experiments. The older reliable pattern was to put the run and dashboard commands in `.cmd` files and launch those files with Windows `cmd start` outside the sandbox. Example:

```powershell
C:\Windows\System32\cmd.exe /c "start ""Emergent Rulesfix Run"" /min ""C:\Codex_Demot\Emergent_money\run_rulesfix_3000_30_100_2000.cmd"" & start ""Emergent Rulesfix Dashboard"" /min ""C:\Codex_Demot\Emergent_money\dashboard_rulesfix_3000_30_100_2000.cmd"""
```

Verification commands:

```powershell
netstat -ano | Select-String -Pattern ':8057'
Invoke-WebRequest -Uri 'http://127.0.0.1:8057/' -UseBasicParsing -TimeoutSec 10
.\.venv\Scripts\python.exe -c "import json; print(json.load(open(r'C:\Codex_Demot\Emergent_money\runs\RUN_NAME\checkpoint_latest.json', encoding='utf-8'))['cycle'])"
```

## Migration Checklist

Before moving serious runs to another machine:

- Confirm `python --version` is `3.11.x`.
- Create and use `.venv`; do not depend on global site-packages.
- Confirm `rustc --version`, `cargo --version`, and `python -m maturin --version`.
- Confirm Visual Studio `vcvars64.bat` exists and the Rust wheel builds.
- Install the Rust wheel into `.venv` and confirm `_legacy_native_search` imports from `.venv`.
- Run the native regression gate before trusting accelerated exact runs.
- For GPU work, confirm `nvidia-smi`, CuPy import, and a small `--backend cuda` smoke test.
- Use the same scenario parameters, seed, checkpoint interval, and acceleration flags when comparing machines.
