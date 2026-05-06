# legacy_search Rust backend

This crate implements the optional `_legacy_native_search` module used by `emergent_money.legacy_search_backend`.

Goal:

- keep Python exact legacy mechanics as the semantic reference
- port only the exchange-search hot loop first
- compare native results against the Python backend before any wider port
- support opt-in phenomenon-screening paths that parallelize read-only candidate scoring without replacing the exact reference

Typical build from the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e .[dev,native]
maturin develop --manifest-path native\legacy_search\Cargo.toml
```

For a full machine-transfer setup, including the Windows C++ build tools, Rust `PATH`, wheel installation, and validation commands, see `DEVELOPMENT_ENVIRONMENT.md` in the repository root.

The `plan_parallel_phenomenon_candidates` export is used by the deprecated `--experimental-parallel-phenomenon-exchange` wave path. It scores active agents' locally visible exchange candidates in parallel and returns execution-ready plans to the conflict scheduler. It does not perform event-exact sequential commit and should be validated at phenomenon level against selected exact checkpoints.

New phenomenon-screening work should prefer `--experimental-session-clearing-phenomenon-exchange`. That path uses native local basket sessions: an agent scans locally visible acquaintances and goods into a ranked shopping list, each candidate is revalidated before commit, and no global market aggregates are exposed to the agent heuristic. The wave path is kept only as a short-term comparison and rollback baseline.
