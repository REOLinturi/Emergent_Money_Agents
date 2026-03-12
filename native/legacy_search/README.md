# legacy_search Rust backend

This crate implements the optional `_legacy_native_search` module used by `emergent_money.legacy_search_backend`.

Goal:

- keep Python exact legacy mechanics as the semantic reference
- port only the exchange-search hot loop first
- compare native results against the Python backend before any wider port

Typical build from the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e .[dev,native]
maturin develop --manifest-path native\legacy_search\Cargo.toml
```
