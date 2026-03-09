from __future__ import annotations

from .base import BackendUnavailableError, BaseBackend
from .cuda_backend import CudaBackend
from .numpy_backend import NumPyBackend


def create_backend(name: str = "numpy") -> BaseBackend:
    normalized = name.lower()
    if normalized == "numpy":
        return NumPyBackend()
    if normalized == "cuda":
        return CudaBackend()
    raise BackendUnavailableError(f"Unknown backend: {name}")


def available_backend_names() -> list[str]:
    names = ["numpy"]
    if CudaBackend.available():
        names.append("cuda")
    return names
