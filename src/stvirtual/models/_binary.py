"""Load the separately distributed binary model implementation."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any, MutableMapping


_INSTALL_HINT = (
    "The stVirtual binary model core is not installed. Install the Linux/Python "
    "3.12 wheel bundled in the repository with `pip install "
    "wheels/stvirtual_core-0.1.0.dev0-cp312-cp312-linux_x86_64.whl`."
)


def export_binary_module(namespace: MutableMapping[str, Any], module_name: str) -> ModuleType:
    """Expose a compiled implementation through its stable public module path."""
    try:
        implementation = import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == "_stvirtual_core" or (exc.name or "").startswith("_stvirtual_core."):
            raise ImportError(_INSTALL_HINT) from exc
        raise

    exported = {
        name: getattr(implementation, name)
        for name in dir(implementation)
        if not name.startswith("__")
    }
    namespace.update(exported)
    namespace["__all__"] = sorted(name for name in exported if not name.startswith("_"))
    namespace["__doc__"] = implementation.__doc__

    def module_getattr(name: str) -> Any:
        return getattr(implementation, name)

    def module_dir() -> list[str]:
        return sorted(set(namespace) | set(dir(implementation)))

    namespace["__getattr__"] = module_getattr
    namespace["__dir__"] = module_dir
    namespace["_implementation"] = implementation
