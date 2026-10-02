"""Load the identically named module from the compiled model package."""
from importlib import import_module
from typing import Any, MutableMapping
from stvirtual import __version__

_INSTALL_HINT = (
    'Install the matching stVirtual model core with `pip install '
    'wheels/stvirtual_core-0.2.0-cp312-cp312-linux_x86_64.whl`.'
)

def export_binary_module(namespace: MutableMapping[str, Any], module_name: str) -> None:
    """Expose the native model without translating names or call arguments."""
    try:
        core = import_module('_stvirtual_core')
    except ModuleNotFoundError as error:
        if error.name == '_stvirtual_core':
            raise ImportError(_INSTALL_HINT) from error
        raise
    if getattr(core, '__version__', None) != __version__:
        raise ImportError(_INSTALL_HINT)
    native_name = module_name.replace('stvirtual.models.', '_stvirtual_core.', 1)
    implementation = import_module(native_name)
    exports = {name: getattr(implementation, name) for name in dir(implementation)
               if not name.startswith('__')}
    namespace.update(exports)
    namespace['__all__'] = sorted(name for name in exports if not name.startswith('_'))
    namespace['_implementation'] = implementation
