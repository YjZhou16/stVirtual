from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from stvirtual.perturb import prepare_runtime
prepare_runtime(Path(__file__).resolve().parents[1])
"""Current stVirtual transition API for the migrated LUAD experiment."""
from stvirtual.models import stage2_3d_transition as _model
def __getattr__(name):
    return getattr(_model,name)
__all__ = [n for n in dir(_model) if not n.startswith('_')]
globals().update({n:getattr(_model,n) for n in __all__})
