"""Run this dataset perturbation using the current stVirtual API."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
from stvirtual.perturb import main
if __name__ == "__main__":
    if not any(a == "--config" or a.startswith("--config=") for a in sys.argv[1:]):
        sys.argv.extend(["--config", str(HERE / "config.yaml")])
    main()
