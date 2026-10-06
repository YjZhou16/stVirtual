"""Run gain=1.2 extrapolation from the completed training manifest."""
import argparse
import json
import subprocess
import sys
sys.dont_write_bytecode = True
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    sys.path.insert(0, str(root.parent))
    from workflow import read_training_manifest
    def path(key): return str((config_path.parent / config[key]).resolve())
    if not args.dry_run:
        read_training_manifest(path('training_manifest'))
    for seed in config['seeds']:
        for fraction in config['fractions']:
            command = [sys.executable, str(root / 'simulation.py'),
                '--training-manifest', path('training_manifest'),
                '--output-root', path('output_root'),
                '--ablation-fraction', str(fraction), '--birth-hazard-max', '0.5',
                '--death-hazard-max', '0.04', '--message-passing-substeps', '3',
                '--latent-update-gain', str(config['latent_update_gain']),
                '--coord-step-fraction', '0.15', '--seed', str(seed), '--device', args.device,
                '--ablation-only', '--out-subdir', f'gain1p2_fraction{fraction:g}_seed{seed}']
            if args.dry_run:
                print(command)
            else:
                subprocess.run(command, cwd=root, check=True)

if __name__ == '__main__': main()
