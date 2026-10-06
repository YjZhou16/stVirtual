"""Decode saved simulation frames using an experiment's decoder and Stage 1 model."""
from pathlib import Path
import argparse
import json
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True, help='Experiment directory containing config.yaml')
    parser.add_argument('--source', help='Source sample (defaults to the first configured route)')
    parser.add_argument('--target', help='Target sample')
    parser.add_argument('--frame-dir', type=Path, help='Simulation directory; required when multiple runs exist')
    parser.add_argument('--output-dir', type=Path, help='Output directory (default: decoded directory beside simulation)')
    parser.add_argument('--decoder-checkpoint', type=Path)
    parser.add_argument('--stage1-checkpoint', type=Path, help='Stage 1 checkpoint used to generate these frames')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Check paths and list frames without decoding')
    args = parser.parse_args()
    import yaml
    experiment = args.experiment.expanduser().resolve()
    config = yaml.safe_load((experiment / 'config.yaml').read_text())
    def resolve(value):
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (experiment / path).resolve()
    if bool(args.source) != bool(args.target):
        parser.error('--source and --target must be provided together')
    routes = config.get('routes')
    if not routes:
        ids = config.get('route_ids', [])
        routes = list(zip(ids[:-1], ids[1:]))
    if not routes:
        defaults = {'Mcardiac': [('E9.5h', 'E11.5h')], 'human_lung_cancer': [('AAH', 'LUAD')],
                    'human_gastric_cancer': [('normal', 'cancer')]}
        routes = defaults.get(experiment.name, [])
    if args.source:
        src, tgt = args.source, args.target
    elif routes:
        src, tgt = map(str, routes[0])
    else:
        parser.error('No route in config.yaml; provide --source and --target')
    route = f'{src}_to_{tgt}'
    run_root = resolve(config['run_root'])
    if args.frame_dir:
        frames = args.frame_dir.expanduser().resolve()
    else:
        candidates = [run_root / 'simulation' / route]
        candidates += sorted(run_root.glob(f'{route}_*/simulation'))
        candidates = [p for p in candidates if p.is_dir() and any(p.glob('*.h5ad'))]
        if len(candidates) != 1:
            parser.error(f'Expected one simulation directory for {route}; found {len(candidates)}. Set --frame-dir.')
        frames = candidates[0]
    if args.output_dir:
        output = args.output_dir.expanduser().resolve()
    elif frames.parent.name == 'simulation':
        output = frames.parent.parent / 'decoded' / route
    else:
        output = frames.parent / 'decoded'
    decoder = args.decoder_checkpoint.expanduser().resolve() if args.decoder_checkpoint else resolve(config['decoder_checkpoint'].format(src=src, tgt=tgt))
    stage1 = args.stage1_checkpoint.expanduser().resolve() if args.stage1_checkpoint else resolve(config['stage1_checkpoint']) / route / 'checkpoints' / 'best.pt'
    def numeric_key(path):
        return [int(x) if x.isdigit() else x for x in re.split(r'(\d+)', path.name)]
    paths = sorted(frames.glob('*.h5ad'), key=numeric_key)
    if not paths:
        raise FileNotFoundError(f'No simulation frames in {frames}')
    for checkpoint in (decoder, stage1):
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    if output == frames or output.is_relative_to(frames) or frames.is_relative_to(output):
        raise ValueError('Input and output directories must be separate')
    existing = [output / p.name for p in paths if (output / p.name).exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f'{len(existing)} decoded files already exist; use a new --output-dir or --overwrite')
    print(json.dumps({'route': route, 'frames': len(paths), 'frame_dir': str(frames),
                      'output_dir': str(output), 'decoder': str(decoder), 'stage1': str(stage1)}, indent=2))
    if args.dry_run:
        return
    from stvirtual.decoder.infer import decode_h5ad
    for i, path in enumerate(paths, 1):
        decode_h5ad(input_path=path, output_path=output / path.name,
                    checkpoint_path=decoder, latent_key='X_latent', latent_is_normalized=True,
                    normalization_checkpoint=stage1, device_name=args.device, overwrite=args.overwrite)
        print(f'[{i}/{len(paths)}] {path.name}', flush=True)


if __name__ == '__main__':
    main()
