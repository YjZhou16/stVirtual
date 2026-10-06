"""Connect heart Slide-seq preprocessing, training and extrapolation artifacts."""
import json
from pathlib import Path

ARTIFACT_KEYS = (
    'adata_path', 'stage1_trace', 'bound_dir', 'stage1_checkpoint',
    'policy_checkpoint', 'decoder_checkpoint', 'lr_pairs', 'transition_prior',
)


def read_training_manifest(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'{path}: run preprocess.ipynb, then train.ipynb first')
    manifest = json.loads(path.read_text())
    for key in ARTIFACT_KEYS:
        value = Path(manifest[key])
        if not value.is_absolute():
            value = path.parent / value
        if not (value.is_dir() if key == 'bound_dir' else value.is_file()):
            raise FileNotFoundError(f'Missing training artifact {key}: {value}')
        manifest[key] = str(value.resolve())
    return manifest


def write_training_manifest(path, *, source, target, steps, latent_key, **artifacts):
    if set(artifacts) != set(ARTIFACT_KEYS):
        raise ValueError(f'Expected artifact fields: {ARTIFACT_KEYS}')
    manifest = dict(source=source, target=target, steps=steps, latent_key=latent_key)
    manifest.update({key: str(Path(value).resolve()) for key, value in artifacts.items()})
    for key in ARTIFACT_KEYS:
        value = Path(manifest[key])
        if not (value.is_dir() if key == 'bound_dir' else value.is_file()):
            raise FileNotFoundError(f'Missing training artifact {key}: {value}')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    return path


def train_route_decoders(config, experiment_dir, *, device):
    """Train route decoders against this run's input and Stage-1 normalization."""
    from stvirtual.decoder.config import TrainConfig
    from stvirtual.decoder.train import train_decoder

    experiment_dir = Path(experiment_dir)
    def resolve(value):
        path = Path(value)
        return path if path.is_absolute() else experiment_dir / path

    checkpoints = {}
    for source, target in config['routes']:
        checkpoint = resolve(config['decoder_checkpoint'].format(src=source, tgt=target))
        normalization = resolve(config['stage1_checkpoint']) / f'{source}_to_{target}' / 'checkpoints/best.pt'
        settings = TrainConfig(
            data_path=resolve(config['input_path']), output_dir=checkpoint.parents[1],
            latent_key=config['latent_key'], counts_key='counts', sample_key='sample',
            sample_times={source: 0.0, target: 1.0}, checkpoint_name=checkpoint.name,
            simulation_normalization_checkpoint=normalization, device=str(device),
            **config.get('decoder_training', {}),
        )
        actual = train_decoder(settings).resolve()
        if actual != checkpoint.resolve():
            raise RuntimeError(f'Decoder output path mismatch: {actual} != {checkpoint}')
        checkpoints[f'{source}_to_{target}'] = actual
    return checkpoints
