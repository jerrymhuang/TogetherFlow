"""Validation-only ablation on the frozen matched reference; never loads test data."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from dataclasses import replace

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'outputs/regularized-retraining-2026-10-22'
PROFILES = {
    'baseline': {},
    'dropout': {'summary_dropout': .4, 'inference_dropout': .25},
    'decay': {'weight_decay': .15},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile', choices=PROFILES)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=120)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('epochs must be positive')
    # Import the frozen simulator, networks, adapter and training loop, preserving
    # the completed comparison even if the working tree changes during the pilot.
    source = REFERENCE / 'source'
    sys.path[:0] = [str(source / 'experiments'), str(source)]
    import run_main_paper_offline as runner
    from variable_radius_variants import BASE
    import jax
    if jax.default_backend() != 'gpu':
        raise RuntimeError('Research pilots require GPU')
    runner.OFFLINE_REGULARIZATION.update(PROFILES[args.profile])
    directory = args.output.resolve() / args.profile
    directory.mkdir(parents=True, exist_ok=True)
    specification = {
        'profile': args.profile, 'settings': runner.OFFLINE_REGULARIZATION,
        'epochs': args.epochs, 'training_budget': 3000, 'test_evaluated': False,
        'source': str(source),
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_manifest_sha256': hashlib.sha256((REFERENCE / 'source_sha256.json').read_bytes()).hexdigest(),
        'selection_metric': 'mean validation loss over final 30 epochs',
        'design': 'same initialization, banks, batch size and epoch budget; one seed, exploratory',
    }
    spec_path = directory / 'pilot.json'
    if spec_path.exists() and json.loads(spec_path.read_text()) != specification:
        raise RuntimeError('Pilot specification changed; choose a fresh output directory')
    runner.write_json(spec_path, specification)
    options = SimpleNamespace(regularized_offline=True, regularized_online=False,
        online=False, patience=0, epochs=args.epochs, batch_size=32, n_eval=300,
        draws=1000, matched=True, smoke=False, training_only=True, restart_training=False)
    variant = replace(BASE, online=False, epochs=args.epochs, batch_size=32,
                      early_stopping_patience=0, n_val=300, n_test=300)
    validation = runner.bank(REFERENCE / 'banks/validation', variant, 300, BASE.seed + 100000)
    train = runner.bank(REFERENCE / 'banks/train', variant, 30000, BASE.seed)
    runner.run_budget(variant, 3000, train, validation, None, directory, options)


if __name__ == '__main__':
    main()
