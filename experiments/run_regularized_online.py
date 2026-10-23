"""Run the regularized online reference using the original evaluation banks."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/online-regularization-2026-10-22')
    parser.add_argument('--banks', type=Path, default=ROOT / 'outputs/matched-retraining-2026-10-21/banks')
    parser.add_argument('--python', type=Path, default=ROOT / '.venv/bin/python')
    args = parser.parse_args()
    # A completed bank.json alone is insufficient in a source-only checkout.
    missing = []
    if not args.python.is_file():
        missing.append(f'Training interpreter missing: {args.python}')
    for name in ('validation', 'test'):
        directory = args.banks / name
        if not (directory / 'bank.json').exists():
            missing.append(f'Bank metadata missing: {directory}')
            continue
        state = json.loads((directory / 'bank.json').read_text())
        if state['completed'] != state['spec']['size']:
            missing.append(f'Bank incomplete: {directory}')
        for key in state['spec']['infer'] + state['spec']['channels']:
            if not (directory / f'{key}.npy').is_file():
                missing.append(f'Bank array missing: {directory / (key + ".npy")}')
    if missing:
        sys.exit('\n'.join(missing))
    env = dict(os.environ, KERAS_BACKEND='jax', JAX_PLATFORMS='cuda,cpu',
               XLA_PYTHON_CLIENT_PREALLOCATE='false', PYTHONUNBUFFERED='1')
    subprocess.run([str(args.python), '-c',
        'import jax; import jax.numpy as j; assert jax.default_backend() == "gpu"; print((j.ones((16,16)) @ j.ones((16,16))).block_until_ready())'],
        cwd=ROOT, env=env, check=True)
    subprocess.run([str(args.python), 'experiments/run_matched_retraining.py',
        'transformer-flow_matching', '--regularized-online', '--epochs', '300',
        '--banks', str(args.banks.resolve()), '--output', str(args.output.resolve())],
        cwd=ROOT, env=env, check=True)


if __name__ == '__main__':
    main()
