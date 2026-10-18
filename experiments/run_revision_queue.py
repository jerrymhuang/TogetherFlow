"""Sequential, resumable full-budget queue. Requires a working CUDA device."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/revision'
MAIN = ROOT / '.venv/bin/python'
SUPER = ROOT / '.venv-superstats/bin/python'
SLUGS = ['rev-fixed-reference', 'rev-fixed-switching', 'switching-transfer',
         'rev-fixed-partial-control', 'rev-fixed-partial', 'superstats-fixed-path',
         'rev-fixed-timevarying', 'rev-fixed-radius-1.25', 'rev-fixed-radius-4',
         'rev-fixed-prior-wide', 'rev-fixed-prior-tight', 'rev-fixed-prior-slow-velocity',
         'rev-fixed-prior-eta-smooth', 'rev-fixed-prior-eta-diffuse',
         'rev-fixed-offline-3k', 'rev-fixed-legacy-heading', 'rev-fixed-legacy-noise']


def command(slug):
    if slug == 'superstats-fixed-path':
        return [str(SUPER), 'experiments/run_v5b_path.py', '--fixed-radius']
    return [str(MAIN), 'experiments/run_revision.py',
            '--cross-evaluate' if slug == 'switching-transfer' else slug]


def complete(slug):
    p = OUT / slug
    if slug == 'superstats-fixed-path':
        return all((p / n).exists() for n in ['per_step_metrics.csv', 'time_varying_posterior.png'])
    if slug == 'switching-transfer':
        return (p / 'design.json').exists() and len(list(p.glob('*__on__*.csv'))) == 4
    return all((p / n).exists() for n in ['report.md', 'metrics.csv', 'checkpoints/model.keras'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    if args.dry_run:
        for slug in SLUGS:
            print('complete' if complete(slug) else 'pending', slug)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / 'queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = dict(os.environ, KERAS_BACKEND='jax', JAX_PLATFORMS='cuda,cpu',
               PYTHONPATH=str(ROOT), PYTHONUNBUFFERED='1',
               XLA_PYTHON_CLIENT_PREALLOCATE='false')
    # No silent CPU fallback. Both package environments must actually compute on GPU.
    probe = 'import jax; import jax.numpy as j; assert jax.default_backend() == "gpu", jax.devices(); print(jax.devices()); jax.debug.callback(lambda x: None, j.ones(1)); print((j.ones((32,32)) @ j.ones((32,32))).block_until_ready().sum())'
    for python in [MAIN, SUPER]:
        subprocess.run([str(python), '-c', probe], cwd=ROOT, env=env, check=True)
    logs = OUT / 'logs'
    logs.mkdir(exist_ok=True)
    status = {}
    def save():
        tmp = OUT / 'queue-status.tmp'
        tmp.write_text(json.dumps(status, indent=2))
        tmp.replace(OUT / 'queue-status.json')
    for slug in SLUGS:
        if complete(slug):
            status[slug] = {'status': 'complete (existing)'}; save(); continue
        if slug == 'switching-transfer' and not all(complete(s) for s in SLUGS[:2]):
            status[slug] = {'status': 'blocked by training failure'}; save(); continue
        stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        status[slug] = {'status': 'running', 'started_utc': stamp}; save()
        with (logs / f'{slug}.log').open('a') as log:
            log.write(f'\nQueue start: {stamp}\n'); log.flush()
            result = subprocess.run(command(slug), cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        status[slug].update(status='complete' if result.returncode == 0 and complete(slug) else 'failed',
                            returncode=result.returncode,
                            ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save()
    sys.exit(any(s['status'] in ['failed','blocked by training failure'] for s in status.values()))


if __name__ == '__main__':
    main()
