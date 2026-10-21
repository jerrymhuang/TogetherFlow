"""Run the two regularized offline budgets, preserving their original results."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / 'outputs/matched-retraining-2026-10-21'


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def report(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    rows = {}
    for ax, n in zip(axes, (3000, 30000)):
        original = json.loads((ORIGINAL / f'offline-{n}/history.json').read_text())
        updated = json.loads((output / f'offline-{n}/history.json').read_text())
        for name, history in [('Original', original), ('Regularized', updated)]:
            ax.plot(np.arange(1, len(history['val_loss'])+1), history['val_loss'], label=name)
        ax.set(title=f'{n:,} simulations', xlabel='Epoch', ylabel='Validation loss')
        ax.legend()
        rows[str(n)] = dict(original_final_validation=original['val_loss'][-1],
            revised_final_validation=updated['val_loss'][-1],
            revised_best_validation=min(updated['val_loss']),
            revised_final_10_mean=float(np.mean(updated['val_loss'][-10:])),
            original_final_10_mean=float(np.mean(original['val_loss'][-10:])),
            epochs=len(updated['loss']))
    fig.tight_layout()
    for suffix in ('pdf', 'png'):
        fig.savefig(output / f'validation_comparison.{suffix}', dpi=160)
    plt.close(fig)
    write_json(output / 'validation_comparison.json', rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/offline-regularization-2026-10-21')
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    if args.report_only:
        report(args.output)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / 'queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = dict(os.environ, KERAS_BACKEND='jax', JAX_PLATFORMS='cuda,cpu',
               XLA_PYTHON_CLIENT_PREALLOCATE='false', PYTHONUNBUFFERED='1', NUMBA_NUM_THREADS='12')
    if '580.173.02' in Path('/proc/driver/nvidia/version').read_text():
        library = Path('/tmp/togetherflow-nvidia-580.173.02/lib')
        if not (library / 'libcuda.so.1').exists():
            raise RuntimeError('Matching NVIDIA workaround is missing')
        env['LD_LIBRARY_PATH'] = str(library) + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    python = str(ROOT / '.venv/bin/python')
    subprocess.run([python, '-c', 'import jax; import jax.numpy as j; assert jax.default_backend() == "gpu"; print((j.ones((16,16)) @ j.ones((16,16))).block_until_ready().sum())'], cwd=ROOT, env=env, check=True)
    snapshot = args.output / 'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    for name in ('run_main_paper_offline.py', 'run_matched_retraining.py', 'run_regularized_offline.py'):
        shutil.copy2(ROOT / 'experiments' / name, snapshot / name)
    logs = args.output / 'logs'
    logs.mkdir(exist_ok=True)
    status = {}
    for n in (3000, 30000):
        arm = f'offline-{n}'
        if (args.output / arm / 'complete.json').exists():
            status[arm] = {'status': 'complete'}
            continue
        command = [python, 'experiments/run_matched_retraining.py', arm,
            '--regularized-offline', '--epochs', '300', '--banks', str(ORIGINAL / 'banks'),
            '--output', str(args.output)]
        status[arm] = dict(status='running', command=command,
            started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write_json(args.output / 'queue-status.json', status)
        with (logs / f'{arm}.log').open('a') as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        complete = result.returncode == 0 and (args.output / arm / 'complete.json').exists()
        if complete:
            selection = json.loads((args.output / arm / 'training_complete.json').read_text())
            complete = selection['epochs'] == selection['selected_epoch'] == 300 and selection['selection'] == 'final epoch'
        status[arm].update(status='complete' if complete else 'failed', returncode=result.returncode,
            ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write_json(args.output / 'queue-status.json', status)
    if all(row['status'] == 'complete' for row in status.values()):
        subprocess.run([python, __file__, '--report-only', '--output', str(args.output)], cwd=ROOT, env=env, check=True)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
