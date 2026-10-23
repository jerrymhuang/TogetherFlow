"""Serial retraining queue with explicit status, GPU checks, and isolated failures."""
import argparse
import csv
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STATIC = [f"{s}-{i}" for s in ("transformer", "bdlstm") for i in ("flow_matching", "diffusion")]
ARMS = STATIC + ["offline-3000", "offline-30000", "superstats-fixed", "superstats-variable"]


def write_json(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def command(arm, output, smoke):
    if arm.startswith('superstats'):
        cmd = [str(ROOT / '.venv-superstats/bin/python'), 'experiments/run_time_varying.py',
               '--fixed-radius' if arm.endswith('fixed') else '--variable-radius',
               '--match-baseline', '--output-dir', str(output / arm)]
    else:
        cmd = [str(ROOT / '.venv/bin/python'), 'experiments/run_matched_retraining.py',
               arm, '--output', str(output)]
    return cmd + (['--smoke'] if smoke else [])


def summarize(output, status):
    lines = ['# Matched retraining status', '', '| Arm | Status |', '|---|---|']
    for arm in ARMS:
        lines.append(f"| {arm} | {status.get(arm, {}).get('status', 'pending')} |")
    lines += ['', 'Static arms share simulation definitions and held-out banks. Offline arms use final weights after all 300 epochs, without early stopping.',
              '', 'Superstats uses the dynamic-weight model; fixed radius conditions on r=2.5. Its path metrics are separate from scalar baseline metrics.', '']
    (output / 'STATUS.md').write_text('\n'.join(lines))
    rows = []
    for arm in ARMS:
        if status.get(arm, {}).get('status') != 'complete':
            continue
        path = output / arm / ('invariant_metrics.csv' if arm.startswith('superstats') else 'metrics.csv')
        if path.exists():
            with path.open() as file:
                reader = csv.reader(file)
                header = next(reader)
                for row in reader:
                    for parameter, value in zip(header[1:], row[1:]):
                        rows.append([arm, row[0], {"r_t": "r", "v_t": "v", "eta_t": "noise", "tau_t": "tau"}.get(parameter, parameter), value])
    with (output / 'comparison.csv').open('w') as file:
        writer = csv.writer(file)
        writer.writerow(['arm', 'metric', 'parameter', 'value'])
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/matched-retraining-2026-10-21')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.dry_run:
        for arm in ARMS:
            print(' '.join(command(arm, args.output, args.smoke)))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / 'queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = dict(os.environ, KERAS_BACKEND='jax', PYTHONPATH=str(ROOT),
               PYTHONUNBUFFERED='1', XLA_PYTHON_CLIENT_PREALLOCATE='false',
               NUMBA_NUM_THREADS='12', MPLBACKEND='Agg',
               JAX_PLATFORMS='cpu' if args.smoke else 'cuda,cpu')
    if not args.smoke:
        driver_version = Path('/proc/driver/nvidia/version').read_text()
        if '580.173.02' in driver_version:
            driver_lib = Path('/tmp/togetherflow-nvidia-580.173.02/lib')
            if not (driver_lib / 'libcuda.so.1').exists():
                raise RuntimeError('Matching NVIDIA library workaround missing')
            env['LD_LIBRARY_PATH'] = str(driver_lib) + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
        probe = 'import jax; import jax.numpy as j; assert jax.default_backend() == "gpu", jax.devices(); print(jax.devices()); print((j.ones((32,32)) @ j.ones((32,32))).block_until_ready().sum())'
        for executable in ('.venv/bin/python', '.venv-superstats/bin/python'):
            subprocess.run([str(ROOT / executable), '-c', probe], env=env, cwd=ROOT, check=True)
    logs = args.output / 'logs'
    logs.mkdir(exist_ok=True)
    state_path = args.output / 'queue-status.json'
    status = json.loads(state_path.read_text()) if state_path.exists() else {}
    for arm in ARMS:
        if (args.output / arm / 'complete.json').exists():
            status[arm] = {'status': 'complete'}
            continue
        status[arm] = dict(status='running', started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                           command=command(arm, args.output, args.smoke))
        write_json(state_path, status)
        summarize(args.output, status)
        with (logs / f'{arm}.log').open('a') as log:
            result = subprocess.run(command(arm, args.output, args.smoke), cwd=ROOT,
                                    env=env, stdout=log, stderr=subprocess.STDOUT)
        status[arm].update(status='complete' if result.returncode == 0 and (args.output / arm / 'complete.json').exists() else 'failed',
                           returncode=result.returncode, ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write_json(state_path, status)
        summarize(args.output, status)
    write_json(state_path, status)
    summarize(args.output, status)
    sys.exit(any(value['status'] != 'complete' for value in status.values()))


if __name__ == '__main__':
    main()
