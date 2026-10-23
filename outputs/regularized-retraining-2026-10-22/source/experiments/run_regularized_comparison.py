"""Regenerate reference banks and retrain regularized offline/online models."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'outputs/matched-retraining-2026-10-21'
DEFAULT_OUTPUT = ROOT / 'outputs/regularized-retraining-2026-10-22'


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def prepare(output):
    source = output / 'source'
    if source.exists():
        raise RuntimeError('Source already exists; choose a fresh output directory')
    source.mkdir(parents=True)
    # Copy code only; never copy large historical outputs or environments.
    for folder in ('experiments', 'togetherflow'):
        for path in (ROOT / folder).rglob('*.py'):
            if '__pycache__' in path.parts:
                continue
            target = source / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    # Restore the exact original simulator, model architecture and data adapter.
    for name in ('togetherflow/simulator.py', 'togetherflow/priors.py',
                 'togetherflow/networks/transformer.py', 'togetherflow/networks/bdlstm.py',
                 'experiments/run_variant.py', 'experiments/variants.py',
                 'experiments/variable_radius_variants.py'):
        shutil.copy2(REFERENCE / 'source_snapshot' / name, source / name)
    manifest = json.loads((REFERENCE / 'transformer-flow_matching/manifest.json').read_text())
    for name, expected in manifest['source_sha256'].items():
        if name in ('experiments/run_main_paper_offline.py', 'experiments/run_matched_retraining.py'):
            continue  # These runners intentionally add the new regularization profile.
        path = source / name
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Reference source mismatch: {name}')
    save(output / 'source_sha256.json', {
        str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in source.rglob('*.py')})
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--prepared', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = output / 'source' if args.prepared else prepare(output)
    if args.prepare_only:
        print(source)
        return
    import fcntl
    lock = (output / 'queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    python = str(ROOT / '.venv/bin/python')
    logs = output / 'logs'
    logs.mkdir(exist_ok=True)
    banks = output / 'banks'
    env = dict(os.environ, KERAS_BACKEND='jax', PYTHONUNBUFFERED='1',
               XLA_PYTHON_CLIENT_PREALLOCATE='false', MPLBACKEND='Agg',
               MPLCONFIGDIR='/tmp/togetherflow-comparison-mpl',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    threads = max(1, len(os.sched_getaffinity(0)) // 2)
    status = {'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'stage': 'regenerating evaluation banks', 'arms': {}}
    save(output / 'queue-status.json', status)
    # The new bank directories have no stale "completed" metadata.
    code = '''import json,sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()/'experiments'))
from run_main_paper_offline import bank
from variable_radius_variants import BASE
root,reference=map(Path,sys.argv[1:])
for name,offset in [('validation',100000),('test',200000)]:
    bank(root/name,BASE,300,BASE.seed+offset)
    actual=json.loads((root/name/'bank.json').read_text())
    expected=json.loads((reference/'banks'/name/'bank.json').read_text())
    assert actual==expected, f'Regenerated bank specification mismatch: {name}'
    print(name, 'complete; original specification matched',flush=True)
'''
    with (logs / 'banks.log').open('a') as log:
        subprocess.run([python, '-c', code, str(banks), str(REFERENCE)], cwd=source,
            env=dict(env, JAX_PLATFORMS='cpu', NUMBA_NUM_THREADS=str(threads * 2)),
            stdout=log, stderr=subprocess.STDOUT, check=True)
    status['stage'] = 'training and evaluation'
    save(output / 'queue-status.json', status)
    processes = {}
    handles = []

    def start(arm, gpu):
        profile = '--regularized-online' if arm == 'transformer-flow_matching' else '--regularized-offline'
        command = [python, 'experiments/run_matched_retraining.py', arm, profile,
            '--epochs', '300', '--banks', str(banks), '--output', str(output)]
        log = (logs / f'{arm}.log').open('a')
        handles.append(log)
        proc = subprocess.Popen(command, cwd=source, env=dict(env,
            JAX_PLATFORMS='cuda', CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=str(gpu),
            NUMBA_NUM_THREADS=str(threads), OMP_NUM_THREADS=str(threads)),
            stdout=log, stderr=subprocess.STDOUT)
        processes[arm] = proc
        status['arms'][arm] = {'status': 'running', 'pid': proc.pid, 'gpu': gpu, 'command': command}
        save(output / 'queue-status.json', status)

    start('transformer-flow_matching', 1)
    start('offline-3000', 0)
    second_offline_started = False
    while processes:
        for arm, proc in list(processes.items()):
            code = proc.poll()
            if code is None:
                continue
            complete = code == 0 and (output / arm / 'complete.json').exists()
            status['arms'][arm].update(status='complete' if complete else 'failed', returncode=code)
            del processes[arm]
            save(output / 'queue-status.json', status)
            if arm == 'offline-3000' and complete and not second_offline_started:
                second_offline_started = True
                start('offline-30000', 0)
        if processes:
            time.sleep(5)
    for handle in handles:
        handle.close()
    success = len(status['arms']) == 3 and all(v['status'] == 'complete' for v in status['arms'].values())
    status['stage'] = 'complete' if success else 'failed'
    save(output / 'queue-status.json', status)
    if not success:
        sys.exit(1)


if __name__ == '__main__':
    main()
