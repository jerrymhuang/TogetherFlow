"""Detach online training into a systemd user service with host memory protection."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
UNIT = 'togetherflow-online-regularized'


def command():
    threads = len(os.sched_getaffinity(0))
    total_memory = os.sysconf('SC_PHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
    # Leave RAM for the OS; CPU and GPU compute are not quota-limited.
    memory_high = int(total_memory * .75)
    memory_max = int(total_memory * .85)
    return [
        'systemd-run', '--user', '--collect', f'--unit={UNIT}',
        f'--working-directory={ROOT}',
        '--property=CPUWeight=10',
        f'--property=MemoryHigh={memory_high}', f'--property=MemoryMax={memory_max}',
        '--property=MemorySwapMax=0', '--property=OOMPolicy=stop',
        '--property=Nice=10', '--property=IOWeight=10',
        '--property=TimeoutStopSec=30s',
        '--property=KillMode=control-group',
        '--setenv=CUDA_DEVICE_ORDER=PCI_BUS_ID', '--setenv=CUDA_VISIBLE_DEVICES=1',
        f'--setenv=NUMBA_NUM_THREADS={threads}', f'--setenv=OMP_NUM_THREADS={threads}',
        '--setenv=OPENBLAS_NUM_THREADS=1', '--setenv=MKL_NUM_THREADS=1',
        '--setenv=XLA_PYTHON_CLIENT_PREALLOCATE=false',
        '--setenv=MPLCONFIGDIR=/tmp/togetherflow-online-matplotlib',
        sys.executable, str(ROOT / 'experiments/run_regularized_online.py'),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    cmd = command()
    print(shlex.join(cmd), flush=True)
    if args.dry_run:
        return
    # systemd-run submits a detached service; all descendants share its limits.
    subprocess.run(cmd, check=True)
    print(f'Logs: journalctl --user -u {UNIT} -f')
    print(f'Stop: systemctl --user stop {UNIT}')


if __name__ == '__main__':
    main()
