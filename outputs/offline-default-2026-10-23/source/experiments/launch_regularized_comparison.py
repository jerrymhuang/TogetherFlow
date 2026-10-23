"""Launch detached bank regeneration and regularized online/offline retraining."""
import argparse
from pathlib import Path
import shlex
import subprocess
from launch_regularized_online import command

ROOT = Path(__file__).resolve().parents[1]
UNIT = 'togetherflow-regularized-comparison'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    cmd = command()[:-2]
    cmd = [f'--unit={UNIT}' if part.startswith('--unit=') else part for part in cmd]
    cmd += [str(ROOT / '.venv/bin/python'),
            str(ROOT / 'experiments/run_regularized_comparison.py'), '--prepared']
    print(shlex.join(cmd), flush=True)
    if not args.dry_run:
        subprocess.run(cmd, check=True)
        print(f'Logs: journalctl --user -u {UNIT} -f')
        print(f'Stop: systemctl --user stop {UNIT}')


if __name__ == '__main__':
    main()
