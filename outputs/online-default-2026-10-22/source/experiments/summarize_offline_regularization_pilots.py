"""Summarize completed validation-only pilots against the matched baseline."""
import argparse
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/offline-regularization-pilots-2026-10-22')
    args = parser.parse_args()
    baseline = ROOT / 'outputs/regularized-retraining-2026-10-22/offline-3000'
    rows = {}
    for name, directory in [('baseline', baseline), *[(n, args.output/n) for n in ('dropout', 'decay')]]:
        h = json.loads((directory / 'history.json').read_text())
        v = np.asarray(h['val_loss'][:120])
        if len(v) != 120 or not np.isfinite(v).all():
            raise ValueError(f'Incomplete history: {name}')
        rows[name] = {'epochs_compared': 120, 'mean_validation_epochs_91_120': float(v[-30:].mean()),
            'validation_window_means': [float(v[i:i+30].mean()) for i in range(0,120,30)],
            'best_validation': float(v.min()), 'final_validation': float(v[-1])}
    reference = rows['baseline']['mean_validation_epochs_91_120']
    for row in rows.values():
        row['relative_improvement'] = 1-row['mean_validation_epochs_91_120']/reference
    winner = min(rows, key=lambda n: rows[n]['mean_validation_epochs_91_120'])
    result = {'test_evaluated': False, 'exploratory_single_seed': True, 'rows': rows,
              'lowest_mean_validation': winner}
    (args.output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    lines = ['# Offline regularization pilot results', '',
        'Validation-only, 3K fixed training bank, identical initialization and frozen reference source. Baseline uses the first 120 epochs of the completed reference run. Primary metric: mean validation loss over epochs 91–120.', '',
        '| Profile | Mean validation, epochs 91–120 | Change from baseline |', '|---|---:|---:|']
    for name,row in rows.items():
        lines.append(f"| {name} | {row['mean_validation_epochs_91_120']:.5f} | {row['relative_improvement']:+.1%} improvement |")
    lines += ['', f'Lowest mean validation: {winner}.', '',
        'These exploratory single-seed comparisons do not guarantee zero overfitting or establish statistical significance. Any promising candidate needs a longer validation run before replacing the completed results. Test posteriors were not evaluated. Training loss gaps across dropout settings are not directly comparable.', '']
    (args.output/'REPORT.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
