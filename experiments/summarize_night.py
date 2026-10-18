"""Summarize saved historical metrics without importing the training runtime.

Run: python experiments/summarize_night.py
No generated heuristic establishes identifiability or real-world validity.
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'outputs' / 'variants'
DEST = ROOT / 'docs' / 'revision'


def main():
    records = []
    for path in sorted(OUT.glob('*/metrics.csv')):
        with path.open() as f:
            rows = list(csv.reader(f))
        for row in rows[1:]:
            for parameter, value in zip(rows[0][1:], row[1:]):
                records.append({'variant': path.parent.name, 'metric': row[0],
                                'parameter': parameter, 'value': float(value)})
    DEST.mkdir(parents=True, exist_ok=True)
    with (DEST / 'existing_metrics.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['variant','metric','parameter','value'])
        writer.writeheader(); writer.writerows(records)
    params = list(dict.fromkeys(r['parameter'] for r in records))
    variants = list(dict.fromkeys(r['variant'] for r in records))
    lookup = {(r['variant'],r['metric'],r['parameter']): r['value'] for r in records}
    lines = ['# Saved experiment results', '',
             'Historical artifacts, not new runs. Reviewer numbering in old reports is stale.', '',
             'NRMSE normalization and simulator versions must be verified before comparing historical and new runs. '
             'Most historical arms use features computed at the true radius. '
             'Use fixed-radius or radius-independent analyses for deployable inference claims.', '',
             'No metrics were found for `v8-salience-hidden`. The superstats path run has a diagnostic PNG but no numeric metrics archive locally.', '']
    for metric in ['NRMSE', 'ECE', 'Post. Contraction']:
        lines += [f'## {metric}', '', '| Variant | ' + ' | '.join(params) + ' |',
                  '|---|' + '|'.join(['---']*len(params)) + '|']
        for v in variants:
            values = [f'{lookup[v,metric,p]:.4f}' if (v,metric,p) in lookup else '—' for p in params]
            lines.append('| '+v+' | '+' | '.join(values)+' |')
        lines.append('')
    lines += ['## Artifact availability', '', '| Variant | Metrics | Checkpoint | Manifest |', '|---|---|---|---|']
    for v in variants:
        folder = OUT/v
        lines.append(f'| {v} | yes | {"yes" if (folder/"checkpoints/model.keras").exists() else "missing"} | {"yes" if (folder/"manifest.json").exists() else "missing"} |')
    (DEST / 'existing_results.md').write_text('\n'.join(lines)+'\n')
    print(f'Wrote {len(variants)} variants to {DEST}')


if __name__ == '__main__':
    main()
