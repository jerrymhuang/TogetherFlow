# Beacon-switching recovery and calibration appendix

`beacon_switching.tex` presents the beacon-switching model extension and its
parameter recovery and calibration, with the variable-radius reference for
comparison. The text describes the dynamic-salience mechanism, followed by
paired recovery, coverage, and marginal rank-calibration figures, one numerical
table, and a three-paragraph Simulation Study.

## Sources and interpretation

The figures and table use the original diagnostics from:

- `outputs/variable-radius/rev-variable-reference/`
- `outputs/variable-radius/rev-variable-switching/`

Each estimator is trained and evaluated under its own generative model on a
separate bank of 300 held-out datasets. These figures and metrics are not the
2x2 matched transfer evaluation. All table values come from the same run folders
as their figures. The switching arm with observed salience is not used here:
the evaluated estimator does not receive the realized salience trajectories.

`table_values.csv` records all 24 unrounded table values and source paths.
`figure_sources.csv` records the six copied figure assets, sources, and hashes.
The two metric files and run manifests are copied to `source_metrics/`.

## Integration and rebuilding

Copy `beacon_switching.tex` and the six PNG figures listed in
`figure_sources.csv` to the manuscript and its graphics directory. The fragment
requires amsmath, graphicx, booktabs, multirow, and hyperref. `preview.tex` supplies a
standalone wrapper and points to the local `figures/` folder.

Rebuild the table and copy the matching figure assets from the repository root:

```bash
python3 outputs/docs/beacon_switching_revision/build_appendix.py
```

No model retraining or posterior sampling is performed. The source values,
figure hashes, table structure, and internal references were checked. No TeX
compiler is installed in the current environment, so the revised fragment has
not been compiled. With pdflatex installed, compile `preview.tex` twice from
this directory.

## Separate transfer analysis

The earlier draft is retained as `beacon_switching_transfer.tex`, with its
original figure `figures/ext-switching-transfer.pdf`, copied transfer metrics,
and `transfer_table_values.csv`. Rebuild these separate materials using
`build_transfer_appendix.py`. They are not included by the current preview.
