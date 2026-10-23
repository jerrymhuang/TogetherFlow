# Offline regularization pilot results

Validation-only, 3K fixed training bank, identical initialization and frozen reference source. Baseline uses the first 120 epochs of the completed reference run. Primary metric: mean validation loss over epochs 91–120.

| Profile | Mean validation, epochs 91–120 | Change from baseline |
|---|---:|---:|
| baseline | 1.16949 | +0.0% improvement |
| dropout | 1.15542 | +1.2% improvement |
| decay | 1.16608 | +0.3% improvement |

Lowest mean validation: dropout.

These exploratory single-seed comparisons do not guarantee zero overfitting or establish statistical significance. Any promising candidate needs a longer validation run before replacing the completed results. Test posteriors were not evaluated. Training loss gaps across dropout settings are not directly comparable.
