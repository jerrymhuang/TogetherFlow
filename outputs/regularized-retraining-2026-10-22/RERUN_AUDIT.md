# Regularized flat-model rerun audit

Existing reruns verified on 2026-10-22. All three arms completed 300 epochs and evaluated the final weights on the regenerated shared test bank (300 datasets, 1,000 posterior draws per dataset). Original results are preserved.

| Regime | Best validation loss | Final validation loss | Mean last 10 epochs |
|---|---:|---:|---:|
| offline-3000 | 1.0280 | 1.2555 | 1.1678 |
| offline-30000 | 0.5704 | 0.5923 | 0.6561 |
| transformer-flow_matching | 0.8613 | 0.9326 | 0.9745 |

The regularization profile uses AdamW (learning rate 0.0002, weight decay 0.05), summary/inference dropout 0.25/0.15, and validation-plateau learning-rate reductions. No early stopping or best-weight restoration is used.

The original offline final validation losses were 3.0855 (3K) and 1.0666 (30K). Comparing the final epoch with the minimum of 300 noisy validation measurements is insufficient evidence of overfitting. The regularized runs do not show the original sustained validation deterioration.

| Regime | Mean validation, epochs 51–100 | Mean validation, epochs 251–300 |
|---|---:|---:|
| Offline 3K | 1.16572 | 1.17802 |
| Offline 30K | 0.67117 | 0.66152 |
| Online | 1.01779 | 0.97811 |

The offline 3K curve plateaus with a small increase (about 1.1%), while offline 30K and online improve over these windows. These descriptive averages do not establish statistical significance or guarantee zero overfitting. Online training uses newly simulated training data rather than repeatedly fitting a fixed bank; its curve gives no evidence of sustained validation deterioration. Training and validation losses also differ in dropout behavior, so their gap alone is not a reliable overfitting diagnostic.

The 3K posterior diagnostics also report ECE 0.122–0.174 and negative-infinite log-gamma for all four parameters; calibration requires separate assessment and does not itself diagnose overfitting. See each arm’s metrics.csv and diagnostic figures.

No additional duplicate training was launched during this audit because completed regularized offline and online reruns already exist. The earlier characterization of residual overfitting based on best-versus-final loss was unsupported and has been corrected.
