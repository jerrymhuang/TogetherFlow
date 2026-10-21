# Offline regularization correction

The original no-early-stopping runs show deteriorating validation performance:

| Bank | Best validation loss (epoch) | Final validation loss |
|---|---|---|
| 3,000 | 1.0008 (23) | 3.0855 |
| 30,000 | 0.6387 (41) | 1.0666 |

Training losses continue decreasing. This supports overfitting rather than a
failure to optimize. Inspection confirms the installed BayesFlow offline dataset
shuffles each epoch and the configured dropout reaches the transformer blocks.
No implementation defect in those mechanisms was found. The original settings
were insufficient to control generalization across 300 repeated passes.

The correction is an **explicit optional offline profile**:

- AdamW learning rate 0.0002 instead of the default 0.0005 cosine peak.
- Weight decay 0.05 instead of 0.005.
- Summary dropout 0.25 instead of 0.10; inference dropout 0.15 instead of 0.05.
- Reduce learning rate by half after 8 epochs without a validation-loss
  improvement of at least 0.001, down to 0.000001.
- Gradient clip norm remains 1.5.
- All 300 epochs still run. No early stopping or restoration of earlier weights.
  Posterior evaluation uses the final weights.

Both budgets receive the same profile. Original architectures, layer widths,
parameter constraints, simulator, nested training banks, validation/test banks,
initialization seed, posterior draw count and metric definitions are preserved.
The online 2x2 comparison and Superstats results are unchanged. Dropout and the
optimizer policy intentionally differ from the online reference; this remains a
comparison of training regimes, not an isolation of simulation budget alone.

A 60-epoch pilot on the actual 3K bank uses the existing validation bank and does
not load the test bank or sample test posteriors. It checks whether the proposed
settings improve the original validation trajectory before full retraining.
The smoke test verifies end-to-end training, checkpoint saving and diagnostics.
Regression tests verify plateau LR reductions do not stop training or change
weights, and that the profile cannot accidentally be applied online.

Replacement results are written under `outputs/offline-regularization-2026-10-21/`;
original results are retained under `outputs/matched-retraining-2026-10-21/`.
The replacement queue reuses the exact existing banks. Its status is in
`queue-status.json`, per-epoch status is in each arm's `status.json`, and
`validation_comparison.pdf`/`.json` are produced when both arms complete.

```bash
python3 experiments/run_regularized_offline.py
```

The correction is a candidate regularization policy, not a guarantee that all
300-epoch generalization gaps disappear. Full-run validation and posterior
calibration must be assessed after training. Training losses are not directly
comparable across different dropout rates; validation curves are the relevant
comparison here.

## Pilot outcome

The 3K pilot completed all 60 epochs. Mean validation loss over epochs 51–60
fell from 1.40646 to 1.13877 (19.0% lower). Best single-epoch validation loss
was 1.03733 versus 1.00079 originally; the evidence supports improved late-run
stability, not a better peak result. The LR reached 0.000025 through plateau
reductions. No test posteriors were evaluated during selection. Both full
300-epoch replacement runs were launched after this check; their final results
are pending.
