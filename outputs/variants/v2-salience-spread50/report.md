# Beacon salience inferred

**Variant:** `v2-salience-spread50`  
**Addresses:** Reviewer point 8

## Why this arm exists

The Unity twin lets an agent re-target between beacons; the Python training simulator selected the nearest beacon unconditionally. The unified score s_b^alpha / d_ib closes that gap, and promoting alpha from a fixed setting to an inferred parameter turns the discrepancy into a measurable quantity: if alpha is recoverable, the training simulator can represent the twin's switching behaviour rather than merely approximating it. The paired spread-10 arm was dropped: beacons are virtual and always outside the room, and at spread 10 roughly 80% of sampled beacons fell inside it, so that arm was exercising a configuration the apparatus cannot produce. The beacon sampler now rejects the interior.

## Generative Configuration

| Setting | Value |
|---------|-------|
| Reviewer point addressed | Reviewer point 8 |
| Prior | salience_prior |
| Inferred parameters | w, r, v, noise, alpha |
| Observable channels | positions, rotations, neighbors, distances, angular_velocities, neighbor_fluctuations |
| Reference radii (r-free counts) | — |
| Beacon strengths | [1.0, 1.0, 1.0, 8.0] |
| Beacon spread | 50 |
| Heading update | relative bearing (rotationally invariant) |
| Agents / beacons | 49 / 4 |
| Time horizon / dt | 60s / 0.1 |
| Seed | 20260803 |

## Training and Network Configuration

| Setting | Value |
|---------|-------|
| Inference network | FlowMatching (Base, widths 256x4, time_embedding_dim 32) |
| Summary network | TransformerSummaryNet — TimeSeriesTransformer, Base (summary_dim=15) |
| Epochs | 300 |
| Batch size | 32 |
| Validation data | 300 simulations |
| Test data | 300 simulations |
| Training mode | online |
| Early stopping | off |
| Simulation budget | 300 x 50 x 32 (online, fresh each batch) |
| Wall-clock | 35.2 min |

## Convergence

![Training loss](loss.png)

The training loss curve shows the optimization objective over epochs. A healthy curve decreases smoothly and plateaus. Key warning signs: (i) a growing gap between training and validation loss indicates overfitting; (ii) loss still visibly decreasing at the final epoch means the network could benefit from more epochs; (iii) NaN spikes indicate numerical instability, often caused by extreme simulator outputs or missing standardization.

**Assessment:** Training converged without detected issues: the loss decreased and plateaued, and no divergence between training and validation loss was flagged.

## Parameter Recovery

![Parameter recovery](recovery.png)

Each panel plots the posterior median (point estimate) against the true parameter value across held-out simulations. Points falling on the diagonal indicate perfect recovery. Vertical bars represent 95% posterior credible intervals — their width reflects estimation uncertainty. Systematic deviations from the diagonal reveal bias; wide intervals indicate the data are only weakly informative for that parameter.

**Assessment:** r show recovery in the acceptable range; w, v, noise, alpha fall short.

## Calibration and Coverage

![Calibration ECDF](calibration_ecdf.png)

**Calibration ECDF** — Simulation-based calibration (SBC) plots show the empirical CDF of posterior ranks. Well-calibrated posteriors produce ECDFs close to the uniform diagonal. Lines consistently above the diagonal indicate overconfident (too narrow) posteriors; lines below indicate underconfident (too wide) posteriors.

![Coverage](coverage.png)

**Coverage** — Shows the fraction of held-out true values falling within nominal credible intervals (e.g., 50%, 80%, 95%). Well-calibrated models yield empirical coverage matching the nominal level. Under-coverage means the credible intervals are too narrow; over-coverage means they are too wide.

**Assessment:** w, r, v, noise, alpha show calibration in the acceptable range.

## Posterior Z-Score and Contraction

![Z-score and contraction](z_score_contraction.png)

The z-score–contraction plot summarizes posterior quality in two dimensions. The x-axis shows **posterior contraction** — the fraction by which the posterior variance has shrunk relative to the prior variance. Values near 0 indicate no information gain (the data are uninformative for that parameter); values near 1 indicate near-complete information gain. The y-axis shows the **posterior z-score** — the average standardized deviation between the posterior mean and the true value. Symmetric values around 0 indicate an unbiased (Gaussian-like) posterior. The ideal region is the middle-right corner (z-scores distributed around 0, high contraction).

**Assessment:** w, r, v, noise show contraction in the acceptable range; alpha fall short.

## Numerical Diagnostic Summary

| Metric | w | r | v | noise | alpha |
|--------|-----|-----|-----|-----|-----|
| NRMSE | 0.237 | 0.058 | 0.157 | 0.410 | 0.967 |
| Log-gamma | 3.053 | -0.131 | 3.602 | 4.340 | 1.791 |
| ECE | 0.027 | 0.061 | 0.014 | 0.010 | 0.015 |
| Post. Contraction | 0.935 | 0.997 | 0.976 | 0.828 | 0.094 |

**w** — excellent calibration; poor recovery; medium contraction

**r** — fair calibration; good recovery; high contraction

**v** — excellent calibration; poor recovery; high contraction

**noise** — excellent calibration; poor recovery; medium contraction

**alpha** — excellent calibration; poor recovery; low contraction

## Suggested Next Steps

1. Poor recovery for w, v, noise, alpha — increase network capacity and training duration; if no improvement, these parameters may be weakly identifiable.
2. Low contraction for alpha — the data may not be informative for these parameters; consider a more informative prior or a richer summary network.
3. Fair calibration for r — consider more training epochs or a slight capacity increase.
