# Separation at the calibrated setting

**Variant:** `v1-collision`  
**Addresses:** R1 point 1

## Why this arm exists

The reviewers' primary concern is that Vicsek carries no forces, so agents pass through one another and the trajectories are not usable as navigation. The separation term answers that directly. Behavioural evidence lives in outputs/scenarios/; these arms answer the different question of whether adding the mechanism costs anything in parameter recovery. Read against v0-reference, which differs only in that separation is off. rho and kappa are held at the scenario-calibrated values: kappa=0.6 keeps a converging 30-agent group as one connected cluster, while kappa=1.5 fragments it into six.

## Generative Configuration

| Setting | Value |
|---------|-------|
| Reviewer point addressed | R1 point 1 |
| Prior | complete_pooling_prior |
| Inferred parameters | w, r, v, noise |
| Observable channels | positions, rotations, neighbors, distances, angular_velocities, neighbor_fluctuations |
| Reference radii (r-free counts) | — |
| Beacon strengths | uniform |
| Beacon spread | 50 |
| Heading update | relative bearing (rotationally invariant) |
| Agents / beacons | 49 / 4 |
| Time horizon / dt | 60s / 0.1 |
| Seed | 20260803 |

## Training and Network Configuration

| Setting | Value |
|---------|-------|
| Inference network | FlowMatching (Base, widths 256x4, time_embedding_dim 32) |
| Summary network | TransformerSummaryNet — TimeSeriesTransformer, Base (summary_dim=12) |
| Epochs | 300 |
| Batch size | 32 |
| Validation data | 300 simulations |
| Test data | 300 simulations |
| Training mode | online |
| Early stopping | off |
| Simulation budget | 300 x 50 x 32 (online, fresh each batch) |
| Wall-clock | 35.7 min |

## Convergence

![Training loss](loss.png)

The training loss curve shows the optimization objective over epochs. A healthy curve decreases smoothly and plateaus. Key warning signs: (i) a growing gap between training and validation loss indicates overfitting; (ii) loss still visibly decreasing at the final epoch means the network could benefit from more epochs; (iii) NaN spikes indicate numerical instability, often caused by extreme simulator outputs or missing standardization.

**Assessment:** Training converged without detected issues: the loss decreased and plateaued, and no divergence between training and validation loss was flagged.

## Parameter Recovery

![Parameter recovery](recovery.png)

Each panel plots the posterior median (point estimate) against the true parameter value across held-out simulations. Points falling on the diagonal indicate perfect recovery. Vertical bars represent 95% posterior credible intervals — their width reflects estimation uncertainty. Systematic deviations from the diagonal reveal bias; wide intervals indicate the data are only weakly informative for that parameter.

**Assessment:** r show recovery in the acceptable range; w, v, noise fall short.

## Calibration and Coverage

![Calibration ECDF](calibration_ecdf.png)

**Calibration ECDF** — Simulation-based calibration (SBC) plots show the empirical CDF of posterior ranks. Well-calibrated posteriors produce ECDFs close to the uniform diagonal. Lines consistently above the diagonal indicate overconfident (too narrow) posteriors; lines below indicate underconfident (too wide) posteriors.

![Coverage](coverage.png)

**Coverage** — Shows the fraction of held-out true values falling within nominal credible intervals (e.g., 50%, 80%, 95%). Well-calibrated models yield empirical coverage matching the nominal level. Under-coverage means the credible intervals are too narrow; over-coverage means they are too wide.

**Assessment:** w, r, v, noise show calibration in the acceptable range.

## Posterior Z-Score and Contraction

![Z-score and contraction](z_score_contraction.png)

The z-score–contraction plot summarizes posterior quality in two dimensions. The x-axis shows **posterior contraction** — the fraction by which the posterior variance has shrunk relative to the prior variance. Values near 0 indicate no information gain (the data are uninformative for that parameter); values near 1 indicate near-complete information gain. The y-axis shows the **posterior z-score** — the average standardized deviation between the posterior mean and the true value. Symmetric values around 0 indicate an unbiased (Gaussian-like) posterior. The ideal region is the middle-right corner (z-scores distributed around 0, high contraction).

**Assessment:** w, r, v, noise show contraction in the acceptable range.

## Numerical Diagnostic Summary

| Metric | w | r | v | noise |
|--------|-----|-----|-----|-----|
| NRMSE | 0.222 | 0.049 | 0.161 | 0.435 |
| Log-gamma | 2.321 | 0.762 | 0.379 | 2.245 |
| ECE | 0.025 | 0.031 | 0.013 | 0.026 |
| Post. Contraction | 0.950 | 0.998 | 0.976 | 0.820 |

**w** — excellent calibration; poor recovery; medium contraction

**r** — excellent calibration; excellent recovery; high contraction

**v** — excellent calibration; poor recovery; high contraction

**noise** — excellent calibration; poor recovery; medium contraction

## Suggested Next Steps

1. Poor recovery for w, v, noise — increase network capacity and training duration; if no improvement, these parameters may be weakly identifiable.
