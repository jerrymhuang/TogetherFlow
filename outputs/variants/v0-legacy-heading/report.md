# Reference model with the legacy (absolute-bearing) heading update

**Variant:** `v0-legacy-heading`  
**Addresses:** Navigation realism (Part 1)

## Why this arm exists

The published heading update applies absolute world-frame bearings directly as a turn rate, so the dynamics are not rotationally invariant: an agent whose beacon lies due east receives no steering, and an agent facing away from a due-east beacon never turns around. This arm keeps that behaviour while every other arm uses the corrected relative-bearing update, quantifying what the correction changes — which is the quantitative part of the answer to the reviewers' concern about unrealistic collective motion.

## Generative Configuration

| Setting | Value |
|---------|-------|
| Reviewer point addressed | Navigation realism (Part 1) |
| Prior | complete_pooling_prior |
| Inferred parameters | w, r, v, noise |
| Observable channels | positions, rotations, neighbors, distances, angular_velocities, neighbor_fluctuations |
| Reference radii (r-free counts) | — |
| Beacon strengths | uniform |
| Beacon spread | 50 |
| Heading update | absolute bearing (legacy; NOT rotationally invariant) |
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
| Wall-clock | 32.5 min |

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

**Assessment:** w, v, noise show calibration in the acceptable range; r fall short.

## Posterior Z-Score and Contraction

![Z-score and contraction](z_score_contraction.png)

The z-score–contraction plot summarizes posterior quality in two dimensions. The x-axis shows **posterior contraction** — the fraction by which the posterior variance has shrunk relative to the prior variance. Values near 0 indicate no information gain (the data are uninformative for that parameter); values near 1 indicate near-complete information gain. The y-axis shows the **posterior z-score** — the average standardized deviation between the posterior mean and the true value. Symmetric values around 0 indicate an unbiased (Gaussian-like) posterior. The ideal region is the middle-right corner (z-scores distributed around 0, high contraction).

**Assessment:** v show contraction in the acceptable range; w, r, noise fall short.

## Numerical Diagnostic Summary

| Metric | w | r | v | noise |
|--------|-----|-----|-----|-----|
| NRMSE | 1.014 | 0.029 | 0.128 | 0.972 |
| Log-gamma | 3.819 | -11.221 | 1.715 | 1.481 |
| ECE | 0.008 | 0.129 | 0.011 | 0.023 |
| Post. Contraction | 0.003 | 0.999 | 0.982 | 0.103 |

**w** — excellent calibration; poor recovery; low contraction

**r** — poor calibration; excellent recovery; poor — overconfident

**v** — excellent calibration; fair recovery; high contraction

**noise** — excellent calibration; poor recovery; low contraction

## Suggested Next Steps

1. Poor calibration for r — increase summary network capacity or train for more epochs.
2. Poor recovery for w, noise — increase network capacity and training duration; if no improvement, these parameters may be weakly identifiable.
3. Overconfident posteriors for r — inspect the simulator for potential issues and consider increasing the simulation budget.
4. Low contraction for w, noise — the data may not be informative for these parameters; consider a more informative prior or a richer summary network.
