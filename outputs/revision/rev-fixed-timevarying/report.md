# Non-stationary influence weight (logit random walk)

**Variant:** `rev-fixed-timevarying`  
**Addresses:** R2 point 5

## Why this arm exists

The published model fixes w for the duration of a trial, asserting that an agent's balance between beacon-driven and neighbour-driven motion never shifts. This arm replaces the parameter with a parameter path, logit w_t = logit w_{t-1} + tau sqrt(dt) xi_t, and infers (w0, tau) instead of a single w. The point is that tau = 0 recovers the stationary model exactly, so stationarity stops being an assumption and becomes a restriction the data can reject — or fail to reject, which is equally reportable. Read the contraction on tau first: if the data cannot distinguish a walk from a constant, that is the honest answer to the reviewer, and it is a statement about identifiability rather than about human behaviour.

## Generative Configuration

| Setting | Value |
|---------|-------|
| Reviewer point addressed | R2 point 5 |
| Prior | prior_nonstationary_fixed_r_2.5 |
| Inferred parameters | w0, v, noise, tau |
| Observable channels | positions, rotations, neighbors, distances, angular_velocities, neighbor_fluctuations |
| Reference radii (r-free counts) | — |
| Beacon strengths | uniform |
| Beacon spread | 50 |
| Beacon salience | static |
| Heading update | relative bearing (rotationally invariant) |
| Agents / beacons | 49 / 4 |
| Time horizon / dt | 60s / 0.1 |
| Seed | 20260803 |

## Training and Network Configuration

| Setting | Value |
|---------|-------|
| Inference network | flow_matching (widths 256x4, time_embedding_dim 32) |
| Summary network | TransformerSummaryNet — TimeSeriesTransformer, Base (summary_dim=12) |
| Epochs | 300 |
| Batch size | 32 |
| Validation data | 300 simulations |
| Test data | 300 simulations |
| Training mode | online |
| Early stopping | off |
| Simulation budget | 300 x 50 x 32 (online, fresh each batch) |
| Wall-clock | 29.8 min |

## Convergence

![Training loss](loss.png)

The training loss curve shows the optimization objective over epochs. A healthy curve decreases smoothly and plateaus. Key warning signs: (i) a growing gap between training and validation loss indicates overfitting; (ii) loss still visibly decreasing at the final epoch means the network could benefit from more epochs; (iii) NaN spikes indicate numerical instability, often caused by extreme simulator outputs or missing standardization.

**Assessment:** No non-finite losses detected. This check does not establish convergence; inspect the loss curves.

## Parameter Recovery

![Parameter recovery](recovery.png)

Each panel plots the posterior median (point estimate) against the true parameter value across held-out simulations. Points falling on the diagonal indicate perfect recovery. Vertical bars represent 95% posterior credible intervals — their width reflects estimation uncertainty. Systematic deviations from the diagonal reveal bias; wide intervals indicate the data are only weakly informative for that parameter.

**Assessment:** No per-parameter ratings were available.

## Calibration and Coverage

![Calibration ECDF](calibration_ecdf.png)

**Calibration ECDF** — Simulation-based calibration (SBC) plots show the empirical CDF of posterior ranks. Well-calibrated posteriors produce ECDFs close to the uniform diagonal. Systematic deviations indicate non-uniform ranks; their shape can reflect bias or dispersion errors. Use the coverage plot to assess interval width rather than assigning dispersion from the sign alone.

![Coverage](coverage.png)

**Coverage** — Shows the fraction of held-out true values falling within nominal credible intervals (e.g., 50%, 80%, 95%). Well-calibrated models yield empirical coverage matching the nominal level. Under-coverage means the credible intervals are too narrow; over-coverage means they are too wide.

**Assessment:** No per-parameter ratings were available.

## Posterior Z-Score and Contraction

![Z-score and contraction](z_score_contraction.png)

The z-score–contraction plot summarizes posterior quality in two dimensions. The x-axis shows **posterior contraction** — the fraction by which the posterior variance has shrunk relative to the prior variance. Values near 0 indicate no information gain (the data are uninformative for that parameter); values near 1 indicate near-complete information gain. The y-axis shows the **posterior z-score** — the average standardized deviation between the posterior mean and the true value. Symmetric values around 0 indicate an unbiased (Gaussian-like) posterior. The ideal region is the middle-right corner (z-scores distributed around 0, high contraction).

**Assessment:** No per-parameter ratings were available.

## Numerical Diagnostic Summary

| Metric | w0 | v | noise | tau |
|--------|-----|-----|-----|-----|
| NRMSE | 0.382 | 0.232 | 0.470 | 0.880 |
| Log-gamma | -0.596 | 3.748 | 3.898 | 3.034 |
| ECE | 0.012 | 0.023 | 0.023 | 0.020 |
| Post. Contraction | 0.870 | 0.944 | 0.788 | 0.185 |

**w0** — NRMSE=0.382; Log-gamma=-0.596; ECE=0.012; Post. Contraction=0.870

**v** — NRMSE=0.232; Log-gamma=3.748; ECE=0.023; Post. Contraction=0.944

**noise** — NRMSE=0.470; Log-gamma=3.898; ECE=0.023; Post. Contraction=0.788

**tau** — NRMSE=0.880; Log-gamma=3.034; ECE=0.020; Post. Contraction=0.185

## Suggested Next Steps

1. Inspect recovery, coverage, and rank plots jointly; no automatic threshold establishes identifiability or convergence.
