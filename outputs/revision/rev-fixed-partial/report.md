# Per-agent influence weight, population-level inference

**Variant:** `rev-fixed-partial`  
**Addresses:** R2 point 4

## Why this arm exists

Every published configuration gives all 49 agents one parameter vector, which the reviewers single out as the model's least defensible assumption about people. Here each agent draws its own weight from a population distribution, w_i = sigmoid(logit(mu_w) + sigma_w z_i), and the inference targets are the population parameters. sigma_w is the quantity that carries the argument: if it is recoverable the model detects heterogeneity, and sigma_w = 0 is exactly complete pooling, so the published specification becomes a restriction the data can test rather than an assumption it inherits.

Two configuration choices are forced by what comes next rather than by this arm. The inference network is a DiffusionModel and the adapter does not constrain, because CompositionalWorkflow accepts only the former and refuses a non-zero log-det Jacobian; a checkpoint trained any other way cannot be composed across trials afterwards. Read this arm against v0-diffusion rather than v0-reference, since the estimator differs from the flow-matching arms.

## Generative Configuration

| Setting | Value |
|---------|-------|
| Reviewer point addressed | R2 point 4 |
| Prior | partial_pooling_prior_fixed_r_2.5 |
| Inferred parameters | mu_w, v, noise, sigma_w |
| Observable channels | positions, rotations, neighbors, distances, angular_velocities, neighbor_fluctuations |
| Reference radii (r-free counts) | — |
| Beacon strengths | uniform |
| Beacon spread | 50 |
| Beacon salience | static |
| Heading update | relative bearing (rotationally invariant) |
| Agents / beacons | 49 / 4 |
| Time horizon / dt | 20s / 0.1 |
| Seed | 20260803 |

## Training and Network Configuration

| Setting | Value |
|---------|-------|
| Inference network | diffusion (widths 256x4, time_embedding_dim 32) |
| Summary network | SummaryNet — Conv1D + BiLSTM (summary_dim=12) |
| Epochs | 300 |
| Batch size | 32 |
| Validation data | 300 simulations |
| Test data | 300 simulations |
| Training mode | online |
| Early stopping | off |
| Simulation budget | 300 x 50 x 32 (online, fresh each batch) |
| Wall-clock | 11.4 min |

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

| Metric | mu_w | v | noise | sigma_w |
|--------|-----|-----|-----|-----|
| NRMSE | 0.811 | 1.668 | 1.103 | 0.901 |
| Log-gamma | -106.679 | -inf | -231.627 | -4.327 |
| ECE | 0.243 | 0.477 | 0.228 | 0.087 |
| Post. Contraction | 0.748 | 0.892 | 0.419 | 0.222 |

**mu_w** — NRMSE=0.811; Log-gamma=-106.679; ECE=0.243; Post. Contraction=0.748

**v** — NRMSE=1.668; Log-gamma=-inf; ECE=0.477; Post. Contraction=0.892

**noise** — NRMSE=1.103; Log-gamma=-231.627; ECE=0.228; Post. Contraction=0.419

**sigma_w** — NRMSE=0.901; Log-gamma=-4.327; ECE=0.087; Post. Contraction=0.222

## Suggested Next Steps

1. Inspect recovery, coverage, and rank plots jointly; no automatic threshold establishes identifiability or convergence.
