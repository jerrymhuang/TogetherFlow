# Matched retraining, 2026-10-21

The reference is `variable_radius_variants.BASE`, the four-parameter
(w, r, v, eta) reference from `outputs/variable-radius/rev-variable-reference`.
The new transformer/flow-matching arm retrains that reference in the same suite.
Original research outputs are preserved.

## Shared physical specification

- 49 agents, 4 beacons, 60 seconds, dt=0.1, beacon spread=50.
- Relative-heading steering and diffusive heading noise.
- Observables: positions, rotations, angular velocities, and counts at fixed
  reference radii [0.5, 1, 2, 4]. No true-radius neighborhood features.
- Baseline priors: w ~ Beta(2,2), r ~ LogNormal(0,0.5),
  v ~ 2 Beta(2,2), eta ~ Beta(2,2).
- Parameter support constraints are applied before inference and inverted for
  saved posterior draws, including the Superstats runs.
- Seed 20260803; independent validation/test seed offsets +100000/+200000.
- 300 validation and 300 test datasets, 1000 posterior draws per test dataset.
- Scalar diagnostics use BayesFlow 2.0.14, prior-normalized NRMSE and its default
  aggregation. Raw posteriors, truths, histories and selected models are saved.

## Eight arms

| Arms | Training | Epochs | Selection |
|---|---|---|---|
| Transformer × flow matching/diffusion | Online, 50 batches × 32 simulations per epoch | 300 | Final weights, no early stopping |
| BiLSTM × flow matching/diffusion | Identical online budget and simulator seed | 300 | Final weights, no early stopping |
| Offline 3,000 | Fixed bank, exact prefix of 30K bank | 300 | Final weights, **no early stopping** |
| Offline 30,000 | Fixed bank | 300 | Final weights, **no early stopping** |
| Superstats fixed radius | 30K fixed bank; r=2.5 known | Up to 300 | Best validation loss, patience 20 |
| Superstats variable radius | 30K fixed bank; r inferred | Up to 300 | Best validation loss, patience 20 |

The 2×2 factors are summary network (transformer/BiLSTM) and inference network
(flow matching/diffusion). The subnet widths and time embedding are matched,
with summary dimension 12. The offline arms use the transformer/flow-matching
reference architecture. All six static-model arms use the **same saved validation
and test banks**. The online reference uses 480,000 fresh training simulations;
the offline budgets reuse their banks and receive different numbers of gradient
updates at the same epoch count. This is not an equal-simulation-budget or
equal-update-count comparison of online and offline training. Optimizer defaults
also differ: online Adam versus offline AdamW/cosine scheduling.

## Superstats comparison limits

Superstats is pinned to development commit
e9142e5a6d7916e971bdab7d8dbe59452683aacf. It uses recurrent embeddings and separate
coupling networks for trajectory and scalar targets, in marginal smoothing mode.
Only w varies with time, following a logit random walk with
constant tau ~ 0.4 Beta(1,3). Shared physical settings, priors, observable inputs,
support constraints and evaluation sizes match the baseline, but this is a
dynamic-model extension, not another cell of the static network comparison.
Its generated datasets differ from the static model's shared test bank.

The fixed-radius extension conditions on a known r=2.5; it is not a matched
four-unknown-parameter comparison. The variable-radius extension includes five
targets: the weight trajectory plus r, v, eta and tau. Trajectory diagnostics
remain in separate per-step exports. `invariant_metrics.csv` uses the same scalar
metric definitions as the baseline, but parameters and data-generating models
must still be identified when comparing results. Posterior figures separate
weight trajectories from constant-parameter marginals.

## Execution and monitoring

```bash
python3 experiments/run_retraining_queue.py
```

Outputs are under `outputs/matched-retraining-2026-10-21/`:
`STATUS.md`, `queue-status.json`, per-arm logs, and `comparison.csv`.
GPU execution is serialized. Both environments must pass actual GPU matrix
multiplication. The existing process-local NVIDIA library workaround is applied
only while its corresponding old kernel module remains loaded.

Completed runs are skipped. Static posterior batches are persisted individually;
completed training can resume diagnostics. Incomplete static training requires
an explicit restart rather than silently changing its epoch budget. Superstats
can resume diagnostics from saved selected weights, but interrupted training
should use a new output directory. Changed manifests are rejected.

Validation uses small runs in `/tmp/togetherflow-retraining-validation` plus a
full-trajectory Superstats GPU check; these are not scientific results.
