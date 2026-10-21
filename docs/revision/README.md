# Reviewer 2 revision work

Status: 2026-10-17. Points use the review supplied in chat; point 1 is excluded.

## Working specification

Use fixed radius 2.5 room units, as in the existing fixed-radius analysis. This is a specified model constant, not a measured or known human sensing radius. The reviewer explicitly allows fixing radius. Neighbor features are computable at that specified radius. Compare separately trained models at 1.25 and 4.0 to assess design sensitivity. These values bracket the current choice; they are not empirically validated human ranges. Do not claim structural non-identifiability from a failed neural recovery experiment.

Existing variable-radius results remain historical secondary evidence. New fixed-radius analyses are stored under `outputs/revision/`, preserving original results. New runs must be compared to new matched controls, especially after the BayesFlow upgrade. Smoke runs use four agents, two seconds, and one epoch: they cannot support scientific conclusions.

## Checklist in execution order

- [x] Choose and document a working fixed-radius formulation.
- [x] Consolidate 24 historical metric files, including all extension parameters: [results](existing_results.md), [CSV](existing_metrics.csv).
- [x] Replace missing private report-helper imports; remove unsupported automatic quality grades and correct rank-ECDF interpretation.
- [x] Add configuration/package manifests to new runs. Historical reports are preserved and may contain stale estimator labels.
- [x] Include superstats trajectory inference, fixed-radius mode, numerical exports, and bounded example plotting.
- [x] Prepare a static/switching 2x2 training/testing evaluation using identical test banks across estimators.
- [x] Prepare matched hierarchical control, fixed-radius prior sensitivity, and calibration ablations.
- [x] Draft [2D/3D discussion](3d_discussion.md).
- [ ] Run full-budget fixed-radius reference and switching training, followed by transfer evaluation.
- [ ] Run full-budget hierarchical control and extension; interpret heterogeneity recovery against the control.
- [ ] Run full-budget superstats path analysis under the fixed-radius specification.
- [ ] Run fixed-radius sensitivity and matched calibration controls. Repeat promising contrasts across seeds before making strong comparative claims.
- [ ] Obtain Unity implementation/settings to verify what constitutes actual beacon switching. Current transfer study is a Python proxy, not an established Unity validation.
- [ ] Integrate selected results and draft text into the manuscript/response letter (not available locally).

## Evidence and limits by reviewer point

2. Existing fixed-radius results: w/v/eta NRMSE 0.179/0.171/0.355. Radius-independent features yield r NRMSE 0.844, contraction 0.293. Fixing r is the practical response; validate sensitivity to the constant.

3. Original final 3K CSV in `togetherflow/results/final/` contains speed ECE 0.20658. Revised reference ECE is 0.0128, but simulator, training and priors changed. New offline-3K, legacy-heading, and legacy-noise arms isolate individual contrasts around the corrected reference. They do not reconstruct the entire original experiment and do not estimate interactions. Audit metric normalization: BayesFlow's NRMSE defaults changed across versions; historical NRMSE values are not automatically comparable to new ones.

4. Existing hierarchical sigma_w contraction 0.092 is weak. The model varies individual w only, with population parameters inferred; it does not estimate individual posterior weights or heterogeneity in every parameter. New matched control shares duration, architecture family, estimator, constraints and budget. Summary dimension necessarily changes with the number of targets. Also, sigmoid(logit(mu_w)+sigma_w*z) makes mu_w the population median on the probability scale, not generally its arithmetic mean. Use that interpretation in the paper.

5. Existing superstats plot supports informative w_t recovery, approximately 0.7–0.82 contraction, but exact numerical arrays are missing. Volatility recovery is weak. Do not infer exact numbers from the plot. New fixed-radius run exports metrics and examples. A random walk permits changing influence; it does not encode a directional exploration-to-following transition or validate that mechanism in humans.

6. Six historical prior variants exist. The new fixed-radius family preserves alternative priors for w/v/eta and replaces uncertainty over r with fixed-value sensitivity. Prior predictive averages under different priors are not the same as comparing posteriors on identical observations.

7. Static nearest-beacon selection may change targets when geometry changes. Describe the proposed comparison precisely as static salience versus time-varying salience with hysteresis, not as an absolute no-switch/switch contrast. Cross-test both estimators under both generators. Test data use a separate seed and the same bank for both estimators. Unity equivalence remains unverified.

8. Discussion draft is complete as a starting point; apparatus wording and literature citations still need alignment with the manuscript.

## Runtime and commands

Main project: BayesFlow 2.0.14 (latest stable verified on PyPI), with CPU JAX by default. Install using `uv sync` or `uv pip install --python <environment-python> -e .`. For GPU execution, provision a compatible JAX CUDA runtime and working driver explicitly.

Superstats is pinned to upstream dev commit `e9142e5a6d7916e971bdab7d8dbe59452683aacf` (verified 2026-10-21), with BayesFlow 2.0.14 and NumPy >=2.5.1,<2.6. Its package version still reads 0.0.2; the commit identifies the development code. Keep a dedicated environment:

```bash
uv venv /tmp/togetherflow-superstats --python python3.12
uv pip install --python /tmp/togetherflow-superstats/bin/python -r environments/superstats/requirements.txt
PYTHONPATH=. KERAS_BACKEND=jax /tmp/togetherflow-superstats/bin/python experiments/run_time_varying.py --fixed-radius
```

Current session environments: `/tmp/togetherflow-latest-venv` (main), `/tmp/togetherflow-revision-venv` (superstats). They are temporary; recreate from the tracked specifications if removed.

```bash
python3 experiments/summarize_night.py
PYTHONPATH=. KERAS_BACKEND=jax python experiments/run_revision.py --list
PYTHONPATH=. KERAS_BACKEND=jax python experiments/run_revision.py rev-fixed-reference
PYTHONPATH=. KERAS_BACKEND=jax python experiments/run_revision.py rev-fixed-switching
PYTHONPATH=. KERAS_BACKEND=jax python experiments/run_revision.py --cross-evaluate
PYTHONPATH=. KERAS_BACKEND=jax python experiments/run_revision.py rev-fixed-partial-control
PYTHONPATH=. KERAS_BACKEND=jax python experiments/run_revision.py rev-fixed-partial
```

Append `--smoke` and set `JAX_PLATFORMS=cpu` for small validation runs. Full experiments have not been completed in this session. NVIDIA driver initialization fails with a driver/library mismatch; no driver changes have been attempted. Historical variant checkpoints are absent locally, so new full transfer evaluations require training or recovering those checkpoints.

## Validation completed in this session

- BayesFlow 2.0.14 reference: simulation, one training batch, checkpoint save/load, metrics and plots.
- BayesFlow 2.0.14 hierarchical diffusion extension: simulation, training, checkpoint, metrics and plots.
- BayesFlow 2.0.14 switching: training and all four train/test combinations completed.
- Superstats 0.0.2 / BayesFlow 2.0.12: fixed-radius trajectory simulation, training, posterior sampling, per-step CSV/NPZ and three example trajectories completed.
- Main `.venv` upgraded to BayesFlow 2.0.14; temporary superstats environment remains compatible with its released pins.

The original metrics and paper figures were not overwritten. Smoke artifacts are retained locally but ignored by Git to avoid treating them as scientific results.

## GPU workaround and full-analysis queue (2026-10-17)

The loaded kernel module is 580.173.02, while the installed system libraries and on-disk module are 580.178.04. The user requested no reboot. The official NVIDIA 580.173.02 bundle was downloaded and integrity-checked by its extractor, without installing it; matching compute libraries are isolated under `/tmp/togetherflow-nvidia-580.173.02/lib`. A process-local `LD_LIBRARY_PATH` restores NVML access to the RTX 5000 Ada (32 GB). System libraries, modules, and desktop sessions are unchanged. This is temporary, not a system-wide driver repair; a later reboot should load the installed module. The launcher stops using the private libraries once that old module is no longer loaded.

GPU installations: main `.venv` with BayesFlow 2.0.14; persistent `.venv-superstats` now uses the pinned Superstats development commit/BayesFlow 2.0.14; both use JAX CUDA 13. To preserve the main GPU packages when syncing, use `uv sync --extra cuda`. To recreate the trajectory environment, install its requirements plus `jax[cuda13]==0.11.2`.

Launch/resume all 17 full-budget stages using:

```bash
bash experiments/launch_revision_gpu.sh
```

The queue validates actual GPU matrix multiplication in both environments, runs one analysis at a time, and records `outputs/revision/queue-status.json` plus per-stage logs in `outputs/revision/logs/`. Completed stages are skipped. A checkpoint with completed training history can resume diagnostics for ordinary variants. Failed stages remain explicit; failed reference/switching training blocks the transfer calculation. Interrupted superstats runs may require manual checkpoint recovery. No smoke results are counted as full analyses.

The queue was launched in tmux session `togetherflow-revision` on October 17, 2026. Initial startup required enabling both `cuda,cpu` JAX platforms because BayesFlow uses CPU callbacks; GPU remains the asserted default. Early startup failures are retained in append-only logs, so consult the latest queue start and `queue-status.json` for current state.

## Separate posterior plots (2026-10-21)

Only `w_t` appears in `time_varying_posterior.png`. Constant parameters appear
in `time_invariant_posterior_1.png` through `_3.png`, one held-out dataset per
figure, using Superstats `plot_marginals`. Fixed-radius runs include v, eta,
and tau; variable-radius runs also include r. The weight plot uses pointwise
95%/65% equal-tailed intervals, without temporal smoothing.

Regenerate the figures without training:

```bash
JAX_PLATFORMS=cpu .venv-superstats/bin/python experiments/plot_path_posteriors.py outputs/revision/superstats-fixed-path outputs/variable-radius/superstats-variable-path
```

The existing samples were trained with release 0.0.2 and predicted constants
at every step. Superstats pools those stepwise marginals within each dataset
and subsamples to the original draw count for the constant-parameter plots.
These are mixtures of stepwise marginals, not newly inferred scalar posteriors.
Original training manifests, checkpoints, samples, and metrics remain unchanged;
`posterior_plot_manifest.json` records the plotting source separately.

Future runs use the development API's separate varying/invariant networks:
only w is a sequence target, and the other parameters are scalar targets.
Scalar draws are broadcast across steps only for the existing export/metric
format. This architecture change requires new training, not loading old
checkpoints. Use `--output-dir` to keep new runs separate from existing results.

See [the matched retraining design](matched_retraining_2026-10-21.md) for the new eight-arm suite, including full-epoch offline budgets without early stopping.
