# Main-paper regeneration: 3K and 30K offline training

The requested research jobs were started on October 20, 2026, on the RTX 5070 Ti Laptop GPU. See `console.log` for live progress. This directory preserves the original main-paper results elsewhere in the repository.

Both training runs and all held-out diagnostics are complete. The 3K run stopped after 43 epochs and selected epoch 23; the 30K run stopped after 55 epochs and selected epoch 35. The original 30K posterior sampling pass stopped advancing after 150 of 300 cases; it was restarted from the saved checkpoint using `experiments/resume_main_paper_sampling.py`, which saves completed posterior batches and their random-generator states. This recovery preserves the original sampler, batch size, and seed sequence; a small equivalence and cache-resume check passed. All 300 test cases were subsequently completed. See `offline-30000/status.json` and `offline-30000/sampling_recovery.json` for the completion status and provenance.

Both runs infer `(w, r, v, noise)` under the revised variable-radius reference (`rev-variable-reference`): 49 agents, 60 seconds, relative-heading steering, diffusive heading noise, and observable trajectories plus neighbor counts at fixed reference distances. Inputs exclude statistics evaluated at the unknown true radius. The transformer summary and flow-matching posterior architectures match the online extension reference.

- Training: fixed banks of 3,000 and 30,000 simulations. The smaller bank is an exact subset of the larger bank.
- Validation and test: separate banks of 300 simulations each, shared between runs, with distinct seed ranges from training.
- Optimization: batch size 32, maximum 300 epochs, validation-loss early stopping with patience 20 and restoration of the best weights. BayesFlow's offline default uses AdamW; the online reference uses Adam.
- Evaluation: 1,000 posterior draws per test simulation. Plots and metrics use the same saved draws. Primary NRMSE uses the prior-only prediction baseline, with an identical empirical-prior bootstrap seed for both budgets; contraction and NRMSE use median aggregation across test simulations. Range-normalized scores are saved separately and retain median aggregation.
- One training seed; no repeated-seed uncertainty estimate. The online reference consumes 480,000 fresh simulations, so these are not equal-budget offline-versus-online comparisons.

Each run writes `offline-3000/` or `offline-30000/`, with a manifest, per-epoch status/history, validation-selected checkpoint, posterior draws with paired truths, recovery/coverage/calibration/contraction figures, and metrics. `training_complete.json` marks completion of training; `complete.json` marks completion of held-out diagnostics. Interim epoch losses are not final recovery results.

The isolated environment is `/tmp/togetherflow-main-paper-venv`; installed versions are recorded in `environment.txt`. The reference versions of BayesFlow (2.0.14), JAX (0.11.2), Keras (3.15.1), and NumPy (2.5.4) are preserved. The original project `.venv` is unchanged.

The runner is `experiments/run_main_paper_offline.py`. Bank generation resumes after interruption. Completed training is reused for diagnostics; an interrupted training run requires `--restart-training` to explicitly restart it. Keep this output directory's source code and configuration unchanged when resuming; manifests reject mismatches.

`experiments/build_main_paper_results.py` assembled the table, comparison figures, revised LaTeX simulation-study fragment, PDF preview, and ZIP in `outputs/docs/main_paper_revision/` and `outputs/docs/main_paper_revision.zip`. The larger training budget reduces NRMSE and increases contraction for all four parameters, but has higher ECE and marginal undercoverage. The fragment discusses both findings. The original `outputs/docs/sim_study.tex` remains available.

Tiny runs in `smoke/` and `smoke-full-shape/` only validate the pipeline and GPU capacity. They are excluded from manuscript assembly.
