# Variable-radius companion experiments

Prepared October 18, 2026. These are additional analyses, not replacements for
`outputs/revision/` or historical results.

Radius is drawn once per trial from LogNormal(0, 0.5) (log-scale parameters),
then inferred jointly with the other parameters. It is constant within a trial
and shared across agents. All ordinary arms retain the same radius prior.

Inputs are positions, rotations, angular velocities, and neighbor counts at
fixed reference distances [0.5, 1, 2, 4]. These distances define observable
summaries, not the simulated sensing radius. True-radius neighbor counts,
distances, and neighbor fluctuations are excluded. The observed-switching arm
additionally receives the salience schedule.

The suite includes baseline, fixed-strength and inferred-strength collision
avoidance, static salience, hidden/observed dynamic salience, time-varying
influence, hierarchical influence and a matched complete-pooling control,
a 2x2 static/switching transfer evaluation, and Superstats trajectory inference.
The hierarchical pair preserves the previous matched diffusion/BiLSTM setup;
its earlier calibration problems remain unresolved and results require review.
The trajectory model uses its separate pinned environment and offline training.

Outputs: `outputs/variable-radius/`. Ordinary models infer radius explicitly;
Superstats treats r as time-invariant. Historical runs predicted it at every
step; new development-API runs infer it as a scalar. See the revision README
for separate trajectory/marginal plots and saved-run plotting semantics.
Smoke outputs are pipeline checks, not scientific results. Historical and
fixed-radius results are not automatically numerically comparable to this suite.

Launch/resume:

```bash
bash experiments/launch_variable_radius_gpu.sh
```

Queue status: `outputs/variable-radius/queue-status.json`.
Stage logs: `outputs/variable-radius/logs/`.
Full runs are sequential and require actual GPU execution in both environments.
Existing completed stages are skipped; Superstats partial training recovery
may need manual attention. No manuscript claims are updated automatically.
