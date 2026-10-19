# Partial pooling of individual agent weights

This controlled synthetic example estimates each agent's stable influence
weight together with the population distribution. It keeps the coupled group
trajectory as the source of evidence and does not treat agents as independent
datasets.

## Model and interpretation

For each group:

\[
\mu_w\sim\operatorname{Beta}(2,2),\quad
\sigma_w\sim1.5\operatorname{Beta}(1,3),\quad
z_i\sim\mathcal N(0,1),\quad
w_i=\operatorname{sigmoid}(\operatorname{logit}(\mu_w)+\sigma_w z_i).
\]

The simulator uses each realized w_i for that agent throughout the trial.
The joint inference targets are mu_w, sigma_w, shared speed v, shared heading
noise eta, and all eight w_i. Individual posteriors therefore reflect both the
agent's observed motion and the shared population model: this is partial
pooling. The local weights are retained in joint posterior draws, so uncertainty
in contrasts can be computed without pretending their posteriors are independent.

**Weight direction matters:** in this simulator w_i weights beacon attraction;
1-w_i weights alignment with neighboring agents. A higher w_i does not mean
stronger social alignment. mu_w is the population **median** weight, and sigma_w
is dispersion on the logit scale, not the standard deviation of bounded weights.

The complete-pooling control sets sigma_w=0, infers one shared w plus v and eta,
and assigns each sampled shared w to every agent. Both arms use the same prior
for the common parameters, simulator settings, observable features, network
family and hidden widths, training budget, and held-out observations. Their
output dimensions necessarily differ.

## Controlled observation design

Eight agents move for 20 seconds, sampled at dt=0.1. Radius is fixed at 2.5;
four known beacons are placed at (-3,-4), (-3,4), (3,-4), and (3,4) in the
8-by-10 room. Speed and heading noise remain unknown and are inferred. The
relative-heading and diffusive-noise dynamics match the corrected simulator.

For each agent, features describe how its observed turns respond to the known
beacon direction and neighboring agents' observed headings. Neighbor relations
use the specified fixed radius. The implementation follows the simulator's
arithmetic-angle alignment rule, including wrapping, rather than substituting
a circular mean. It extracts turning-regression moments, noise and motion
summaries, and neighbor counts. All agents' features jointly condition the
posterior; no true weights or population parameters enter the inputs.

Turning summaries exclude steps starting within 0.21 room units of a wall.
Given the prior's maximum speed of 2 and dt=0.1, retained steps cannot reflect
at a wall. These features are not asserted to be sufficient statistics or an
exact transition likelihood. Amortized inference is trained on their simulated
distribution.

This design changes the beacon layout, sampling frequency, summary features,
estimator and agent count from the first compositional pilot. It is a new
pedagogical example, not a matched demonstration that any one change improved
that pilot. It assumes exact observed trajectories and known geometry; human
measurement noise and unknown sensing radii are outside its scope.

## Inference and checks

The example uses BayesFlow BasicWorkflow with FlowMatching, targeting the joint
population-and-agent posterior. Partial pooling is defined by the hierarchical
model and joint targets, not by choosing CompositionalWorkflow. No across-agent
composition is performed.

The completed comparison used two independent seeds (831, 832), each with 4,096 training and 512 validation
groups per arm, 100 training epochs and 256 posterior samples. Separate seed
ranges supply 128 partial-pooling and 128 complete-pooling test groups per seed.
Both fitted estimators see exactly the same test observations for each generator.
Evaluation on the other generator is an intentional model-mismatch check; it
is not simulation-based calibration under the fitted model.

Exports include:

- Individual weight RMSE and marginal 90% interval coverage.
- Within-group centered RMSE and correlation, separating recovery of individual
  differences from recovery of the group mean.
- A zero-difference baseline for centered RMSE, and complete-pooling predictions.
- Joint-draw centered intervals and their empirical coverage.
- Population spread recovery, per-agent posterior ranks and intervals.
- A plot of the first two held-out heterogeneous groups, selected by index,
  comparing partial and complete pooling with the same simulated truth.

Coverage summaries and ranks are diagnostics, not certification of calibration.
Agents in one group are dependent; 128 groups are not 1,024 independent datasets.
The fixed-size flattened feature encoder is not guaranteed permutation-equivariant,
though the feature extraction itself is checked for correct agent permutation.

## Run

```bash
PYTHONPATH=. .venv/bin/python experiments/run_agent_partial_pooling.py \
  --out outputs/agent-partial-pooling/new-example
```

The output directory must be new, protecting existing results. Defaults now run
on CPU with one seed (831). The earlier two-seed comparison is retained below. Use `--help` to change sample sizes, seeds, or the number of agents; changes
require retraining. Main results are under
`outputs/agent-partial-pooling/example/`: `comparison.csv`, `manifest.json`,
`pooling_comparison.png`, and per-seed/per-estimator evaluation directories.
Models and posterior arrays are saved locally under existing Git ignore rules.
The separate `smoke` output is only an execution check.

```bash
PYTHONPATH=. NUMBA_NUM_THREADS=4 .venv/bin/python -m unittest discover \
  -s tests -p 'test_agent*.py'
```

Tests cover latent export without changed observable trajectories, collapse at
zero population spread, recovery of a known noiseless turning coefficient,
agent permutation in features, and matched feature shapes across pooling arms.

The partial-pooling prior is continuous on positive sigma_w and assigns no point
mass to sigma_w=0. Its posterior cannot establish exact equality of agents or
provide a posterior probability of the complete-pooling model. The homogeneous
test bank checks behavior at that boundary; formal model selection would require
a separate model-comparison design or a prior with a point mass at zero.

## Completed example: October 18, 2026

Both seeds completed all four train/test combinations on CPU.
For heterogeneous held-out groups:

| Metric | Seed 831 | Seed 832 |
| --- | ---: | ---: |
| Partial-pooling individual RMSE | 0.0482 | 0.0521 |
| Complete-pooling individual RMSE | 0.0913 | 0.0879 |
| Partial-pooling within-group RMSE | 0.0430 | 0.0458 |
| Zero-difference within-group RMSE | 0.0885 | 0.0837 |
| Partial-pooling within-group correlation | 0.874 | 0.837 |
| Partial-pooling individual 90% coverage | 85.4% | 82.8% |
| Partial-pooling centered 90% coverage | 87.3% | 84.8% |

The example now recovers individual differences beyond group-level influence,
and partial pooling improves point estimates over the matched complete-pooling
control on these heterogeneous groups. This is a controlled simulation result,
not evidence of empirical validity in people.

The tradeoff is visible on homogeneous groups: complete pooling has individual
RMSE 0.0261/0.0230, versus partial pooling's 0.0410/0.0375. Partial pooling can
still infer spurious differences there (centered RMSE 0.0312/0.0295), so it should
not be advertised as automatically detecting the absence of heterogeneity.

**Calibration remains unfinished.** Partial-pooling individual intervals
undercover on heterogeneous groups; the population-spread interval coverage is
also imperfect (see per-parameter exports). Even complete pooling under its own
generator has only 65.6%/81.3% coverage for nominal 90% individual intervals.
Thus neither estimator is ready for calibrated scientific uncertainty claims.
The complete-pooling model's much lower coverage on heterogeneous groups is
additionally affected by deliberate model misspecification. More training,
architecture and calibration work, plus more seeds, are needed before use in
manuscript claims or human inference.

The first two heterogeneous held-out groups are plotted in
[the comparison figure](../../outputs/agent-partial-pooling/example/pooling_comparison.png).
The full table is [comparison.csv](../../outputs/agent-partial-pooling/example/comparison.csv).
Six agent-related regression tests passed, including the three checks added for
this example. Syntax and whitespace checks passed. Earlier experiments and the
manuscript discussion were not modified by this example.

### Single-seed BayesFlow parameter recovery

[Recovery plot](../../outputs/agent-partial-pooling/example/parameter_recovery.png)
([PDF](../../outputs/agent-partial-pooling/example/parameter_recovery.pdf)) uses
`bf.diagnostics.plots.recovery` on seed **831 only**: 128 held-out heterogeneous
groups and 256 posterior draws per group. The 12 panels show mu_w, v, eta,
sigma_w, and the eight individual weights. BayesFlow plots posterior medians
against ground truth with 95% credible intervals and correlation annotations.
These intervals differ from the 90% intervals in the earlier evaluation tables.
Radius is fixed at 2.5 and is therefore not an inference target or panel.

Figure labels use weights 1–8; CSV agent indices remain 0–7. Each weight panel
represents an agent slot across simulated groups, not repeated measurements of
one persistent person. The runner now defaults to one seed and generates this
BayesFlow diagnostic from the first requested seed. The earlier custom two-seed
figure remains a historical artifact; it is no longer generated by the runner.
`parameter_recovery.json` records the data source and plotting settings.
