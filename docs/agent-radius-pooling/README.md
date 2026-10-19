# Twelve-agent partial pooling of weights and sensing radii

This extension gives each of 12 agents its own stable influence weight and
sensing radius. Both are inferred jointly with six population/shared parameters
from the complete interacting group's observed motion. It preserves the earlier
fixed-radius examples and outputs.

## Hierarchy

For each group, draw:

- mu_w ~ Beta(2,2), sigma_w ~ 1.5 Beta(1,3).
- mu_r ~ LogNormal(log(2.5), 0.35), sigma_r ~ 0.75 Beta(2,2).
- Shared speed v ~ 2 Beta(2,2), heading noise eta ~ Beta(2,2).

Then draw, independently conditional on these globals:

\[
\operatorname{logit}(w_i)\sim\mathcal N(\operatorname{logit}(\mu_w),\sigma_w^2),
\qquad \log r_i\sim\mathcal N(\log\mu_r,\sigma_r^2).
\]

Both local parameters are constant within a 20-second trial. mu_w and mu_r
are population **medians** on their respective physical scales. sigma_w is a
logit-scale standard deviation and sigma_r a log-radius standard deviation.
The radius population prior is centered around the earlier fixed radius 2.5;
it is a new prior specification, not the earlier shared-radius LogNormal(0,0.5)
experiment. There is no inferred population correlation between weight and
radius in this first extension, although posterior dependencies are retained.

Each receiver i uses its own r_i for selecting neighbors, so i can sense j even
when j cannot sense i. w_i is the weight on beacon attraction, and 1-w_i the
weight on neighbor alignment. It must not be interpreted as an increasing
measure of social alignment.

## Observations and inference

The controlled beacon layout, 8-by-10 room, dt=0.1 and corrected relative-heading,
diffusive-noise dynamics follow the preceding individual-weight example.
Four beacons are at (-3,-4), (-3,4), (3,-4), (3,4).

All features are computed from observed positions/headings and the known layout.
Turning summaries are evaluated at fixed reference radii
[0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 8, 12]. These are feature
definitions, **not the simulated latent radii**. No true-radius neighbor counts,
latent weights, latent radii, or population parameters enter the inputs.
The reference grid is not a discrete prior: inferred radii remain continuous
and positive. Steps starting within 0.21 units of a wall are excluded from the
turning summaries to avoid reflections given maximum speed 2 and dt=0.1.

The posterior is joint over 30 parameters, conditioned on features from all
12 agents. BayesFlow BasicWorkflow with FlowMatching uses three 256-unit hidden
layers. Targets are transformed to unbounded coordinates (log for radius centers
and individual radii, scaled logit for bounded parameters); diagnostics are on
the original physical scales. No agent-wise conditional independence or
compositional sampling assumption is introduced.

This is a controlled, exact-observation simulation example. Different latent
parameters can produce similar motion, especially when w_i is near 1 (little
neighbor alignment) or radii include almost all agents. Hierarchical shrinkage
and a convincing recovery scatterplot alone do not establish individual-radius
identifiability. The unknown radius makes this harder than the preceding example.

## Evaluation budget and requested layout

One seed: 831. Defaults use 5,000 training and 300 validation simulations, 300
epochs maximum, batch size 32, and CPU execution. The minimum-validation-loss
checkpoint is restored, with early-stopping patience 25; the actual and selected
epoch counts are recorded in the manifest. Training, validation, and test seeds
occupy separate ranges.

Recovery uses **300 newly simulated held-out groups and 1,000 posterior draws
per group**, matching `Variant.n_test=300` in `experiments/variants.py` and the
BayesFlow `BasicWorkflow` diagnostic default used by `experiments/run_variant.py`.
These are the numbers of independent test simulations and posterior draws,
respectively; matching those counts does not make architectures or training
protocols identical to the other suites.

`bf.diagnostics.plots.recovery` generates a **5-row by 6-column** figure:

1. mu_w, sigma_w, mu_r, sigma_r, v, eta.
2. w_1 through w_6.
3. w_7 through w_12.
4. r_1 through r_6.
5. r_7 through r_12.

Global data use #132a70, individual weights #6969ff and individual radii #4e2a84 (RGB 78, 42, 132; regenerated for `bounded-radius/` only). There is no overall
figure title. Panels show posterior medians and 95% credible intervals. The
agent labels refer to ordered agent slots across different simulated groups,
not repeated observations of persistent people.

The evaluation also exports per-parameter RMSE, prior-median baseline error,
95% coverage and contraction; within-group centered recovery for weights and
radii; and within-agent posterior weight/radius correlations. Centering separates
recovery of individual differences from merely learning the group center.
Coverage is empirical and evaluated at the group level; the 12 agents within
each group are not independent replications.

## Reproduce

```bash
PYTHONPATH=. .venv/bin/python experiments/run_agent_radius_pooling.py \
  --out outputs/agent-radius-pooling/new-example
```

The script now defaults to the bounded radius prior (`--radius-max 10`, see
below); `--radius-max inf` reproduces the unbounded lognormal hierarchy above.
Output directories must not already exist. The unbounded full run is under
`outputs/agent-radius-pooling/validation-selected/`; the bounded one under
`outputs/agent-radius-pooling/bounded-radius/`. `parameter_recovery.png` and
`parameter_recovery.pdf` are the requested diagnostics. `manifest.json` records
the settings and completion status; `metrics.csv`, `summary.json`, and
`agent_posteriors.csv` contain the numeric evaluation. Checkpoints and full
posterior/evaluation arrays are retained locally under existing Git ignore rules.
The separate `smoke/` directory is only a pipeline check.

```bash
PYTHONPATH=. NUMBA_NUM_THREADS=4 .venv/bin/python -m unittest discover \
  -s tests -p 'test_agent*.py'
```

Tests verify asymmetric receiver neighborhoods, equivalence to a shared radius
when the radius vector is constant, zero-spread collapse, latent export without
changes to observations, reproducibility, feature permutation, and round-trip
transforms for all 30 parameters, and that a finite radius bound keeps every
radius inside (0, R) while preserving its median. The earlier individual-weight
checks also run.


## Training and validation notes

The initial final-epoch run in `outputs/agent-radius-pooling/example/` was stopped
at epoch 194 after validation loss rose while training loss fell. It did not
reach held-out posterior evaluation. The replacement run uses the same seed,
simulated data design, and recovery budget, but chooses weights exclusively by
validation loss. The test set is not used for checkpoint selection. The stopped
run's manifest is retained; it is not a completed result.

All 11 agent-related tests, four existing revision tests, and three existing
shared-variable-radius tests passed (18 tests total). These checks establish
implementation invariants and compatibility, not inferential calibration.


## Bounded-radius rerun

Under the lognormal hierarchy, individual radii had an unbounded upper tail:
held-out true radii reached 27.3 and posterior 95% upper bounds 36.3, in an
8-by-10 room whose diagonal is about 12.8. Radii that large all mean "sense
every agent" and cannot be told apart, so they inflated the radius panels'
scale without adding information.

The rerun replaces the local radius prior by a scaled logit-normal with upper
bound R = 10 (`TogetherFlowSimulator(radius_upper=10)`):

\[
\operatorname{logit}(r_i/R)\sim\mathcal N(\operatorname{logit}(\mu_r/R),\sigma_r^2).
\]

mu_r is still the population median, and its LogNormal(log 2.5, 0.35) prior
is truncated below R (it essentially never binds). For radii well below R the
logit is close to the log, so the hierarchy is nearly unchanged where most mass
lies. sigma_r is now a logit(r/R)-scale standard deviation: at r = 2.5 a given
sigma_r implies about 25% less log-radius spread than before. Its 0.75 Beta(2,2)
prior is unchanged. mu_r and r_i are learned as logit(r/R) instead of log r, so
posterior draws also respect the bound. Everything else, including seed 831,
data budgets, architecture and checkpoint selection, is identical. Training
stopped at epoch 82 with epoch 57 selected.

| | lognormal (`validation-selected`) | bounded, R = 10 (`bounded-radius`) |
|---|---|---|
| max true r_i (held out) | 27.3 | 8.9 |
| max posterior draw of r_i | 66.9 | 9.5 |
| max 95% upper bound of r_i | 36.3 | 9.0 |
| r_i RMSE (mean over agents) | 1.59 | 0.95 |
| r_i prior-median RMSE | 1.89 | 1.27 |
| r_i 95% coverage | 0.83 | 0.84 |
| r_i contraction | 0.69 | 0.72 |
| mu_r RMSE / coverage | 0.65 / 0.85 | 0.60 / 0.85 |
| centered radius correlation | 0.30 | 0.25 |
| centered weight correlation | 0.74 | 0.73 |

RMSEs are lower because the prior is narrower, as the matching drop in the
prior-median baseline shows. Recovery of radius *differences* within a group
remains weak. Radius coverage, like that of mu_r and sigma_r, stays below the
nominal 95%. Bounding the radius fixes the scale problem; it does not make
individual radii better identified.
