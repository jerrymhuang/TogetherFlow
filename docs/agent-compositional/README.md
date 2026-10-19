# Agent-level CompositionalWorkflow pilot

This experiment asks whether joint inference from an interacting group's
trajectory can recover individual influence weights. It uses BayesFlow 2.0.14
`CompositionalWorkflow`, with all population parameters and individual weights
as one joint target vector.

## Scope and API finding

Agents are coupled by the simulator. They are not treated as conditionally
independent composition blocks. A direct `compositional_sample` call with one
whole-group block failed with:

> At least two conditioning variables are required for compositional sampling,
> got 1. Use 'sample' instead.

The completed experiment therefore uses `CompositionalWorkflow.sample`.
This tests agent-level inference with this workflow class, **not a benefit from
compositional sampling**. BasicWorkflow could fit the same joint target.
Actual composition needs multiple evidence blocks that are independent given
the shared targets, such as appropriately modeled independent group trials.
Repeated trials with persistent agent weights would require explicitly retaining
those weights across simulations and specifying the corresponding joint prior.
Splitting a single interacting group into agent blocks is not justified here.

## Design

- Four agents, four beacons, 20-second trajectories, radius fixed at 2.5.
- Same hierarchical prior family as the fixed-radius partial-pooling arm.
- Joint targets: population median mu_w, speed, noise, population logit-scale
  spread sigma_w, and each realized agent weight.
- Full ordered group observations: positions, sine/cosine heading, angular
  velocity, sampled every 0.5 seconds. No latent weights enter the observations.
- Small convolution/GRU summary and diffusion network; CPU execution.
- 4,096 training groups, 512 validation groups, 128 independent test groups;
  50 epochs and 256 posterior samples per test group. Separate seed ranges.
- Bounded targets are manually logit-transformed and transformed back for
  diagnostics. The adapter itself has no nonlinear constraint transform.
- All runs are exploratory, with fewer agents and different architecture and
  observation features than the previous suite. Numerical comparison to those
  runs would not be a matched comparison.

Individual marginal RMSE, 90% interval coverage, and contraction are exported.
Within-group centered recovery is also evaluated: recovering only the group
mean must not be mistaken for identifying which agents differ. The comparison
is zero within-group differences. These are marginal intervals, not simultaneous
coverage guarantees. No individual human conclusions follow from this pilot.

## Reproduce and inspect

```bash
PYTHONPATH=. .venv/bin/python experiments/run_agent_compositional.py \
  --out outputs/agent-compositional/new-pilot
PYTHONPATH=. NUMBA_NUM_THREADS=4 .venv/bin/python -m unittest discover \
  -s tests -p 'test_agent_weights.py'
```

The output directory must not already exist. `manifest.json` records settings;
`metrics.csv`, `summary.json`, `agent_posteriors.csv`, and
`agent_posteriors.png` contain the evaluation. The checkpoint and posterior
samples are retained locally under the repository's existing ignore rules.
The `smoke` directory records the initial API probe; `smoke-joint` is an
end-to-end execution check only, not a scientific result.

## Pilot results (October 18, 2026)

The run completed on CPU. Each agent position was evaluated across 128 held-out
groups (512 agent marginals total; agents within a group are dependent).

| Quantity | Result |
| --- | --- |
| Individual-weight RMSE, by agent position | 0.188–0.197 |
| Prior-mean RMSE, by agent position | 0.231–0.240 |
| Individual 90% interval coverage | 83.6–89.1% |
| Individual posterior contraction | 28.6–29.6% |
| Within-group centered RMSE | 0.07552 |
| Zero-difference baseline RMSE | 0.07537 |
| Within-group centered correlation | 0.0517 |
| Population-spread sigma_w RMSE / prior baseline | 0.3061 / 0.3037 |

**Interpretation:** marginal agent-weight predictions improve over the prior,
but this does not establish recovery of individual differences. Centered
predictions do not beat assigning all agents the same weight, and sigma_w is
not recovered better than the prior-mean baseline. Most apparent improvement is
consistent with learning group-level influence. Some marginal intervals also
undercover; this single reduced-budget seed cannot establish calibration.

The next useful experiment would improve agent-specific observation encoding
while retaining the complete group context, evaluate a matched group-only
baseline, and repeat across seeds and group sizes. Compositional sampling would
be a separate extension requiring independent trials and an explicit model of
which individual parameters persist. This result is neither evidence that
individual influence is fundamentally unidentifiable nor evidence that
compositional inference would fix the problem.

Validation: three latent-export regression tests and four existing revision
configuration/reproducibility tests passed. Opt-in latent export preserves the
existing observable arrays exactly for the tested seeds. The end-to-end smoke
and pilot both completed with the ordinary joint sampler.
