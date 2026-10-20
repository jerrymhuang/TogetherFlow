# Updated model-extension draft

Revised October 19, 2026 from the attachment supplied in chat. Python only.

`model_extensions.tex` is the manuscript fragment. It preserves the original
section order and existing labels; each Simulation Study passage
has at most three short paragraphs. The added Sensing radius identification
subsection contains a comparison table and a short two-paragraph discussion. ECE rows were added to the
extension comparisons. Training and evaluation details are consolidated in the introduction, and each prose paragraph occupies one source line. No model was retrained. New figures use saved results and one deterministic salience illustration.

## Results selected

- Collision, salience, and scalar time-varying influence: `outputs/variable-radius/rev-variable-{reference,collision,salience,timevarying}/`.
- Time-varying path: `outputs/variable-radius/superstats-variable-path/per_step_metrics.csv`; the reported path metrics are means over exported time points, not fixed-radius results.
- Hierarchy: `outputs/agent-radius-pooling/bounded-radius-online/`, the completed online, 12-agent, bounded-radius run. This replaces the old population-only diffusion analysis, including its fixed-radius equations and unsuccessful control table. The hierarchy table uses raw RMSE of posterior medians, not the other suites' NRMSE.
- Radius comparison: the variable-radius reference and `outputs/revision/rev-fixed-{reference,radius-1.25,radius-4}/`. This is a comparison of full specifications; inputs and test populations differ as explained in the caption. It is not an identical-data intervention on inference dimension alone.
- Prior sensitivity: `outputs/revision/rev-fixed-reference/` and `rev-fixed-prior-{wide,tight,slow-velocity,eta-smooth,eta-diffuse}/`. The appended subsection specifies all priors, reports recovery/calibration, and compares reference/wide/concentrated recovery plots. These runs fix radius at 2.5; the section does not claim unknown-radius prior robustness or posterior sensitivity on identical observations. All three discussion paragraphs occupy one source line each, and training details remain in the introduction.

`table_values.csv` records the unrounded source values, metric names, parameters,
and source CSV paths for every numeric table cell. The two individual hierarchy
rows average the 12 agent-specific metrics; they are not pooled RMSEs.

## Changes that affect interpretation

1. Radius metrics from the older true-radius input design are replaced by the current fixed-reference-distance observation design. Radius is inferred in the first three extensions.
2. Collision avoidance no longer claims a universal recovery improvement: noise recovery is weaker in the new results. The proximity-weight index and the separation equation were corrected to describe the implemented heading drift, and repulsion radius is independent of sensing radius. The evaluated gain is 0.6.
3. Salience exponent recovery is described as weak. The experiment does not prove structural non-identifiability or show that a narrower prior fixes it.
4. The time-varying section now reports the unknown-radius scalar and trajectory runs. The trajectory analysis has a different estimator, duration, and training design.
5. The hierarchy now estimates individual weights AND radii with population parameters. Radius uses the bounded logit-normal hierarchy with R=10, not an unbounded local lognormal. There is no matched complete-pooling control for this online design, so no such comparison is claimed. Within-group radius recovery remains weaker and centered intervals undercover.
6. The intro completes the training description and distinguishes prior-error NRMSE from the original manuscript's range-normalized definition. It also states the revised heading/noise dynamics used by these runs; the corresponding main-model description must be consistent.

## Figures and compilation

Recovery figures compare each behavioral extension with the variable-radius reference: collision avoidance (`m1`), beacon salience (`m2`), time-varying influence (`m3`), and individual heterogeneity (`m4`). The hierarchy is shown in full, including all 30 targets, with the reference above it for descriptive context; the caption explicitly identifies the different group size, priors, and observation design. It is not a matched complete-pooling comparison. The sensing-radius subsection uses a table and a short discussion without adding figures.

The original six PNG assets retain the existing `ext-recovery-m0/m1/m2` and `ext-coverage-m0/m1/m2` names. Added assets are `ext-recovery-m3.png` from `outputs/variable-radius/rev-variable-timevarying/recovery.png` and `ext-recovery-m4.pdf` from `outputs/agent-radius-pooling/bounded-radius-online/parameter_recovery.pdf`. Replace the figure assets when integrating the text so plots and tables use the same selected results.

Each behavioral extension also has its own mechanism or posterior figure, discussed in its Simulation Study passage:

- `ext-collision-crossing.png`: the saved crossing-groups scenario, with no repulsion, the evaluated repulsion setting, and stronger repulsion. The caption records the conditional scenario settings; rates quoted in the text come from `outputs/scenarios/report.md`, not from the recovery experiment.
- `ext-salience-selection.pdf`: selection regions and the strongest beacon's selection response in an explicitly illustrative fixed layout. This is computed directly from the score rule, not from new posterior inference, and cannot establish identifiability in the training population.
- `ext-timevarying-paths.png`: the saved Superstats posterior trajectories for the first three held-out datasets, including tracking errors and uncertainty. Pointwise intervals are not simultaneous path bands.
- `ext-individual-posteriors.pdf`: medians, marginal 95% intervals, and truths for all agents in the first two held-out groups from the selected online hierarchy. Groups are selected by index, not by recovery quality.

`figure_sources.csv` and `figure_data/provenance.json` identify sources and selections. `figure_data/` also retains the plotted individual rows, the salience response and layout, and the crossing-scenario report excerpt. Rebuild the four extension-specific figures from the repository root using `.venv/bin/python outputs/docs/model_extensions_revision/build_extension_figures.py`. The script reads saved outputs and performs no model training.

Copy the fragment and the contents of `figures/` into your manuscript's source
and graphics locations. The fragment needs amsmath, amssymb, graphicx, booktabs,
multirow, and hyperref (for autoref). All section/equation/figure labels from
the supplied draft are preserved, including the corrected separation equation.
References to the main manuscript resolve when integrated; they appear as ??
in the standalone preview. `preview.tex` is only a compilation wrapper.

To compile the standalone preview from this directory:

```
pdflatex -interaction=nonstopmode -halt-on-error preview.tex
pdflatex -interaction=nonstopmode -halt-on-error preview.tex
```

The hierarchical recovery figure uses the y-axis label `Estimate` on all five rows, matching the other recovery plots. The plotting function in `experiments/run_agent_radius_pooling.py` uses the same label for future exports.

## Individual posterior violins

The figure generator supports genuine violin plots from physical-scale posterior draws, with posterior medians, marginal 95% intervals, and ground-truth crosses overlaid. It retains the two-by-two layout and the first two held-out groups. Each violin is normalized to the same maximum width; its width represents density within that marginal, not probability mass across agents.

The selected run's `posterior.npz` and `model.keras` are currently absent from this workspace. Its CSV contains medians and interval endpoints only, so it cannot determine posterior shape. The current manuscript figure and caption therefore still show intervals. No density is reconstructed from those summaries.

After restoring the selected run's `posterior.npz` (arrays `samples` with shape groups-by-draws-by-30 and `truth` with shape groups-by-30), run from the repository root:

```
.venv/bin/python outputs/docs/model_extensions_revision/build_extension_figures.py --individual-only --individual-style violin
```

Use `--posterior-file /path/to/posterior.npz` for a different location. The generator verifies physical-scale support and checks the first two groups' truths and agent ordering against the original CSV before replacing the figure. It also updates the manuscript caption and figure provenance, and saves the plotted draws in `figure_data/individual_posterior_draws.npz`. Recompile the preview using the commands above. Explicit violin mode stops before changing the existing figure if draws are unavailable; the default `auto` mode uses intervals until the posterior file exists. No retraining is performed.
