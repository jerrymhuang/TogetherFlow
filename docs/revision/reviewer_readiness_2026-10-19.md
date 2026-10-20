# Reviewer evidence audit — October 19, 2026

Scope: requests 2, 3, 4, 6, 7, 8, and 9 supplied in chat. This is an audit of
saved experiments, source code, and writing, not a new experiment or a claim of
human validation. Numbering below follows the current reviewer requests;
several older configuration labels use different numbering.

Most requested analyses exist. The remaining work is chiefly selecting the
appropriate evidence and correcting the writing, with substantive limits on
radius identification, prior robustness, and Python/Unity equivalence.

| Request | Evidence status | Writing decision |
| --- | --- | --- |
| 2. Sensing radius | Fixed-radius analysis and sensitivity completed; unknown-radius analyses also exist | Use the fixed specification as the clearest response. State that 2.5 is a specified model constant, not a measured human sensing range. Unknown-radius recovery remains limited. |
| 3. Speed calibration | Training-regime, simulator-mechanism, horizon, and prior analyses completed | Report improved calibration and the matched contrasts; avoid claiming a uniquely established cause of the original error. |
| 4. Partial pooling | Demonstrated, including individual weights and a later joint weight/radius hierarchy | Prefer the later controlled individual-level examples. Earlier population-only diffusion fits are poorly calibrated. |
| 6. Time-varying weight | Scalar and trajectory analyses completed | Demonstrate changing weights and quantify recovery; volatility is weakly recovered and no human transition mechanism is validated. |
| 7. Prior sensitivity | Alternative-prior runs exist; strongest current suite fixes radius | Describe prior-predictive recovery sensitivity. Historical radius-prior runs use true-radius features; they do not establish deployable unknown-radius robustness. |
| 8. Beacon switching | Full Python 2x2 transfer analyses completed | The discrepancy affects calibration of influence. Unity equivalence and the historical reason for simplifying it still need explicit reconciliation. |
| 9. 2D/3D discussion | A substantive draft exists | Integrate it, resolve its citation placeholder, and align it with the apparatus. A 3D experiment was not requested and has not been demonstrated here. |

## 2. Sensing radius

The fixed-radius queue completed all 17 stages, including reference training,
radius sensitivity, switching transfer, hierarchy, trajectories, and calibration
ablations ([status](../../outputs/revision/queue-status.json)). The companion
unknown-radius queue completed all 11 stages
([status](../../outputs/variable-radius/queue-status.json)). Completion means
execution succeeded; it does not certify inference quality.

The current fixed-radius reference sets r = 2.5 room units. Its w/v/eta ECE is
0.0120/0.0163/0.0089. Separately trained r = 1.25 and r = 4 analyses are saved.
For those three constants, w/v/eta contraction is respectively
0.954/0.968/0.884, 0.968/0.957/0.847, and 0.970/0.954/0.820.
These are recovery results under each assumed radius, not a test of applying
the wrong radius to identical data.

The unknown-radius reference uses counts at fixed observable distances
[0.5, 1, 2, 4], excluding true-radius counts, distances, and neighbor
fluctuations. Radius NRMSE is 0.8267, contraction 0.3020, and ECE 0.0091
([metrics](../../outputs/variable-radius/rev-variable-reference/metrics.csv)).
This is limited recovery in this experiment, not a formal proof of structural
non-identifiability. Low ECE alone cannot establish posterior accuracy.

The later 12-agent hierarchy uses a different prior, controlled beacon layout,
and richer turning features. Its radius results cannot be presented as a
matched improvement over the shared-radius reference; see point 4.

## 3. Movement speed

The original final 3K CSV confirms speed ECE = 0.20658
([source](../../togetherflow/results/final/tflow_complete_pooling_bdlstm_fm100_3e3_final.csv)).
The current fixed-radius controls provide the following evidence:

| Analysis | Speed ECE |
| --- | ---: |
| Revised online reference | 0.0163 |
| Revised offline 3K, validation-selected weights | 0.0634 |
| Revised reference with legacy heading update | 0.0085 |
| Revised reference with legacy noise mechanism | 0.0106 |
| Restricted speed prior | 0.0096 |
| Wide joint prior | 0.0261 |
| Tight joint prior | 0.0145 |

Sources: the corresponding `outputs/revision/rev-fixed-*/metrics.csv` files.
The offline/online contrast supports sensitivity to training regime and
simulation exposure. The online reference sees 480,000 simulations; offline
training reuses a bank of 3,000. This is not an equal-simulation-budget test,
and the revised 3K arm is not an exact reconstruction of the original paper.
Do not attribute the entire 0.207-to-0.016 difference solely to online training,
the summary architecture, or one simulator correction.

Historical 20/40/60/80/100-second arms report speed ECE
0.0236/0.0208/0.0128/0.0216/0.0204. They show no monotonic calibration improvement
with duration. They use the historical observation design and one training seed
per arm, so they cannot settle horizon sensitivity for the revised observation
model or prove that longer trajectories contain no additional information.
The reviewer asks about possible remedies; it is reasonable to report that
longer horizons were tested without a consistent benefit, while modified
priors and revised training yielded low speed ECE. Additional-observable
analyses exist but are not a matched isolation of the original speed problem.

## 4. Partial pooling

There is now a concrete secondary demonstration, beyond population-only fits:

* The eight-agent, fixed-radius comparison was repeated for seeds 831 and 832.
  Individual-weight RMSE under partial pooling is 0.0482/0.0521 versus
  0.0913/0.0879 under the matched complete-pooling estimator on the same
  heterogeneous observations. Within-group centered correlations are
  0.874/0.837. Nominal 90% individual coverage is only 85.4%/82.8%, and the
  complete-pooling control also undercovers under its own generator
  ([comparison](../../outputs/agent-partial-pooling/example/comparison.csv)).
* The later online, 12-agent bounded-radius hierarchy jointly estimates
  population parameters, individual weights and radii, speed, and noise from
  300 held-out groups with 1,000 posterior draws each. Centered weight
  correlation is 0.827, centered RMSE 0.0494 versus a zero-difference baseline
  of 0.0879, and centered 95% coverage 96.4%. Centered radius correlation is
  0.422, centered RMSE 0.6848 versus baseline 0.7551, and centered 95% coverage
  91.4% ([summary](../../outputs/agent-radius-pooling/bounded-radius-online/summary.json)).
  Per-agent marginal 95% coverage ranges from 94–97% for weights and
  90.3–96.3% for radii
  ([metrics](../../outputs/agent-radius-pooling/bounded-radius-online/metrics.csv)).
  This is one seed and a different observation/prior design; it has no matched
  complete-pooling comparison at this same design.

These support a partial-pooling demonstration with explicit limits. Weight
differences are more informative than radius differences; population radius
spread remains weakly constrained (contraction 0.155). Do not claim all
individual differences or all uncertainty estimates are resolved.

The earlier fixed/variable population-only diffusion hierarchies and their
controls have severe calibration errors (speed ECE approximately 0.44–0.48).
Do not use those as the successful demonstration simply because their queues
finished. Also, mu_w and mu_r are population medians under the implemented
hierarchies, and w weights beacon attraction; 1-w weights neighbor alignment.

## 6. Time-varying influence

The fixed-radius scalar model recovers w0 with contraction 0.870, but tau only
0.185. The unknown-radius companion gives 0.797 and 0.090, respectively.
Both permit a logit random walk of w during a trial.

The numerical trajectory exports now exist. Averaging the exported per-step
metrics over the 30-second window gives:

| Trajectory model | Mean w_t contraction | Mean w_t ECE | Mean tau contraction |
| --- | ---: | ---: | ---: |
| Fixed radius | 0.8963 | 0.0449 | 0.0565 |
| Unknown radius, fixed-distance features | 0.7943 | 0.0315 | 0.1655 |

Sources: [fixed](../../outputs/revision/superstats-fixed-path/per_step_metrics.csv)
and [variable](../../outputs/variable-radius/superstats-variable-path/per_step_metrics.csv)
per-step CSVs. Time points share trajectories and are not independent
replications. Path inference uses a separate estimator, software environment,
and training design from the scalar model; their scores are not a controlled
comparison. The path model demonstrates changing influence, while tau is weakly
recovered. A drift-free random walk does not impose a directional transition
from exploration toward social following or establish that people make it.

## 7. Prior sensitivity

The revised fixed-radius family includes wide/tight joint priors, a slower
speed prior, and smoother/more diffuse noise priors. Across those five arms,
w/v/eta ECE ranges are 0.0084–0.0276, 0.0096–0.0261, and 0.0063–0.0228.
Historical runs additionally change the radius prior, including a bounded
alternative, but their true-radius input features limit their relevance to
unknown-radius inference.

Distinguish three claims: recovery across different prior-predictive
populations, sensitivity to a fixed radius constant, and sensitivity of
posteriors for the same observations to changing priors. The first two have
been tested. This audit did not find a common-observation posterior-prior
sensitivity analysis. If the paper retains an inferred-radius main analysis,
a radius-prior comparison using only observable inputs remains the clearest
experimental gap. If it adopts fixed radius, explain that radius uncertainty
has been replaced by an explicit constant and design sensitivity.

## 8. Beacon switching and Unity

Both suites include a 2x2 train/test transfer study with the same held-out
observations supplied to both estimators. Use these cross-evaluation files
for comparisons; the ordinary diagnostic files use different test banks.

| Radius specification | Static-trained w ECE on static data | Static-trained w ECE on switching data | Switching-trained w ECE on switching data |
| --- | ---: | ---: | ---: |
| Fixed | 0.0475 | 0.1028 | 0.0071 |
| Unknown | 0.0197 | 0.1239 | 0.0161 |

Sources: [fixed transfer](../../outputs/revision/switching-transfer/)
and [variable transfer](../../outputs/variable-radius/switching-transfer/).
The experiments support sensitivity of influence-weight calibration to
unmodelled switching. They do not establish unchanged generalizability;
switching-aware training also has tradeoffs on static data.

Unity source is available locally, contrary to the older checklist's wording.
[GenericAgent.cs](../../interactables/together-unity/Assets/Scripts/GenericAgent.cs)
recomputes the nearest target in FixedUpdate and interpolates noisy look
directions over 30 steps after a target change. Other experimental Unity
scripts use different attention/locomotion rules. The Python extension tests
time-varying salience with hysteresis, not that exact Unity implementation.
Confirm the scene/prefab/version used in the paper before asserting parity.

The baseline Python rule can also change its nearest beacon as geometry
changes. Describe the tested contrast as static versus dynamic salience,
rather than absolute absence versus presence of all target changes. An
explicit rationale for the original simplification still needs to be written
and confirmed by the authors; computational tractability should not be
invented as a historical fact. The existing transfer study is a useful Python
sensitivity assessment, not direct Unity-to-Python parameter recovery.

## 9. Planar model

[The draft discussion](3d_discussion.md) explains omitted vertical movement,
gaze versus locomotor heading, possible consequences for parameter
interpretation, and a staged extension from planar locomotion with 3D gaze to
full spatial states and visibility rules. It proposes testing the planar
limit, spatial calibration, and full-trajectory versus projected inference.
This meets the requested discussion/proposed-development scope once integrated
and referenced. Its [A] placeholder still needs an actual apparatus-relevant
reference or a carefully rewritten unsupported sentence. No empirical size or
direction of projection bias is established here.

## Writing corrections before integration

* The unchecked full-run items in [the old checklist](README.md) are stale.
* `outputs/docs/model_extension_results.tex` and several individual extension
  files mix different historical metric sets and contain conflicting readings
  of radius/noise recovery. Select one explicitly defined main specification
  and regenerate the corresponding tables from its saved CSVs.
* The original simulation-study NRMSE equation uses range normalization.
  Revised ordinary runs explicitly use BayesFlow's prior-error normalization;
  do not compare their magnitudes without harmonizing definitions. Trajectory
  metrics come from Superstats and also need their definition stated.
* Fix coverage direction: empirical coverage below nominal indicates
  undercoverage/overconfidence; above nominal indicates overcoverage. ECE alone
  does not determine direction. The current simulation-study text associates
  coverage above nominal with overconfidence.
* Remove claims that calibration proves identifiability, that a failed neural
  recovery fit proves structural non-identifiability, or that a small ECE
  proves the full posterior is correct. Contraction also requires calibration
  and recovery checks before it can be interpreted as useful information.
* Do not claim training-seed variability or numerical equivalence has been
  established by one seed per arm. Most current suites have that limitation.
* The later online hierarchy's manifest records final-weight checkpointing
  but also a `selected_epoch` of 268, which is the minimum-validation-loss epoch
  computed by the script. The online code does not restore that epoch. Describe
  the evaluated weights as final epoch 300, and avoid treating 268 as the
  evaluated checkpoint.

Verification for this audit: checked queue states, manifests, source settings,
raw metric CSVs, per-step averages, hierarchy comparison exports, and the local
Unity scripts. No new training or manuscript replacement was performed.
