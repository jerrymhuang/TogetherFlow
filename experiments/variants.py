"""Variant definitions for the reviewer-response experiment night.

Each Variant is a self-contained description of one amortized-inference run:
which generative model to simulate from, which observables the network is
allowed to see, and which parameters it must infer. The runner
(`run_variant.py`) turns a Variant into a results folder following the
amortized-workflow reporting layout.

The point of holding these as data rather than as separate scripts is that
every arm shares one code path, so a difference between two reports is
attributable to the fields that differ here — not to drift between scripts.
"""

from dataclasses import dataclass, field

from togetherflow.priors import (
    complete_pooling_prior,
    salience_prior,
    fixed_radius_prior,
    collision_prior,
    prior_wide,
    prior_tight,
    prior_bounded_radius,
    prior_slow_velocity,
    prior_eta_smooth,
    prior_eta_diffuse,
    prior_nonstationary,
    partial_pooling_prior,
)

# The six channels used by every run to date.
BASE_CHANNELS = [
    "positions",
    "rotations",
    "neighbors",
    "distances",
    "angular_velocities",
    "neighbor_fluctuations",
]

# `neighbors` and `distances` are computed at the TRUE sensing radius, so they
# leak the parameter r into the observables. This set removes them and supplies
# neighbour counts at fixed reference radii instead — information about spatial
# scale that does not presuppose the value being inferred.
R_FREE_CHANNELS = [
    "positions",
    "rotations",
    "angular_velocities",
    "radii_counts",
]

REFERENCE_RADII = [0.5, 1.0, 2.0, 4.0]

# The six base channels plus the beacon salience schedule. Salience is a
# property of the apparatus rather than of the agents, so it enters as four
# extra features per timestep (one per beacon) rather than per-agent ones.
SALIENCE_CHANNELS = BASE_CHANNELS + ["salience"]

# Support bounds used by the adapter's .constrain() calls.
PARAM_BOUNDS = {
    "w":     {"lower": 0.0, "upper": 1.0},
    "r":     {"lower": 0.0},
    "v":     {"lower": 0.0, "upper": 2.0},
    "noise": {"lower": 0.0, "upper": 1.0},
    "alpha": {"lower": 0.0},
    "kappa": {"lower": 0.0},
    "w0":    {"lower": 0.0, "upper": 1.0},
    "tau":   {"lower": 0.0, "upper": 0.4},
    "mu_w":    {"lower": 0.0, "upper": 1.0},
    "sigma_w": {"lower": 0.0, "upper": 1.5},
}


@dataclass
class Variant:
    slug: str
    title: str
    addresses: str                      # which reviewer point(s) this answers
    rationale: str                      # why this arm exists, for the report

    prior: object = complete_pooling_prior
    param_names: tuple = ("w", "r", "v", "noise")
    infer: tuple = ("w", "r", "v", "noise")   # subset routed to inference_variables
    channels: list = field(default_factory=lambda: list(BASE_CHANNELS))
    reference_radii: list | None = None
    beacon_strengths: list | None = None
    beacon_spread: float = 50.0
    time_horizon: float = 60.0

    # Time-varying beacon salience. None leaves strengths static, which is every
    # arm before V8. When set, beacon salience follows a shared log-OU process
    # and is emitted as the `salience` observable channel: the room renders the
    # beacons, so the schedule is a known experimental input, not a latent.
    # These are DESIGN constants, deliberately not inference targets — see the
    # V8 rationale for why inferring them was the wrong move.
    salience_sigma: float | None = None      # stationary SD of log-salience
    salience_tau: float = 40.0               # OU reversion timescale, seconds
    salience_sensitivity: float = 0.0        # alpha; 1.0 = score is s_b(t)/d_ib
    switch_margin: float = 1.0               # hysteresis; 1.0 = plain argmax

    num_agents: int = 49
    num_beacons: int = 4
    dt: float = 0.1

    # "bdlstm"      — Conv1D downsampling + bidirectional LSTM (the published net)
    # "transformer" — TimeSeriesTransformer with Time2Vec embedding
    #
    # Default is the transformer: at matched budget it roughly halves velocity
    # NRMSE and calibration error against the BiLSTM, with fewer parameters.
    # v0-bdlstm-online retains the BiLSTM so the comparison stays available.
    summary_net: str = "transformer"

    # Steer by the wrapped difference between target bearing and current heading.
    # The legacy alternative (False) applies absolute world-frame bearings as a
    # turn rate, which is not rotationally invariant — an agent whose beacon lies
    # due east receives no steering at all. Kept only for the comparison arm.
    relative_heading: bool = True

    # Collision avoidance. Off by default so the reference arm keeps the
    # published dynamics apart from the corrections; the V1 arms switch it on.
    repulsion_radius: float = 0.0
    repulsion_gain: float = 0.0

    # eta as the diffusion coefficient of the heading process rather than a
    # perturbation of the alignment target. On by default: under the old
    # mechanism eta was effectively inert (a 17x change in eta moved the mean
    # per-step turn by under 10%), which is the most likely explanation for its
    # near-zero posterior contraction in the published arms.
    diffusive_heading: bool = True

    # "flow_matching" — the published estimator, used everywhere by default
    # "diffusion"     — required by CompositionalWorkflow, so the partial-pooling
    #                   work will need it; v0-diffusion is the matched control
    #                   that establishes what switching costs on its own.
    inference_net: str = "flow_matching"

    # Adapter support bounds. None means PARAM_BOUNDS; an arm whose prior leaves
    # the default support must override, or .constrain() will map draws the prior
    # can actually produce outside the region the flow is allowed to place mass.
    param_bounds: dict | None = None

    # How prior draws become the per-timestep parameter path.
    # "static"        — every timestep identical; all stationary arms
    # "random_walk_w" — w follows a logit random walk scaled by tau
    expander: str = "static"

    # Whether the adapter maps bounded parameters to an unconstrained space.
    # Must be False for any arm whose checkpoint will be used compositionally:
    # BayesFlow refuses compositional sampling with a non-zero log-det Jacobian
    # (approximators/helpers/compositional.py), and .constrain() has one.
    constrain_parameters: bool = True

    n_train: int = 5000
    n_val: int = 300
    n_test: int = 300
    epochs: int = 300
    batch_size: int = 32
    seed: int = 20260803

    # Online training draws fresh simulations for every batch, so overfitting is
    # impossible by construction and every arm can run its full course. Early
    # stopping is therefore disabled (0): it exists for the offline path, where a
    # 5k bank overfits at a val/train ratio of ~9.5x, and it would otherwise cut
    # arms off at different epochs, making them non-comparable.
    early_stopping_patience: int = 0
    online: bool = True
    num_batches_per_epoch: int = 50   # online only


# ─────────────────────────────────────────────────────────────────────────────
# V0 — reference point
# ─────────────────────────────────────────────────────────────────────────────

V0 = Variant(
    slug="v0-reference",
    title="Reference model (transformer summary net, relative heading)",
    addresses="Reference arm (no reviewer point directly)",
    rationale=(
        "The arm every other variant is read against. Carries the current best "
        "configuration — online training, transformer summary network, "
        "rotationally invariant heading update — so that differences elsewhere "
        "are attributable to the variable each arm changes rather than to "
        "training or architecture."
    ),
    online=True,
)

V0_BDLSTM = Variant(
    slug="v0-bdlstm-online",
    title="Reference model with the published BiLSTM summary network",
    addresses="Summary-network justification",
    rationale=(
        "The published summary network is a bidirectional LSTM behind two strided "
        "convolutions; nothing in the paper justifies that choice against an "
        "alternative. Paired with v0-reference, which differs only in the sequence "
        "model — same prior, observables, seed and online budget — so the gap is "
        "attributable to architecture alone."
    ),
    summary_net="bdlstm",
    online=True,
)

V0_LEGACY_HEADING = Variant(
    slug="v0-legacy-heading",
    title="Reference model with the legacy (absolute-bearing) heading update",
    addresses="Navigation realism (Part 1)",
    rationale=(
        "The published heading update applies absolute world-frame bearings "
        "directly as a turn rate, so the dynamics are not rotationally invariant: "
        "an agent whose beacon lies due east receives no steering, and an agent "
        "facing away from a due-east beacon never turns around. This arm keeps "
        "that behaviour while every other arm uses the corrected relative-bearing "
        "update, quantifying what the correction changes — which is the "
        "quantitative part of the answer to the reviewers' concern about "
        "unrealistic collective motion."
    ),
    relative_heading=False,
    online=True,
)


V0_DIFFUSION = Variant(
    slug="v0-diffusion",
    title="Reference model with a diffusion inference network",
    addresses="Estimator control for the partial-pooling work",
    rationale=(
        "Partial pooling needs BayesFlow's CompositionalWorkflow, which accepts "
        "only a DiffusionModel inference network — FlowMatching is a sibling "
        "class, not a subclass, so the isinstance check rejects it regardless of "
        "how close the two are theoretically. That leaves the paper mixing two "
        "estimators, and this arm is what licenses it: identical prior, "
        "observables, seed, summary network and subnet width, differing only in "
        "the generative process. If it lands close to v0-reference, the "
        "partial-pooling result is not confounded by the estimator switch. Note "
        "'matched' means matched subnet: DiffusionModel brings a noise schedule "
        "and prediction type with no FlowMatching counterpart."
    ),
    inference_net="diffusion",
    online=True,
)


V0_BDLSTM_DIFFUSION = Variant(
    slug="v0-bdlstm-diffusion",
    title="BiLSTM summary network with a diffusion inference network",
    addresses="Estimator control for the partial-pooling work",
    rationale=(
        "Completes the summary-network x inference-network grid. With "
        "v0-reference (transformer + flow matching) and v0-diffusion "
        "(transformer + diffusion), this arm separates an estimator effect from "
        "an interaction: if the diffusion penalty is similar under both summary "
        "networks it is a property of the inference network, and if it is not, "
        "the two components are not independently swappable and the "
        "partial-pooling configuration has to be chosen as a pair."
    ),
    summary_net="bdlstm",
    inference_net="diffusion",
    online=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# V1 — R1 part 1: collision avoidance
# ─────────────────────────────────────────────────────────────────────────────

_V1_RATIONALE = (
    "The reviewers' primary concern is that Vicsek carries no forces, so agents "
    "pass through one another and the trajectories are not usable as navigation. "
    "The separation term answers that directly. Behavioural evidence lives in "
    "outputs/scenarios/; these arms answer the different question of whether "
    "adding the mechanism costs anything in parameter recovery."
)

V1_COLLISION = Variant(
    slug="v1-collision",
    title="Separation at the calibrated setting",
    addresses="R1 point 1",
    rationale=_V1_RATIONALE + (
        " Read against v0-reference, which differs only in that separation is "
        "off. rho and kappa are held at the scenario-calibrated values: kappa=0.6 "
        "keeps a converging 30-agent group as one connected cluster, while "
        "kappa=1.5 fragments it into six."
    ),
    repulsion_radius=0.4,
    repulsion_gain=0.6,
    online=True,
)

V1_COLLISION_KAPPA = Variant(
    slug="v1-collision-kappa",
    title="Separation gain inferred",
    addresses="R1 point 1",
    rationale=_V1_RATIONALE + (
        " Promotes kappa from a setting we chose to a quantity the data speaks "
        "to. The prior spans the over-dispersed regime deliberately: if the "
        "posterior excludes it, the calibrated value is supported by the data "
        "rather than by our judgement of the figures. If kappa turns out "
        "non-identifiable, that is the honest result and fixing it is justified."
    ),
    prior=collision_prior,
    param_names=("w", "r", "v", "noise", "kappa"),
    infer=("w", "r", "v", "noise", "kappa"),
    repulsion_radius=0.4,
    online=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# V4 — reviewer point 3: does a longer horizon carry more information?
# ─────────────────────────────────────────────────────────────────────────────

V4_SHORT_HORIZON = Variant(
    slug="v4-short-horizon",
    title="Twenty-second observation window",
    addresses="Reviewer point 3",
    rationale=(
        "Reviewer point 3 asks whether longer horizons would fix the velocity "
        "calibration. The prior predictive says the opposite: because beacons sit "
        "outside the room, agents reach a wall within roughly 15 s and then mill, "
        "and trajectory statistics stop discriminating — full-window tortuosity "
        "is 17.9 at eta=0 against 19.6 at eta=1.5, non-monotone in between, while "
        "the same statistic over the first 10 s recovers a clean trend. This arm "
        "tests the consequence: if 20 s recovers as well as 60 s, the extra 40 s "
        "is cost without signal, and the answer to the reviewer is that the "
        "horizon is already past saturation rather than too short."
    ),
    time_horizon=20.0,
    online=True,
)

V4_LONG_HORIZON = Variant(
    slug="v4-long-horizon",
    title="Hundred-second observation window",
    addresses="Reviewer point 3",
    rationale=(
        "The other side of the horizon question. v4-short-horizon tests whether "
        "cutting the window to 20 s costs anything; this arm tests whether "
        "extending it to 100 s buys anything. Taken with v0-reference at 60 s the "
        "three arms trace information against duration directly, which is what "
        "reviewer point 3 asks about: the prior-predictive saturation argument "
        "predicts 100 s should recover no better than 60 s, and if it does recover "
        "better, the saturation reading is wrong and the reviewer is right. "
        "Everything but the horizon is matched to v0-reference, so the comparison "
        "is attributable to duration alone. The extra 40 s is spent in the milling "
        "regime past the wall encounter, and attention cost grows with the square "
        "of sequence length: 1000 timesteps against the reference 600."
    ),
    time_horizon=100.0,
    online=True,
)

V4_HORIZON_80 = Variant(
    slug="v4-horizon-80",
    title="Eighty-second observation window",
    addresses="Reviewer point 3",
    rationale=(
        "The interpolating point between v0-reference (60 s) and v4-long-horizon "
        "(100 s). With 20/60/80/100 s the horizon curve has four points rather than "
        "three, which matters for the two effects the 100 s arm produced: w "
        "improved slightly and monotonically (NRMSE 0.371 -> 0.342 -> 0.328) while "
        "eta degraded (0.442 -> 0.560, contraction 0.795 -> 0.687). Both were "
        "measured on single runs at one seed, and a monotone trend through three "
        "points spaced 20/60/100 is weak evidence. If 80 s falls between its "
        "neighbours on both, the trends are real and the milling-regime dilution "
        "reading of eta holds; if it does not, the differences are seed noise and "
        "should be reported as such. Matched to v0-reference apart from the "
        "horizon: 800 timesteps against the reference 600."
    ),
    time_horizon=80.0,
    online=True,
)

V4_HORIZON_40 = Variant(
    slug="v4-horizon-40",
    title="Forty-second observation window",
    addresses="Reviewer point 3",
    rationale=(
        "Fills the 20-to-60 s gap, the widest remaining one on the horizon curve. "
        "It arrives after the 80 s arm has already answered the question the 80 s "
        "arm was built to ask: 80 s beat BOTH its neighbours on every parameter "
        "(w NRMSE 0.266 against 0.342 at 60 s and 0.328 at 100 s), so it is not an "
        "interpolating point and the apparent 20->60->100 monotone trend in w was "
        "not real. The arm-to-arm spread at one seed (0.062 between 80 and 100 s) "
        "is about four times the 60->100 s 'trend' it was supposed to confirm. "
        "This arm therefore measures the noise floor rather than a trend: five "
        "points at one seed each will show how much of the variation across the "
        "curve is horizon and how much is run-to-run. The honest fix is replicate "
        "seeds per horizon, not more horizons; this is the cheaper diagnostic that "
        "bounds the problem first."
    ),
    time_horizon=40.0,
    online=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# V8 — reviewer point 8, second attempt: switching driven by a salience schedule
# ─────────────────────────────────────────────────────────────────────────────

_V8_RATIONALE = (
    "v2-salience-spread50 was built to measure beacon switching and contained "
    "almost none: replaying its exact configuration and recomputing the "
    "selection rule at every step gives 0.005 switches per agent per trial, with "
    "99.7% of agents never re-targeting at all. The reason is structural. Under "
    "a STATIC salience field the score s_b^alpha / d_ib changes only through "
    "d_ib, and an agent moves toward its own target, so the choice is "
    "self-reinforcing and the room is partitioned into fixed cells each agent "
    "walks into. A posterior contraction of 0.094 on alpha was therefore not a "
    "weak signal but a nearly inert parameter. (The same replay showed alpha did "
    "not even control the initial partition: with strengths [1,1,1,8] and alpha "
    "~ Gamma(2,1), s^alpha reaches 8-500x while beacons outside the room afford "
    "distance ratios of only 1.5-3x, so the strong beacon won for 99.8% of "
    "agents at every alpha in the prior.)\n\n"
    "This arm makes salience time-varying instead: log s_b(t) is a shared "
    "Ornstein-Uhlenbeck process, one path per beacon, identical for every agent "
    "in a trial because salience is a property of the beacon rather than of the "
    "observer. Now the world changes underneath the agents, and re-targeting is "
    "driven by the beacon rather than by the agent's own geometry."
)

V8_SALIENCE_OBSERVED = Variant(
    slug="v8-salience-observed",
    title="Beacon switching driven by an observed salience schedule",
    addresses="Reviewer point 8",
    rationale=_V8_RATIONALE + (
        "\n\nCrucially the salience schedule is an OBSERVABLE, not an inference "
        "target. The immersive room renders the beacons, so what each beacon was "
        "doing at each moment is something the experiment sets and records — "
        "treating it as a latent to be recovered would model away information "
        "the apparatus actually has. So sigma_s and tau_s are design constants, "
        "the inferred parameters go back to the published four, and the question "
        "becomes whether knowing the drive improves their recovery. It should "
        "help w most: a turn that coincides with a beacon brightening is "
        "attributable to the beacon rather than to unexplained noise, and w is "
        "exactly the beacon-versus-neighbour balance.\n\n"
        "This also avoids the trap the superstatistics route would have walked "
        "into. Inferring the volatility of a parameter process has returned a "
        "null three times here — tau in v5-timevarying-w at contraction 0.076, "
        "alpha in v2 at 0.094, eta at ~0.09 — so an arm whose headline number "
        "was the posterior for sigma_s had a poor prior of saying anything."
    ),
    channels=list(SALIENCE_CHANNELS),
    # Chosen by prior predictive, not by guess. At sigma_s 0.7 / tau_s 40 s /
    # margin 1.5 the model gives 2.0 switches per agent per 60 s trial (90th
    # percentile 4, 9% of agents never switching, median dwell 14.5 s), which is
    # the 3-4 maximum the switching should show. Higher sigma_s or lower tau_s
    # runs it up: 5.25 switches/agent at sigma_s 1.0, tau_s 20, margin 1.5.
    salience_sigma=0.7,
    salience_tau=40.0,
    salience_sensitivity=1.0,
    # Hysteresis is required, not cosmetic. With a plain argmax over a
    # continuously varying field the leader flickers at the discretization
    # scale rather than at tau_s — measured median dwell 0.30 s against a
    # tau_s of 20-40 s, and 59% of dwells under half a second — because the
    # difference between two OU paths recrosses zero densely near a tie. A
    # challenger must beat the incumbent by 1.5x to take over. Note this does
    # NOT rescue the trajectories: straightness is 0.102 with switching against
    # 0.090 with none, so agents mill either way (see gotcha 7 on the room not
    # containing them); the margin is about the switching being real, not about
    # the motion.
    switch_margin=1.5,
    online=True,
)

V8_SALIENCE_HIDDEN = Variant(
    slug="v8-salience-hidden",
    title="Same switching dynamics, salience schedule unobserved",
    addresses="Reviewer point 8",
    rationale=(
        "The control that makes v8-salience-observed readable. Identical "
        "generative process — same OU salience schedule, same hysteresis, same "
        "switching — but the `salience` channel is withheld, so the network sees "
        "exactly the six published observables while the world changes "
        "underneath the agents for reasons it cannot see.\n\n"
        "The pair isolates the value of the observable rather than the value of "
        "the mechanism. Without it, an improvement in the observed arm could "
        "just as well come from the switching dynamics themselves making "
        "trajectories more informative. Read w first: the difference between "
        "these two arms is what the room's knowledge of its own beacons is "
        "worth, and it is the honest form of the answer to reviewer point 8. If "
        "the hidden arm also does poorly against v0-reference, that is a "
        "separate and reportable finding — that unmodelled switching degrades "
        "recovery, which is an argument for instrumenting the schedule."
    ),
    channels=list(BASE_CHANNELS),
    salience_sigma=0.7,
    salience_tau=40.0,
    salience_sensitivity=1.0,
    switch_margin=1.5,
    online=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# V2 — reviewer point 8: Python omits beacon switching, Unity includes it
# ─────────────────────────────────────────────────────────────────────────────

_V2_RATIONALE = (
    "The Unity twin lets an agent re-target between beacons; the Python training "
    "simulator selected the nearest beacon unconditionally. The unified score "
    "s_b^alpha / d_ib closes that gap, and promoting alpha from a fixed setting "
    "to an inferred parameter turns the discrepancy into a measurable quantity: "
    "if alpha is recoverable, the training simulator can represent the twin's "
    "switching behaviour rather than merely approximating it."
)

V2_SPREAD50 = Variant(
    slug="v2-salience-spread50",
    title="Beacon salience inferred",
    addresses="Reviewer point 8",
    rationale=_V2_RATIONALE + (
        " The paired spread-10 arm was dropped: beacons are virtual and always "
        "outside the room, and at spread 10 roughly 80% of sampled beacons fell "
        "inside it, so that arm was exercising a configuration the apparatus "
        "cannot produce. The beacon sampler now rejects the interior."
    ),
    prior=salience_prior,
    param_names=("w", "r", "v", "noise", "alpha"),
    infer=("w", "r", "v", "noise", "alpha"),
    beacon_strengths=[1.0, 1.0, 1.0, 8.0],
    beacon_spread=50.0,
    online=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# V3 — reviewer point 2: sensing radius must be fixed, richer-estimated, or
#      shown non-identifiable
# ─────────────────────────────────────────────────────────────────────────────

V3A_FIXED = Variant(
    slug="v3a-fixed-radius",
    title="Sensing radius held as a known constant",
    addresses="Reviewer point 2 (option: fix r)",
    rationale=(
        "Takes the reviewer's first option literally: r is pinned to a known "
        "constant and dropped from the inference targets. The question this arm "
        "answers is what the other three parameters cost in recovery and "
        "calibration when r is no longer competing with them."
    ),
    prior=fixed_radius_prior,
    infer=("w", "v", "noise"),
    online=True,
)

V3C_RFREE = Variant(
    slug="v3c-rfree-observables",
    title="Sensing radius inferred from r-free multi-scale observables",
    addresses="Reviewer point 2 (option: richer observables)",
    rationale=(
        "Two of the six published channels — neighbour count and average "
        "neighbour distance — are computed at the true sensing radius, so the "
        "observables are a direct function of the parameter being inferred. Any "
        "identifiability claim built on them is therefore circular. This arm "
        "replaces those channels with neighbour counts at four fixed reference "
        "radii, which describe spatial scale without presupposing r. If r "
        "recovers here, it is genuinely identifiable; if it does not, that is the "
        "formal negative result the reviewer's third option asks for."
    ),
    channels=list(R_FREE_CHANNELS),
    reference_radii=list(REFERENCE_RADII),
    online=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# V5 — reviewer point 6: the modulation weight is stationary
# ─────────────────────────────────────────────────────────────────────────────

V5_TIMEVARYING_W = Variant(
    slug="v5-timevarying-w",
    title="Non-stationary influence weight (logit random walk)",
    addresses="Reviewer point 6",
    rationale=(
        "The published model fixes w for the duration of a trial, asserting that "
        "an agent's balance between beacon-driven and neighbour-driven motion "
        "never shifts. This arm replaces the parameter with a parameter path, "
        "logit w_t = logit w_{t-1} + tau sqrt(dt) xi_t, and infers (w0, tau) "
        "instead of a single w. The point is that tau = 0 recovers the "
        "stationary model exactly, so stationarity stops being an assumption and "
        "becomes a restriction the data can reject — or fail to reject, which is "
        "equally reportable. Read the contraction on tau first: if the data "
        "cannot distinguish a walk from a constant, that is the honest answer to "
        "the reviewer, and it is a statement about identifiability rather than "
        "about human behaviour."
    ),
    prior=prior_nonstationary,
    param_names=("w0", "r", "v", "noise", "tau"),
    infer=("w0", "r", "v", "noise", "tau"),
    expander="random_walk_w",
    online=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# V6 — reviewer point 4: partial pooling
# ─────────────────────────────────────────────────────────────────────────────

V6A_PARTIAL_POOLING = Variant(
    slug="v6a-partial-pooling",
    title="Per-agent influence weight, population-level inference",
    addresses="Reviewer point 4 / R2 weakness 2",
    rationale=(
        "Every published configuration gives all 49 agents one parameter vector, "
        "which the reviewers single out as the model's least defensible "
        "assumption about people. Here each agent draws its own weight from a "
        "population distribution, w_i = sigmoid(logit(mu_w) + sigma_w z_i), and "
        "the inference targets are the population parameters. sigma_w is the "
        "quantity that carries the argument: if it is recoverable the model "
        "detects heterogeneity, and sigma_w = 0 is exactly complete pooling, so "
        "the published specification becomes a restriction the data can test "
        "rather than an assumption it inherits.\n\n"
        "Two configuration choices are forced by what comes next rather than by "
        "this arm. The inference network is a DiffusionModel and the adapter "
        "does not constrain, because CompositionalWorkflow accepts only the "
        "former and refuses a non-zero log-det Jacobian; a checkpoint trained "
        "any other way cannot be composed across trials afterwards. Read this "
        "arm against v0-diffusion rather than v0-reference, since the estimator "
        "differs from the flow-matching arms."
    ),
    prior=partial_pooling_prior,
    param_names=("mu_w", "r", "v", "noise", "sigma_w"),
    infer=("mu_w", "r", "v", "noise", "sigma_w"),
    # BiLSTM, not the transformer default. The estimator grid found the two
    # components are not independently swappable: moving from flow matching to
    # diffusion costs the transformer heavily (w NRMSE 0.342 -> 0.706,
    # contraction 0.890 -> 0.522) but the BiLSTM barely at all (0.290 -> 0.347,
    # 0.916 -> 0.887). Since compositional inference forces the diffusion
    # network, the summary network has to be chosen to suit it.
    summary_net="bdlstm",
    inference_net="diffusion",
    constrain_parameters=False,
    # Twenty seconds, not the usual sixty. Individual differences are visible
    # only while agents are still navigating: the across-agent spread of a
    # beacon-following index grows by a factor of 1.23-1.32 across the full
    # sigma_w range at a 15 s horizon but only 1.09-1.11 at 40 s, because after
    # roughly 15 s agents have reached the walls and everyone looks alike
    # regardless of their weight. Training on 60 s would spend three quarters of
    # every sequence on the saturated regime that carries no information about
    # the quantity this arm exists to estimate.
    time_horizon=20.0,
    online=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# V7 — reviewer point 7: prior sensitivity
# ─────────────────────────────────────────────────────────────────────────────

_V7 = "Reviewer point 7"

V7_WIDE = Variant(
    slug="v7-prior-wide",
    title="Prior sensitivity — weakly informative",
    addresses=_V7,
    rationale=(
        "Widens every marginal toward uniform. Diagnostics that hold here are not "
        "artefacts of the reference prior's concentration."
    ),
    prior=prior_wide,
    online=True,
)

V7_TIGHT = Variant(
    slug="v7-prior-tight",
    title="Prior sensitivity — informative",
    addresses=_V7,
    rationale=(
        "Concentrates every marginal. Sets the optimistic end of the range: "
        "recovery that fails even here is a property of the model, not the prior."
    ),
    prior=prior_tight,
    online=True,
)

V7_BOUNDED_R = Variant(
    slug="v7-prior-bounded-radius",
    title="Prior sensitivity — sensing radius bounded to the room",
    addresses="Reviewer points 7 and 2",
    rationale=(
        "The reference prior lognormal(0, 0.5) puts mass beyond the room diagonal "
        "(~6.4 for an 8x10 room), where every agent already senses every other "
        "agent and r stops being distinguishable. Restricting r to [0.25, 4.0] "
        "tests whether poor recovery of r is a prior-support artefact rather than "
        "a structural non-identifiability."
    ),
    prior=prior_bounded_radius,
    online=True,
)

V7_SLOW_V = Variant(
    slug="v7-prior-slow-velocity",
    title="Prior sensitivity — slower velocity range",
    addresses="Reviewer points 7 and 3",
    rationale=(
        "At the reference prior's upper tail an agent traverses the room well "
        "within the observation window, after which the trajectory carries little "
        "further information about speed. Restricting v to [0.05, 1.0] tests "
        "whether the reported calibration error on v is driven by that saturated "
        "regime."
    ),
    prior=prior_slow_velocity,
    online=True,
)


V7_ETA_SMOOTH = Variant(
    slug="v7-prior-eta-smooth",
    title="Prior sensitivity — eta at the published specification",
    addresses="Reviewer point 7 (eta)",
    rationale=(
        "Reviewer point 7 names w, v, r and eta, but the original sensitivity "
        "arms covered only wide, tight, radius and velocity. eta is also the one "
        "parameter whose meaning changed: it is now the diffusion coefficient of "
        "the heading process rather than a perturbation of the alignment target, "
        "so Beta(2,5) no longer carries the argument it was chosen for. Keeping "
        "it as the smooth end of the sensitivity range gives a documented path "
        "from the published results to these."
    ),
    prior=prior_eta_smooth,
    online=True,
)

V7_ETA_DIFFUSE = Variant(
    slug="v7-prior-eta-diffuse",
    title="Prior sensitivity — eta into the strongly diffusive regime",
    addresses="Reviewer point 7 (eta)",
    rationale=(
        "The upper bracket, eta ~ 1.5*Beta(2,2), where heading noise starts to "
        "compete with goal-directed steering. Together with v7-prior-eta-smooth "
        "this brackets the new reference Beta(2,2), which was itself chosen by "
        "prior predictive check rather than inherited."
    ),
    prior=prior_eta_diffuse,
    # The prior reaches 1.5, beyond the default [0, 1] support for noise.
    param_bounds=PARAM_BOUNDS | {"noise": {"lower": 0.0, "upper": 1.5}},
    online=True,
)


# ─────────────────────────────────────────────────────────────────────────────

ALL_VARIANTS = [
    V0, V0_BDLSTM, V0_LEGACY_HEADING, V0_DIFFUSION, V0_BDLSTM_DIFFUSION,
    V1_COLLISION, V1_COLLISION_KAPPA,
    V2_SPREAD50,
    V8_SALIENCE_OBSERVED, V8_SALIENCE_HIDDEN,
    V3A_FIXED, V3C_RFREE,
    V4_SHORT_HORIZON, V4_HORIZON_40, V4_HORIZON_80, V4_LONG_HORIZON,
    V5_TIMEVARYING_W,
    V6A_PARTIAL_POOLING,
    V7_WIDE, V7_TIGHT, V7_BOUNDED_R, V7_SLOW_V,
    V7_ETA_SMOOTH, V7_ETA_DIFFUSE,
]

BY_SLUG = {v.slug: v for v in ALL_VARIANTS}
