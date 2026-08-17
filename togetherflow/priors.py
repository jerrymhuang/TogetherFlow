import numpy as np
from numba import njit

# Priors are @njit so they can be called at simulation speed and so that seeding
# them goes through numba's RNG (see TogetherFlowSimulator.seed).
#
# Every prior returns theta in the canonical slot order:
#     [w, r, v, noise]  (+ alpha in slot 4 when beacon selection is inferred)
# The simulator reads these slots positionally; `param_names` on the simulator
# decides which slots are exposed as named outputs.

FIXED_RADIUS = 2.5  # the "known constant" used by the fixed-r arm

# eta is the diffusion coefficient of the heading process, not the perturbation
# of an alignment target it used to be, so its prior was re-derived from a prior
# predictive check (experiments/prior_predictive_eta.py) rather than inherited.
# Beta(2,2) puts the visually calibrated operating point (eta ~ 0.6, mean
# per-step turn ~0.25 rad) around its 65th percentile; the old Beta(2,5) put it
# near the 90th and concentrated the budget on near-deterministic motion.


@njit
def complete_pooling_prior():
    """Sample prior parameters [w, r, v, noise] for the complete-pooling model."""
    weight = np.random.beta(2., 2.)
    radius = np.random.lognormal(0., 0.5)
    v      = np.random.beta(2., 2.) * 2.
    focus  = np.random.beta(2., 2.)
    return np.array([weight, radius, v, focus], dtype=np.float32)


# ── Reviewer point 8: beacon switching as an inferred quantity ────────────────

@njit
def salience_prior():
    """[w, r, v, noise, alpha] — alpha is the beacon-salience exponent.

    alpha ~ Gamma(2, 1) covers the regimes in the beacon-selection write-up:
    mass near 0 (nearest-beacon), around 1 (salience/distance balanced), and a
    right tail above 1 (salience-dominant).
    """
    weight = np.random.beta(2., 2.)
    radius = np.random.lognormal(0., 0.5)
    v      = np.random.beta(2., 2.) * 2.
    focus  = np.random.beta(2., 2.)
    alpha  = np.random.gamma(2., 1.)
    return np.array([weight, radius, v, focus, alpha], dtype=np.float32)


# ── Reviewer point 2: sensing radius held as a known constant ─────────────────

@njit
def fixed_radius_prior():
    """[w, r=FIXED_RADIUS, v, noise] — r is a known constant, not inferred.

    The slot is kept so the simulator contract is unchanged; the adapter simply
    does not route it into the inference variables.
    """
    weight = np.random.beta(2., 2.)
    v      = np.random.beta(2., 2.) * 2.
    focus  = np.random.beta(2., 2.)
    return np.array([weight, FIXED_RADIUS, v, focus], dtype=np.float32)


# ── Reviewer point 7: alternative priors for the sensitivity analysis ─────────
#
# Each varies ONE aspect of the reference prior so that any diagnostic shift is
# attributable. Ranges stay physically admissible for an 8x10 room.

@njit
def prior_wide():
    """Weakly-informative: every marginal widened toward uniform."""
    weight = np.random.beta(1., 1.)             # uniform on [0, 1]
    radius = np.random.lognormal(0., 1.0)       # heavier tail than reference
    v      = np.random.beta(1., 1.) * 2.
    focus  = np.random.beta(1., 1.)
    return np.array([weight, radius, v, focus], dtype=np.float32)


@njit
def prior_tight():
    """Informative: mass concentrated near the centre of each reference marginal."""
    weight = np.random.beta(5., 5.)
    radius = np.random.lognormal(0., 0.25)
    v      = np.random.beta(5., 5.) * 2.
    focus  = np.random.beta(5., 5.)
    return np.array([weight, radius, v, focus], dtype=np.float32)


@njit
def prior_bounded_radius():
    """Reference prior, except r is bounded to the room.

    The reference lognormal(0, 0.5) places mass beyond the room diagonal (~6.4
    for an 8x10 room), where the sensing radius stops being distinguishable
    because every agent already sees every other agent. This arm restricts r to
    a range where it can in principle be identified.
    """
    weight = np.random.beta(2., 2.)
    radius = 0.25 + np.random.beta(2., 3.) * 3.75   # r in [0.25, 4.0]
    v      = np.random.beta(2., 2.) * 2.
    focus  = np.random.beta(2., 2.)
    return np.array([weight, radius, v, focus], dtype=np.float32)


@njit
def prior_slow_velocity():
    """Reference prior, except v is restricted to a slower, better-resolved range.

    Targets reviewer point 3: at the reference prior's upper tail an agent
    crosses the room within the observation window, so the trajectory carries
    little additional information about speed.
    """
    weight = np.random.beta(2., 2.)
    radius = np.random.lognormal(0., 0.5)
    v      = 0.05 + np.random.beta(2., 2.) * 0.95   # v in [0.05, 1.0]
    focus  = np.random.beta(2., 2.)
    return np.array([weight, radius, v, focus], dtype=np.float32)


# ── Reviewer point 1: separation strength as an inferred quantity ─────────────

@njit
def collision_prior():
    """[w, r, v, eta, kappa] — kappa is the separation gain.

    Gamma(2, 0.75) has mode 0.75 and reaches ~3.5 at the 95th percentile, so it
    spans the gentle setting used in the scenarios (0.6), the strong one (1.5),
    and the regime above ~3 where separation fragments a converging group. The
    upper tail is deliberately included: whether the data can rule out
    over-dispersion is the question the arm exists to answer.
    """
    weight = np.random.beta(2., 2.)
    radius = np.random.lognormal(0., 0.5)
    v      = np.random.beta(2., 2.) * 2.
    focus  = np.random.beta(2., 2.)
    kappa  = np.random.gamma(2., 0.75)
    return np.array([weight, radius, v, focus, kappa], dtype=np.float32)


# ── Reviewer point 7: sensitivity on eta specifically ────────────────────────
#
# Reviewer point 7 names w, v, r AND eta, but the original four sensitivity arms
# covered only wide/tight/radius/velocity. These two close that gap, and they
# bracket the reference Beta(2,2) on the axis the prior predictive measured.

@njit
def prior_eta_smooth():
    """Reference, with eta ~ Beta(2,5) — the published specification.

    Retained as the smooth end of the range precisely because it is what the
    paper reported, so there is a documented path from the published results to
    the re-run ones.
    """
    weight = np.random.beta(2., 2.)
    radius = np.random.lognormal(0., 0.5)
    v      = np.random.beta(2., 2.) * 2.
    focus  = np.random.beta(2., 5.)
    return np.array([weight, radius, v, focus], dtype=np.float32)


@njit
def prior_eta_diffuse():
    """Reference, with eta ~ 1.5 * Beta(2,2).

    Reaches the strongly diffusive regime where heading noise starts to compete
    with goal-directed steering. Note this exceeds the [0, 1] support the other
    arms use, so the arm overrides the adapter bound on eta.
    """
    weight = np.random.beta(2., 2.)
    radius = np.random.lognormal(0., 0.5)
    v      = np.random.beta(2., 2.) * 2.
    focus  = np.random.beta(2., 2.) * 1.5
    return np.array([weight, radius, v, focus], dtype=np.float32)


# ── Reviewer point 6: non-stationary influence weight ────────────────────────

@njit
def prior_nonstationary():
    """[w0, r, v, eta, tau] — tau is the logit random-walk scale for w.

    tau ~ 0.4 * Beta(1, 3) has positive density AT zero, which matters here:
    tau = 0 is the stationary model, so the prior has to give the data a real
    chance to conclude that w does not move. Its upper reach corresponds to w
    traversing roughly two thirds of [0, 1] within a 60 s trial (median
    trajectory range 0.65 at tau = 0.4), which is the strongest non-stationarity
    that still leaves a recognisable navigation task.

    Note tau is per unit time, not per step: the expander scales increments by
    sqrt(dt), so the same tau means the same behaviour at any dt or horizon.
    """
    weight = np.random.beta(2., 2.)
    radius = np.random.lognormal(0., 0.5)
    v      = np.random.beta(2., 2.) * 2.
    focus  = np.random.beta(2., 2.)
    tau    = np.random.beta(1., 3.) * 0.4
    return np.array([weight, radius, v, focus, tau], dtype=np.float32)


# ── Reviewer point 8: time-varying beacon salience ───────────────────────────

@njit
def salience_ou_prior():
    """[w, r, v, eta, sigma_s, tau_s] — hyper-parameters of the salience process.

    Beacon salience follows a shared log-Ornstein-Uhlenbeck process (see
    `simulator.make_ou_salience_process`); these two govern it.

    sigma_s ~ 2.0 * Beta(1.5, 2) is the stationary SD of log-salience. It has
    density at zero because sigma_s = 0 collapses every beacon to s = 1, which
    makes the selection rule exactly nearest-beacon — the published model. So
    "salience does not vary" is a restriction the data can reject rather than an
    assumption, the same structure prior_nonstationary gives tau. The upper reach
    matters for a different reason: switching needs the log-salience gap between
    two beacons to exceed log(d_ratio), and with beacons outside the room those
    ratios are only about 1.5-3x, i.e. log-gaps of 0.4-1.1. A gap between two
    independent paths has SD sigma_s*sqrt(2), so sigma_s near 1 is where
    crossings start, and the prior must reach well past it.

    tau_s ~ 2 + 18 * Beta(2, 2) seconds is the reversion timescale, spanning
    roughly 2-20 s: at the bottom the field flickers faster than an agent can
    steer toward anything, at the top a 60 s trial contains only a few crossings.
    It is bounded away from zero because tau_s -> 0 degenerates to white noise,
    which is not a switching model at all — it would just add heading noise and
    duplicate eta.

    Note alpha is NOT inferred here and is held at 1.0 by the variant: with a
    time-varying field the exponent is redundant, since any large alpha makes the
    rule a pure argmax over salience regardless of its value. Inferring both is
    what made v2-salience-spread50 unidentifiable.
    """
    weight  = np.random.beta(2., 2.)
    radius  = np.random.lognormal(0., 0.5)
    v       = np.random.beta(2., 2.) * 2.
    focus   = np.random.beta(2., 2.)
    sigma_s = np.random.beta(1.5, 2.) * 2.0
    tau_s   = 2.0 + np.random.beta(2., 2.) * 18.0
    return np.array([weight, radius, v, focus, sigma_s, tau_s], dtype=np.float32)


# ── Reviewer point 4: partial pooling ────────────────────────────────────────

@njit
def partial_pooling_prior():
    """[mu_w, r, v, eta, sigma_w] — a population distribution over w.

    Each agent draws its own weight as
        w_i = sigmoid(logit(mu_w) + sigma_w * z_i),   z_i ~ N(0, 1)
    once per trial. The inference targets are the population parameters, not the
    individual weights: a single trial contains 49 agents, so the spread is
    estimable from one trial without needing per-agent posteriors, which is the
    "simplified form" the reviewer explicitly allowed.

    sigma_w ~ 1.5 * Beta(1, 3) has positive density at zero, because sigma_w = 0
    is exactly the complete-pooling model and the data must be able to conclude
    that agents do not differ. At sigma_w = 1 and mu_w = 0.5 the middle 95% of
    agents span w in roughly [0.12, 0.88], which is strong heterogeneity.
    """
    mu_w    = np.random.beta(2., 2.)
    radius  = np.random.lognormal(0., 0.5)
    v       = np.random.beta(2., 2.) * 2.
    focus   = np.random.beta(2., 2.)
    sigma_w = np.random.beta(1., 3.) * 1.5
    return np.array([mu_w, radius, v, focus, sigma_w], dtype=np.float32)
