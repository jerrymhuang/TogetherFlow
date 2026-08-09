"""Prior predictive check on eta, now that it is a heading diffusion coefficient.

    uv run python experiments/prior_predictive_eta.py

eta changed meaning: it was the standard deviation of a perturbation applied to
the Vicsek alignment target, and it is now the diffusion coefficient of the
heading process. Beta(2,5) was chosen for the former and carries no argument for
the latter, so this script asks two questions before any training time is spent:

  1. What does eta actually do to trajectories? A sweep at fixed w, r, v.
  2. Under each candidate prior, what distribution of trajectory behaviour do we
     end up training on? A prior that concentrates on smooth or on unusable
     motion wastes the simulation budget either way.

Two summaries, both cheap and both interpretable without reference to the
inference machinery:

  mean |turn|   per-step heading change in radians. Direct read on wobble.
  tortuosity    path length / net displacement, over the FIRST 10 s only.
                Measured over the full 60 s it does not discriminate at all
                (17.9 at eta=0 against 19.6 at eta=1.5, non-monotone in between):
                beacons are outside the room, so agents reach a wall within
                ~15 s and then mill, leaving net displacement dominated by where
                they happened to settle rather than by how they got there. That
                saturation is worth knowing on its own — it means late-trajectory
                data carries little information about any parameter — but it
                makes the full-horizon ratio useless as a summary.
"""

import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from togetherflow.simulator import TogetherFlowSimulator
from scenarios import make_fixed_prior, ROOM, REPULSION_RADIUS, REPULSION_GAIN

OUT = pathlib.Path(__file__).parent.parent / "outputs" / "scenarios"

# The training configuration, not the figure configuration: whatever we conclude
# has to hold for the data the networks will actually see.
NUM_AGENTS = 49
NUM_BEACONS = 4
TIME_HORIZON = 60.0
DT = 0.1
BEACON_SPREAD = 50.0
SEED = 20260808
EARLY_WINDOW_S = 10.0    # tortuosity window, before agents reach the walls

# Reference-prior medians for the parameters we are not varying.
W_REF, R_REF, V_REF = 0.5, 1.0, 1.0


def trajectory_stats(positions, rotations):
    """Per-simulation (mean |turn|, tortuosity), averaged over agents."""
    dtheta = np.abs((np.diff(rotations, axis=1) + np.pi) % (2 * np.pi) - np.pi)
    mean_turn = dtheta.mean(axis=(1, 2))

    # Restricted to the pre-saturation window; see the module docstring.
    k = int(EARLY_WINDOW_S / DT)
    early = positions[:, :k]
    step = np.linalg.norm(np.diff(early, axis=1), axis=-1)
    path = step.sum(axis=1)                                   # (B, A)
    net = np.linalg.norm(early[:, -1] - early[:, 0], axis=-1)
    tort = (path / np.maximum(net, 1e-6)).mean(axis=1)
    return mean_turn, tort


def simulate(prior, batch, seed=SEED):
    sim = TogetherFlowSimulator(
        num_agents=NUM_AGENTS, num_beacons=NUM_BEACONS, room_size=ROOM,
        dt=DT, time_horizon=TIME_HORIZON, output_mode="raw", prior=prior,
        relative_heading=True, diffusive_heading=True,
        beacon_spread=BEACON_SPREAD, seed=seed,
        repulsion_radius=REPULSION_RADIUS, repulsion_gain=REPULSION_GAIN,
    )
    out = sim.sample(batch)
    return trajectory_stats(out["positions"], out["rotations"])


# ── Candidate priors ─────────────────────────────────────────────────────────
# Each returns [w, r, v, eta] with w, r, v at the reference specification so the
# only thing varying between candidates is the marginal on eta.

def _make_prior(eta_sampler):
    from numba import njit

    @njit
    def _prior():
        w = np.random.beta(2., 2.)
        r = np.random.lognormal(0., 0.5)
        v = np.random.beta(2., 2.) * 2.
        return np.array([w, r, v, eta_sampler()], dtype=np.float32)
    return _prior


def build_candidates():
    from numba import njit

    @njit
    def eta_current():        # Beta(2,5): the prior inherited from the old meaning
        return np.random.beta(2., 5.)

    @njit
    def eta_centred():        # Beta(2,2): mass around 0.5
        return np.random.beta(2., 2.)

    @njit
    def eta_uniform():        # Beta(1,1) on [0, 1]
        return np.random.beta(1., 1.)

    @njit
    def eta_wide():           # 1.5 * Beta(2,2): reaches the strongly diffusive regime
        return np.random.beta(2., 2.) * 1.5

    return [
        ("Beta(2,5)  (current)", _make_prior(eta_current)),
        ("Beta(2,2)", _make_prior(eta_centred)),
        ("Beta(1,1)", _make_prior(eta_uniform)),
        ("1.5*Beta(2,2)", _make_prior(eta_wide)),
    ]


def main():
    # ── 1. Sweep: what does eta do, holding everything else fixed? ────────────
    grid = [0.0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.25, 1.5]
    print(f"eta sweep at w={W_REF}, r={R_REF}, v={V_REF}  "
          f"({NUM_AGENTS} agents, {TIME_HORIZON:g}s, 8 sims each)\n")
    print(f"{'eta':>6}  {'mean |turn|':>12}  {'tortuosity(10s)':>16}")
    sweep = []
    for eta in grid:
        mt, tt = simulate(make_fixed_prior(W_REF, R_REF, V_REF, eta), 8)
        sweep.append((eta, mt.mean(), tt.mean()))
        print(f"{eta:6.2f}  {mt.mean():12.4f}  {tt.mean():16.2f}")

    # ── 2. Prior predictive: what will we train on? ───────────────────────────
    print(f"\nprior predictive, 64 simulations per candidate")
    print(f"{'prior':>22}  {'mean |turn| (5/50/95%)':>28}  {'tortuosity 10s (5/50/95%)':>28}")
    results = []
    for name, prior in build_candidates():
        mt, tt = simulate(prior, 64)
        q_mt = np.percentile(mt, [5, 50, 95])
        q_tt = np.percentile(tt, [5, 50, 95])
        results.append((name, mt, tt))
        print(f"{name:>22}  {q_mt[0]:7.3f} {q_mt[1]:7.3f} {q_mt[2]:7.3f}       "
              f"{q_tt[0]:7.2f} {q_tt[1]:7.2f} {q_tt[2]:7.2f}")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    e, m, t = zip(*sweep)
    axes[0].plot(e, m, "o-", color="#3B2A63")
    axes[0].set_xlabel(r"$\eta$"); axes[0].set_ylabel("mean |turn| per step (rad)")
    axes[0].set_title(r"What $\eta$ does")
    ax0b = axes[0].twinx()
    ax0b.plot(e, t, "s--", color="#EE2C8B", alpha=0.8)
    ax0b.set_ylabel("tortuosity", color="#EE2C8B")

    for ax, idx, label in ((axes[1], 1, "mean |turn| per step (rad)"),
                           (axes[2], 2, "tortuosity (first 10 s)")):
        for name, mt, tt in results:
            ax.hist(mt if idx == 1 else tt, bins=18, alpha=0.5, label=name)
        ax.set_xlabel(label)
        ax.set_title("Prior predictive")
    axes[2].legend(fontsize=8)

    fig.suptitle(r"Prior predictive check on $\eta$ as heading diffusion", fontsize=13)
    fig.tight_layout()
    path = OUT / "figures" / "prior_predictive_eta.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
