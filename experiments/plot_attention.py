"""Render which beacon each agent is attending to, in the style of sim_48_final.

    uv run python experiments/plot_attention.py

Beacon selection is not an output of the simulator — the kernel picks a beacon
per agent per step and only the resulting motion is recorded. This script
recomputes the selection from the recorded positions using the same score the
kernel uses, so the attention lines drawn here are the ones the agents actually
followed rather than a plausible-looking reconstruction.

Two panels, because "attends to 8 beacons" means different things either side of
the salience exponent:

  alpha = 0    nearest beacon. The room is partitioned into Voronoi cells and
               attention is decided by position alone.
  alpha > 0    salience-weighted. A bright beacon draws agents out of the cell
               they stand in, which is what breaks the partition.
"""

import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Circle

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from togetherflow.simulator import TogetherFlowSimulator, _seed_numba_rng
from togetherflow.initialization import initialize_beacons
from scenarios import make_fixed_prior, ROOM, HALF_X, HALF_Y, REPULSION_RADIUS, REPULSION_GAIN

OUT = pathlib.Path(__file__).parent.parent / "outputs" / "scenarios" / "figures"

NUM_AGENTS = 48
NUM_BEACONS = 8
SNAPSHOT_S = 3.0                 # when to sample the scene
NOISE = 0.60                     # upper tail of Beta(2,5); see note in main()
BEACON_SEED = 4417               # controls the beacon layout only
TRAIL = 30                       # timesteps of trajectory history to draw

# Palette lifted from the reference figure.
BG = "#EAEAF2"
ROOM_FILL = "#C6D5E3"
AGENT_EDGE = "#3B2A63"
BEACON_ON = "#EE2C8B"
BEACON_OFF = "#6B7FD7"
ATTEND = "#EE6FB0"
CLUSTER = "#E060A8"


def select_beacons(positions, beacons, strengths, alpha):
    """Reproduce the kernel's selection: argmax_b  s_b^alpha / d_ib."""
    d = np.linalg.norm(positions[:, None, :] - beacons[None, :, :], axis=-1) + 1e-8
    return np.argmax(strengths[None, :] ** alpha / d, axis=1)


def draw(ax, title, positions, rotations, beacons, strengths, alpha):
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])

    final = positions[-1]
    chosen = select_beacons(final, beacons, strengths, alpha)

    # Room, as a rounded panel rather than a hard outline.
    ax.add_patch(FancyBboxPatch(
        (-HALF_X, -HALF_Y), ROOM[0], ROOM[1],
        boxstyle="round,pad=0,rounding_size=0.9",
        facecolor=ROOM_FILL, edgecolor="none", alpha=0.75, zorder=1,
    ))

    # Attention clusters: agents sharing a beacon, drawn as a soft disc so the
    # grouping reads before any individual agent does.
    for b in range(len(beacons)):
        members = np.where(chosen == b)[0]
        if len(members) < 3:
            continue
        centre = final[members].mean(axis=0)
        # 75th percentile, not the maximum: a group strung out along a wall has
        # one far member that would otherwise inflate the disc across the room
        # and stop it describing where the group actually is.
        spread = np.percentile(np.linalg.norm(final[members] - centre, axis=1), 75)
        spread = min(spread, 3.2)      # a disc larger than the room describes nothing
        ax.add_patch(Circle(
            centre, spread + 0.35, facecolor=CLUSTER, alpha=0.15,
            edgecolor=CLUSTER, linestyle="--", linewidth=1.0, zorder=2,
        ))

    # Trails, one pastel per agent.
    trail_colors = plt.cm.Set3(np.linspace(0, 1, NUM_AGENTS))
    hist = positions[-TRAIL:]
    for a in range(NUM_AGENTS):
        ax.plot(hist[:, a, 0], hist[:, a, 1], color=trail_colors[a],
                lw=1.1, alpha=0.75, zorder=3, solid_capstyle="round")

    # Attention lines from each agent to the beacon it selected.
    for a in range(NUM_AGENTS):
        b = beacons[chosen[a]]
        ax.plot([final[a, 0], b[0]], [final[a, 1], b[1]],
                color=ATTEND, lw=0.7, alpha=0.45, linestyle="--", zorder=2)

    # Headings.
    ax.quiver(
        final[:, 0], final[:, 1], np.cos(rotations[-1]), np.sin(rotations[-1]),
        color=AGENT_EDGE, width=0.0042, headwidth=3.6, headlength=4.0,
        scale=34, zorder=5, alpha=0.95,
    )
    ax.scatter(final[:, 0], final[:, 1], s=34, facecolor="white",
               edgecolor=AGENT_EDGE, linewidth=1.3, zorder=6)

    # Beacons: attended ones large and hot, ignored ones small and cool. The
    # marker size also carries salience, so a bright unattended beacon is visible
    # as a failure of the selection rule rather than an absence of one.
    counts = np.bincount(chosen, minlength=len(beacons))
    for b, (pos, s, n) in enumerate(zip(beacons, strengths, counts)):
        attended = n > 0
        ax.scatter(*pos, s=120 + 60 * s if attended else 46,
                   c=BEACON_ON if attended else BEACON_OFF,
                   edgecolor="white", linewidth=1.2 if attended else 0.6,
                   zorder=7)
        ax.annotate(f"{n}" if attended else "", pos, textcoords="offset points",
                    xytext=(0, -3), ha="center", va="center",
                    fontsize=7.5, color="white", zorder=8, weight="bold")

    ax.set_title(title, fontsize=12, color="#333333", pad=10)
    ax.set_aspect("equal")

    lim = np.abs(beacons).max() + 1.5
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    return counts


def main():
    rng_seed = 20260808
    # initialize_beacons runs under numba, whose RNG is independent of NumPy's
    # Python-level state — seeding only np.random leaves the beacon layout
    # different on every run.
    _seed_numba_rng(rng_seed)

    # Beacons outside the room, at a range that keeps them on the page. The
    # rejection sampler guarantees none land inside.
    _seed_numba_rng(BEACON_SEED)
    beacons = initialize_beacons(NUM_BEACONS, room_sensing_range=14.0, room_size=ROOM)
    beacons = np.asarray(beacons, dtype=np.float64)

    # One clearly brighter beacon, so the salience panel has something to show.
    strengths = np.ones(NUM_BEACONS)
    strengths[3] = 3.0
    strengths[6] = 2.2

    panels = [
        ("Nearest beacon  ($\\alpha = 0$)", 0.0),
        ("Salience-weighted  ($\\alpha = 1.2$)", 1.2),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.6))
    fig.patch.set_facecolor("white")

    for ax, (title, alpha) in zip(axes, panels):
        sim = TogetherFlowSimulator(
            num_agents=NUM_AGENTS,
            num_beacons=NUM_BEACONS,
            room_size=ROOM,
            dt=0.1,
            time_horizon=SNAPSHOT_S,
            output_mode="raw",
            prior=make_fixed_prior(0.75, 2.0, 1.0, NOISE),
            relative_heading=True,
            seed=rng_seed,
            fixed_beacons=beacons,
            beacon_strengths=strengths,
            salience_sensitivity=alpha,
            repulsion_radius=REPULSION_RADIUS,
            repulsion_gain=REPULSION_GAIN,
            diffusive_heading=True,
        )
        out = sim.sample(1)
        counts = draw(ax, title, out["positions"][0], out["rotations"][0],
                      beacons, strengths, alpha)
        print(f"{title:<40} beacons attended: {int((counts > 0).sum())}/{NUM_BEACONS}  "
              f"agents per beacon: {counts.tolist()}")

    fig.suptitle(f"{NUM_AGENTS} agents attending to {NUM_BEACONS} beacons",
                 fontsize=15, color="#222222")
    # The generative model has several switches now (heading update, noise
    # channel, separation, boundary), and a figure that does not name them cannot
    # be told apart from one produced by the published model.
    fig.text(
        0.5, 0.055,
        f"$t = {SNAPSHOT_S:g}$ s   ·   $w = 0.75$, $r = 2.0$, $v = 1.0$, "
        f"$\\eta = {NOISE:g}$ as heading diffusion   ·   "
        f"separation $\\rho = {REPULSION_RADIUS:g}$, $\\kappa = {REPULSION_GAIN:g}$   ·   "
        "reflecting walls, beacons outside the room",
        ha="center", va="top", fontsize=9.5, color="#666666",
    )
    fig.tight_layout()
    path = OUT / "attention_8_beacons.png"
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
