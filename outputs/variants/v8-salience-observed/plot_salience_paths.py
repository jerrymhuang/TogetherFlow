"""Render salience_paths.png — beacon salience paths and the agent response.

Run from the project root:

    uv run python outputs/variants/v8-salience-observed/plot_salience_paths.py

Top panel per trial: the four beacon salience paths on a log axis. The process
is log-OU, so log is its natural scale and s = 1 is the marginal median.
Bottom strip: the share of agents committed to each beacon, recomputed with the
same hysteresis rule `simulator_fun` applies. The pairing is the point — the
world changes first and the agents follow, all at once, because the salience
field is shared.

Tweak the KNOBS block below; everything downstream reads from it.
"""
import os

# Must precede the keras import chain; the project has no tensorflow installed.
os.environ.setdefault("KERAS_BACKEND", "jax")

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from togetherflow import TogetherFlowSimulator
from togetherflow.simulator import make_ou_salience_process
from togetherflow.priors import complete_pooling_prior

# ── KNOBS ────────────────────────────────────────────────────────────────────
# The three most-flipped knobs also read from the environment, so one script
# can render both figures without editing:
#   SP_TRIALS=1 SP_BEACONS=8 SP_OUT=salience_paths_8beacons.png uv run python ...
N_TRIALS    = int(os.environ.get("SP_TRIALS", 4))   # one block per trial
NUM_BEACONS = int(os.environ.get("SP_BEACONS", 4))
SEED        = 20260803       # v8 arm seed; change for a different draw

# v8-salience-observed generative settings. Keep these in sync with the Variant
# in experiments/variants.py, or the figure stops describing the arm.
SIGMA_S, TAU_S, MARGIN, ALPHA = 0.7, 40.0, 1.5, 1.0
NUM_AGENTS, DT, HORIZON = 49, 0.1, 60.0

OUT = pathlib.Path(__file__).parent / os.environ.get("SP_OUT", "salience_paths.png")

# Reference categorical palette in its documented order. Sliced to NUM_BEACONS.
# At 4 beacons this takes slots 1/2/3/4 — note that ordering, not the 1/2/3/7 an
# all-pairs form would need, because a line chart is scored on the ADJACENT
# pairlist. All 8 pass adjacent in light mode (worst CVD dE 9.1, normal-vision
# 19.6). Re-run scripts/validate_palette.js rather than eyeballing any change.
#
# Caveat at 8: ~7 is the documented ceiling for colour classes carrying meaning,
# so the 8-beacon figure leans on the direct labels for identity, not on hue
# alone. If it reads as spaghetti, the fix is small multiples — one panel per
# beacon with the other seven in grey — not more hues.
SERIES_ALL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
              "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SERIES = SERIES_ALL[:NUM_BEACONS]
# Slots 3, 4 and 5 sit below 3:1 contrast on this surface; direct labels are the
# required relief, which is why every line carries one.
SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#b5b4ae"

FIG_W, ROW_H = 9.6, 3.0      # inches per trial block
DPI = 150
# Header is a fixed number of INCHES, not a fraction: at one trial a fractional
# header ate the whole figure and the title, subtitle and legend overprinted.
# Two legend rows are needed past four series.
HEADER_H = 1.30 + (0.26 if NUM_BEACONS > 4 else 0.0)
FIG_H = HEADER_H + ROW_H * N_TRIALS

# Beacons are pinned so the selection rule can be replayed exactly from the
# recorded positions — the simulator does not return sampled beacon positions.
# The salience process itself does not depend on where the beacons are, so the
# top panels are unaffected by this choice; only the agent strips are.
BEACON_RING_RADIUS = 23.0    # well outside the 8x10 room; beacons are virtual
_ang = np.arange(NUM_BEACONS) * (2 * np.pi / NUM_BEACONS) + np.pi / NUM_BEACONS
FIXED_BEACONS = np.stack(
    [BEACON_RING_RADIUS * np.cos(_ang), BEACON_RING_RADIUS * np.sin(_ang)], axis=-1
).astype(np.float32)
# ─────────────────────────────────────────────────────────────────────────────


def simulate():
    sim = TogetherFlowSimulator(
        num_agents=NUM_AGENTS, num_beacons=NUM_BEACONS, dt=DT,
        time_horizon=HORIZON, output_mode="raw", prior=complete_pooling_prior,
        param_names=("w", "r", "v", "noise"),
        salience_process=make_ou_salience_process(
            num_beacons=NUM_BEACONS, dt=DT, sigma_s=SIGMA_S, tau_s=TAU_S),
        include_salience_paths=True, salience_sensitivity=ALPHA,
        switch_margin=MARGIN, beacon_spread=50.0, fixed_beacons=FIXED_BEACONS,
        relative_heading=True, diffusive_heading=True, seed=SEED,
    )
    d = sim.sample(batch_size=N_TRIALS)
    return np.asarray(d["positions"]), np.asarray(d["salience"])


def replay_targets(pos, sal):
    """Committed beacon per agent per step, mirroring simulator_fun's rule."""
    B, T, A, _ = pos.shape
    dist = np.linalg.norm(
        pos[:, :, :, None, :] - FIXED_BEACONS[None, None, None, :, :], axis=-1
    ) + 1e-8
    score = sal[:, :, None, :] ** ALPHA / dist
    tgt = np.empty((B, T, A), dtype=np.int64)
    tgt[:, 0, :] = np.argmax(score[:, 0], axis=-1)
    for k in range(1, T):
        cur = tgt[:, k - 1, :]
        cur_s = np.take_along_axis(score[:, k], cur[:, :, None], axis=-1)[:, :, 0]
        masked = score[:, k].copy()
        np.put_along_axis(masked, cur[:, :, None], -np.inf, axis=-1)
        chal = np.argmax(masked, axis=-1)
        chal_s = np.take_along_axis(score[:, k], chal[:, :, None], axis=-1)[:, :, 0]
        tgt[:, k, :] = np.where(chal_s > cur_s * MARGIN, chal, cur)
    return tgt


def main():
    pos, sal = simulate()
    tgt = replay_targets(pos, sal)
    T = pos.shape[1]
    t = np.arange(T) * DT
    share = np.stack([(tgt == b).mean(axis=2) for b in range(NUM_BEACONS)], axis=-1)
    n_sw = (tgt[:, 1:, :] != tgt[:, :-1, :]).sum(axis=1).mean(axis=1)

    # Limits from the data: a fixed guess clipped the tallest path.
    ylo, yhi = float(sal.min()) * 0.82, float(sal.max()) * 1.55

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=SURFACE)
    # Third row of each block is an invisible spacer, so the blocks breathe
    # while each salience panel stays glued to its own agent strip.
    gs = GridSpec(N_TRIALS * 3, 1, figure=fig,
                  height_ratios=[3.0, 0.85, 0.62] * N_TRIALS, hspace=0.0)

    for i in range(N_TRIALS):
        ax = fig.add_subplot(gs[3 * i, 0])
        ax_s = fig.add_subplot(gs[3 * i + 1, 0], sharex=ax)
        fig.add_subplot(gs[3 * i + 2, 0]).set_axis_off()

        for a in (ax, ax_s):
            a.set_facecolor(SURFACE)
            for side in ("top", "right"):
                a.spines[side].set_visible(False)
            a.spines["left"].set_color(MUTED)
            a.spines["bottom"].set_color(MUTED)
            a.tick_params(colors=INK2, labelsize=9, length=3, color=MUTED)

        ax.axhline(1.0, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.set_yscale("log")
        ticks = [v for v in (0.25, 0.5, 1, 2, 4, 8) if ylo <= v <= yhi]
        ax.set_yticks(ticks)
        ax.set_yticklabels(["%g" % v for v in ticks])
        # Log scale labels decade minors by default, which crowds the axis.
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_ylim(ylo, yhi)
        ax.grid(axis="y", color=MUTED, lw=0.6, alpha=0.35, zorder=0)
        ax.set_axisbelow(True)

        lw = 2.0 if NUM_BEACONS <= 4 else 1.6   # thinner marks when crowded
        for b in range(NUM_BEACONS):
            ax.plot(t, sal[i, :, b], color=SERIES[b], lw=lw, zorder=3,
                    solid_capstyle="round")

        # Direct labels — the relief the low-contrast hues require, and the
        # secondary encoding that carries identity when hue alone cannot. Nudged
        # apart in log space so they never overprint each other: with 8 beacons
        # several paths finish within a few percent of one another.
        finals = np.array([sal[i, -1, b] for b in range(NUM_BEACONS)])
        order = np.argsort(finals)
        # Min separation in log units, as a fraction of the axis range. Scaled
        # by trial count because a taller panel affords tighter fractions.
        gap = (np.log(yhi) - np.log(ylo)) * (0.035 if N_TRIALS > 1 else 0.055)
        placed = np.log(finals.copy())
        for j in range(1, NUM_BEACONS):
            lo_i, hi_i = order[j - 1], order[j]
            if placed[hi_i] - placed[lo_i] < gap:
                placed[hi_i] = placed[lo_i] + gap
        for b in range(NUM_BEACONS):
            ax.annotate(f"B{b + 1}", xy=(t[-1], np.exp(placed[b])),
                        xytext=(5, 0), textcoords="offset points",
                        color=SERIES[b], fontsize=9, fontweight="bold",
                        va="center", clip_on=False)

        ax.set_ylabel("salience  $s_b(t)$", color=INK2, fontsize=10)
        # Inside the axes: above would collide with the strip of the trial above.
        ax.text(0.011, 0.93, f"Trial {i + 1}", transform=ax.transAxes,
                color=INK, fontsize=11, fontweight="bold", va="top")
        ax.text(0.989, 0.93, f"{n_sw[i]:.1f} switches per agent",
                transform=ax.transAxes, color=INK2, fontsize=9.5,
                ha="right", va="top")
        plt.setp(ax.get_xticklabels(), visible=False)

        # 2px surface gaps between stacked bands.
        ax_s.stackplot(t, *[share[i, :, b] for b in range(NUM_BEACONS)],
                       colors=SERIES, edgecolor=SURFACE, linewidth=2, zorder=2)
        ax_s.set_ylim(0, 1)
        ax_s.set_yticks([0, 1])
        ax_s.set_yticklabels(["0", "all"])
        ax_s.set_ylabel("agents", color=INK2, fontsize=10)
        ax_s.margins(x=0)
        if i == N_TRIALS - 1:
            ax_s.set_xlabel("time (s)", color=INK2, fontsize=10)
        else:
            plt.setp(ax_s.get_xticklabels(), visible=False)

    # Header laid out in inches converted to figure fraction, so it holds at any
    # trial count rather than assuming a tall figure.
    def frac(inches_from_top):
        return 1.0 - inches_from_top / FIG_H

    handles = [plt.Line2D([], [], color=c, lw=2.5, solid_capstyle="round")
               for c in SERIES]
    fig.legend(handles, [f"Beacon {b + 1}" for b in range(NUM_BEACONS)],
               loc="upper center", bbox_to_anchor=(0.5, frac(0.80)), ncol=4,
               frameon=False, fontsize=10, labelcolor=INK2, columnspacing=2.4,
               handlelength=1.6)
    fig.text(0.5, frac(0.22), "Beacon salience paths and the agent response",
             ha="center", va="top", color=INK, fontsize=14, fontweight="bold")
    fig.text(0.5, frac(0.50),
             f"shared log-OU field, $\\sigma_s$={SIGMA_S}, $\\tau_s$={TAU_S:.0f}s, "
             f"$\\alpha$={ALPHA:.0f}, switch margin {MARGIN}×  ·  "
             f"{NUM_AGENTS} agents, {NUM_BEACONS} beacons, {HORIZON:.0f}s, "
             f"beacons fixed for replay",
             ha="center", va="top", color=INK2, fontsize=9.5)

    fig.subplots_adjust(top=frac(HEADER_H), bottom=0.50 / FIG_H,
                        left=0.088, right=0.955)
    fig.savefig(OUT, dpi=DPI, facecolor=SURFACE)
    print(f"wrote {OUT}")
    print("mean switches/agent per trial:", np.round(n_sw, 2))


if __name__ == "__main__":
    main()
