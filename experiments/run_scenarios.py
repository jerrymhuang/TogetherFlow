"""Run the navigation scenarios and report trajectory realism.

    uv run python experiments/run_scenarios.py            # all scenarios
    uv run python experiments/run_scenarios.py crossing-groups
    uv run python experiments/run_scenarios.py --list

This is deliberately not part of `run_variant.py`. That harness reports SBI
diagnostics — recovery, calibration, contraction — which say nothing about
whether a trajectory looks like people walking. R1 asks about the trajectories
themselves, so the output here is behavioural: collision counts, clearances,
exit times, and the figures a reader can check by eye.

Every scenario runs twice, with the separation term off and on, and the report
is the paired difference.
"""

import argparse
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from togetherflow.simulator import TogetherFlowSimulator
from scenarios import (
    ALL_SCENARIOS, BY_SLUG, BODY_DIAMETER, REPULSION_RADIUS, REPULSION_GAIN,
    STRONG_RADIUS, STRONG_GAIN, HALF_X, HALF_Y, ROOM,
)

ROOT = pathlib.Path(__file__).parent.parent
OUT = ROOT / "outputs" / "scenarios"

CONDITIONS = [
    ("published", 0.0, 0.0),
    ("separation", REPULSION_RADIUS, REPULSION_GAIN),
    ("separation (strong)", STRONG_RADIUS, STRONG_GAIN),
]


def simulate(scenario, repulsion_radius, repulsion_gain):
    sim = TogetherFlowSimulator(
        num_agents=scenario.num_agents,
        num_beacons=scenario.num_beacons,
        room_size=ROOM,
        dt=scenario.dt,
        time_horizon=scenario.time_horizon,
        output_mode="raw",
        prior=scenario.prior,
        relative_heading=True,
        seed=scenario.seed,
        fixed_beacons=scenario.beacons,
        init_positions=scenario.init_positions,
        init_rotations=scenario.init_rotations,
        obstacles=scenario.obstacles,
        beacon_assignment=scenario.beacon_assignment,
        door_wall=scenario.door_wall,
        door_center=scenario.door_center,
        door_half_width=scenario.door_half_width,
        repulsion_radius=repulsion_radius,
        repulsion_gain=repulsion_gain,
        diffusive_heading=True,
    )
    out = sim.sample(1)
    return out["positions"][0], out["rotations"][0]      # (T, A, 2), (T, A)


def pairwise_distances(positions):
    """(T, A, A) pairwise distances, with the diagonal set to infinity."""
    diff = positions[:, :, None, :] - positions[:, None, :, :]
    d = np.linalg.norm(diff, axis=-1)
    idx = np.arange(d.shape[1])
    d[:, idx, idx] = np.inf
    return d


def compute_metrics(scenario, positions, rotations):
    T, A, _ = positions.shape
    d = pairwise_distances(positions)

    # Collisions: unordered pairs closer than one body diameter. Counting pairs
    # rather than agents avoids double-counting each encounter.
    iu = np.triu_indices(A, k=1)
    pair_d = d[:, iu[0], iu[1]]
    collision_rate = float((pair_d < BODY_DIAMETER).mean())
    min_clearance = float(pair_d.min())

    # Nearest-neighbour clearance is the quantity a reader can sanity-check
    # against personal-space intuitions.
    nn_d = d.min(axis=2)
    p05_clearance = float(np.percentile(nn_d, 5))

    inside = (np.abs(positions[..., 0]) <= HALF_X) & (np.abs(positions[..., 1]) <= HALF_Y)
    outside_fraction = float((~inside).mean())

    # Vicsek order parameter: 1 is a perfectly aligned flock, 0 is disorder.
    order = np.abs(np.exp(1j * rotations).mean(axis=1)).mean()

    metrics = {
        "collision_rate": collision_rate,
        "min_clearance": min_clearance,
        "p05_clearance": p05_clearance,
        "outside_room": outside_fraction,
        "order_parameter": float(order),
    }

    if scenario.obstacles is not None and len(scenario.obstacles):
        pen = np.zeros((T, A), dtype=bool)
        for cx, cy, rad in scenario.obstacles:
            dist = np.linalg.norm(positions - np.array([cx, cy]), axis=-1)
            # A small tolerance: the projection puts agents exactly on the
            # surface, and floating point can land a hair inside.
            pen |= dist < (rad - 1e-6)
        metrics["obstacle_penetration"] = float(pen.mean())

    if scenario.door_wall >= 0:
        # Exited = outside the room at the final step, having left via the door.
        exited = ~inside
        metrics["exited_fraction"] = float(exited[-1].mean())
        # First exit time per agent, in seconds; NaN for agents still inside.
        first = np.where(exited.any(axis=0), exited.argmax(axis=0) * scenario.dt, np.nan)
        if np.isfinite(first).any():
            metrics["median_exit_time_s"] = float(np.nanmedian(first))
        else:
            metrics["median_exit_time_s"] = float("nan")

    if scenario.group_labels is not None:
        # Did the groups actually swap sides? Mean x per group, start vs end.
        g0 = scenario.group_labels == 0
        g1 = scenario.group_labels == 1
        start_gap = positions[0, g0, 0].mean() - positions[0, g1, 0].mean()
        end_gap = positions[-1, g0, 0].mean() - positions[-1, g1, 0].mean()
        metrics["groups_swapped"] = float(np.sign(start_gap) != np.sign(end_gap))
        # Cross-group clearance during the encounter is the number that matters:
        # within-group crowding is expected, between-group interpenetration is not.
        cross = d[:, g0][:, :, g1]
        metrics["min_cross_group_clearance"] = float(cross.min())
        metrics["cross_group_collision_rate"] = float((cross < BODY_DIAMETER).mean())

    return metrics


def plot_scenario(scenario, runs, path):
    """Trajectories side by side, published against separation."""
    fig, axes = plt.subplots(1, len(runs), figsize=(5.2 * len(runs), 6.0), sharex=True, sharey=True)
    if len(runs) == 1:
        axes = [axes]

    for ax, (label, positions, _rot, _m) in zip(axes, runs):
        # Room outline, with the door drawn as a gap in the wall.
        ax.plot([-HALF_X, HALF_X, HALF_X, -HALF_X, -HALF_X],
                [-HALF_Y, -HALF_Y, HALF_Y, HALF_Y, -HALF_Y],
                color="0.25", lw=1.5, zorder=1)
        if scenario.door_wall == 2:
            ax.plot([scenario.door_center - scenario.door_half_width,
                     scenario.door_center + scenario.door_half_width],
                    [HALF_Y, HALF_Y], color="white", lw=4.0, zorder=2)

        if scenario.obstacles is not None:
            for cx, cy, rad in scenario.obstacles:
                ax.add_patch(plt.Circle((cx, cy), rad, color="0.75", zorder=2))

        colors = (["#3b7dd8", "#d1495b"] if scenario.group_labels is not None
                  else ["#3b7dd8"])
        labels = (scenario.group_labels if scenario.group_labels is not None
                  else np.zeros(scenario.num_agents, dtype=int))

        for a in range(positions.shape[1]):
            ax.plot(positions[:, a, 0], positions[:, a, 1],
                    color=colors[labels[a]], lw=0.7, alpha=0.55, zorder=3)
        ax.scatter(positions[0, :, 0], positions[0, :, 1], s=14,
                   c="0.35", marker="o", zorder=4, label="start")
        ax.scatter(positions[-1, :, 0], positions[-1, :, 1], s=22,
                   c="#111111", marker="x", zorder=5, label="end")
        ax.scatter(scenario.beacons[:, 0], scenario.beacons[:, 1], s=140,
                   marker="*", c="#e8a33d", edgecolor="0.2", zorder=6, label="beacon")

        ax.set_title(label)
        ax.set_aspect("equal")
        ax.margins(0.08)

    axes[0].legend(loc="lower left", fontsize=8, framealpha=0.9)
    fig.suptitle(f"{scenario.title}", y=0.98)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


METRIC_LABELS = {
    "collision_rate": "Collision rate (pairs < body diameter)",
    "min_clearance": "Minimum clearance",
    "p05_clearance": "5th pct nearest-neighbour clearance",
    "outside_room": "Fraction of agent-steps outside the room",
    "order_parameter": "Order parameter (mean)",
    "obstacle_penetration": "Fraction of agent-steps inside an obstacle",
    "exited_fraction": "Fraction exited by the final step",
    "median_exit_time_s": "Median exit time (s)",
    "groups_swapped": "Groups swapped sides (1 = yes)",
    "min_cross_group_clearance": "Minimum cross-group clearance",
    "cross_group_collision_rate": "Cross-group collision rate",
}


def write_report(results, path):
    L = ["# Navigation scenarios", ""]
    L.append(
        "Behavioural checks on the generative model, run at fixed parameters. "
        f"Two agents are counted as colliding below a body diameter of {BODY_DIAMETER} "
        "room units. `published` is the model without a separation term; "
        f"`separation` adds it at rho={REPULSION_RADIUS}, kappa={REPULSION_GAIN}, and "
        f"`separation (strong)` at rho={STRONG_RADIUS}, kappa={STRONG_GAIN} to show "
        "where the mechanism over-disperses. All three use the corrected reflecting "
        "boundary, and all beacons lie outside the room."
    )
    L.append("")

    for scenario, runs in results:
        L.append(f"## {scenario.title}")
        L.append("")
        L.append(f"*Addresses:* {scenario.addresses}  ")
        L.append(f"*Question:* {scenario.question}")
        L.append("")
        L.append(f"![{scenario.slug}](figures/{scenario.slug}.png)")
        L.append("")

        keys = list(runs[0][3].keys())
        L.append("| Metric | " + " | ".join(r[0] for r in runs) + " |")
        L.append("|---|" + "---|" * len(runs))
        for k in keys:
            label = METRIC_LABELS.get(k, k)
            cells = []
            for _, _, _, m in runs:
                val = m[k]
                cells.append("—" if val != val else f"{val:.3f}")
            L.append(f"| {label} | " + " | ".join(cells) + " |")
        L.append("")

    path.write_text("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slug", nargs="?", help="scenario to run (default: all)")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for s in ALL_SCENARIOS:
            print(f"  {s.slug:<20} {s.addresses:<22} {s.title}")
        return

    scenarios = [BY_SLUG[args.slug]] if args.slug else ALL_SCENARIOS
    if args.slug and args.slug not in BY_SLUG:
        ap.error(f"unknown scenario '{args.slug}'. Use --list.")

    (OUT / "figures").mkdir(parents=True, exist_ok=True)

    results = []
    for scenario in scenarios:
        runs = []
        for label, rho, kappa in CONDITIONS:
            positions, rotations = simulate(scenario, rho, kappa)
            metrics = compute_metrics(scenario, positions, rotations)
            runs.append((label, positions, rotations, metrics))
            print(f"  {scenario.slug:<18} {label:<11} "
                  + "  ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
        plot_scenario(scenario, runs, OUT / "figures" / f"{scenario.slug}.png")
        results.append((scenario, runs))

    write_report(results, OUT / "report.md")
    print(f"\nwrote {OUT / 'report.md'}")


if __name__ == "__main__":
    main()
