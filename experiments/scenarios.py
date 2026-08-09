"""Navigation scenarios for the R1 response.

R1 asks for concrete navigation examples rather than inference diagnostics:
crossing groups, obstacle avoidance, group formation around a beacon, and
exiting a room. These are demonstrations of the generative model, so the
parameters are fixed rather than sampled and the quantities of interest are
behavioural (collisions, clearances, exit times) rather than recovery-based.

Each scenario is run twice by `run_scenarios.py` — separation off, which is the
published model, and separation on — so every figure and every number in the
report is a paired comparison attributable to the collision term alone.
"""

from dataclasses import dataclass, field

import numpy as np
from numba import njit

ROOM = (8., 10.)
HALF_X, HALF_Y = ROOM[0] * 0.5, ROOM[1] * 0.5

# Personal space and body size. BODY_DIAMETER is the distance below which two
# agents are counted as having collided; REPULSION_RADIUS is how far out the
# separation term starts to act. Both are in room units, where the room is 8x10.
BODY_DIAMETER = 0.4

# Separation strong enough to open personal space but not to break the group up.
# Beacons sit outside the room, so a converging group consolidates against a wall
# with only half the space it would have around a free-standing point; at
# kappa=1.5 that pressure fragments a 30-agent group into six clusters, while at
# kappa=0.6 it stays a single cluster with a neighbour density close to the
# published model's. STRONG_* keeps the over-dispersed setting available as the
# contrast that shows where the mechanism breaks.
REPULSION_RADIUS = 0.4
REPULSION_GAIN = 0.6

STRONG_RADIUS = 0.6
STRONG_GAIN = 1.5


def make_fixed_prior(w, r, v, noise):
    """A 'prior' that always returns the same parameter vector.

    Scenarios demonstrate behaviour at a known operating point, so the generative
    model is run at fixed theta. Keeping the simulator's prior interface means
    the scenarios exercise exactly the same code path as the inference runs.
    """
    @njit
    def _prior():
        return np.array([w, r, v, noise], dtype=np.float32)
    return _prior


@dataclass
class Scenario:
    slug: str
    title: str
    addresses: str
    question: str            # what the figure is supposed to answer

    num_agents: int
    beacons: np.ndarray                       # (B, 2) fixed positions
    init_positions: np.ndarray                # (A, 2)
    init_rotations: np.ndarray                # (A,)

    w: float = 0.75
    r: float = 2.0
    v: float = 1.0
    # Median of the new reference prior Beta(2,2). eta is a heading diffusion
    # coefficient now, so the old 0.1 sits in the far lower tail and would show
    # the scenarios under near-deterministic motion the arms never see.
    noise: float = 0.5

    obstacles: np.ndarray | None = None       # (K, 3) as (x, y, radius)
    beacon_assignment: np.ndarray | None = None
    door_wall: int = -1
    door_center: float = 0.0
    door_half_width: float = 0.0

    time_horizon: float = 40.0
    dt: float = 0.1
    seed: int = 20260808

    # Groups for the crossing diagnostic; empty when the scenario has one group.
    group_labels: np.ndarray | None = None

    @property
    def prior(self):
        return make_fixed_prior(self.w, self.r, self.v, self.noise)

    @property
    def num_beacons(self):
        return int(self.beacons.shape[0])


def _grid_cluster(n, x_range, y_range, rng):
    """Scatter n agents uniformly in a rectangle."""
    xs = rng.uniform(x_range[0], x_range[1], size=n)
    ys = rng.uniform(y_range[0], y_range[1], size=n)
    return np.stack([xs, ys], axis=-1)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Crossing groups — R1 point 1 and point 3
# ─────────────────────────────────────────────────────────────────────────────
#
# Two groups start against opposite walls and are given opposing goals, so their
# straight-line paths intersect in the middle of the room. This is the scenario
# the reviewer asked about directly ("two groups of agents cross paths while
# moving toward opposite beacons").
#
# Note this REQUIRES scripted goals. Under nearest-beacon selection the room is
# partitioned into Voronoi cells around the beacons and every agent moves into
# its own cell, so paths from different cells never intersect — the absence of
# crossings in the published Figure 7 is a structural consequence of the
# selection rule, not a property of the trajectories that happened to be drawn.

def _crossing_groups():
    rng = np.random.default_rng(11)
    n_per = 16
    left  = _grid_cluster(n_per, (-3.6, -2.4), (-2.0, 2.0), rng)
    right = _grid_cluster(n_per, (2.4, 3.6), (-2.0, 2.0), rng)
    positions = np.concatenate([left, right], axis=0)

    # Facing their targets: left group east, right group west.
    rotations = np.concatenate([np.zeros(n_per), np.full(n_per, np.pi)])

    # Beacons are virtual and always outside the room, so each group walks to the
    # far wall rather than to a point it can stand on.
    beacons = np.array([[7.0, 0.0], [-7.0, 0.0]])
    # Left group -> beacon 0 (east), right group -> beacon 1 (west).
    assignment = np.concatenate([np.zeros(n_per, dtype=np.int64),
                                 np.ones(n_per, dtype=np.int64)])
    labels = np.concatenate([np.zeros(n_per, dtype=np.int64),
                             np.ones(n_per, dtype=np.int64)])

    return Scenario(
        slug="crossing-groups",
        title="Two groups crossing with opposing goals",
        addresses="R1 point 1, R1 point 3",
        question=(
            "When two groups must pass through each other, do they interpenetrate "
            "as overlapping point particles, or do they resolve the encounter?"
        ),
        num_agents=2 * n_per,
        beacons=beacons,
        init_positions=positions,
        init_rotations=rotations,
        beacon_assignment=assignment,
        group_labels=labels,
        time_horizon=30.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Obstacle field — R1 point 3
# ─────────────────────────────────────────────────────────────────────────────

def _obstacle_field():
    rng = np.random.default_rng(12)
    n = 24
    positions = _grid_cluster(n, (-3.5, 3.5), (-4.5, -3.0), rng)
    rotations = np.full(n, np.pi / 2)          # facing the beacon

    beacons = np.array([[0.0, 8.0]])
    obstacles = np.array([
        [-2.0,  0.5, 0.9],
        [ 1.6, -0.8, 1.1],
        [ 2.2,  2.4, 0.7],
        [-1.4,  3.0, 0.6],
    ])

    return Scenario(
        slug="obstacle-field",
        title="Group traversing a field of obstacles",
        addresses="R1 point 3",
        question=(
            "Do agents route around solid obstacles, or does the model rely on "
            "the obstacle being invisible to the dynamics?"
        ),
        num_agents=n,
        beacons=beacons,
        init_positions=positions,
        init_rotations=rotations,
        obstacles=obstacles,
        time_horizon=40.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Beacon formation — R1 point 3
# ─────────────────────────────────────────────────────────────────────────────

def _beacon_formation():
    rng = np.random.default_rng(13)
    n = 30
    positions = _grid_cluster(n, (-3.7, 3.7), (-4.7, 4.7), rng)
    rotations = rng.uniform(0, 2 * np.pi, size=n)

    # Beyond the east wall. Because the beacon is virtual the group cannot reach
    # it and instead consolidates against the wall facing it, which is the
    # physical analogue of "formation around a beacon" in an immersive room —
    # and a harsher crowding test than a free-standing point, since the wall
    # removes half the space the group could otherwise spread into.
    beacons = np.array([[8.5, 0.0]])

    return Scenario(
        slug="beacon-formation",
        title="Group formation at the wall facing a beacon",
        addresses="R1 point 3",
        question=(
            "Convergence on one region is the worst case for crowding. Does the "
            "group consolidate into a plausible cluster, or collapse into a pile?"
        ),
        num_agents=n,
        beacons=beacons,
        init_positions=positions,
        init_rotations=rotations,
        time_horizon=40.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Room exit — R1 point 3
# ─────────────────────────────────────────────────────────────────────────────

def _room_exit():
    rng = np.random.default_rng(14)
    n = 28
    positions = _grid_cluster(n, (-3.5, 3.5), (-4.5, 2.0), rng)
    rotations = rng.uniform(0, 2 * np.pi, size=n)

    # One beacon just outside the door, so the whole group funnels through a gap
    # narrower than the group is wide — the congestion case.
    beacons = np.array([[0.0, 6.5]])

    return Scenario(
        slug="room-exit",
        title="Group exiting a room through a single door",
        addresses="R1 point 3",
        question=(
            "Does the group form a queue at a door narrower than itself, and does "
            "everyone actually get out?"
        ),
        num_agents=n,
        beacons=beacons,
        init_positions=positions,
        init_rotations=rotations,
        door_wall=2,                 # +y wall
        door_center=0.0,
        door_half_width=0.7,
        time_horizon=50.0,
    )


ALL_SCENARIOS = [
    _crossing_groups(),
    _obstacle_field(),
    _beacon_formation(),
    _room_exit(),
]

BY_SLUG = {s.slug: s for s in ALL_SCENARIOS}
