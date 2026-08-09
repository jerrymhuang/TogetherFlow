import numpy as np
from numba import njit

from .utils import bound_agent_state


@njit
def wrap_angle(angle):
    """Wrap an angle to (-pi, pi]. Required whenever angles are differenced."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


@njit
def external_influence(agent_position, beacon_position):
    """
    Unit vector pointing from an agent toward its nearest beacon.

    Parameters
    ----------
    agent_position  : np.ndarray of shape (2,)
    beacon_position : np.ndarray of shape (2,)

    Returns
    -------
    np.ndarray of shape (2,)
    """
    direction = np.arctan2(
        beacon_position[1] - agent_position[1],
        beacon_position[0] - agent_position[0],
    )
    return np.array([np.cos(direction), np.sin(direction)], dtype=np.float32)


@njit
def internal_influence(neighbor_rotations, focus):
    """
    Vicsek alignment vector from pre-collected neighbor rotations.

    Parameters
    ----------
    neighbor_rotations : np.ndarray of shape (N,)
        Rotations of neighbors already within the sensing radius.
    focus : float
        Standard deviation of Gaussian rotational noise.

    Returns
    -------
    np.ndarray of shape (2,)
    """
    if len(neighbor_rotations) == 0:
        return np.array([0.0, 0.0], dtype=np.float32)
    avg = np.sum(neighbor_rotations) / len(neighbor_rotations)
    direction = avg + np.random.normal(0.0, focus)
    return np.array([np.cos(direction), np.sin(direction)], dtype=np.float32)


@njit
def combined_influences(
    agent_positions,
    agent_rotations,
    beacon_positions,
    beacon_strengths,
    salience_sensitivity,
    reference_radii,
    room_size=(8., 10.),
    velocity=1.0,
    sensing_radius=2.5,
    dt=0.1,
    influence_weights=None,
    internal_focus=0.1,
    relative_heading=False,
    repulsion_radius=0.0,
    repulsion_gain=0.0,
    obstacles=None,
    max_turn_rate=6.283185307179586,
    door_wall=-1,
    door_center=0.0,
    door_half_width=0.0,
    beacon_assignment=None,
    diffusive_heading=False,
):
    """
    Advance all agents by one time step under beacon attraction and Vicsek alignment.

    Performs a single neighbor scan per agent, reusing the result for both the
    summary statistics (num_neighbors, average_distance) and the alignment update.

    Beacon selection uses the unified score s_b^salience_sensitivity / d_ib:
      salience_sensitivity = 0.0  ->  nearest beacon (default)
      salience_sensitivity = 1.0  ->  saliency-weighted (s_b / d_ib)
      other values                ->  continuous interpolation / extrapolation

    Parameters
    ----------
    agent_positions  : np.ndarray of shape (A, 2)
    agent_rotations  : np.ndarray of shape (A,)
    beacon_positions : np.ndarray of shape (B, 2)
    beacon_strengths : np.ndarray of shape (B,) — per-beacon salience weights
    salience_sensitivity : float — saliency exponent; 0.0 = nearest, 1.0 = saliency
    reference_radii  : np.ndarray of shape (R,) — FIXED radii for r-free neighbour
        counts. These do not depend on `sensing_radius`, so the resulting channel
        can be observed without knowing the parameter being inferred. Pass an
        empty array to skip.
    room_size        : tuple (width, height)
    velocity         : float
    sensing_radius   : float  — used consistently for both stats and dynamics
    dt               : float
    influence_weights: np.ndarray of shape (A,) — per-agent weight on beacon
        attraction vs. Vicsek alignment. An array rather than a scalar so that
        complete pooling (every entry identical) and partial pooling (entries
        drawn per agent from a population distribution) run through one code
        path, and any difference between them is the parameter, not the code.
    internal_focus   : float  — std of Gaussian rotational noise
    repulsion_radius : float  — personal-space radius rho. Neighbours and obstacle
        surfaces closer than this push the agent away, with a strength that grows
        linearly from 0 at rho to 1 on contact. Zero disables separation entirely,
        which reproduces the published dynamics.
    repulsion_gain   : float  — gain kappa on the separation term. Unlike the
        beacon/alignment pair this sits *outside* the convex combination: an agent
        about to collide should be able to override both of them, which is the
        whole point of a collision-avoidance term.
    obstacles        : np.ndarray of shape (K, 3) — circular obstacles as
        (centre_x, centre_y, radius). Repel like neighbours do, and additionally
        block: a step that ends inside one is projected back to its surface.
        Pass an empty (0, 3) array for an empty room.
    max_turn_rate    : float  — cap on |delta| in rad/s. Agents cannot turn
        arbitrarily fast, and without a cap the unbounded separation term can spin
        an agent through a large angle in one step. The default 2*pi does not bind
        for the published dynamics, where |delta| <= pi by construction.
    door_wall        : int    — see `bound_agent_state`; -1 for a sealed room.
    door_center      : float
    door_half_width  : float
    diffusive_heading: bool   — where eta enters the model.

        False (published): eta is the standard deviation of a Gaussian
        perturbation applied to the *neighbour-average bearing* inside the Vicsek
        term. That perturbation is then weighted by (1 - w) and scaled by dt, and
        redrawn independently each step, so it low-pass filters away instead of
        accumulating: raising eta from 0.29 to 5.0 changes the mean per-step turn
        by under 10%, and non-monotonically. An agent with no neighbours, or one
        with w = 1, is perfectly deterministic. This is almost certainly why eta
        shows near-zero posterior contraction.

        True: eta is the diffusion coefficient of the heading process,

            theta_t = theta_{t-1} + (w d_beacon + (1-w) d_vicsek) dt
                      + eta sqrt(dt) xi_t,     xi_t ~ N(0, 1)

        which is the drift-diffusion form the paper describes in
        "External Influence: Individual Motion as Spatial Drift Diffusion".
        The increment accumulates as a random walk, applies whether or not the
        agent has neighbours, and is not attenuated by (1 - w) — so the beacon
        channel is stochastic too, as a drift-diffusion process requires. The
        alignment target is left unperturbed in this mode so eta is not counted
        twice.

    Returns
    -------
    new_positions  : np.ndarray of shape (A, 2)
    new_rotations  : np.ndarray of shape (A,)
    num_neighbors  : np.ndarray of shape (A,)
    average_dists  : np.ndarray of shape (A,)
    radii_counts   : np.ndarray of shape (A, R) — neighbour counts at each fixed
        reference radius; independent of `sensing_radius`.
    """
    num_agents = agent_positions.shape[0]
    num_beacons = beacon_positions.shape[0]
    num_radii = reference_radii.shape[0]
    num_obstacles = obstacles.shape[0]
    separating = repulsion_gain > 0.0 and repulsion_radius > 0.0

    new_positions = np.zeros((num_agents, 2))
    new_rotations = np.zeros((num_agents,))
    num_neighbors = np.zeros((num_agents,))
    average_dists = np.zeros((num_agents,))
    radii_counts = np.zeros((num_agents, num_radii))

    for i in range(num_agents):

        # Single neighbor scan — shared by statistics, Vicsek update, and the
        # fixed-radius counts. The r-free counts ride along on the same pass.
        nbr_rots = []
        nbr_dists = []
        # Accumulated "get away from here" vector, summed over crowding
        # neighbours and obstacle surfaces on the same pass as everything else.
        rep_x = 0.0
        rep_y = 0.0
        for j in range(num_agents):
            dx = agent_positions[j, 0] - agent_positions[i, 0]
            dy = agent_positions[j, 1] - agent_positions[i, 1]
            d = (dx ** 2 + dy ** 2) ** 0.5
            if d > 0.0:
                for k in range(num_radii):
                    if d <= reference_radii[k]:
                        radii_counts[i, k] += 1.0
            if 0.0 < d <= sensing_radius:
                nbr_rots.append(agent_rotations[j])
                nbr_dists.append(d)
            if separating and 0.0 < d <= repulsion_radius:
                # Unit vector away from j, weighted by how far inside personal
                # space j has come. Linear in d so the term is continuous at rho.
                strength = 1.0 - d / repulsion_radius
                rep_x -= dx / d * strength
                rep_y -= dy / d * strength

        if separating:
            for k in range(num_obstacles):
                ox = agent_positions[i, 0] - obstacles[k, 0]
                oy = agent_positions[i, 1] - obstacles[k, 1]
                d_centre = (ox * ox + oy * oy) ** 0.5
                # Measure to the surface, not the centre, so a wide pillar and a
                # narrow post feel the same at the same clearance.
                d_surface = d_centre - obstacles[k, 2]
                if d_surface <= repulsion_radius and d_centre > 0.0:
                    if d_surface < 0.0:
                        d_surface = 0.0
                    strength = 1.0 - d_surface / repulsion_radius
                    rep_x += ox / d_centre * strength
                    rep_y += oy / d_centre * strength

        num_neighbors[i] = float(len(nbr_rots))
        if len(nbr_dists) > 0:
            average_dists[i] = np.mean(np.array(nbr_dists))
        else:
            average_dists[i] = 0.0

        # Beacon selection: score = s_b^salience_sensitivity / d_ib  (0 -> nearest)
        #
        # A scripted assignment overrides the rule. This exists for the
        # navigation scenarios: under nearest-beacon selection the room is
        # partitioned into Voronoi cells around the beacons and every agent moves
        # *into* its own cell, so trajectories originating in different cells
        # cannot intersect. Two groups with opposing goals are therefore not
        # expressible by the selection rule at all, and scripting the goal is the
        # only way to stage the crossing the reviewers asked to see.
        if beacon_assignment.shape[0] > 0 and beacon_assignment[i] >= 0:
            beacon_id = beacon_assignment[i]
        else:
            beacon_id = 0
            best_score = -np.inf
            for b in range(num_beacons):
                bx = beacon_positions[b, 0] - agent_positions[i, 0]
                by = beacon_positions[b, 1] - agent_positions[i, 1]
                d_b = (bx * bx + by * by) ** 0.5 + 1e-8
                score = beacon_strengths[b] ** salience_sensitivity / d_b
                if score > best_score:
                    best_score = score
                    beacon_id = b

        ddm_vec = external_influence(agent_positions[i], beacon_positions[beacon_id])

        if len(nbr_rots) > 0:
            # In diffusive mode the alignment target is clean and eta is applied
            # once, to the heading state below; otherwise eta perturbs the target
            # here, which is the published behaviour.
            align_noise = 0.0 if diffusive_heading else internal_focus
            vicsek_vec = internal_influence(np.array(nbr_rots), align_noise)
        else:
            vicsek_vec = np.array([0.0, 0.0], dtype=np.float32)

        ddm_angle = np.arctan2(ddm_vec[1], ddm_vec[0])
        vicsek_angle = np.arctan2(vicsek_vec[1], vicsek_vec[0])

        if relative_heading:
            # Steer by the wrapped difference between target bearing and current
            # heading. This is rotationally invariant: rotating the whole scene
            # rotates the trajectories and changes nothing else.
            d_beacon = wrap_angle(ddm_angle - agent_rotations[i])
            if len(nbr_rots) > 0:
                d_vicsek = wrap_angle(vicsek_angle - agent_rotations[i])
            else:
                # No neighbours means no alignment torque. Leaving this at
                # wrap(0 - theta) would instead steer the agent toward world-east,
                # reintroducing exactly the bias this mode exists to remove.
                d_vicsek = 0.0
            w_i = influence_weights[i]
            delta = w_i * d_beacon + (1.0 - w_i) * d_vicsek

            # Separation rides on top of the convex combination rather than
            # inside it. The magnitude of the accumulated vector carries how
            # crowded the agent is, so one neighbour at the edge of personal
            # space barely deflects it while a wall of neighbours dominates.
            rep_mag = (rep_x * rep_x + rep_y * rep_y) ** 0.5
            if rep_mag > 0.0:
                rep_angle = np.arctan2(rep_y, rep_x)
                d_rep = wrap_angle(rep_angle - agent_rotations[i])
                delta += repulsion_gain * rep_mag * d_rep
        else:
            # Legacy behaviour: absolute world-frame bearings used directly as a
            # turn rate. Retained so published results remain reproducible, but
            # note it is not rotationally invariant — an agent whose beacon lies
            # due east receives no steering at all.
            delta = influence_weights[i] * ddm_angle + (1.0 - influence_weights[i]) * vicsek_angle

        # Agents cannot pivot arbitrarily fast. This does not bind for the
        # published dynamics (|delta| <= pi there) but keeps the unbounded
        # separation term from spinning an agent through a large angle in a step.
        if delta > max_turn_rate:
            delta = max_turn_rate
        elif delta < -max_turn_rate:
            delta = -max_turn_rate

        # Drift, then the Wiener increment. The turn-rate cap above applies to the
        # drift only: it models how fast an agent can deliberately pivot, and
        # clipping the diffusion as well would truncate the noise distribution.
        heading = agent_rotations[i] + delta * dt
        if diffusive_heading:
            heading += internal_focus * np.sqrt(dt) * np.random.normal(0.0, 1.0)
        rotation = np.mod(heading, 2.0 * np.pi)

        px = agent_positions[i, 0] + velocity * np.cos(rotation) * dt
        py = agent_positions[i, 1] + velocity * np.sin(rotation) * dt

        # Obstacles block regardless of whether separation is switched on:
        # solidity is geometry, not a behavioural parameter. A step that ends
        # inside a pillar is projected back onto its surface.
        for k in range(num_obstacles):
            ox = px - obstacles[k, 0]
            oy = py - obstacles[k, 1]
            d_centre = (ox * ox + oy * oy) ** 0.5
            if d_centre < obstacles[k, 2]:
                if d_centre > 0.0:
                    px = obstacles[k, 0] + ox / d_centre * obstacles[k, 2]
                    py = obstacles[k, 1] + oy / d_centre * obstacles[k, 2]
                else:
                    px = obstacles[k, 0] + obstacles[k, 2]

        new_positions[i, 0] = px
        new_positions[i, 1] = py
        new_positions[i], new_rotations[i] = bound_agent_state(
            agent_positions[i],
            new_positions[i],
            rotation,
            room_size=room_size,
            door_wall=door_wall,
            door_center=door_center,
            door_half_width=door_half_width,
        )

    return new_positions, new_rotations, num_neighbors, average_dists, radii_counts
