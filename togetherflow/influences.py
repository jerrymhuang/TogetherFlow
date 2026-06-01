import numpy as np
from numba import njit

from .utils import bound_agent_position


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
    room_size=(8., 10.),
    velocity=1.0,
    sensing_radius=2.5,
    dt=0.1,
    influence_weight=0.5,
    internal_focus=0.1,
):
    """
    Advance all agents by one time step under beacon attraction and Vicsek alignment.

    Performs a single neighbor scan per agent, reusing the result for both the
    summary statistics (num_neighbors, average_distance) and the alignment update.

    Parameters
    ----------
    agent_positions  : np.ndarray of shape (A, 2)
    agent_rotations  : np.ndarray of shape (A,)
    beacon_positions : np.ndarray of shape (B, 2)
    room_size        : tuple (width, height)
    velocity         : float
    sensing_radius   : float  — used consistently for both stats and dynamics
    dt               : float
    influence_weight : float  — weight on beacon attraction vs. Vicsek alignment
    internal_focus   : float  — std of Gaussian rotational noise

    Returns
    -------
    new_positions  : np.ndarray of shape (A, 2)
    new_rotations  : np.ndarray of shape (A,)
    num_neighbors  : np.ndarray of shape (A,)
    average_dists  : np.ndarray of shape (A,)
    """
    num_agents = agent_positions.shape[0]
    num_beacons = beacon_positions.shape[0]

    new_positions = np.zeros((num_agents, 2))
    new_rotations = np.zeros((num_agents,))
    num_neighbors = np.zeros((num_agents,))
    average_dists = np.zeros((num_agents,))

    for i in range(num_agents):

        # Single neighbor scan — shared by statistics and Vicsek update
        nbr_rots = []
        nbr_dists = []
        for j in range(num_agents):
            dx = agent_positions[j, 0] - agent_positions[i, 0]
            dy = agent_positions[j, 1] - agent_positions[i, 1]
            d = (dx ** 2 + dy ** 2) ** 0.5
            if 0.0 < d <= sensing_radius:
                nbr_rots.append(agent_rotations[j])
                nbr_dists.append(d)

        num_neighbors[i] = float(len(nbr_rots))
        if len(nbr_dists) > 0:
            average_dists[i] = np.mean(np.array(nbr_dists))
        else:
            average_dists[i] = 0.0

        # Nearest beacon
        beacon_id = 0
        min_dist = np.inf
        for b in range(num_beacons):
            bx = beacon_positions[b, 0] - agent_positions[i, 0]
            by = beacon_positions[b, 1] - agent_positions[i, 1]
            d_b = (bx * bx + by * by) ** 0.5
            if d_b < min_dist:
                min_dist = d_b
                beacon_id = b

        ddm_vec = external_influence(agent_positions[i], beacon_positions[beacon_id])

        if len(nbr_rots) > 0:
            vicsek_vec = internal_influence(np.array(nbr_rots), internal_focus)
        else:
            vicsek_vec = np.array([0.0, 0.0], dtype=np.float32)

        ddm_angle = np.arctan2(ddm_vec[1], ddm_vec[0])
        vicsek_angle = np.arctan2(vicsek_vec[1], vicsek_vec[0])

        new_rotations[i] = np.mod(
            agent_rotations[i] + (
                influence_weight * ddm_angle + (1.0 - influence_weight) * vicsek_angle
            ) * dt,
            2.0 * np.pi,
        )

        new_positions[i, 0] = agent_positions[i, 0] + velocity * np.cos(new_rotations[i]) * dt
        new_positions[i, 1] = agent_positions[i, 1] + velocity * np.sin(new_rotations[i]) * dt
        new_positions[i] = bound_agent_position(new_positions[i], room_size=room_size)

    return new_positions, new_rotations, num_neighbors, average_dists
