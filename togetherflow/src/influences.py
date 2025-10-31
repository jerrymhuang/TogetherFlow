import numpy as np
from numba import njit
from .utils import count_neighbors, bound_agent_position

@njit
def external_influence(
    agent_position,
    beacon_position,
    noise=False,
    noise_amplitude=0.01
):
    """
    Generate a drift-diffusion vector in 2D space for a single agent
    based on a target location (in this case, the position of a beacon).

    Parameters
    ----------
    agent_position : np.ndarray
        The position of the agent.
    beacon_position : np.ndarray
        The position of the target beacon.
    noise_amplitude : float, optional
        The dispersion of a von Mises distribution for rotational noise influenced by the neighbors.
        The higher the value is, the less perturbation there would be.
    noise: bool, optional
        Whether the focus is interpreted as noise amplitude.

    Returns
    -------
    np.ndarray
        A 2D vector representing the drift-diffusion process towards the target (beacon).
    """
    # Calculate the angle towards the beacon (in radian)
    beacon_direction = np.arctan2(
        beacon_position[1] - agent_position[1],
        beacon_position[0] - agent_position[0]
    )

    # Generate a random direction with drift around the target angle
    if noise:
        beacon_direction = beacon_direction + (np.random.random() - 0.5) * noise_amplitude
        # beacon_direction = beacon_direction + np.random.vonmises(0., 8.) * noise_amplitude

    # Convert the angle to a unit vector in 2D space
    v = np.array([np.cos(beacon_direction), np.sin(beacon_direction)], dtype=np.float32)

    return v


@njit
def internal_influence(
    self_position,
    other_positions,
    other_rotations,
    sensing_radius=1.5,
    focus=0.01,
    noise=False
):
    """
    Generate an influence vector for a single agent
    based on the angular component of the Vicsek model.

    Parameters
    ----------
    self_position : np.ndarray of shape (2,)
        A 2D vector representing the position of the agent
    other_positions : np.ndarray of shape (2,)
        A 2D vector representing the positions of the neighboring agents.
    other_rotations : np.ndarray of shape (2,)
        A 2D vector representing the rotations of the neighboring agents.
    sensing_radius : float
        The sensing radius within which agents interact with their neighbors.
    focus : float, optional
        The dispersion of a von Mises distribution for rotational noise influenced by the neighbors.
        The higher the value is, the less perturbation there would be.
    noise: bool, optional
        Whether the focus is interpreted as noise amplitude.

    Returns
    -------
    np.ndarray
        A 2D unit vector representing the averaged influence direction with added noise.
    """

    neighbor_rotations = []

    for i in range(len(other_positions)):
        dx = other_positions[i, 0] - self_position[0]
        dy = other_positions[i, 1] - self_position[1]
        d = (dx ** 2 + dy ** 2) ** 0.5

        if d <= sensing_radius:
            neighbor_rotations.append(other_rotations[i])

    if len(neighbor_rotations) == 0:
        return np.array([0.0, 0.0], dtype=np.float32)

    neighbor_rotations = np.array(neighbor_rotations)
    averaged_rotation = np.sum(neighbor_rotations) / len(neighbor_rotations)

    if noise:
        deviation = (np.random.random() - 0.5) * focus
    else:
        deviation = np.random.normal(loc=0.0, scale=focus)
    direction = averaged_rotation + deviation

    v = np.array([np.cos(direction), np.sin(direction)], dtype=np.float32)

    return v


@njit
def combined_influences(
    agent_positions: np.ndarray = None,
    agent_rotations: np.ndarray = None,
    beacon_positions: np.ndarray = None,
    room_size: tuple = (8, 10),
    velocity: float = 1.0,
    sensing_radius: float = 2.5,
    dt: float = 0.1,
    influence_weight: float = 0.5,
    internal_focus: float = 0.1
):
    """
    Update the positions and orientations of a single agent
    based on velocity and influence vectors.

    Parameters
    ----------
    agent_positions : np.ndarray
        Current positions of the agents.
    agent_rotations : np.ndarray
        Current orientations of the agents.
    beacon_positions : np.ndarray
        Positions of the beacons.
    room_size : tuple
        Size of the room's boundary.
    velocity : float, optional
        The speed at which agents move (default is 1.0).
    sensing_radius : float, optional
        The sensing radius within which agents interact with their neighbors.
    dt : float, optional
        The time step for updating positions and orientations (default is 0.1).
    influence_weight : float, optional
        The weight of influence_vector1 in determining new orientations (default is 0.7).
    internal_focus : float, optional
        Concentration of the agent's rotational noise influenced by the neighbors

    Returns
    -------
    tuple of np.ndarray
        Updated positions (np.ndarray) and orientations (np.ndarray) of the agents.
    """

    assert (len(agent_positions) == len(agent_rotations))

    num_agents = agent_positions.shape[0]
    num_beacons = beacon_positions.shape[0]

    # Create new numpy arrays for the updated agent positions and rotations
    new_agent_positions = np.zeros((num_agents, 2))
    new_agent_rotations = np.zeros((num_agents,))
    num_neighbors = np.zeros((num_agents,))
    average_dists = np.zeros((num_agents,))
    #max_dists = np.zeros((num_agents,))

    for i in range(num_agents):

        num_neighbors[i], average_dists[i] = count_neighbors(agent_positions[i], agent_positions)

        # Generate the ddm vector for the agent based on its closest beacon
        distance_to_beacon = []

        for b in range(num_beacons):
            bx = beacon_positions[b, 0] - agent_positions[i, 0]
            by = beacon_positions[b, 1] - agent_positions[i, 1]
            distance_to_beacon.append((bx * bx + by * by) ** 0.5)

        beacon_id = np.argmin(np.array(distance_to_beacon))

        ddm_vector = external_influence(
            agent_positions[i],
            beacon_positions[beacon_id],
            # focus=external_focus
        )

        # Generate the vicsek vector for the agent based on its neighbors (all agents)
        vicsek_vector = internal_influence(
            self_position=agent_positions[i],
            other_positions=agent_positions,
            other_rotations=agent_rotations,
            sensing_radius=sensing_radius,
            focus=internal_focus
        )

        # Update orientations based on two influence vectors
        ddm_influence = np.arctan2(ddm_vector[1], ddm_vector[0])
        vicsek_influence = np.arctan2(vicsek_vector[1], vicsek_vector[0])

        # Combine influences to update orientations with different weights
        new_agent_rotations[i] = agent_rotations[i] + (
                    influence_weight * ddm_influence + (1 - influence_weight) * vicsek_influence) * dt

        # Ensure orientations are within the range [0, 2*pi]
        new_agent_rotations[i] = np.mod(new_agent_rotations[i], 2 * np.pi)

        # Update positions based on current orientations
        new_agent_positions[i, 0] = agent_positions[i, 0] + velocity * np.cos(new_agent_rotations[i].item()) * dt
        new_agent_positions[i, 1] = agent_positions[i, 1] + velocity * np.sin(new_agent_rotations[i].item()) * dt

        new_agent_positions[i] = bound_agent_position(new_agent_positions[i], room_size=room_size)

    return new_agent_positions, new_agent_rotations, num_neighbors, average_dists#, max_dists
