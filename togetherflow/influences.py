import numpy as np
from numba import njit

from utils import bound, world2local, relative_angle


@njit
def position_influence(
        agent_position,
        beacon_position,
        noise=0.01
):
    """
    Generate a position drift-diffusion vector in 2D space for a single agent at a single time step
    based on a target location (in this case, the position of a beacon).

    Parameters
    ----------
    agent_position : np.ndarray
        The position of the agent.
    beacon_position : np.ndarray
        The position of the target beacon.
    noise : float, optional
        The amplitude of rotational noise influenced by the neighbors.

    Returns
    -------
    np.ndarray
        A 2D vector representing the drift-diffusion process towards the target (beacon).
    """

    beacon_direction = beacon_position - agent_position

    influence = np.arctan2(beacon_direction[1], beacon_direction[0]) + (np.random.random() - 0.5) * noise
    # Noise is already here. No need to add it again in the simulation.

    return influence


@njit
def rotation_influence(
        agent_position,
        agent_rotation,
        beacon_position,
        noise=0.01
):
    """
    Generate a rotation drift-diffusion vector in 2D space for a single agent at a single time step
    based on a target orientation (in this case, the orientation of a beacon).

    Parameters
    ----------
    agent_position : np.ndarray
        The position of the agent.
    agent_rotation : np.ndarray
        The orientation of the agent.
    beacon_position : np.ndarray
        The position of the target beacon.

    Returns
    -------
    np.ndarray
        A 2D rotation vector representing the drift-diffusion process towards the target (beacon).
    """

    beacon_orientation = world2local(
        agent_position, agent_rotation, beacon_position
    )

    influence = relative_angle(beacon_orientation) + (np.random.random() - 0.5) * noise

    return influence


@njit
def alignment_influence(
        self_position,
        other_positions,
        other_rotations,
        sensing_radius=1.5,
        noise=0.1
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

        # Exclude 0 so that the agent itself is not taken into account.
        if d <= sensing_radius and d > 0:
            neighbor_rotations.append(other_rotations[i])

    ## No neighbor, no influence
    if len(neighbor_rotations) == 0:
        return np.array([0., 0.], dtype=np.float32)

    # Compute neighbor rotation average
    neighbor_rotations = np.array(neighbor_rotations)
    averaged_rotation = np.sum(neighbor_rotations) / len(neighbor_rotations)

    # Add noise
    direction = averaged_rotation + (np.random.random() - 0.5) * noise

    # Decompose
    influence = np.array([np.cos(direction), np.sin(direction)], dtype=np.float32)

    return influence


@njit
def cohesion_influence(
        self_position,
        other_positions,
        sensing_radius=1.5,
        noise=0.01
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

    neighbor_positions = []

    for i in range(len(other_positions)):
        dx = other_positions[i, 0] - self_position[0]
        dy = other_positions[i, 1] - self_position[1]
        d = (dx ** 2 + dy ** 2) ** 0.5

        if d <= sensing_radius and d > 0:
            neighbor_positions.append(other_positions[i])

    if len(neighbor_positions) == 0:
        return np.array([0.0, 0.0], dtype=np.float32)

    neighbor_positions = np.array(neighbor_positions)
    averaged_position = np.sum(neighbor_positions) / len(neighbor_positions)

    direction = averaged_position + (np.random.random() - 0.5) * noise

    influence = np.array([np.cos(direction), np.sin(direction)], dtype=np.float32)

    return influence
