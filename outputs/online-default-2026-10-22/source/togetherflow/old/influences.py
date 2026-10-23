import numpy as np
from numba import njit

from .utils import world2local, relative_angle
# bound


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
    noise : float, optional
        The amplitude of rotational noise influenced by the neighbors.

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
    self_position : np.ndarray of shape (2)
        A 2D vector representing the position of the agent
    other_positions : np.ndarray of shape (2)
        A 2D vector representing the positions of the neighboring agents.
    other_rotations : np.ndarray of shape (2)
        A 2D vector representing the rotations of the neighboring agents.
    sensing_radius : float
        The sensing radius within which agents interact with their neighbors.
    noise: bool, optional
        Whether the focus is interpreted as noise amplitude.

    Returns
    -------
    np.ndarray
        A 2D unit vector representing the averaged influence direction with added noise.
    """

    assert other_positions.shape[0] == other_rotations.shape[0]
    num_agents = other_positions.shape[0]

    # Since the number of agents is known, instead of list initialization,
    # we can just initialize it directly as a zero array to account for the
    # maximum number of neighbors without getting into Numba troubles.
    # If a neighbor is not within the sensing radius, then their respective
    # value in this array would not change.
    neighbor_rotations = np.zeros((num_agents, 1))

    # The downside of this is that now we need a counter to keep track of
    # the neighbors.
    influence = 0.
    num_neighbors = 0

    for i in range(len(other_positions)):
        dx = other_positions[i, 0] - self_position[0]
        dy = other_positions[i, 1] - self_position[1]
        d = (dx ** 2 + dy ** 2) ** 0.5

        # Exclude 0 so that the agent itself is not taken into account.
        if 0 < d <= sensing_radius:
            neighbor_rotations[i] = other_rotations[i]
            num_neighbors += 1

    # No neighbor, no influence
    if num_neighbors != 0:
        # Compute neighbor rotation average
        averaged_rotation = np.sum(neighbor_rotations) / num_neighbors

        # Add noise
        influence = averaged_rotation + (np.random.random() - 0.5) * noise

    return influence


@njit
def cohesion_influence(
    self_position,
    other_positions,
    sensing_radius=1.5,
    noise=0.1
):
    """
    Generate an influence vector for a single agent
    based on the angular component of the Vicsek model.

    Parameters
    ----------
    self_position : np.ndarray of shape (2)
        A 2D vector representing the position of the agent
    other_positions : np.ndarray of shape (2)
        A 2D vector representing the positions of the neighboring agents.
    sensing_radius : float
        The sensing radius within which agents interact with their neighbors.
    noise: bool, optional
        Whether the focus is interpreted as noise amplitude.

    Returns
    -------
    np.ndarray
        A 2D unit vector representing the averaged influence direction with added noise.
    """

    num_agents = len(other_positions)
    # Since the number of agents is known, instead of list initialization,
    # we can just initialize it directly as a zero array to account for the
    # maximum number of neighbors without getting into Numba troubles.
    # If a neighbor is not within the sensing radius, then their respective
    # value in this array would not change.
    neighbor_positions = np.zeros((num_agents, 2), dtype=np.float32)

    # The downside of this is that now we need a counter to keep track of
    # the neighbors.
    influence = 0.
    num_neighbors = 0

    for i in range(len(other_positions)):
        dx = other_positions[i, 0] - self_position[0]
        dy = other_positions[i, 1] - self_position[1]
        d = (dx ** 2 + dy ** 2) ** 0.5

        if 0 < d <= sensing_radius:
            neighbor_positions[i] = other_positions[i]
            num_neighbors += 1

    if num_neighbors != 0:

        averaged_position = np.sum(neighbor_positions, axis=0) / num_neighbors
        averaged_position -= self_position

        influence = np.arctan2(
            averaged_position[1], averaged_position[0]
        ) + (np.random.random() - 0.5) * noise

    return influence


@njit
def separation_influence(
    self_position,
    other_positions,
    repulsion_radius=1.5,
    noise=0.1
):
    num_agents = len(other_positions)

    neighbor_positions = np.zeros((num_agents, 2), dtype=np.float32)

    influence = 0.0
    num_neighbors = 0

    for i in range(len(other_positions)):
        dx = other_positions[i, 0] - self_position[0]
        dy = other_positions[i, 1] - self_position[1]
        d = (dx ** 2 + dy ** 2) ** 0.5

        if 0 < d <= repulsion_radius:
            neighbor_positions[i] = other_positions[i]
            num_neighbors += 1

    return influence


# For debugging only
if __name__ == "__main__":
    pass
