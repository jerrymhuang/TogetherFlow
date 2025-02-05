import numpy as np

from numba import njit


@njit
def count_neighbors(self_position, other_positions, sensing_radius=1.5):
    """
    Helper function that counts the number of neighbors

    Parameters
    ----------
    self_position   : np.ndarray of size (2)
        The position of the agent itself
    other_positions : np.ndarray of size (num_agents, 2)
        The positions of all agents
    sensing_radius  : float, default: 1.5
        The sensing radius of the agent

    Returns
    -------
    num_neighbors   : int, default: 0
        The number of neighbors within the agent's sensing radius.
    """

    num_neighbors = 0

    for i in range(len(other_positions)):
        dx = other_positions[i, 0] - self_position[0]
        dy = other_positions[i, 1] - self_position[1]
        d = (dx ** 2 + dy ** 2) ** 0.5

        if d <= sensing_radius and d > 0:
            num_neighbors += 1

    return num_neighbors

@njit
def bound(direction):

    if direction > np.pi:
        d = direction - 2 * np.pi
    elif direction < -np.pi:
        d = direction + 2 * np.pi
    else:
        d = direction

    return d


@njit
def world2local(
        agent_position,
        agent_rotation,
        beacon_position
):
    """
    Convert the orientation of beacon in world coordinates
    to its relative direction in agent coordinates

    Parameters
    ----------
    agent_position : np.ndarray
        Position of agent in world coordinates
    beacon_position : np.ndarray
        Position of beacon in world coordinates
    agent_rotation : np.float32
        Rotation of agent in world coordinates

    Returns
    -------
    beacon_direction : np.ndarray
        Direction of beacon in agent coordinates
    """

    # Compute beacon direction in world coordinates
    beacon_orientation = beacon_position - agent_position

    # Rotation matrix to convert beacon orientation to agent coordinates
    rotation_matrix = np.array([[np.cos(agent_rotation.item()), np.sin(agent_rotation.item())],
                                [-np.sin(agent_rotation.item()), np.cos(agent_rotation.item())]], dtype=np.float32)

    # Apply the rotation to get the beacon direction in local coordinates
    beacon_direction = rotation_matrix @ beacon_orientation
    return beacon_direction.astype(np.float32)


@njit
def relative_angle(v):
    # Calculate the angle of the vector relative to the agent's orientation
    return np.arctan2(v[1], v[0])


@njit
def adaptive_drift_rate(angle, drift_rate=0.1):
    # Drift rate is proportional to the relative angle, scaled by a maximum drift rate
    return np.sign(angle) * drift_rate
