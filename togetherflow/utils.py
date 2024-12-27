import numpy as np

from numba import njit


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
