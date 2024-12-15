import numpy as np
from numba import njit


@njit
def external_influence(
        agent_position,
        beacon_position,
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
    direction = beacon_direction + (np.random.random() - 0.5) * noise_amplitude
    # direction = beacon_direction + np.random.vonmises(0., 8.) * noise_amplitude

    # Convert the angle to a unit vector in 2D space
    v = np.array([np.cos(direction), np.sin(direction)], dtype=np.float32)

    return v


@njit
def internal_influence(
        self_position,
        other_positions,
        other_rotations,
        sensing_radius=1.5,
        focus=0.125,
        noise=False
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

        if sensing_radius >= d > 0:
            neighbor_rotations.append(other_rotations[i])

    if len(neighbor_rotations) == 0:
        return np.array([0.0, 0.0], dtype=np.float32)

    neighbor_rotations = np.array(neighbor_rotations)
    averaged_rotation = np.sum(neighbor_rotations) / len(neighbor_rotations)

    if noise:
        deviation = (np.random.random() - 0.5) * focus
    else:
        deviation = np.random.vonmises(mu=0., kappa=4.) * focus
    direction = averaged_rotation + deviation

    v = np.array([np.cos(direction), np.sin(direction)], dtype=np.float32)

    return v
