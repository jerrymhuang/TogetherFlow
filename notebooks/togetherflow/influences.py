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
    target_position : np.ndarray
        The position of the target beacon.
    focus : float, optional
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
    direction = beacon_direction + (np.random.random() - 0.5) * noise_amplitude
    # direction = beacon_direction + np.random.vonmises(0., 8.) * noise_amplitude

    # Convert the angle to a unit vector in 2D space
    v = np.array([np.cos(direction), np.sin(direction)], dtype=np.float32)

    return v

