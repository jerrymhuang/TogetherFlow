import numpy as np
from numba import njit


@njit
def count_neighbors(self_position, other_positions, sensing_radius):
    """
    Count agents within sensing_radius and compute their average distance.

    Parameters
    ----------
    self_position   : np.ndarray of shape (2,)
    other_positions : np.ndarray of shape (N, 2)
    sensing_radius  : float

    Returns
    -------
    num_neighbors    : int
    average_distance : float
    """
    num_neighbors = 0
    distances = []

    for i in range(len(other_positions)):
        dx = other_positions[i, 0] - self_position[0]
        dy = other_positions[i, 1] - self_position[1]
        d = (dx ** 2 + dy ** 2) ** 0.5
        if 0.0 < d <= sensing_radius:
            distances.append(d)
            num_neighbors += 1

    if len(distances) > 0:
        average_distance = np.mean(np.array(distances))
    else:
        average_distance = 0.0

    return num_neighbors, average_distance


@njit
def bound_agent_position(agent_position, room_size=(8., 10.), boundary_noise=0.01):
    """
    Nudge an agent back inside the rectangular room boundary.

    Parameters
    ----------
    agent_position : np.ndarray of shape (2,)
    room_size      : tuple (width, height)
    boundary_noise : float
    """
    bounded_position = agent_position.copy()
    if np.abs(agent_position[0]) > room_size[0] * 0.5:
        bounded_position[0] = agent_position[0] + boundary_noise
    if np.abs(agent_position[1]) > room_size[1] * 0.5:
        bounded_position[1] = agent_position[1] + boundary_noise
    return bounded_position
