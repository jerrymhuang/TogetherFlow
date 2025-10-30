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
    distances = []

    for i in range(len(other_positions)):
        dx = other_positions[i, 0] - self_position[0]
        dy = other_positions[i, 1] - self_position[1]
        d = (dx ** 2 + dy ** 2) ** 0.5

        if d <= sensing_radius and d > 0:
            distances.append(d)
            num_neighbors += 1

    average_distance = np.mean(np.array(distances)) if len(distances) > 0 else 0.0
    max_distance = np.max(distances) if len(distances) > 0 else 0.0

    return num_neighbors, average_distance, max_distance


@njit
def bound_agent_position(
        agent_position,
        room_size = (8., 10.),
        boundary_noise=0.01
):
    """
    Prevent the agent from going out of the room's boundary.

    Parameters
    ----------
    agent_position : np.ndarray
        Position of the agent
    room_size : tuple
        Size of the room's boundary


    """
    bounded_position = agent_position.copy()

    if isinstance(room_size, float):
        distance_to_room_center = agent_position[0] ** 2 + agent_position[1] ** 2
        if distance_to_room_center > room_size ** 2:
            bounded_position = agent_position / distance_to_room_center * room_size
    else:
        if np.abs(agent_position[0]) > room_size[0] * 0.5:
            bounded_position[0] = agent_position[0] + boundary_noise
        if np.abs(agent_position[1]) > room_size[1] * 0.5:
            bounded_position[1] = agent_position[1] + boundary_noise

    return bounded_position
