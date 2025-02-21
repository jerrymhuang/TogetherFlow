import numpy as np
from numba import njit


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

@njit
def bound_agent_rotation(direction):

    if (direction > np.pi):
        d = direction - 2 * np.pi
    elif direction < -np.pi:
        d = direction + 2 * np.pi
    else:
        d = direction

    return d
