import numpy as np

from numba import njit
from influences import (
    position_influence,
    rotation_influence,
    alignment_influence,
    cohesion_influence
)
from utils import adaptive_drift_rate, bound


@njit
def look_at_beacon(
    agent_position,
    agent_rotation,
    beacon_position,
    drift=0.5,
    noise=0.1,
    timesteps=1000,
    dt=0.1
):
    """
    Simulate reorientation as influenced by the relative position
    between an agent and its target beacon.

    Parameters
    ----------
    agent_rotation : np.ndarray or float
        Rotation of the agent
    beacon_influence : np.ndarray or float
        Influence of the beacon
    dt : float
        Time interval per time step
    drift_rate : float
        Rotational drift rate of the agent, interpreted as the speed of head movement
    noise_amplitude : float
        Noise amplitude of the agent's rotation
    timesteps : int
        Number of time steps to simulate

    Returns
    -------
    np.ndarray or float
        Time series of agent rotation influenced by the beacon
    """

    # Revert orientation vector to angular value
    # beacon_orientation = np.arctan2(beacon_influence[1], beacon_influence[0])

    # Initialize time series of the agent's rotation
    rotations = np.zeros((timesteps, 1), dtype=np.float32)
    rotations[0] = agent_rotation


    for t in range(1, timesteps):
        # Calculate the relative angle between the agent's current orientation and
        # the direction of the beacon relative to the agent
        direction = rotation_influence(agent_position, rotations[t-1], beacon_position, noise)
        rotations[t] = (rotations[t - 1] + direction * drift * dt)


    # Normalization
    rotations = rotations % (2. * np.pi)
    return rotations


@njit
def move_to_beacon(
    agent_position,
    beacon_position,
    drift=0.5,
    noise=0.1,
    timesteps=1000,
    dt = 0.1
):
    positions = np.zeros((timesteps, 2), dtype=np.float32)
    positions[0] = agent_position

    for i in range(1, timesteps):
        direction = position_influence(positions[i - 1], beacon_position, noise)
        positions[i] = positions[i - 1] + direction * drift * dt + (np.random.random(size=2) - 0.5) * noise

    return positions


@njit
def look_with_neighbors(
    agent_positions,
    agent_rotations,
    sensing_radius = 1.5,
    drift=0.5,
    noise=0.1,
    timesteps=1000,
    dt = 0.1
):
    assert len(agent_positions) == len(agent_rotations)
    num_agents = len(agent_positions)

    rotations = np.zeros((timesteps, num_agents, 1), dtype=np.float32)
    rotations[0] = agent_rotations

    for t in range(1, timesteps):
        for a in range(num_agents):

            direction = alignment_influence(
                agent_positions[a],
                agent_positions,
                rotations[t-1],
                sensing_radius,
                noise
            )

            rotations[t, a] = rotations[t-1, a] + direction * drift * dt

    # Normalization
    rotations = rotations % (2. * np.pi)
    return rotations


@njit
def move_with_neighbors(
    agent_position,
    sensing_radius=1.5,
    noise=0.1,
    timesteps=1000,
    dt=0.1
):

    positions = np.zeros((timesteps, 2), dtype=np.float32)


@njit
def individual_motion(
    agent_position,
    agent_rotation,
    beacon_position,
    position_drift=0.5,
    rotation_drift=0.5,
    position_noise=0.1,
    rotation_noise=0.1,
    timesteps=1000,
    dt=0.1
):
    positions = np.zeros((timesteps, 2), dtype=np.float32)
    rotations = np.zeros((timesteps, 1), dtype=np.float32)
    positions[0] = agent_position
    rotations[0] = agent_rotation

    for t in range(1, timesteps):
        walk_direction = position_influence(positions[t-1], beacon_position, position_noise)
        look_direction = rotation_influence(positions[t-1], rotations[t-1], beacon_position, rotation_noise)

        positions[t] = positions[t - 1] + walk_direction * position_drift * dt
        rotations[t] = rotations[t - 1] + look_direction * rotation_drift * dt

    # normalization
    rotations = rotations % (2. * np.pi)

    return positions, rotations


@njit
def collective_motion():
    raise NotImplementedError


@njit
def main_simulator(

):
    raise NotImplementedError


# @njit
# def rotational_update(
#         agent_position,
#         agent_rotation,
#         beacon_position,
#         drift_rate = np.pi * 0.1,
#         dt = 0.1,
#         noise_amplitude = 0.01
# ):
#
#     rotate_vec = rotation_influence(agent_position, agent_rotation, beacon_position)
#
#     v = adaptive_drift_rate(rotate_vec, drift_rate)
#
#     noise = np.random.normal(0., noise_amplitude)
#
#     new_agent_rotation = bound(agent_rotation + v * dt + noise)
#
#     return new_agent_rotation
