import numpy as np

from numba import njit
from influences import position_influence, rotation_influence
from utils import adaptive_drift_rate, bound


@njit
def rotational_update(
        agent_position,
        agent_rotation,
        beacon_position,
        drift_rate = np.pi * 0.1,
        dt = 0.1,
        noise_amplitude = 0.01
):

    rotate_vec = rotation_influence(agent_position, agent_rotation, beacon_position)

    v = adaptive_drift_rate(rotate_vec, drift_rate)

    noise = np.random.normal(0., noise_amplitude)

    new_agent_rotation = bound(agent_rotation + v * dt + noise)

    return new_agent_rotation


@njit
def look_at_beacon(
        agent_rotation,
        beacon_influence,
        dt=0.1,
        drift_rate=0.1,
        noise_amplitude=0.001,
        timesteps=100
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

    # Initialize the agent as not being in a steady state (i.e., not fixated to a beacon)
    steady_state = False

    for t in range(1, timesteps):
        # Calculate the relative angle between the agent's current orientation and
        # the direction of the beacon relative to the agent
        rotation_diff = (beacon_influence - rotations[t - 1]) % (2 * np.pi) - np.pi

        # Calculate the angle to rotate based upon the relative angle
        reorientation_angle = drift_rate * rotation_diff

        # If steady state is reached, then the agent will only be influenced by noise.
        # Otherwise, the reorientation happens until the steady state is reached.
        if steady_state:
            rotations[t] = beacon_influence + (np.random.random() - 0.5) * noise_amplitude
        else:
            rotations[t] = rotations[t - 1] + reorientation_angle * dt + noise_amplitude * np.random.randn()

            if np.abs(rotation_diff) < noise_amplitude:
                steady_state = True

    return rotations


@njit
def walk_to_beacon(
        agent_position,
        beacon_position,
        beacon_influence=None,
        drift_rate=0.5,
        dt=0.1,
        noise_amplitude=None,
        timesteps=1001,
        bounded=False,
        room_size=(8., 10.),
):
    """
    Simulate an agent's locomotion pattern influenced by a beacon.

    Parameters
    ----------
    agent_position : np.ndarray
        The starting position of the agent.
    beacon_position : np.ndarray
        The position of the beacon.
    beacon_influence : np.ndarray
        The influence of the beacon.
    drift_rate : float
        The drift rate of the agent.
    dt : float
        The time interval per time step for the simulation.
    noise_amplitude : float
        The amplitude of the noise.
    timesteps : int
        The number of timesteps to simulate.
    bounded : bool
        Whether the agent is bounded.
    room_size : tuple
        The size of the room.

    Returns
    -------
    positions : np.ndarray
        The positions of the agent's locomotion pattern as time series influenced by a beacon.
    """

    # Initialize a numpy array for the position time series
    positions = np.zeros((timesteps, 2), dtype=np.float32)
    positions[0] = agent_position
    steady_state = False
    steady_state_position = np.empty(2, dtype=np.float32)

    # Calculate the influence of beacons if there is none
    if beacon_influence is None:
        beacon_influence = rotation_influence(agent_position, beacon_position)

    for t in range(1, timesteps):
        # Calculate noise for each timestep
        noise = noise_amplitude * (np.random.random(size=2) - 0.5)

        if steady_state:
            # If the agent reaches a steady state, then it would stay at the vicinity
            # of its steady state position.
            if np.linalg.norm(beacon_position - positions[t - 1]) < noise_amplitude * 2.:
                positions[t] = beacon_position + noise
            else:
                positions[t] = steady_state_position + noise
        else:
            # Otherwise, position should be updated so that the agent continues to approach the beacon
            drift = drift_rate * beacon_influence * dt
            positions[t] = positions[t - 1] + drift + noise

            # If we assume the agent positions to be bounded, then we check it against the boundary
            # If it exceeds the boundary before reaching a beacon, then it reaches a steady state
            if bounded:
                if np.abs(positions[t, 0]) > room_size[0] * 0.5 or np.abs(positions[t, 1]) > room_size[1] * 0.5:
                    steady_state = True
                    steady_state_position = positions[t]

            # If the agent reaches a beacon, it also reaches a steady state
            if np.linalg.norm(beacon_position - positions[t]) < noise_amplitude * 2.:
                steady_state = True
                steady_state_position = beacon_position

    return positions


@njit
def move_to_beacon(
        agent_position,
        beacon_position,
        noise_amplitude=0.01,
        drift_rate=0.5,
        dt = 0.1,
        proximity = 1.,
        timesteps=1001
):
    """

    """

    # Initialize a numpy array for the position time series
    positions = np.zeros((timesteps, 2), dtype=np.float32)
    positions[0] = agent_position

    for t in range(1, timesteps):
        # Update beacon direction (as angles)
        beacon_direction = rotation_influence(positions[t-1], beacon_position)

        noise = noise_amplitude * (np.random.random(size=2) - 0.5)
        drift = drift_rate * beacon_direction * dt

        positions[t] = positions[t-1] + drift + noise

        if np.linalg.norm(beacon_position - positions[t]) < proximity:
            positions = beacon_position + (np.random.random(size=2) - 0.5) * proximity

    return positions


