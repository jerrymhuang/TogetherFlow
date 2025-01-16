import numpy as np

from numba import njit

from initializations import (
    initialize_agents,
    initialize_beacons
)

from influences import (
    position_influence,
    rotation_influence,
    alignment_influence,
    cohesion_influence
)
# from utils import adaptive_drift_rate, bound


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
    agent_position : array_like
        Position of the agent
    agent_rotation : np.ndarray or float
        Rotation of the agent
    beacon_position : np.ndarray or float
        position of the beacon
    drift : float
        Rotational drift rate of the agent, interpreted as the speed of head movement
    noise : float
        Noise amplitude of the agent's rotation
    timesteps : int
        Number of time steps to simulate
    dt : float
        Time interval per time step

    Returns
    -------
    np.ndarray or float
        Time series of agent rotation influenced by the beacon
    """

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
    dt=0.1
):
    """
    Simulate motion (locomotion) as influenced by the relative position
    between an agent and its target beacon.

    Parameters
    ----------
    agent_position : np.ndarray or float
        Rotation of the agent
    beacon_position : np.ndarray or float
        Influence of the beacon
    drift : float
        Rotational drift rate of the agent, interpreted as the speed of head movement
    noise : float
        Noise amplitude of the agent's rotation
    timesteps : int
        Number of time steps to simulate
    dt : float
        Time interval per time step

    Returns
    -------
    np.ndarray or float
        Time series of agent rotation influenced by the beacon
    """
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
        sensing_radius=1.5,
        timesteps=100,
        dt=0.1,
        drift=0.5,
        noise=0.1
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
                rotations[t - 1],
                sensing_radius,
                noise
            )

            rotations[t, a] = rotations[t - 1, a] + direction * drift * dt

    rotations = rotations % (2. * np.pi)
    return rotations


@njit
def move_with_neighbors(
    agent_positions,
    sensing_radius=1.5,
    drift=0.5,
    noise=0.1,
    timesteps=1000,
    dt=0.1
):

    num_agents = agent_positions.shape[0]
    positions = np.zeros((timesteps, num_agents, 2), dtype=np.float32)
    positions[0] = agent_positions

    for t in range(1, timesteps):
        for a in range(num_agents):
            direction = cohesion_influence(
                agent_positions[a],
                agent_positions,
                sensing_radius,
                noise
            )

            positions[t, a] = positions[t - 1, a] + direction * drift * dt

    return positions


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
        move_direction = position_influence(positions[t-1], beacon_position, position_noise)
        look_direction = rotation_influence(positions[t-1], rotations[t-1], beacon_position, rotation_noise)

        positions[t] = positions[t - 1] + move_direction * position_drift * dt
        rotations[t] = rotations[t - 1] + look_direction * rotation_drift * dt

    # normalization
    rotations = rotations % (2. * np.pi)

    return positions, rotations


@njit
def collective_motion(
    agent_positions,
    agent_rotations,
    alignment_drift=0.5,
    cohesion_drift=0.5,
    alignment_noise=0.1,
    cohesion_noise=0.1,
    sensing_radius=1.5,
    timesteps=1000,
    dt=0.1
):

    assert len(agent_positions) == len(agent_rotations)
    num_agents = len(agent_positions)

    positions = np.zeros((timesteps, num_agents, 2), dtype=np.float32)
    rotations = np.zeros((timesteps, num_agents, 1), dtype=np.float32)
    positions[0] = agent_positions
    rotations[0] = agent_rotations

    for t in range(1, timesteps):
        for a in range(num_agents):

            move_direction = cohesion_influence(
                positions[t-1, a],
                positions[t-1],
                sensing_radius,
                cohesion_noise
            )

            look_direction = alignment_influence(
                positions[t-1, a],
                positions[t-1],
                rotations[t-1],
                sensing_radius,
                alignment_noise
            )

            positions[t, a] = positions[t - 1, a] + move_direction * cohesion_drift * dt
            rotations[t, a] = rotations[t - 1, a] + look_direction * alignment_drift * dt

    rotations = rotations % (2. * np.pi)
    return positions, rotations


@njit
def motion_simulation(
    theta=None,
    agent_positions=None,
    agent_rotations=None,
    beacon_positions=None,
    position_drift=0.5,
    rotation_drift=0.5,
    alignment_drift=0.5,
    cohesion_drift=0.5,
    position_noise=0.1,
    rotation_noise=0.1,
    alignment_noise=0.1,
    cohesion_noise=0.1,
    sensing_radius=1.5,
    influence_weight=0.5,
    timesteps=1000,
    dt=0.1
):

    assert len(agent_positions) == len(agent_rotations)
    num_agents = len(agent_positions)
    num_beacons = len(beacon_positions)

    if theta is not None:
        influence_weight = theta[0]
        sensing_radius = theta[1]
        # All drifts are the same, for now
        position_drift = theta[2]
        rotation_drift = theta[2]
        alignment_drift = theta[2]
        cohesion_drift = theta[2]

    # Initialization
    positions = np.zeros((timesteps, num_agents, 2), dtype=np.float32)
    rotations = np.zeros((timesteps, num_agents, 1), dtype=np.float32)
    positions[0] = agent_positions
    rotations[0] = agent_rotations

    for t in range(1, timesteps):
        for a in range(num_agents):

            distances_to_beacons = np.zeros((num_beacons, 1), dtype=np.float32)

            for b in range(num_beacons):
                bx = beacon_positions[b, 0] - positions[t, a, 0]
                by = beacon_positions[b, 1] - positions[t, a, 1]
                distances_to_beacons[b] = (bx * bx + by * by) ** 0.5

            beacon_id = np.argmin(distances_to_beacons)

            move_direction = position_influence(
                positions[t-1, a],
                beacon_positions[beacon_id],
                position_noise
            )

            look_direction = rotation_influence(
                positions[t-1, a],
                rotations[t-1, a],
                beacon_positions[beacon_id],
                rotation_noise
            )

            follow_direction = cohesion_influence(
                positions[t-1, a],
                positions[t-1],
                sensing_radius,
                cohesion_noise
            )

            align_direction = alignment_influence(
                positions[t-1, a],
                positions[t-1],
                rotations[t-1],
                sensing_radius,
                alignment_noise
            )

            position_shift = (
                influence_weight * move_direction * position_drift +
                (1 - influence_weight) * follow_direction * cohesion_drift
            )

            rotation_shift = (
                influence_weight * look_direction * rotation_drift +
                (1 - influence_weight) * align_direction * alignment_drift
            )

            positions[t, a] = positions[t - 1, a] + position_shift * dt
            rotations[t, a] = rotations[t - 1, a] + rotation_shift * dt

    rotations = rotations % (2. * np.pi)
    return positions, rotations


if __name__ == "__main__":
    agent_positions, agent_rotations = initialize_agents()
    beacon_positions = initialize_beacons()

    sim_positions, sim_rotations = motion_simulation(
        agent_positions=agent_positions,
        agent_rotations=agent_rotations,
        beacon_positions=beacon_positions
    )
