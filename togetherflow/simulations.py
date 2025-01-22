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
        motion_vector = np.array([np.cos(direction), np.sin(direction)], dtype=np.float32)
        positions[i] = positions[i - 1] + motion_vector * drift * dt

    return positions


@njit
def look_with_neighbors(
        agent_positions,
        agent_rotations,
        sensing_radius=1.5,
        timesteps=1000,
        dt=0.1,
        drift=1.,
        noise=0.1
):
    """
    Simulate rotation (orientation) as influenced by the neighboring agents

    Parameters
    ----------
    agent_positions : np.ndarray or float
        Positions of the agents
    agent_rotations : np.ndarray or float
        Rotations of the agents
    sensing_radius : float
        The sensing radius of the agents
    timesteps : int
        Number of time steps to simulate
    dt : float
        Time interval per time step
    drift: float
        Rotational drift rate of the agent as influenced by their neighbors
    noise : float
        Noise amplitude of the agent's rotation

    Returns
    -------
    np.ndarray or float
        Time series of agent rotation influenced by the neighboring agents
    """

    assert len(agent_positions) == len(agent_rotations)
    num_agents = len(agent_positions)

    # Position update is a result of rotation update!

    positions = np.zeros((timesteps, num_agents, 2), dtype=np.float32)
    positions[0] = agent_positions

    rotations = np.zeros((timesteps, num_agents, 1), dtype=np.float32)
    rotations[0] = agent_rotations

    for t in range(1, timesteps):

        for a in range(num_agents):
            direction = alignment_influence(
                positions[t - 1, a],
                positions[t - 1],
                rotations[t - 1],
                sensing_radius,
                noise
            )
            rotations[t, a] = direction

            if direction == 0.0:
                motion_vector = np.array([0.,0.], dtype=np.float32)
            else:
                motion_vector = np.array([np.cos(direction), np.sin(direction)], dtype=np.float32)

            positions[t, a] = positions[t - 1, a] + motion_vector * drift * dt

    return np.concatenate((positions, rotations), axis=-1)


@njit
def move_with_neighbors(
    agent_positions,
    sensing_radius=1.5,
    drift=0.5,
    noise=0.1,
    timesteps=1000,
    dt=0.1
):
    """
    Simulate motion (locomotion) as influenced by the neighboring agents

    Parameters
    ----------
    agent_positions : np.ndarray or float
        Positions of the agents
    sensing_radius : float
        The sensing radius of the agents
    timesteps : int
        Number of time steps to simulate
    dt : float
        Time interval per time step
    drift: float
        Locomotive drift rate of the agent as influenced by their neighbors
    noise : float
        Noise amplitude of the agent's position

    Returns
    -------
    np.ndarray or float
        Time series of agent position influenced by the neighboring agents
    """

    num_agents = agent_positions.shape[0]

    positions = np.zeros((timesteps, num_agents, 2), dtype=np.float32)
    positions[0] = agent_positions

    for t in range(1, timesteps):
        for a in range(num_agents):
            direction = cohesion_influence(
                positions[t-1, a],
                positions[t-1],
                sensing_radius,
                noise
            )

            if direction == 0.0:
                motion_vector = np.array([0.,0.], dtype=np.float32)
            else:
                motion_vector = np.array([np.cos(direction), np.sin(direction)], dtype=np.float32)
            
            positions[t, a] = positions[t - 1, a] + motion_vector * drift * dt
            # positions[t, a, 0] = positions[t - 1, a, 0] + np.cos(direction) * drift * dt
            # positions[t, a, 1] = positions[t-1, a, 1] + np.sin(direction) * drift * dt

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
    """
    Simulate agent motion as influenced by the beacons only.

    Parameters
    ----------
    agent_position : np.ndarray or float
        Position of the agent
    agent_rotation : np.ndarray or float
        Rotation of the agent
    beacon_position : np.ndarray or float
        Position of the beacon
    position_drift : float
        Positional drift rate of the beacon
    rotation_drift : float
        Rotational drift rate of the beacon
    position_noise : float
        Positional noise amplitude of the beacon
    rotation_noise : float
        Rotational noise amplitude of the beacon
    timesteps : int
        Number of time steps to simulate
    dt : float
        Time interval per time step

    Returns
    -------
    np.ndarray or float
        Time series of agent positions and rotations influenced by the beacons
    """

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
    """
    Simulate agent motion as influenced by the neighboring agents only.

    Parameters
    ----------
    agent_positions : np.ndarray or float
        Positions of the agents
    agent_rotations : np.ndarray or float
        Rotations of the agents
    alignment_drift : float
        Drift rate for rotational alignment of the agents with their sensed neighbors
    cohesion_drift : float
        Drift rate for cohesion of the agents with their neighbors
    alignment_noise : float
        Noise amplitude of the agent's alignment
    cohesion_noise : float
        Noise amplitude of the agent's cohesion
    sensing_radius : float
        The sensing radius of the agent
    timesteps : int
        Number of time steps to simulate
    dt : float
        Time interval per time step

    Returns
    -------
    np.ndarray or float
        Time series of agent positions and rotations influenced by the neighboring agents
    """

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

    # rotations = rotations % (2. * np.pi)
    return np.concatenate((positions, rotations), axis=-1)


@njit
def motion_simulation(
    theta=None,
    num_agents: int = 12,
    num_beacons: int = 10,
    room_size: np.ndarray = (8, 10),
    position_drift: float = 0.5,
    rotation_drift: float = 0.5,
    position_noise: float = 0.001,
    rotation_noise: float = 0.001,
    alignment_noise: float = 0.001,
    cohesion_noise: float = 0.001,
    sensing_radius: float = 1.5,
    influence_weight: float = 0.5,
    timesteps=1000,
    dt=0.1
):
    """
    Full simulation of agent motion, incorporating both individual and collective influences.

    Parameters
    ----------
    theta : np.ndarray or float
        Prior for the agents' properties, such as drift rates, modulation weights, and sensing radii.
        If none given, defaults are used.
    num_agents : int
        Number of agents to simulate.
    num_beacons : int
        Number of beacons to simulate.
    room_size : tuple
        Size of the room to simulate the agents.
    position_drift : float
        Positional drift rate of the beacon
    rotation_drift : float
        Rotational drift rate of the beacon
    alignment_drift : float
        Alignment drift rate of the beacon as influenced by their neighbors
    cohesion_drift : float
        Cohesion drift rate of the beacon as influenced by their neighbors
    position_noise : float
        Positional noise amplitude of the agent
    rotation_noise : float
        Rotational noise amplitude of the agent
    alignment_noise : float
        Alignment noise amplitude of the agent
    cohesion_noise : float
        Cohesion noise amplitude of the agent
    sensing_radius : float
        Sensing radius of the agents
    influence_weight : float
        The modulation weight between individual and collective influences for each agents
    timesteps : int
        Number of time steps to simulate
    dt : float
        Time interval per time step

    Returns
    -------
    np.ndarray or float
        Time series of agent positions and rotations
    """

    initial_agent_positions, initial_agent_rotations = initialize_agents(num_agents, room_size=room_size)
    beacon_positions = initialize_beacons(num_beacons)

    if theta is not None:
        influence_weight = theta[0]
        sensing_radius = theta[1]
        # All drifts are the same, for now
        position_drift = theta[2]
        rotation_drift = theta[2]
        # backup
        alignment_drift = theta[2]
        cohesion_drift = theta[2]

    # Initialization
    positions = np.zeros((timesteps, num_agents, 2), dtype=np.float32)
    rotations = np.zeros((timesteps, num_agents, 1), dtype=np.float32)
    positions[0] = initial_agent_positions
    rotations[0] = initial_agent_rotations

    for t in range(1, timesteps):
        for a in range(num_agents):

            # Find the nearest beacon to target
            distances_to_beacons = np.zeros((num_beacons, 1), dtype=np.float32)

            for b in range(num_beacons):
                bx = beacon_positions[b, 0] - positions[t, a, 0]
                by = beacon_positions[b, 1] - positions[t, a, 1]
                distances_to_beacons[b] = (bx * bx + by * by) ** 0.5

            beacon_id = np.argmin(distances_to_beacons)

            # The influences
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

            position_shift = influence_weight * move_direction + (1 - influence_weight) * follow_direction
            rotation_shift = influence_weight * look_direction + (1 - influence_weight) * align_direction

            positions[t, a] = positions[t - 1, a] + np.array(
                [np.cos(position_shift), np.sin(position_shift)], dtype=np.float32
            ) * position_drift * dt
            rotations[t, a] = rotations[t - 1, a] + rotation_shift * rotation_drift * dt

    rotations = rotations % (2. * np.pi)
    return np.concatenate((positions, rotations), axis=-1)


# Backup simulation for debugging purposes only
@njit
def look():
    # for debugging rotation only
    raise NotImplementedError


@njit
def move():
    # for debugging position only
    raise NotImplementedError


# For testing purpose only
#
if __name__ == "__main__":

    sim = motion_simulation(num_agents=49, num_beacons=25)

    print(sim)
