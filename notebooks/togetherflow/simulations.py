import numpy as np
from .influences import internal_influence

def initialize_agents(
        num_agents: int = 100,
        boundary_size: float = 100.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate random positions and orientations for agents.

    Parameters
    ----------
    num_agents : int, optional
        Number of agents to generate (default is 100).
    boundary_size : float, optional
        The size of the boundary within which positions are generated (default is 100.0).

    Returns
    -------
    tuple of np.ndarray
        A tuple containing the positions (np.ndarray) and orientations (np.ndarray) of the agents.
    """

    # Generate random positions within the boundary size centered at 0
    positions = np.random.uniform(-boundary_size * 0.5, boundary_size * 0.5, (num_agents, 2))

    # Generate random orientations (angles in radians between 0 and 2*pi)
    orientations = np.random.uniform(0, 2 * np.pi, num_agents)

    return positions, orientations


def initialize_beacons(
        num_beacons=10,
        room_sensing_range=50.
):
    """
    Initialize beacons following a uniform distribution scaled to the room's sensing boundary

    Parameters
    ----------
    num_beacons : int, default: 10
        Number of beacons to initialize.
    environment_size : float, default: 50.0
        Size of the environment for the generation of beacons.

    Returns
    -------
    beacons      : np.ndarray of shape (num_beacons, 2)
        Initial positions of the beacons.
    """

    beacons = np.random.uniform(-room_sensing_range * 0.5, room_sensing_range * 0.5, size=(num_beacons, 2))
    return beacons


def simulate_internal_influence(
        num_agents: int = 1,
        num_beacons: int = 1,
        num_timesteps: int = 1001
):
    """
    Simulate the influence of the agents' internal influence only.

    Parameters
    ----------
    num_agents      : int, default: 12
        Number of agents to simulate.
    num_beacons      : int, default: 10
        Number of beacons to simulate.
    num_timesteps     : int, default: 1001
        Number of timesteps to simulate.
    """

    agent_positions = np.zeros((num_timesteps, num_agents, 2))
    agent_rotations = np.zeros((num_timesteps, num_agents, 1))

    beacon_positions = initialize_beacons(num_beacons=num_beacons)

    agent_positions[0], agent_rotations[0] = initialize_agents(num_agents=num_agents)

    for i in range(1, num_timesteps):
        old_position = agent_positions[i - 1].copy()
        old_rotation = agent_rotations[i - 1].copy()
        for j in range(num_agents):
            beacon_orientation = np.arctan2(
                beacon_positions[0, 1] - old_position[j, 1],
                beacon_positions[0, 0] - old_position[j, 0]
            )

            new_position, new_rotation = internal_influence(
                beacon_positions[0],  # Set only one beacon to test, for now.
                old_position[j],
                old_rotation[j]
            )

            if np.abs(new_rotation - beacon_orientation) < np.pi * 0.01:
                break

            agent_positions[i, j] = new_position
            agent_rotations[i, j] = new_rotation

    return agent_positions, agent_rotations, beacon_positions


def simulate_external_influence(
        theta,
        num_agents=12,
        num_timesteps=100,
        boundary_size=10.0,
):
    """
    Simulate the movement trajectory of the agent,
    as governed by the Vicsek model.

    Parameters
    ----------
    theta : np.ndarray of shape (2, )
        Sampled priors for the model, including the
        base sensing radius (r) and walking speed (v)
        of the agents.
    num_agents     : int, default: 12
        Number of agents to simulate.
    num_timesteps   : int, default: 100
        Number of timesteps to simulate.
    boundary_size   : float, default: 10.0
        Size of the simulation boundary (in meters).

    Returns
    -------
    A concatenated NumPy array of combined trajectory
    and direction of the agents as timeseries.
    """

    # Unpack priors
    r, v = theta[0], theta[1]

    # Scale radius with half of boundary size (for realism)
    sensing_radius = r * boundary_size * 0.5

    # Store trajectories and headings
    agent_positions = np.zeros((num_timesteps, num_agents, 2))
    agent_rotations = np.zeros((num_timesteps, num_agents, 1))

    # Initialize positions and directions for each agent
    agent_positions[0], agent_rotations[0] = initialize_agents(num_agents, boundary_size)

    # Loop over each timestep
    for t in range(1, num_timesteps):
        # Get previous positions and rotations
        old_positions = agent_positions[t - 1].copy()
        old_rotations = agent_rotations[t - 1].copy()

        # For each timestep, initialize directions for the agents
        new_rotations = np.zeros(num_agents)

        # For each agent, collect neighbors within its sensing range
        for i in range(num_agents):
            neighbor_rotations = []
            # If there are any neighbors, average over their directions
            # and assign it as the new direction.
            for j in range(num_agents):
                if i != j and np.linalg.norm(agent_positions[t - 1, i] - agent_positions[j]) < sensing_radius:
                    neighbor_rotations.append(agent_rotations[j])
            if neighbor_rotations != []:
                average_neighbor_rotation = np.mean(np.array(neighbor_rotations))
                new_rotations[i] = average_neighbor_rotation + np.random.uniform(-0.01, 0.01)
            else:
                new_rotations[i] = agent_rotations[i]

        agent_rotations = np.copy(new_rotations)

        # Update position upon new direction
        agent_positions[:, 0] += v * np.cos(agent_rotations)
        agent_positions[:, 1] += v * np.sin(agent_rotations)

        # Assumes periodic boundary condition (for now)
        agent_positions = np.mod(np.copy(agent_positions), boundary_size)

        # Add timestamps to trajectories and headings
        agent_positions[t + 1] = np.copy(agent_positions)
        agent_rotations[t + 1] = np.copy(agent_rotations[:, np.newaxis])

    return np.concatenate((agent_positions, agent_rotations), axis=-1)
