import numpy as np


def internal_influence(
        beacon_position,
        agent_position,
        agent_rotation,
        movement_speed=1.,
        rotation_speed=0.1,
        noise_level=0.1,
        dt=0.1
):
    """
    Internally influence the agent's position and direction for a single time step
    given the target position and rotation speeds.

    Parameters
    ----------
    target_position : np.ndarray of shape (2,)
        Position of the agent's target.
    agent_position : np.ndarray of shape (2,)
        Position of the agent.
    agent_rotation : float
        Orientation of the agent.
    movement_speed : float
        Speed of the agent's movement.
    rotation_speed : float
        Speed of the agent's rotation.
    noise_level : float
        Rotational noise level of the agent.
    dt : float, default: 1.
        Time step of the simulation.

    Returns
    -------
    new_position : np.ndarray of shape (2,)
        Updated position of the agent.
    new_rotation : float
        Updated orientation of the agent.
    """

    # Declare new agent positions and rotation
    new_position = agent_position
    new_rotation = agent_rotation

    # Compute relative orientation between the agent and the look-at target
    beacon_orientation = np.arctan2(
        beacon_position[1] - agent_position[1],
        beacon_position[0] - agent_position[0]
    )

    # Calculate drift rate and noise
    drift = rotation_speed * (beacon_position - agent_rotation) * dt
    noise = np.random.vonmises(mu=0, kappa=1 / (noise_level * noise_level))

    # Update and normalize agent rotation
    new_rotation += drift + noise
    new_rotation = (new_rotation + np.pi) % (2 * np.pi) - np.pi

    # Update agent position
    new_position[0] += movement_speed * np.cos(new_rotation.item()) * dt
    new_position[1] += movement_speed * np.sin(new_rotation.item()) * dt

    return new_position, new_rotation


def external_influence(
        agent_positions,
        agent_rotations,
        neighbor_positions,
        neighbor_rotations,
        dt=0.1
):
    """
    Externally influence the agent's position and direction for a single time step
    given the neighboring agent's position and orientation.

    Parameters
    ----------
    agent_position : np.ndarray of shape (2,)
        Position of the agent.
    agent_rotation : float
        Orientation of the agent.
    neighbor_positions : np.ndarray of shape (2,)
        Positions of the neighboring agent.
    neighbor_rotation : float
        Orientations of the neighboring agent.
    dt : float, default: 1.
        Time step of the simulation.

    Returns
    -------
    new_position : np.ndarray of shape (2,)
        Updated position of the agent.
    new_rotation : float
        Updated orientation of the agent.
    """
    assert len(neighbor_positions) == len(neighbor_rotations), \
        f"Number of neighboring agents' positions ({len(neighbor_positions)}) " + \
        f"does not match number of their rotations ({len(neighbor_rotations)})."

    new_position = agent_positions
    new_rotation = agent_rotations

    # Compute average positions and rotations for neighbors
    average_neighbor_position = np.mean(neighbor_positions)
    # average_neighbor_rotation = np.mean(neighbor_rotations)

    # Update position and rotation
    new_rotation += average_neighbor_position

    new_position[0] += np.cos(new_rotation.item()) * dt
    new_position[1] += np.sin(new_rotation.item()) * dt

    return new_position, new_rotation

