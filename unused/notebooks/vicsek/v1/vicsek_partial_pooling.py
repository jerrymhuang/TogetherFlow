
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.animation as animation
from IPython.display import HTML

np.set_printoptions(suppress=True)

import bayesflow as bf
import tensorflow as tf

from functools import partial
from numba import njit

from bayesflow.simulation import Prior, Simulator, GenerativeModel


@njit
def prior_fun():
    """
    Sample prior parameters for the agent group,
    i.e., base scale of sensing radius (r) and walking speed (v).

    Returns
    -------
    Sampled priors as a NumPy array.
    """

    alpha_r = np.random.uniform(1., 5.,)
    beta_r = np.random.uniform(1., 5.,)

    return np.array([alpha_r, beta_r], dtype=np.float32)


@njit
def initialize_agents(
        num_agents=12,
        boundary_size=10.,
):
    """
    Initialize agent positions and directions.

    Parameters
    ----------
    num_agents      : int, default: 12
        Number of agents to initialize.
    boundary_size   : float, default: 10.0
        Size of the boundary (in meters).

    Returns
    -------
    positions      : np.ndarray of shape (num_agents, 2)
        Initial positions of the agents.
    directions      : np.ndarray of shape (num_agents, )
        Initial directions of the agents.
    """

    positions = np.random.random(size=(num_agents, 2)).astype(np.float32) * boundary_size
    directions = np.random.random(size=num_agents) * np.pi

    theta = np.empty((num_agents, 2), dtype=np.float32)

    for i in range(num_agents):
        priors = prior_fun()
        r = np.random.beta(priors[0], priors[1])
        v = np.random.beta(2., 2.)

        theta[i,0] = r
        theta[i,1] = v

    return positions, directions, theta


@njit
def simulator_fun(
        theta=None,
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

    # Initialize positions and directions for each agent
    positions, directions, theta = initialize_agents(num_agents, boundary_size)

    r, v = theta[:, 0], theta[:, 1]

    # Scale radius with half of boundary size (for realism)
    radius = r * boundary_size * 0.5

    # Store trajectories and headings
    paths = np.zeros((num_timesteps + 1, num_agents, 2))
    headings = np.zeros((num_timesteps + 1, num_agents, 1))

    paths[0] = positions
    headings[0] = directions[:, np.newaxis]

    # Loop over each timestep
    for t in range(num_timesteps):
        # For each timestep, initialize directions for the agents
        new_directions = np.zeros(num_agents)

        # For each agent, collect neighbors within its sensing range
        for i in range(num_agents):

            neighbors = []
            # If there are any neighbors, average over their directions
            # and assign it as the new direction.
            for j in range(num_agents):
                if i != j and np.linalg.norm(positions[i] - positions[j]) < radius[i]:
                    neighbors.append(directions[j])
            if neighbors:
                avg_direction = np.mean(np.array(neighbors))
                new_directions[i] = avg_direction + np.random.uniform(-0.01, 0.01)
            else:
                new_directions[i] = directions[i]

        directions = np.copy(new_directions)

        # Update position upon new direction
        positions[:, 0] += v * np.cos(directions)
        positions[:, 1] += v * np.sin(directions)

        # Assumes periodic boundary condition (for now)
        positions = np.mod(positions, boundary_size)

        # Add timestamps to trajectories and headings
        paths[t + 1] = np.copy(positions)
        headings[t + 1] = np.copy(directions[:, np.newaxis])

    return np.concatenate((paths, headings), axis=-1)


if __name__ == '__main__':


    # param_names = [r"$\alpha_r", r"$\beta_r"]

    # positions, directions, theta = initialize_agents()
    # print(theta)

    # data = simulator_fun(num_agents=12, num_timesteps=100)

    prior = Prior(prior_fun=prior_fun)
    simulator = Simulator(simulator_fun=simulator_fun)

    model = GenerativeModel(
        prior=prior,
        simulator=simulator,
        simulator_is_batched=False,
        name="Vicsek_Partial_Pooling"
    )