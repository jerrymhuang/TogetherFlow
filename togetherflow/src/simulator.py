import numpy as np
from numba import njit

from .initialization import initialize_agents, initialize_beacons
from .influences import internal_influence, external_influence, combined_influences
from .priors import complete_pooling_prior


@njit
def simulator_fun(
        theta=None,
        num_agents: int = 12,
        num_beacons: int = 1,
        room_size: tuple = (8, 10),
        velocity: float = 1.0,
        dt: float = 0.001,
        influence_weight: float = 0.7,
        sensing_radius: float = 10.0,
        internal_focus: float = 0.1,
        time_horizon: float = 30.
):
    """
    Run the simulation and store the time series of positions and orientations of agents.

    Parameters
    ----------
    theta : np.ndarray
        Prior parameters specifying the internal properties of the agents
    num_agents : int, optional
        Number of agents to generate (default is 100).
    num_beacons : int, optional
        Number of beacons to generate (default is 1).
    room_size : float, optional
        The size of the boundary within which positions are generated (default is 100).
    velocity : float, optional
        The speed at which agents move (default is 1.0).
    dt : float, optional
        The time step for the update (default is 0.1).
    influence_weight : float, optional
        The weight for influence_vector1 in determining new orientations (default is 0.7).
    sensing_radius : float, optional
        The sensing radius for the Vicsek model (default is 10.0).
    internal_focus : float, optional
        Variance of the internal influence.
    time_horizon : int, optional
        The number of steps to simulate (default is 30).

    Returns
    -------
    tuple of np.ndarray
        The time series of positions and orientations of the agents.
    """

    if theta is not None:
        influence_weight = theta[0]
        sensing_radius = theta[1]
        velocity = theta[2]
        # internal_focus = theta[3]

    num_timesteps = int(time_horizon / dt)

    # Apply radial bound with sigmoid transformation for the sensing radius
    # (r_min, r_max) = (1., 5.)
    # sensing_radius = r_min + (r_max - r_min) * (1. / (1. + np.exp(-sensing_radius)))

    # Initialize positions and orientations
    initial_positions, initial_rotations = initialize_agents(num_agents, room_size=room_size)

    # Initialize arrays to store time series of positions and orientations
    positions = np.zeros((num_timesteps, num_agents, 2))
    rotations = np.zeros((num_timesteps, num_agents,))
    neighbors = np.zeros((num_timesteps, num_agents,))
    positions[0] = initial_positions
    rotations[0] = initial_rotations

    # Initialize beacons
    beacon_positions = initialize_beacons(num_beacons)

    # Simulation loop
    for t in range(1, num_timesteps):
        ps, rs, num_neighbors = combined_influences(
            agent_positions=positions[t - 1],
            agent_rotations=rotations[t - 1],
            beacon_positions=beacon_positions,
            velocity=velocity,
            sensing_radius=sensing_radius,
            dt=dt,
            influence_weight=influence_weight,
            internal_focus=internal_focus
        )

        # Store positions and orientations for each time step
        positions[t] = ps
        rotations[t] = rs
        neighbors[t] = num_neighbors

    neighbors[0] = neighbors[1]

    rotations = rotations[:, :, np.newaxis]
    neighbors = neighbors[:, :, np.newaxis]

    return np.concatenate((positions, rotations, neighbors), axis=-1)


class TogetherFlowSimulator:

    def __init__(self,
                 num_agents: int = 12,
                 num_beacons: int = 1,
                 room_size: tuple = (8, 10),
                 dt: float = 0.001,
                 internal_focus: float = 0.1,
                 time_horizon: float = 30.,
                 downsample: bool = True,
                 downsample_factor: int = 10
                 ):
        self.num_agents = num_agents
        self.num_beacons = num_beacons
        self.room_size = room_size
        self.dt = dt
        self.internal_focus = internal_focus
        self.time_horizon = time_horizon
        self.downsample = downsample
        self.downsample_factor = downsample_factor
        self.num_timesteps = int(time_horizon / dt)


    def sample(self, batch_shape: int | tuple = (1,)) -> dict[str, np.ndarray]:

        batch_size = batch_shape[0]
        thetas = []
        samples = []

        for i in range(batch_size):
            theta = complete_pooling_prior()
            sim = simulator_fun(
                theta=theta,
                num_agents=self.num_agents,
                num_beacons=self.num_beacons,
                room_size=self.room_size,
                dt=self.dt,
                internal_focus=self.internal_focus,
                time_horizon=self.time_horizon
            )
            thetas.append(theta)
            samples.append(sim)

        thetas = np.array(thetas)
        samples = np.array(samples)

        if self.downsample:
            samples = samples[:,::self.downsample_factor,:,:]

        B, T, A, D = samples.shape

        positions = samples[:,:,:,0:2].reshape((B, T, A*2))
        rotations = samples[:,:,:,2].reshape((B, T, A))
        neighbors = samples[:,:,:,3].reshape((B, T, A))

        return dict(
            w = thetas[:,0],
            r = thetas[:,1],
            v = thetas[:,2],
            positions = positions,
            rotations = rotations,
            neighbors = neighbors
        )
