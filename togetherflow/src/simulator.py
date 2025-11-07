import numpy as np
from numba import njit, prange

from .initialization import initialize_agents, initialize_beacons
from .influences import combined_influences
from .priors import complete_pooling_prior


@njit(parallel=True)
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
        internal_focus = theta[3]

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
    distances = np.zeros((num_timesteps, num_agents,))
    max_dists = np.zeros((num_timesteps, num_agents,))
    angular_velocities = np.zeros((num_timesteps, num_agents,))
    neighbor_fluctuations = np.zeros((num_timesteps, num_agents,))
    positions[0] = initial_positions
    rotations[0] = initial_rotations

    # Initialize beacons
    beacon_positions = initialize_beacons(num_beacons)

    # Simulation loop
    for t in prange(1, num_timesteps):
        ps, rs, num_neighbors, average_dists = combined_influences(
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
        distances[t] = average_dists
        #max_dists[t] = max_dists
        angular_velocities[t] = rs - rotations[t - 1]
        neighbor_fluctuations[t] = num_neighbors - neighbors[t - 1]

    neighbors[0] = neighbors[1]

    rotations = rotations[:, :, np.newaxis]
    neighbors = neighbors[:, :, np.newaxis]
    distances = distances[:, :, np.newaxis]
    #max_dists = max_dists[:, :, np.newaxis]
    angular_velocities = angular_velocities[:, :, np.newaxis]
    neighbor_fluctuations = neighbor_fluctuations[:, :, np.newaxis]

    return np.concatenate((
        positions,
        rotations,
        neighbors,
        distances,
        #max_dists,
        angular_velocities,
        neighbor_fluctuations
    ), axis=-1)


class TogetherFlowSimulator:

    def __init__(
        self,
        num_agents: int = 12,
        num_beacons: int = 1,
        room_size: tuple = (8, 10),
        dt: float = 0.001,
        internal_focus: float = 0.1,
        time_horizon: float = 30.,
        downsample: bool = False,
        downsample_factor: int = 10,
        gather: bool = True,
        return_summary: bool = True
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
        self.gather = gather
        self.return_summary = return_summary


    def sample(self, batch_size: int | tuple = 1) -> dict[str, np.ndarray]:

        if isinstance(batch_size, tuple):
            if len(batch_size) != 1:
                raise ValueError(f"Expected batch_size as int or (int,), got {batch_size}")
            batch_size = batch_size[0]
        elif not isinstance(batch_size, int):
            raise ValueError(f"batch_size must be int or (int,), got {type(batch_size)}")

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
                time_horizon=self.time_horizon,
                # internal_focus=self.internal_focus,
            )
            thetas.append(theta)
            samples.append(sim)

        thetas = np.array(thetas)
        samples = np.array(samples)

        if self.downsample:
            samples = samples[:,::self.downsample_factor,:,:]

        B, T, A, D = samples.shape

        if self.gather:
            positions = samples[:,:,:,0:2].reshape((B, T, A*2))
            rotations = samples[:,:,:,2].reshape((B, T, A))
            neighbors = samples[:,:,:,3].reshape((B, T, A))
            distances = samples[:,:,:,4].reshape((B, T, A))
            #max_dists = samples[:,:,:,5].reshape((B, T, A))
            angular_velocities = samples[:,:,:,5].reshape((B, T, A))
            neighbor_fluctuations = samples[:,:,:,6].reshape((B, T, A))
        else:
            positions = samples[:,:,:,0:2]
            rotations = samples[:,:,:,2][..., None]
            neighbors = samples[:,:,:,3][..., None]
            distances = samples[:,:,:,4][..., None]
            #max_dists = samples[:,:,:,5][..., None]
            angular_velocities = samples[:,:,:,5][..., None]
            neighbor_fluctuations = samples[:,:,:,6][..., None]

        out = dict(
            w = np.expand_dims(thetas[:,0], axis=-1),
            r = np.expand_dims(thetas[:,1], axis=-1),
            v = np.expand_dims(thetas[:,2], axis=-1),
            noise = np.expand_dims(thetas[:,3], axis=-1),
        )

        if not self.return_summary:
            out = out | dict(
                positions = positions,
                rotations = rotations,
                neighbors = neighbors,
                distances = distances,
                #max_dists = max_dists,
                angular_velocities = angular_velocities,
                neighbor_fluctuations = neighbor_fluctuations
            )
        else:
            x = samples[:,:,:,0].reshape((B, T, A))
            y = samples[:,:,:,1].reshape((B, T, A))
            x_mean, x_std = x.mean(axis=(1, 2)), x.std(axis=(1, 2))
            y_mean, y_std = y.mean(axis=(1, 2)), y.std(axis=(1, 2))
            positions = np.array([[x_mean, x_std], [y_mean, y_std]])
            out = out | dict(
                positions = positions,
                rotations = np.array([rotations.mean(axis=(1, 2)), rotations.std(axis=(1, 2))])[..., None],
                neighbors = np.array([neighbors.mean(axis=(1, 2)), neighbors.std(axis=(1, 2))])[..., None],
                distances = np.array([distances.mean(axis=(1, 2)), distances.std(axis=(1, 2))])[..., None],
                angular_velocities = np.array([angular_velocities.mean(axis=(1, 2)), angular_velocities.std(axis=(1, 2))])[..., None],
                neighbor_fluctuations = np.array([neighbor_fluctuations.mean(axis=(1, 2)), neighbor_fluctuations.std(axis=(1, 2))])[..., None],
            )

        return out
