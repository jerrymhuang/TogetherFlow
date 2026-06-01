import numpy as np
from numba import njit

from .initialization import initialize_agents, initialize_beacons
from .influences import combined_influences
from .priors import complete_pooling_prior


@njit
def simulator_fun(
    theta,
    num_agents: int = 12,
    num_beacons: int = 1,
    room_size: tuple = (8., 10.),
    velocity: float = 1.0,
    dt: float = 0.1,
    influence_weight: float = 0.7,
    sensing_radius: float = 2.5,
    internal_focus: float = 0.1,
    time_horizon: float = 30.,
):
    """
    Run one simulation trajectory and return per-channel time series.

    theta overrides (influence_weight, sensing_radius, velocity, internal_focus)
    when provided; the parameter defaults serve as fallbacks for partial thetas.

    Parameters
    ----------
    theta           : np.ndarray of shape (4,) — [w, r, v, noise]
    num_agents      : int
    num_beacons     : int
    room_size       : tuple (width, height)
    velocity        : float  (fallback if theta has < 3 elements)
    dt              : float
    influence_weight: float  (fallback if theta has < 1 element)
    sensing_radius  : float  (fallback if theta has < 2 elements)
    internal_focus  : float  (fallback if theta has < 4 elements)
    time_horizon    : float

    Returns
    -------
    positions             : np.ndarray of shape (T, A, 2)
    rotations             : np.ndarray of shape (T, A)
    neighbors             : np.ndarray of shape (T, A)
    distances             : np.ndarray of shape (T, A)
    angular_velocities    : np.ndarray of shape (T, A)
    neighbor_fluctuations : np.ndarray of shape (T, A)
    """
    influence_weight = theta[0]
    sensing_radius   = theta[1]
    velocity         = theta[2]
    internal_focus   = theta[3]

    num_timesteps = int(time_horizon / dt)

    positions  = np.zeros((num_timesteps, num_agents, 2))
    rotations  = np.zeros((num_timesteps, num_agents))
    neighbors  = np.zeros((num_timesteps, num_agents))
    distances  = np.zeros((num_timesteps, num_agents))
    ang_vels   = np.zeros((num_timesteps, num_agents))
    nbr_flucts = np.zeros((num_timesteps, num_agents))

    positions[0], rotations[0] = initialize_agents(num_agents, room_size=room_size)
    beacon_positions = initialize_beacons(num_beacons)

    for t in range(1, num_timesteps):
        ps, rs, nn, ad = combined_influences(
            agent_positions=positions[t - 1],
            agent_rotations=rotations[t - 1],
            beacon_positions=beacon_positions,
            room_size=room_size,
            velocity=velocity,
            sensing_radius=sensing_radius,
            dt=dt,
            influence_weight=influence_weight,
            internal_focus=internal_focus,
        )
        positions[t]  = ps
        rotations[t]  = rs
        neighbors[t]  = nn
        distances[t]  = ad
        ang_vels[t]   = rs - rotations[t - 1]
        nbr_flucts[t] = nn - neighbors[t - 1]

    # Backfill t=0 statistics from t=1 (no previous state to diff against)
    neighbors[0]  = neighbors[1]
    nbr_flucts[0] = nbr_flucts[1]

    return positions, rotations, neighbors, distances, ang_vels, nbr_flucts


class TogetherFlowSimulator:
    """
    BayesFlow-compatible simulator for multi-agent motion dynamics.

    Parameters
    ----------
    prior        : callable returning np.ndarray of shape (4,)
                   Defaults to complete_pooling_prior. Inject a different prior
                   to change the generative model without touching this class.
    output_mode  : str, one of "flat" | "raw" | "summary"
                   "flat"    — time series with agents flattened into the feature dim
                               positions (B,T,2A), others (B,T,A)  [default]
                   "raw"     — per-agent arrays: positions (B,T,A,2), others (B,T,A,1)
                   "summary" — mean/std collapsed over T and A:
                               positions (B,4), others (B,2)
    """

    def __init__(
        self,
        num_agents: int = 12,
        num_beacons: int = 1,
        room_size: tuple = (8., 10.),
        dt: float = 0.1,
        time_horizon: float = 30.,
        downsample: bool = False,
        downsample_factor: int = 10,
        prior=None,
        output_mode: str = "flat",
    ):
        self.num_agents = num_agents
        self.num_beacons = num_beacons
        self.room_size = room_size
        self.dt = dt
        self.time_horizon = time_horizon
        self.downsample = downsample
        self.downsample_factor = int(downsample_factor)
        self.prior = prior if prior is not None else complete_pooling_prior
        self.output_mode = output_mode

        if output_mode not in ("flat", "raw", "summary"):
            raise ValueError(f"output_mode must be 'flat', 'raw', or 'summary'; got '{output_mode}'")

    def sample(self, batch_size: int | tuple = 1) -> dict[str, np.ndarray]:
        if isinstance(batch_size, tuple):
            if len(batch_size) != 1:
                raise ValueError(f"Expected batch_size as int or (int,), got {batch_size}")
            batch_size = batch_size[0]
        elif not isinstance(batch_size, int):
            raise ValueError(f"batch_size must be int or (int,), got {type(batch_size)}")

        all_thetas, all_pos, all_rot, all_nbr, all_dst, all_av, all_nf = [], [], [], [], [], [], []

        for _ in range(batch_size):
            theta = self.prior()
            pos, rot, nbr, dst, av, nf = simulator_fun(
                theta=theta,
                num_agents=self.num_agents,
                num_beacons=self.num_beacons,
                room_size=self.room_size,
                dt=self.dt,
                time_horizon=self.time_horizon,
            )
            all_thetas.append(theta)
            all_pos.append(pos)
            all_rot.append(rot)
            all_nbr.append(nbr)
            all_dst.append(dst)
            all_av.append(av)
            all_nf.append(nf)

        thetas    = np.stack(all_thetas)  # (B, 4)
        positions = np.stack(all_pos)     # (B, T, A, 2)
        rotations = np.stack(all_rot)     # (B, T, A)
        neighbors = np.stack(all_nbr)     # (B, T, A)
        distances = np.stack(all_dst)     # (B, T, A)
        ang_vels  = np.stack(all_av)      # (B, T, A)
        nbr_flucts = np.stack(all_nf)     # (B, T, A)

        if self.downsample:
            s = self.downsample_factor
            positions  = positions[:, ::s, :, :]
            rotations  = rotations[:, ::s, :]
            neighbors  = neighbors[:, ::s, :]
            distances  = distances[:, ::s, :]
            ang_vels   = ang_vels[:, ::s, :]
            nbr_flucts = nbr_flucts[:, ::s, :]

        B, T, A, _ = positions.shape

        out = {
            "w":     thetas[:, 0:1],
            "r":     thetas[:, 1:2],
            "v":     thetas[:, 2:3],
            "noise": thetas[:, 3:4],
        }

        if self.output_mode == "flat":
            out |= {
                "positions":             positions.reshape(B, T, A * 2),
                "rotations":             rotations,
                "neighbors":             neighbors,
                "distances":             distances,
                "angular_velocities":    ang_vels,
                "neighbor_fluctuations": nbr_flucts,
            }

        elif self.output_mode == "raw":
            out |= {
                "positions":             positions,
                "rotations":             rotations[..., None],
                "neighbors":             neighbors[..., None],
                "distances":             distances[..., None],
                "angular_velocities":    ang_vels[..., None],
                "neighbor_fluctuations": nbr_flucts[..., None],
            }

        elif self.output_mode == "summary":
            def _summarize(arr):
                return np.stack([arr.mean(axis=(1, 2)), arr.std(axis=(1, 2))], axis=-1)

            out |= {
                "positions": np.concatenate([
                    _summarize(positions[:, :, :, 0]),
                    _summarize(positions[:, :, :, 1]),
                ], axis=-1),
                "rotations":             _summarize(rotations),
                "neighbors":             _summarize(neighbors),
                "distances":             _summarize(distances),
                "angular_velocities":    _summarize(ang_vels),
                "neighbor_fluctuations": _summarize(nbr_flucts),
            }

        return out
