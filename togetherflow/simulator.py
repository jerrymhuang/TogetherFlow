import numpy as np
from numba import njit, prange

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
    beacon_strengths=None,
    salience_sensitivity: float = 0.0,
    reference_radii=None,
    beacon_spread: float = 50.,
    relative_heading: bool = False,
    repulsion_radius: float = 0.0,
    repulsion_gain: float = 0.0,
    obstacles=None,
    max_turn_rate: float = 6.283185307179586,
    door_wall: int = -1,
    door_center: float = 0.0,
    door_half_width: float = 0.0,
    init_positions=None,
    init_rotations=None,
    fixed_beacons=None,
    beacon_assignment=None,
    diffusive_heading: bool = False,
    alpha_slot: int = -1,
    kappa_slot: int = -1,
    sigma_slot: int = -1,
):
    """
    Run one simulation trajectory and return per-channel time series.

    theta overrides (influence_weight, sensing_radius, velocity, internal_focus)
    when provided; the parameter defaults serve as fallbacks for partial thetas.

    Parameters
    ----------
    theta           : np.ndarray of shape (T, P) — per-timestep [w, r, v, noise]
                      and optionally alpha in column 4. Time-invariant parameters
                      are constant down the time axis; see `expand_static` and the
                      `expander` argument on TogetherFlowSimulator.
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
    # theta is (T, P): one parameter row per timestep. Time-invariant parameters
    # are simply constant down the rows, so a fixed-w model and a time-varying-w
    # model run through exactly the same code path.
    # Which theta column carries which inferred quantity. Slots are declared
    # rather than inferred from the array width: a five-column theta used to mean
    # "alpha is in column 4" by convention, so any other fifth parameter — the
    # separation gain, a random-walk scale — would have been silently read as the
    # beacon salience exponent and its own term left at its default.

    num_timesteps = int(time_horizon / dt)
    num_radii = reference_radii.shape[0]

    positions  = np.zeros((num_timesteps, num_agents, 2))
    rotations  = np.zeros((num_timesteps, num_agents))
    neighbors  = np.zeros((num_timesteps, num_agents))
    distances  = np.zeros((num_timesteps, num_agents))
    ang_vels   = np.zeros((num_timesteps, num_agents))
    nbr_flucts = np.zeros((num_timesteps, num_agents))
    ms_counts  = np.zeros((num_timesteps, num_agents, num_radii))

    # Per-agent influence weights, drawn once per trial and held for its
    # duration: individual differences are a property of the person, not noise
    # that resamples every step. Under complete pooling (sigma_slot < 0) the
    # array is simply filled with the shared value each step.
    agent_weights = np.zeros(num_agents)
    if sigma_slot >= 0:
        mu = theta[0, 0]
        if mu < 1e-6:
            mu = 1e-6
        elif mu > 1.0 - 1e-6:
            mu = 1.0 - 1e-6
        logit_mu = np.log(mu / (1.0 - mu))
        sigma = theta[0, sigma_slot]
        for i in range(num_agents):
            z = logit_mu + sigma * np.random.normal(0.0, 1.0)
            agent_weights[i] = 1.0 / (1.0 + np.exp(-z))

    # Scenarios specify their own layout; everything else samples one. An empty
    # array is the sentinel because numba cannot branch on a None-or-array type.
    if init_positions.shape[0] > 0:
        positions[0] = init_positions
        rotations[0] = init_rotations
    else:
        positions[0], rotations[0] = initialize_agents(num_agents, room_size=room_size)

    if fixed_beacons.shape[0] > 0:
        beacon_positions = fixed_beacons
    else:
        beacon_positions = initialize_beacons(
            num_beacons, room_sensing_range=beacon_spread, room_size=room_size
        )

    for t in range(1, num_timesteps):
        # Parameters are read per timestep. For a static model every row is
        # identical and this is equivalent to the previous behaviour.
        if sigma_slot < 0:
            for i in range(num_agents):
                agent_weights[i] = theta[t, 0]
        sensing_radius   = theta[t, 1]
        velocity         = theta[t, 2]
        internal_focus   = theta[t, 3]
        if alpha_slot >= 0:
            salience_sensitivity = theta[t, alpha_slot]
        if kappa_slot >= 0:
            repulsion_gain = theta[t, kappa_slot]

        ps, rs, nn, ad, rc = combined_influences(
            agent_positions=positions[t - 1],
            agent_rotations=rotations[t - 1],
            beacon_positions=beacon_positions,
            beacon_strengths=beacon_strengths,
            salience_sensitivity=salience_sensitivity,
            reference_radii=reference_radii,
            room_size=room_size,
            velocity=velocity,
            sensing_radius=sensing_radius,
            dt=dt,
            influence_weights=agent_weights,
            internal_focus=internal_focus,
            relative_heading=relative_heading,
            repulsion_radius=repulsion_radius,
            repulsion_gain=repulsion_gain,
            obstacles=obstacles,
            max_turn_rate=max_turn_rate,
            door_wall=door_wall,
            door_center=door_center,
            door_half_width=door_half_width,
            beacon_assignment=beacon_assignment,
            diffusive_heading=diffusive_heading,
        )
        positions[t]  = ps
        rotations[t]  = rs
        neighbors[t]  = nn
        distances[t]  = ad
        ang_vels[t]   = rs - rotations[t - 1]
        nbr_flucts[t] = nn - neighbors[t - 1]
        ms_counts[t]  = rc

    # Backfill t=0 statistics from t=1 (no previous state to diff against)
    neighbors[0]  = neighbors[1]
    nbr_flucts[0] = nbr_flucts[1]
    ms_counts[0]  = ms_counts[1]

    return positions, rotations, neighbors, distances, ang_vels, nbr_flucts, ms_counts


def expand_static(thetas, num_timesteps):
    """Broadcast time-invariant parameters to a per-timestep trajectory.

    This is the default expander and the fixed-parameter path: every row of the
    returned (B, T, P) array is the same draw, so the dynamics are identical to
    treating theta as a single constant vector.

    Parameters
    ----------
    thetas        : np.ndarray of shape (B, P) — one prior draw per simulation
    num_timesteps : int

    Returns
    -------
    np.ndarray of shape (B, T, P), C-contiguous (numba requires it)
    """
    B, P = thetas.shape
    return np.ascontiguousarray(
        np.broadcast_to(thetas[:, None, :], (B, num_timesteps, P))
    )


def make_partial_expander(trajectories):
    """Build an expander where some parameter columns vary over time.

    Parameters
    ----------
    trajectories : dict[int, np.ndarray]
        Maps a column index to a (B, T) array of per-timestep values for that
        parameter. Columns not listed stay constant at their prior draw, so a
        model with a time-varying w and fixed r, v, noise is expressed as
        ``{0: w_traj}``.

    Notes
    -----
    Intended for pairing with `superstats.transition.*`, whose `.sample(
    batch_size, num_steps)` returns exactly the (B, T) `local_params` array this
    expects. superstats' own GenerativeModel cannot drive this simulator — it
    assumes observations are conditionally independent given the parameters at
    each step, whereas these dynamics carry state between timesteps — but its
    transition models are usable on their own.
    """
    def expander(thetas, num_timesteps):
        out = expand_static(thetas, num_timesteps).copy()
        for col, traj in trajectories.items():
            traj = np.asarray(traj)
            if traj.shape != (out.shape[0], num_timesteps):
                raise ValueError(
                    f"trajectory for column {col} must have shape "
                    f"({out.shape[0]}, {num_timesteps}); got {traj.shape}"
                )
            out[:, :, col] = traj
        return np.ascontiguousarray(out)
    return expander


def make_logit_random_walk_expander(w_col=0, tau_col=4, dt=0.1):
    """Expander where w follows a Gaussian random walk on the logit scale.

        logit w_t = logit w_{t-1} + tau * sqrt(dt) * xi_t,    xi_t ~ N(0, 1)

    with w_0 and tau read from the prior draw. This is the non-stationary
    influence weight of the extension appendix: tau = 0 reproduces the
    stationary model exactly, so stationarity becomes a testable restriction
    rather than an assumption.

    Note this deliberately does not use `superstats.transition.RandomWalk`.
    That class samples its own initial state and step scale from its internal
    priors and hands them back as `hyper_params`, whereas here (w_0, tau) must
    be the values the simulator's own prior drew — they are the inference
    targets, and a path generated from a different draw would make the reported
    posterior refer to parameters that never entered the simulation.

    Parameters
    ----------
    w_col   : int   — column holding w_0, also the column the path is written to
    tau_col : int   — column holding the walk scale. Nothing reads it inside the
                      kernel; it shapes the path here and is otherwise inert,
                      so it must NOT be named "alpha" or "kappa" (see the slot
                      handling in `simulator_fun`).
    dt      : float — must match the simulator's dt, so tau is per unit time
                      rather than per step and is comparable across horizons.

    Returns
    -------
    callable(thetas, num_timesteps) -> (B, T, P)
    """
    def expander(thetas, num_timesteps):
        out = expand_static(thetas, num_timesteps).copy()
        w0 = np.clip(thetas[:, w_col], 1e-6, 1 - 1e-6)
        tau = np.maximum(thetas[:, tau_col], 0.0)

        logit = np.log(w0 / (1.0 - w0))[:, None]
        steps = np.random.normal(size=(thetas.shape[0], num_timesteps))
        steps[:, 0] = 0.0                       # t=0 is exactly w_0
        increments = tau[:, None] * np.sqrt(dt) * steps
        path = logit + np.cumsum(increments, axis=1)

        out[:, :, w_col] = 1.0 / (1.0 + np.exp(-path))
        return np.ascontiguousarray(out)
    return expander


@njit
def _seed_numba_rng(seed):
    """Seed numba's RNG on the calling thread. Numba's state is independent of
    NumPy's Python-level state, so seeding must happen inside a jitted function."""
    np.random.seed(seed)


@njit(parallel=True)
def _batch_simulator(thetas, num_agents, num_beacons, room_size, dt, time_horizon,
                     beacon_strengths, salience_sensitivity, reference_radii,
                     beacon_spread, relative_heading, base_seed,
                     repulsion_radius, repulsion_gain, obstacles, max_turn_rate,
                     door_wall, door_center, door_half_width,
                     init_positions, init_rotations, fixed_beacons, beacon_assignment,
                     diffusive_heading, alpha_slot, kappa_slot, sigma_slot):
    batch_size    = thetas.shape[0]
    num_timesteps = int(time_horizon / dt)
    num_radii     = reference_radii.shape[0]

    all_pos = np.zeros((batch_size, num_timesteps, num_agents, 2))
    all_rot = np.zeros((batch_size, num_timesteps, num_agents))
    all_nbr = np.zeros((batch_size, num_timesteps, num_agents))
    all_dst = np.zeros((batch_size, num_timesteps, num_agents))
    all_av  = np.zeros((batch_size, num_timesteps, num_agents))
    all_nf  = np.zeros((batch_size, num_timesteps, num_agents))
    all_ms  = np.zeros((batch_size, num_timesteps, num_agents, num_radii))

    for b in prange(batch_size):
        # Seed per simulation, not per thread: numba gives each worker thread its
        # own RNG state, so a thread-level seed would make results depend on how
        # the scheduler happened to distribute iterations. Seeding by index makes
        # simulation b reproducible regardless of thread count or ordering.
        if base_seed >= 0:
            np.random.seed(base_seed + b)

        pos, rot, nbr, dst, av, nf, ms = simulator_fun(
            thetas[b], num_agents, num_beacons, room_size, 1.0, dt, 0.7, 2.5, 0.1, time_horizon,
            beacon_strengths, salience_sensitivity, reference_radii, beacon_spread,
            relative_heading,
            repulsion_radius, repulsion_gain, obstacles, max_turn_rate,
            door_wall, door_center, door_half_width,
            init_positions, init_rotations, fixed_beacons, beacon_assignment,
            diffusive_heading, alpha_slot, kappa_slot, sigma_slot,
        )
        all_pos[b] = pos
        all_rot[b] = rot
        all_nbr[b] = nbr
        all_dst[b] = dst
        all_av[b]  = av
        all_nf[b]  = nf
        all_ms[b]  = ms

    return all_pos, all_rot, all_nbr, all_dst, all_av, all_nf, all_ms


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
        beacon_strengths=None,
        salience_sensitivity: float = 0.0,
        reference_radii=None,
        beacon_spread: float = 50.,
        seed=None,
        param_names=("w", "r", "v", "noise"),
        relative_heading: bool = False,
        expander=None,
        repulsion_radius: float = 0.0,
        repulsion_gain: float = 0.0,
        obstacles=None,
        max_turn_rate: float = 6.283185307179586,
        door_wall: int = -1,
        door_center: float = 0.0,
        door_half_width: float = 0.0,
        init_positions=None,
        init_rotations=None,
        fixed_beacons=None,
        beacon_assignment=None,
        diffusive_heading: bool = False,
        include_parameter_paths: bool = False,
    ):
        self.relative_heading = bool(relative_heading)
        # Whether eta is a diffusion coefficient on the heading (True) or a
        # perturbation of the alignment target (False, the published behaviour).
        self.diffusive_heading = bool(diffusive_heading)
        # Turns prior draws (B, P) into per-timestep parameters (B, T, P).
        # The default broadcasts, i.e. all parameters are time-invariant.
        self.expander = expander if expander is not None else expand_static
        self.param_names = tuple(param_names)
        # Emit the per-timestep parameter path alongside the prior draws. Needed
        # whenever the path itself is an inference target rather than a latent
        # nuisance — the superstatistics formulation, where the estimator
        # produces a posterior over w_t at every step instead of over (w0, tau).
        self.include_parameter_paths = bool(include_parameter_paths)
        # Derived from param_names so a variant declares its parameters once and
        # the kernel is told explicitly where to find them.
        self.alpha_slot = self.param_names.index("alpha") if "alpha" in self.param_names else -1
        self.kappa_slot = self.param_names.index("kappa") if "kappa" in self.param_names else -1
        # Presence of sigma_w is what switches the model from complete to
        # partial pooling; slot 0 is then read as the population mean.
        self.sigma_slot = self.param_names.index("sigma_w") if "sigma_w" in self.param_names else -1
        self.num_agents = num_agents
        self.num_beacons = num_beacons
        self.room_size = room_size
        self.dt = dt
        self.time_horizon = time_horizon
        self.downsample = downsample
        self.downsample_factor = int(downsample_factor)
        self.prior = prior if prior is not None else complete_pooling_prior
        self.output_mode = output_mode
        self.beacon_spread = float(beacon_spread)

        # Fixed radii for r-free neighbour counts. Empty by default so the
        # channel costs nothing unless a variant asks for it.
        if reference_radii is None:
            self.reference_radii = np.zeros(0, dtype=np.float64)
        else:
            self.reference_radii = np.asarray(reference_radii, dtype=np.float64)

        # Collision avoidance. Both default to 0, i.e. off, so every existing
        # variant keeps the published dynamics apart from the boundary fix.
        self.repulsion_radius = float(repulsion_radius)
        self.repulsion_gain = float(repulsion_gain)
        self.max_turn_rate = float(max_turn_rate)

        # Empty arrays rather than None: numba needs a concrete array type, and
        # shape[0] == 0 is the "not supplied" test inside the kernel.
        self.obstacles = (
            np.zeros((0, 3), dtype=np.float64) if obstacles is None
            else np.ascontiguousarray(obstacles, dtype=np.float64)
        )
        if self.obstacles.ndim != 2 or self.obstacles.shape[1] != 3:
            raise ValueError(
                f"obstacles must have shape (K, 3) as (x, y, radius); got {self.obstacles.shape}"
            )

        self.door_wall = int(door_wall)
        self.door_center = float(door_center)
        self.door_half_width = float(door_half_width)

        self.init_positions = (
            np.zeros((0, 2), dtype=np.float64) if init_positions is None
            else np.ascontiguousarray(init_positions, dtype=np.float64)
        )
        self.init_rotations = (
            np.zeros(0, dtype=np.float64) if init_rotations is None
            else np.ascontiguousarray(init_rotations, dtype=np.float64)
        )
        if self.init_positions.shape[0] and self.init_positions.shape[0] != num_agents:
            raise ValueError(
                f"init_positions must have shape ({num_agents}, 2); got {self.init_positions.shape}"
            )
        if self.init_positions.shape[0] != self.init_rotations.shape[0]:
            raise ValueError("init_positions and init_rotations must describe the same agents")

        # float32 to match what `initialize_beacons` returns: inside the kernel
        # both feed the same variable, and numba cannot unify the two dtypes.
        self.fixed_beacons = (
            np.zeros((0, 2), dtype=np.float32) if fixed_beacons is None
            else np.ascontiguousarray(fixed_beacons, dtype=np.float32)
        )
        if self.fixed_beacons.shape[0] and self.fixed_beacons.shape[0] != num_beacons:
            raise ValueError(
                f"fixed_beacons must have shape ({num_beacons}, 2); got {self.fixed_beacons.shape}"
            )

        # Per-agent scripted goals; -1 defers to the selection rule. Empty means
        # every agent uses the rule, which is the published behaviour.
        self.beacon_assignment = (
            np.zeros(0, dtype=np.int64) if beacon_assignment is None
            else np.ascontiguousarray(beacon_assignment, dtype=np.int64)
        )
        if self.beacon_assignment.shape[0]:
            if self.beacon_assignment.shape[0] != num_agents:
                raise ValueError(
                    f"beacon_assignment must have shape ({num_agents},); "
                    f"got {self.beacon_assignment.shape}"
                )
            if self.beacon_assignment.max() >= num_beacons:
                raise ValueError(
                    f"beacon_assignment indexes beacon {self.beacon_assignment.max()} "
                    f"but only {num_beacons} beacons exist"
                )

        # seed=None means non-reproducible (the historical behaviour). An integer
        # seed makes both the prior draws and the trajectories reproducible.
        self.seed = seed
        self._call_count = 0

        if beacon_strengths is None:
            self.beacon_strengths = np.ones(num_beacons, dtype=np.float32)
        else:
            self.beacon_strengths = np.asarray(beacon_strengths, dtype=np.float32)
            if self.beacon_strengths.shape != (num_beacons,):
                raise ValueError(f"beacon_strengths must have shape ({num_beacons},); got {self.beacon_strengths.shape}")

        self.salience_sensitivity = float(salience_sensitivity)

        if output_mode not in ("flat", "raw", "summary"):
            raise ValueError(f"output_mode must be 'flat', 'raw', or 'summary'; got '{output_mode}'")

    def sample(self, batch_size: int | tuple = 1) -> dict[str, np.ndarray]:
        if isinstance(batch_size, tuple):
            if len(batch_size) != 1:
                raise ValueError(f"Expected batch_size as int or (int,), got {batch_size}")
            batch_size = batch_size[0]
        elif not isinstance(batch_size, int):
            raise ValueError(f"batch_size must be int or (int,), got {type(batch_size)}")

        # Advance the seed cursor so repeated sample() calls (as in online
        # training) draw *different* batches while the whole sequence stays
        # reproducible for a given seed.
        if self.seed is None:
            base_seed = -1
        else:
            base_seed = int(self.seed) + self._call_count
            _seed_numba_rng(base_seed)
            # Expanders are numpy, not numba, and the two RNGs are independent.
            np.random.seed(base_seed)
        self._call_count += batch_size

        thetas = np.stack([self.prior() for _ in range(batch_size)])  # (B, P)

        # Prior draws remain the inference targets; the expander turns them into
        # the per-timestep parameter array the kernel consumes.
        num_timesteps = int(self.time_horizon / self.dt)
        thetas_t = np.ascontiguousarray(
            self.expander(thetas, num_timesteps), dtype=np.float64
        )
        if thetas_t.shape[:2] != (batch_size, num_timesteps):
            raise ValueError(
                f"expander must return shape ({batch_size}, {num_timesteps}, P); "
                f"got {thetas_t.shape}"
            )

        all_pos, all_rot, all_nbr, all_dst, all_av, all_nf, all_ms = _batch_simulator(
            thetas_t, self.num_agents, self.num_beacons, self.room_size, self.dt, self.time_horizon,
            self.beacon_strengths, self.salience_sensitivity, self.reference_radii,
            self.beacon_spread, self.relative_heading, base_seed,
            self.repulsion_radius, self.repulsion_gain, self.obstacles, self.max_turn_rate,
            self.door_wall, self.door_center, self.door_half_width,
            self.init_positions, self.init_rotations, self.fixed_beacons,
            self.beacon_assignment, self.diffusive_heading,
            self.alpha_slot, self.kappa_slot, self.sigma_slot,
        )

        positions  = all_pos   # (B, T, A, 2)
        rotations  = all_rot   # (B, T, A)
        neighbors  = all_nbr   # (B, T, A)
        distances  = all_dst   # (B, T, A)
        ang_vels   = all_av    # (B, T, A)
        nbr_flucts = all_nf    # (B, T, A)
        ms_counts  = all_ms    # (B, T, A, R)

        if self.downsample:
            s = self.downsample_factor
            # ascontiguousarray: strided views make every downstream batch slice a
            # strided copy, which dominates training time for the smaller array.
            positions  = np.ascontiguousarray(positions[:, ::s, :, :])
            rotations  = np.ascontiguousarray(rotations[:, ::s, :])
            neighbors  = np.ascontiguousarray(neighbors[:, ::s, :])
            distances  = np.ascontiguousarray(distances[:, ::s, :])
            ang_vels   = np.ascontiguousarray(ang_vels[:, ::s, :])
            nbr_flucts = np.ascontiguousarray(nbr_flucts[:, ::s, :])
            ms_counts  = np.ascontiguousarray(ms_counts[:, ::s, :, :])

        B, T, A, _ = positions.shape
        R = self.reference_radii.shape[0]

        out = {name: thetas[:, i:i + 1] for i, name in enumerate(self.param_names)}
        if self.include_parameter_paths:
            # (B, T, 1) per parameter — the values the kernel actually used at
            # each step, after the expander. For a static expander these are
            # constant down the time axis by construction.
            for i, name in enumerate(self.param_names):
                out[f"{name}_path"] = thetas_t[:, :, i:i + 1].astype(np.float32)

        if self.output_mode == "flat":
            out |= {
                "positions":             positions.reshape(B, T, A * 2),
                "rotations":             rotations,
                "neighbors":             neighbors,
                "distances":             distances,
                "angular_velocities":    ang_vels,
                "neighbor_fluctuations": nbr_flucts,
            }
            if R > 0:
                out["radii_counts"] = ms_counts.reshape(B, T, A * R)

        elif self.output_mode == "raw":
            out |= {
                "positions":             positions,
                "rotations":             rotations[..., None],
                "neighbors":             neighbors[..., None],
                "distances":             distances[..., None],
                "angular_velocities":    ang_vels[..., None],
                "neighbor_fluctuations": nbr_flucts[..., None],
            }
            if R > 0:
                out["radii_counts"] = ms_counts

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
            if R > 0:
                out["radii_counts"] = np.concatenate(
                    [_summarize(ms_counts[:, :, :, k]) for k in range(R)], axis=-1
                )

        return out
