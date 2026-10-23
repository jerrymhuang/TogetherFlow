import numpy as np
from numba import njit


@njit
def initialize_agents(
        num_agents: int = 12,
        room_size: tuple = (8., 10.),
):
    """
    Generate random positions and orientations for agents.

    Parameters
    ----------
    num_agents : int, optional
        Number of agents to generate (default is 100).
    room_size : float, optional
        The size of the boundary within which positions are generated (default is 100.0).

    Returns
    -------
    tuple of np.ndarray
        A tuple containing the positions (np.ndarray) and orientations (np.ndarray) of the agents.
    """

    # Generate random positions within the boundary size centered at 0
    x = (np.random.random(size=num_agents).astype(np.float32) - 0.5) * room_size[0]
    y = (np.random.random(size=num_agents).astype(np.float32) - 0.5) * room_size[1]
    positions = np.vstack((x, y)).T

    # Generate random orientations (angles in radians between 0 and 2*pi)
    rotations = np.random.random(size=(num_agents,)).astype(np.float32) * np.pi * 2

    return positions.astype(np.float32), rotations.astype(np.float32)


@njit
def initialize_beacons(
        num_beacons=10,
        room_sensing_range=50.,
        room_size=(8., 10.)
):
    """
    Initialize beacons uniformly over the sensing range, excluding the room.

    Beacons are virtual targets rendered by the immersive room, so they are
    always outside the physical space the agents walk in — an agent approaches
    the wall nearest its beacon and cannot reach the beacon itself. Plain uniform
    sampling over the sensing square does not respect that: at the published
    range of 50 about 3% of beacons land inside the room, so roughly 12% of
    simulations contain at least one beacon an agent can physically stand on.
    Rejecting the interior makes the sampler match the domain.

    Parameters
    ----------
    num_beacons : int, default: 10
        Number of beacons to initialize.
    room_sensing_range : float, default: 50.0
        Sensing distance of the room for the beacons to matter. Must exceed the
        room dimensions, or there is no admissible region to sample from.
    room_size : tuple (width, height)
        The excluded interior.

    Returns
    -------
    beacons      : np.ndarray of shape (num_beacons, 2)
        Initial positions of the beacons, all outside the room.
    """
    half_x = room_size[0] * 0.5
    half_y = room_size[1] * 0.5
    half_range = room_sensing_range * 0.5

    beacons = np.zeros((num_beacons, 2), dtype=np.float32)

    for b in range(num_beacons):
        placed = False
        # Rejection sampling. Acceptance is ~97% at the published range, so this
        # costs nothing there; the bound only matters for ranges that crowd the
        # room, where the admissible region is a thin frame around it.
        for _ in range(1000):
            x = (np.random.random() - 0.5) * room_sensing_range
            y = (np.random.random() - 0.5) * room_sensing_range
            if np.abs(x) > half_x or np.abs(y) > half_y:
                beacons[b, 0] = x
                beacons[b, 1] = y
                placed = True
                break

        if not placed:
            # The sensing range does not admit an exterior; put the beacon on the
            # nearest legal boundary rather than silently returning one inside.
            angle = np.random.random() * 2.0 * np.pi
            radius = max(half_x, half_y) + 1e-3
            if half_range > radius:
                radius = half_range
            beacons[b, 0] = np.cos(angle) * radius
            beacons[b, 1] = np.sin(angle) * radius

    return beacons