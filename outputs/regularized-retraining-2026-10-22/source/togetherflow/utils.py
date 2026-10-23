import numpy as np
from numba import njit


@njit
def count_neighbors(self_position, other_positions, sensing_radius):
    """
    Count agents within sensing_radius and compute their average distance.

    Parameters
    ----------
    self_position   : np.ndarray of shape (2,)
    other_positions : np.ndarray of shape (N, 2)
    sensing_radius  : float

    Returns
    -------
    num_neighbors    : int
    average_distance : float
    """
    num_neighbors = 0
    distances = []

    for i in range(len(other_positions)):
        dx = other_positions[i, 0] - self_position[0]
        dy = other_positions[i, 1] - self_position[1]
        d = (dx ** 2 + dy ** 2) ** 0.5
        if 0.0 < d <= sensing_radius:
            distances.append(d)
            num_neighbors += 1

    if len(distances) > 0:
        average_distance = np.mean(np.array(distances))
    else:
        average_distance = 0.0

    return num_neighbors, average_distance


@njit
def bound_agent_state(
    previous_position,
    new_position,
    new_rotation,
    room_size=(8., 10.),
    boundary_noise=0.01,
    door_wall=-1,
    door_center=0.0,
    door_half_width=0.0,
):
    """
    Reflect an agent off the rectangular room boundary, heading included.

    The room is centred at the origin, so the walls sit at +/-width/2 and
    +/-height/2. A step that would cross a wall is mirrored back inside and the
    corresponding velocity component is negated, which is what makes the room a
    closed environment: an agent that walks into a wall turns along it instead of
    passing through.

    The predecessor of this function added `+boundary_noise` in the *same*
    direction regardless of which wall was crossed, so it pushed agents further
    out past the +x/+y walls and corrected by only 0.01 past the -x/-y walls.
    Agents therefore left the room and followed their beacons out to the beacon
    spread; the room was a claim of the write-up rather than a property of the
    simulation.

    Parameters
    ----------
    previous_position : np.ndarray of shape (2,)
        Position at t-1. Agents that were already outside are left alone, so an
        agent that has escaped through a door keeps going instead of being
        reflected off a wall it is no longer behind.
    new_position      : np.ndarray of shape (2,)
        Proposed position at t, before boundary handling.
    new_rotation      : float
        Proposed heading at t, before boundary handling.
    room_size         : tuple (width, height)
    boundary_noise    : float
        Inward margin kept after reflection, so an agent never lands exactly on
        a wall where the crossing test is a floating-point tie.
    door_wall         : int
        Wall carrying a gap: -1 none, 0 = +x, 1 = -x, 2 = +y, 3 = -y. A crossing
        within the gap is not reflected, which is what lets agents exit a room.
    door_center       : float
        Centre of the gap along the wall (y for the +/-x walls, x for +/-y).
    door_half_width   : float

    Returns
    -------
    bounded_position : np.ndarray of shape (2,)
    bounded_rotation : float
    """
    half_x = room_size[0] * 0.5
    half_y = room_size[1] * 0.5

    bounded = new_position.copy()

    # An agent that was already outside is not behind any wall, so there is
    # nothing to reflect it off. Without this an escaped agent would be bounced
    # around the *outside* of the room by the same tests that contain the others.
    was_outside = (
        np.abs(previous_position[0]) > half_x or np.abs(previous_position[1]) > half_y
    )
    if was_outside:
        return bounded, new_rotation

    # Work on the velocity vector so the heading reflects with the position.
    cos_r = np.cos(new_rotation)
    sin_r = np.sin(new_rotation)

    if bounded[0] > half_x:
        if not (door_wall == 0 and np.abs(bounded[1] - door_center) <= door_half_width):
            bounded[0] = 2.0 * half_x - bounded[0]
            cos_r = -cos_r
    elif bounded[0] < -half_x:
        if not (door_wall == 1 and np.abs(bounded[1] - door_center) <= door_half_width):
            bounded[0] = -2.0 * half_x - bounded[0]
            cos_r = -cos_r

    if bounded[1] > half_y:
        if not (door_wall == 2 and np.abs(bounded[0] - door_center) <= door_half_width):
            bounded[1] = 2.0 * half_y - bounded[1]
            sin_r = -sin_r
    elif bounded[1] < -half_y:
        if not (door_wall == 3 and np.abs(bounded[0] - door_center) <= door_half_width):
            bounded[1] = -2.0 * half_y - bounded[1]
            sin_r = -sin_r

    # Reflection alone can still leave an agent on the wrong side if a single
    # step somehow exceeded the room, so clamp whatever remains inside — unless
    # it left through the door, in which case it is meant to be outside.
    left_by_door = (
        (door_wall == 0 and bounded[0] > half_x)
        or (door_wall == 1 and bounded[0] < -half_x)
        or (door_wall == 2 and bounded[1] > half_y)
        or (door_wall == 3 and bounded[1] < -half_y)
    )
    if not left_by_door:
        if bounded[0] > half_x - boundary_noise:
            bounded[0] = half_x - boundary_noise
        elif bounded[0] < -half_x + boundary_noise:
            bounded[0] = -half_x + boundary_noise
        if bounded[1] > half_y - boundary_noise:
            bounded[1] = half_y - boundary_noise
        elif bounded[1] < -half_y + boundary_noise:
            bounded[1] = -half_y + boundary_noise

    return bounded, np.mod(np.arctan2(sin_r, cos_r), 2.0 * np.pi)
