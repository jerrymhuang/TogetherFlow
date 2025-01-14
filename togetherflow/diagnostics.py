import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyBboxPatch


# These are old. But I might add more so they become useful again.
def plot_rotation_influence(
        num_agents,
        agent_positions,
        agent_rotations,
        beacon_positions,
        influence
):
    from matplotlib.patches import FancyBboxPatch

    num_col = int(num_agents / 3)
    num_row = 3

    fig, axes = plt.subplots(num_row, num_col, figsize=(num_col * 3, num_row * 3))

    for i, ax in enumerate(axes.flat):
        room = FancyBboxPatch(
            (-5., -6.), 10., 12.,
            boxstyle="round, pad=0.5, rounding_size=1.5",
            alpha=0.1
        )
        ax.add_patch(room)

        ax.quiver(
            agent_positions[i, 0], agent_positions[i, 1],
            influence[i, 0], influence[i, 1],
            scale=0.25, angles='xy', scale_units='xy', color="r"
        )
        ax.quiver(
            agent_positions[i, 0], agent_positions[i, 1],
            np.cos(agent_rotations[i]), np.sin(agent_rotations[i]),
            scale=0.25, angles='xy', scale_units='xy', color="b"
        )

        ax.scatter(x=agent_positions[i, 0], y=agent_positions[i, 1], c='b', marker='o')
        ax.scatter(x=beacon_positions[0, 0], y=beacon_positions[0, 1], c='r', marker='o')

        ax.set_xlim(-15., 15.)
        ax.set_ylim(-15., 15.)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect("equal")

    fig.tight_layout()
    return fig


def animate_reorientation(
        agent_position,
        agent_rotations,
        beacon_position,
        beacon_influence,
        frames=100
):
    f, ax = plt.subplots(1, 1, figsize=(4, 4))
    room = FancyBboxPatch(
        (-5., -6.), 10., 12.,
        boxstyle="round, pad=0.5, rounding_size=1.5",
        alpha=0.1
    )
    ax.add_patch(room)

    influence = ax.quiver(
        agent_position[0], agent_position[1],
        beacon_influence[0], beacon_influence[1],
        scale=0.25, angles='xy', scale_units='xy', color="r"
    )

    orientation = ax.quiver(
        agent_position[0], agent_position[1],
        np.cos(agent_rotations[0]), np.sin(agent_rotations[0]),
        scale=0.25, angles='xy', scale_units='xy', color="b"
    )

    ax.scatter(x=agent_position[0], y=agent_position[1], c='b', marker='o')
    ax.scatter(x=beacon_position[0], y=beacon_position[1], c='r', marker='o')

    ax.set_xlim(-15., 15.)
    ax.set_ylim(-15., 15.)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect("equal")

    def update(frame):
        rotation = agent_rotations[frame]

        orientation.set_UVC(np.cos(rotation), np.sin(rotation))
        return influence, orientation

    animation = FuncAnimation(f, update, frames=frames, blit=True)
    plt.close(f)

    return animation
