import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import FancyBboxPatch


def prepare_figure(figsize=(4, 4), lim=20.):
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Add the static room boundary once
    room = FancyBboxPatch(
        (-5., -6.), 10., 12.,
        boxstyle="round, pad=0.5, rounding_size=1.5",
        alpha=0.1
    )
    ax.add_patch(room)

    # Set axis limits and labels once
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_aspect('equal')

    return fig, ax


def inspect_initialization(agent_positions, agent_rotations, beacon_positions):

    fig, ax = prepare_figure(figsize=(8,8))

    ax.quiver(
        agent_positions[:, 0], agent_positions[:, 1],
        np.cos(agent_rotations), np.sin(agent_rotations),
        color="b"
    )

    ax.scatter(agent_positions[:, 0], agent_positions[:, 1], color='blue')
    ax.scatter(beacon_positions[:, 0], beacon_positions[:, 1], color='red')

    return fig

def inspect_simulation(sim_data, beacons, export_video=False):
    positions = sim_data[:, :, 0:2]
    rotations = sim_data[:, :, -1]

    fig, ax = prepare_figure()

    # Initialize the quiver object once
    quiver = ax.quiver(
        positions[0, :, 0], positions[0, :, 1],
        np.cos(rotations[0]), np.sin(rotations[0])
    )

    ax.scatter(x=beacons[:,0], y=beacons[:,1], c='r', marker='o')

    def update(frame):
        # Update offsets and angles for quiver arrows
        quiver.set_offsets(positions[frame])
        quiver.set_UVC(np.cos(rotations[frame]), np.sin(rotations[frame]))
        return quiver,

    # Use FuncAnimation with blit=True for faster performance
    anim = FuncAnimation(fig, update, frames=len(positions), blit=True, repeat=False)

    if export_video:
        writer = FFMpegWriter()
        anim.save("animation.mp4", writer=writer, fps=30)

    return anim




# These are old. But I might add more so they become useful again.
def inspect_rotation_influence(sim, agent_position, beacon_position):
    f, ax = prepare_figure()

    ax.scatter(x=beacon_position[0], y=beacon_position[1], c='r', marker='o')

    # Initialize the quiver object once
    quiver = ax.quiver(
        agent_position[0], agent_position[1],
        np.cos(sim[0]), np.sin(sim[0])
    )

    def update(frame):
        # Update offsets and angles for quiver arrows
        # quiver.set_offsets(positions[frame])
        quiver.set_UVC(np.cos(sim[frame]), np.sin(sim[frame]))
        return quiver,

    # Use FuncAnimation with blit=True for faster performance
    anim = FuncAnimation(f, update, frames=len(sim), blit=True, repeat=False)

    return anim


def inspect_position_influence(sim, beacon_position):
    f, ax = prepare_figure()

    ax.scatter(x=beacon_position[0], y=beacon_position[1], c='r', marker='o')
    ax.plot(sim[:,0], sim[:,1], c='g')
    # Initialize the quiver object once
    quiver = ax.quiver(
        sim[0, 0], sim[0, 1],
        np.cos(beacon_position[0] - sim[0, 0]),
        np.sin(beacon_position[1] - sim[0, 1])
    )

    def update(frame):
        # Update offsets and angles for quiver arrows
        quiver.set_offsets(sim[frame])
        # quiver.set_offsets(positions[frame])

        relative_pos = beacon_position - sim[frame]

        if (np.linalg.norm(relative_pos) < 0.01):
            unit_direction = relative_pos / np.linalg.norm(relative_pos)
        else:
            unit_direction = np.zeros(2)
        quiver.set_UVC(unit_direction[0], unit_direction[1])
        return quiver,

    # Use FuncAnimation with blit=True for faster performance
    anim = FuncAnimation(f, update, frames=len(sim), blit=True, repeat=False)
    return anim

def inspect_alignment_influence(sim):

    f, ax = prepare_figure()


    # Initialize the quiver object once
    quiver = ax.quiver(
        sim[0, :, 0], sim[0, :, 1],
        np.cos(sim[0,:,-1]), np.sin(sim[0,:,-1])
    )

    scatter = ax.scatter(x=sim[0, :, 0], y=sim[0, :, 1], c='r', marker='o')

    def update(frame):
        # Update offsets and angles for quiver arrows
        quiver.set_offsets(sim[frame, :, 0:2])
        quiver.set_UVC(np.cos(sim[frame,:,-1]), np.sin(sim[frame,:,-1]))
        scatter.set_offsets(sim[frame, :, 0:2])
        # ax.scatter(x=sim[frame, :, 0], y=sim[frame, :, 1], c='r', marker='o')
        return quiver, scatter

    # Use FuncAnimation with blit=True for faster performance
    anim = FuncAnimation(f, update, frames=len(sim), blit=True, repeat=False)

    return anim


def inspect_cohesion_influence(sim):
    f, ax = prepare_figure()

    scatter = ax.scatter(x=sim[0, :, 0], y=sim[0, :, 1], c='r', marker='o')

    def update(frame):
        scatter.set_offsets(sim[frame, :, 0:2])
        return scatter,


    # Use FuncAnimation with blit=True for faster performance
    anim = FuncAnimation(f, update, frames=len(sim), blit=True, repeat=False)
    return anim


def inspect_individual_motion(sim):
    f, ax = prepare_figure()
    def update(frame):
        raise NotImplementedError

    # Use FuncAnimation with blit=True for faster performance
    anim = FuncAnimation(f, update, frames=len(sim), blit=True, repeat=False)
    return anim


def inspect_collective_motion(sim):
    f, ax = prepare_figure()
    def update(frame):
        raise NotImplementedError

    # Use FuncAnimation with blit=True for faster performance
    anim = FuncAnimation(f, update, frames=len(sim), blit=True, repeat=False)
    return anim
