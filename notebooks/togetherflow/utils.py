import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch
from IPython.display import HTML


def set_layout(num_total: int, num_row: int = None, num_col: int = None, stacked: bool = False):
    """
    Determine the number of rows and columns in diagnostics visualizations.

    Parameters
    ----------
    num_total     : int
        Total number of parameters
    num_row       : int, default = None
        Number of rows for the visualization layout
    num_col       : int, default = None
        Number of columns for the visualization layout
    stacked     : bool, default = False
        Boolean that determines whether to stack the plot or not.

    Returns
    -------
    num_row       : int
        Number of rows for the visualization layout
    num_col       : int
        Number of columns for the visualization layout
    """
    if stacked:
        num_row, num_col = 1, 1
    else:
        if num_row is None and num_col is None:
            num_row = int(np.ceil(num_total / 6))
            num_col = int(np.ceil(num_total / num_row))
        elif num_row is None and num_col is not None:
            num_row = int(np.ceil(num_total / num_col))
        elif num_row is not None and num_col is None:
            num_col = int(np.ceil(num_total / num_row))

    return num_row, num_col


def make_figure(num_row: int = None, num_col: int = None, figsize: tuple = None):
    """
    Initialize a set of figures

    Parameters
    ----------
    num_row       : int
        Number of rows in a figure
    num_col       : int
        Number of columns in a figure
    figsize       : tuple
        Size of the figure adjusting to the display resolution
        or the user's choice

    Returns
    -------
    f, axes
        Initialized figures
    """
    if num_row == 1 and num_col == 1:
        f, axes = plt.subplots(1, 1, figsize=figsize)
    else:
        if figsize is None:
            figsize = (int(5 * num_col), int(5 * num_row))

        f, axes = plt.subplots(num_row, num_col, figsize=figsize)

    return f, axes


def animate(positions, rotations):
    """
    Animates N agents with given positions and rotations over T timesteps.

    Parameters:
    - positions: numpy array of shape (T, N, 2), where T is the number of timesteps,
      N is the number of agents, and 2 corresponds to the x and y coordinates.
    - rotations: numpy array of shape (T, N), where T is the number of timesteps
      and N is the number of agents. Each value represents the orientation angle in radians.
    """
    T, N, _ = positions.shape

    # Set up the figure and axis
    fig, ax = plt.subplots()
    ax.set_xlim(np.min(positions[:, :, 0]) - 1, np.max(positions[:, :, 0]) + 1)
    ax.set_ylim(np.min(positions[:, :, 1]) - 1, np.max(positions[:, :, 1]) + 1)

    # Create point objects for each agent
    points = [ax.plot([], [], 'o')[0] for _ in range(N)]
    # Create arrow objects for each agent to indicate orientation
    arrows = [FancyArrowPatch((0, 0), (0, 0), arrowstyle='-|>', mutation_scale=15, color='r') for _ in range(N)]
    for arrow in arrows:
        ax.add_patch(arrow)

    def init():
        # Initialize each point and arrow
        for point in points:
            point.set_data([], [])
        for arrow in arrows:
            arrow.set_visible(False)
        return points + arrows

    def update(frame):
        # Update each point and arrow for the given frame
        for i, (point, arrow) in enumerate(zip(points, arrows)):
            x, y = positions[frame, i, 0], positions[frame, i, 1]
            angle = rotations[frame, i]
            point.set_data(x, y)
            arrow.set_visible(True)
            dx = 0.5 * np.cos(angle)
            dy = 0.5 * np.sin(angle)
            arrow.set_positions((x, y), (x + dx, y + dy))
        return points + arrows

    # Create the animation
    anim = FuncAnimation(fig, update, frames=T, init_func=init, blit=True, interval=50)

    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Positions and Rotations Animation')
    plt.close(fig)  # Prevent duplicate static display in Jupyter Notebook
    return HTML(anim.to_jshtml())