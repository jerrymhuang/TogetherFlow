import matplotlib.pyplot as plt
from utils import make_figure, set_layout


def plot_direction(n, A, B):
    fig, axs = plt.subplots(n, 1, figsize=(n, 1))

    if n == 1:  # If only one subplot, make sure axs is iterable
        axs = [axs]

    for i in range(n):
        ax = axs[i]

        # Plot point A
        ax.scatter(A[i,0], A[i,1], color='blue', label='Point A', zorder=5)

        # Plot point B
        ax.scatter(B[0], B[1], color='red', label='Point B', zorder=5)

        # Calculate the direction vector from A to B
        dx = B[0] - A[i,0]
        dy = B[1] - A[i,1]

        # Plot direction as an arrow
        ax.quiver(A[i,0], A[i,1], dx, dy, angles='xy', scale_units='xy', scale=1, color='black', width=0.005,
                  label='Direction', zorder=4)

        # Adding labels and grid
        ax.set_xlim(min(A[i,0], B[0]) - 1, max(A[i,0], B[0]) + 1)
        ax.set_ylim(min(A[i,1], B[1]) - 1, max(A[i,1], B[1]) + 1)
        ax.set_title(f'Subplot {i + 1}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True)

    fig.tight_layout()
    return fig
