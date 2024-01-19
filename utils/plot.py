"""Utility functions to plot data."""

import matplotlib.pyplot as plt
from matplotlib import pylab

# Define parameter for plotting
params = {
    "legend.fontsize": "medium",
    "figure.figsize": (10, 10),
    "axes.labelsize": "medium",
    "axes.titlesize": "large",
    "xtick.labelsize": "medium",
    "ytick.labelsize": "medium",
    "font.size": 13,
}
pylab.rcParams.update(params)


def plot_fps(det_par, det_count, show_mean: bool):
    """Plot frame rate change in line chart.

    Args:
      det_par: Array of detection parameter to be plotted (y_axis)
      det_count: Array of counter to plot in x axis
      show_mean: whether to show the mean of the graph or not
    """
    # Plot the line
    plt.plot(det_count, det_par, color="red", label="FPS", linewidth=3)

    if show_mean:
        # Calculate mean of the parameter
        par_mean = det_par[1:].mean()
        # Plot it
        plt.axhline(
            par_mean,
            color="blue",
            label="mean",
            linewidth=2,
            linestyle="--"
        )

    # Add title and label
    plt.title("FPS Change during Detection")
    plt.xlabel("Detection Count")
    plt.ylabel("FPS Level")

    # Add grid and legend
    plt.grid(True)
    plt.legend()

    # Show image
    plt.tight_layout()
    plt.show()


def plot_delay(det_par, det_count, show_mean: bool):
    """Plot delay change in line chart.

    Args:
      det_par: Array of detection parameter to be plotted (y_axis)
      det_count: Array of counter to plot (x axis)
      show_mean: whether to show the mean of the graph or not
    """
    # Plot the line
    plt.plot(det_count, det_par, color="red", label="Delay", linewidth=3)

    if show_mean:
        # Calculate mean of the parameter
        par_mean = det_par[1:].mean()
        # Plot it
        plt.axhline(
            par_mean,
            color="blue",
            label="mean",
            linewidth=2,
            linestyle="--"
        )

    # Add title and label
    plt.title("Delay Time during Detection")
    plt.xlabel("Detection Count")
    plt.ylabel("Time (Second)")

    # Add grid and legend
    plt.grid(True)
    plt.legend()

    # Show image
    plt.tight_layout()
    plt.show()


def plot_detection(det_ratio: list, color: list = ["#008fd5", "#fc4f30"]):
    """Plot detection ratio in pie chart.

    Args:
      det_ratio: List of ratio of detected parameters.
      color: list of color for each pie segment.
    """
    # Plot the image
    plt.pie(
        det_ratio,
        labels=["Yes", "No"],
        colors=color,
        wedgeprops={"edgecolor": "black", "linewidth": 2},
        autopct="%.1f%%",
    )

    # Add title
    plt.title("Ratio of Detected Object")

    # Show image
    plt.tight_layout()
    plt.show()
