"""Script to measure velocity of the conveyor belt."""

import time
from tkinter import Tk, Button, Frame, BOTH
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hardware.sensor import prepare_gpio, clean_gpio, Infrared
from constant import LIMIT_IR_TIME, RANGE

# Create an instance of tkinter frame or window
win = Tk()

# Set the size of the window
win.geometry("400x400")

# Allowing root window to change
# it's size according to user's need
win.resizable(True, True)

# Name the window
win.title("Velocity Measurement")

# Set a flag
is_running = False

# Define IR instances
prepare_gpio()
ir = Infrared(right_ir=22, left_ir=16)

# Define time variables for measurement
left_ir_time = 0
right_ir_time = 0
old_left_ir_time = 0
old_right_ir_time = 0

# Define global variables
velocity = [0]
par_arr = np.array(velocity)
par_count = np.arange(0, len(velocity), 1)


def measure():
    """Continously measure conveyor velocity until stop button pressed."""
    global left_ir_time
    global right_ir_time
    global old_left_ir_time
    global old_right_ir_time

    if is_running:
        # If object detected by IR sensor, start timer
        if ir.read_sensor()[1] == 0:
            if (time.time() - old_left_ir_time) >= LIMIT_IR_TIME:
                left_ir_time = time.time()
        if ir.read_sensor()[0] == 0:
            if (time.time() - old_right_ir_time) >= LIMIT_IR_TIME:
                right_ir_time = time.time()

        if (left_ir_time != 0) and (right_ir_time != 0):
            delay = right_ir_time - left_ir_time
            if delay > 0:
                detected_velocity = round((RANGE / delay), 4)
                print(f"Conveyor velocity: {detected_velocity} m/s")

                # Reset the measurement
                velocity.append(detected_velocity)
                old_left_ir_time = left_ir_time
                old_right_ir_time = right_ir_time
                left_ir_time = 0
                right_ir_time = 0

    win.after(1, measure)


def create_csv(columns: list, filename: str):
    """Create a CSV file from stacked array.

    Args:
      columns: List of the name of columns for CSV file
      filename: Name or path of the CSV file
    """
    recorded_data = pd.DataFrame(par_arr[1:], columns=columns)
    # Save recorded data
    recorded_data.to_csv(filename, index=False)


def collect_data():
    """Collect data from detection."""
    # Convert list to numpy array
    global par_arr
    par_arr = np.array(velocity)
    # Create array to count number of detection
    global par_count
    par_count = np.arange(0, len(velocity), 1)


def plot_data(show_mean: bool):
    """Plot measurement data.

    Args:
      show_mean: whether to show the mean of the graph or not
    """
    # Plot the image
    plt.figure(figsize=(10, 10))
    plt.plot(par_count, par_arr, color="red", label="velocity", linewidth=3)

    if show_mean:
        # Calculate mean of the parameter
        par_mean = par_arr[1:].mean()
        # Plot it
        plt.axhline(
            par_mean,
            color="blue",
            label="mean",
            linewidth=2,
            linestyle="--"
        )

    # Add title and label
    plt.title("Velocity Graph")
    plt.xlabel("count")
    plt.ylabel("velocity (m/s)")

    # Add grid
    plt.grid(True)

    # Show legend
    plt.legend()

    # Show image
    plt.tight_layout()
    plt.show()


# Define a function to start the loop
def on_start():
    """Start measuremenet."""
    global is_running
    is_running = True


# Define a function to stop the loop
def on_stop():
    """Stop measuremenet."""
    global is_running
    is_running = False


# crprepare_gpio()eating a Frame which can expand according
# to the size of the window
pane = Frame(win)
pane.pack(fill=BOTH, expand=True)

# Add a Button to start/stop the loop
start = Button(pane, text="Start", command=on_start, background="green", fg="white")
start.pack(padx=10, expand=True, fill=BOTH)
stop = Button(pane, text="Stop", command=on_stop, background="red", fg="white")
stop.pack(padx=10, expand=True, fill=BOTH)

# Add button to collect data, plot, and create csv
collect_button = Button(
    pane, text="Collect", command=collect_data, background="blue", fg="white"
)
collect_button.pack(padx=10, expand=True, fill=BOTH)
plot_button = Button(
    pane,
    text="Plot",
    command=lambda: plot_data(True),
    background="orange",
    fg="white"
)
plot_button.pack(padx=10, expand=True, fill=BOTH)
csv = Button(
    pane,
    text="CSV",
    command=lambda: create_csv(["velocity"], "./csv_data/velocity.csv"),
    background="purple",
    fg="white",
)
csv.pack(padx=10, expand=True, fill=BOTH)

# Add exit button
exit_button = Button(
    pane, text="Exit", command=win.quit, background="black", fg="white"
)
exit_button.pack(padx=10, expand=True, fill=BOTH)


# Run a function to measure velocity
win.after(1, measure)

win.mainloop()

# Clean GPIO
clean_gpio()
