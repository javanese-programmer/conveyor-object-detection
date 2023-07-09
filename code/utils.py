"""Utility functions to process detection results"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pylab
from tflite_support.task import processor

# Define parameter for plotting
params = {'legend.fontsize': 'medium',
          'figure.figsize': (10, 10),
          'axes.labelsize': 'medium',
          'axes.titlesize': 'large',
          'xtick.labelsize': 'medium',
          'ytick.labelsize': 'medium',
          'font.size': 13}
pylab.rcParams.update(params)


def visualize(
    image: np.ndarray,
    detection_result: processor.DetectionResult,
    resolution: tuple = (640, 480),
    box_color: tuple = (255, 0, 0)  # blue
) -> np.ndarray:
    """Draws bounding boxes on the input image and return it.
    Args:
      image: The input RGB image.
      detection_result: The list of all "Detection" entities to be visualize.
      resolution: resolution of image frame,
      box_color: Color oh the bounding box.
    Returns:
      Image with bounding boxes.
    """

    # Define viz properties
    width, height = resolution
    size_ratio = np.sqrt(width**2 + height**2) / np.sqrt(640**2 + 480**2)
    margin = 10      # pixels
    row_size = 10    # pixels
    font_size = 1.3 * size_ratio
    font_thickness = 2 * int(size_ratio)
    box_thickness = 3 * int(size_ratio)

    for detection in detection_result.detections:
        # Draw bounding_box
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(image, start_point, end_point, box_color, box_thickness)

        # Draw label and score
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = category_name + ' (' + str(probability) + ')'
        text_location = (margin + bbox.origin_x,
                         margin + row_size + bbox.origin_y)
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    font_size, box_color, font_thickness)

    return image


def localize(detection_result: processor.DetectionResult):
    """Find object bounding box and return it
    Args:
      detection_result: The list of all "Detection" entities.
    Returns:
      Bounding boxes coordinates
    """
    for detection in detection_result.detections:
        # Collect start and end point location of object
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height

        # Collect the score and index
        probability = round(detection.categories[0].score, 3)
        index = detection.categories[0].index

        return probability, index, start_point, end_point


def measure_dim(
        detection_result: processor.DetectionResult,
        frame_height: int = 480):
    """Measure the height, width, and size of detected object in centimeters
    Args:
      detection_result: The list of all "Detection" entities.
      frame_height: The height of frame resolution
    Returns:
      Height, Width, Size of an object in centimeters
    """
    # Define Pixels Per Inch (PPI)
    if frame_height == 600:
        ppi = 122.34
    elif frame_height == 720:
        ppi = 149.53
    elif frame_height == 960:
        ppi = 198.392
    else:
        ppi = 96        # typical value for monitor screen

    # inch to cm conversion
    inch_to_cm = 2.54   # cm

    # pixels to cm conversion
    pixel_to_cm = inch_to_cm / ppi

    for detection in detection_result.detections:
        # Collect object height and width
        bbox = detection.bounding_box
        object_height = bbox.height  # in pixels
        object_width = bbox.width    # in pixels

        # Convert to cm
        true_height = round(object_height * pixel_to_cm, 3)  # in cm
        true_width = round(object_width * pixel_to_cm, 3)    # in cm
        # Calculate size
        true_size = round(true_height * true_width, 3)  # in cm2

        # Collect the score and index
        probability = round(detection.categories[0].score, 3)
        index = detection.categories[0].index

        # Final Dimension
        dimension = (true_height, true_width, true_size)

        return probability, index, dimension


def categorize(detection_result: processor.DetectionResult):
    """Categorize detected objects and return its name, score, and index
    Args:
      detection_result: The list of all "Detection" entities.
    Returns:
      Name, score, and index of an object in centimeters
    """
    for detection in detection_result.detections:
        # Find the category
        category = detection.categories[0]
        name = category.category_name
        index = category.index
        probability = round(category.score, 3)

        return probability, index, name


def detect_color(detection_result: processor.DetectionResult):
    """Detect the color of an object and return it
    Args:
      detection_result: The list of all "Detection" entities.
    Returns:
      color: Color of the detected object in BGR Tuple.
      index: index to light the led
    """
    for detection in detection_result.detections:
        # Find the category
        category = detection.categories[0]
        name = category.category_name
        probability = round(category.score, 3)

        if name == "green_duck":
            color = (76, 242, 86)
            index = 1
        elif name == "green_frog":
            color = (107, 234, 44)
            index = 2
        elif name == "pink_duck":
            color = (211, 130, 255)
            index = 1
        elif name == "blue_duck":
            color = (205, 172, 73)
            index = 2
        else:
            color = (55, 232, 254)
            index = 0

        return probability, index, color


def collect_data(par_list: list):
    """Collecting data from detection.
    Args:
      par_list: List of recorded parameter.
    Return:
      par_arr: Numpy array of parameter
      par_count: Numpy array of counter
    """

    # Convert list to numpy array
    par_arr = np.array(par_list)
    # Create list of ratio between zero and non-zero values
    non_zero = list(par_arr[par_arr != 0])
    all_zero = list(par_arr[par_arr == 0])
    par_ratio = [len(non_zero), len(all_zero)]
    # Create array of non-zero values
    non_zero.insert(0, 0)
    par_arr = np.array(non_zero)
    # Create array to count number of detection
    par_count = np.arange(0, len(par_arr), 1)
    # Return numpy arrays
    return par_arr, par_count, par_ratio


def plot_line(det_par, det_count, title: str, ylabel: str, xlabel: str,
              label: str, show_mean: bool):
    """plot and save image from detection parameters in line chart.
    Args:
      det_par: Array of detection parameter to be plotted (y_axis)
      det_count: Array of counter to plot in x axis
      title: title of graph
      ylabel: label of y-axis
      xlabel: label of x-axis
      label: label in the graph to name the parameter
      show_mean: whether to show the mean of the graph or not
    """

    # Plot the image
    plt.plot(det_count, det_par, color='red', label=label, linewidth=3)

    if show_mean:
        # Calculate mean of the parameter
        par_mean = det_par[1:].mean()
        # Plot it
        plt.axhline(
            par_mean,
            color='blue',
            label='mean',
            linewidth=2,
            linestyle='--')

    # Add title and label
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Add grid
    plt.grid(True)

    # Show legend
    plt.legend()

    # Show image
    plt.tight_layout()
    plt.show()


def plot_pie(det_ratio: list, det_label: list, title: str,
             color: list = ['#008fd5', '#fc4f30']):
    """plot and save image from detection parameters in pie chart.
    Args:
      det_ratio: List of ratio of detected parameters.
      det_label: Label for each parameter.
      color: list of color for each pie segment.
      title: title of the graph.
    """

    # Plot the image
    plt.pie(
        det_ratio,
        labels=det_label,
        colors=color,
        wedgeprops={
            'edgecolor': 'black',
            'linewidth': 2},
        autopct='%.1f%%')

    # Add title
    plt.title(title)

    # Show image
    plt.tight_layout()
    plt.show()


def stack_array(list_a: list, list_b: list, *args):
    """Stack five list together and convert it to array.
    Args:
      list_a: First list
      list_b: Second list
    Return:
      A numpy array stack of inputs
    """

    array_a = np.array(list_a)
    array_b = np.array(list_b)
    final_stack = np.stack((array_a, array_b), axis=-1)

    for other_list in args:
        array_c = np.array(other_list)
        # Checks if list contains another iterables
        try:
            inside_list_size = len(other_list[0])
        except TypeError:
            array_c = array_c.reshape(len(array_c), 1)
        else:
            if isinstance(other_list[0], str):
                array_c = array_c.reshape(len(array_c), 1)
            else:
                array_c = array_c.reshape(len(array_c), inside_list_size)

        final_stack = np.hstack((final_stack, array_c))

    return final_stack


def create_csv(stacked_arr, columns: list, filename: str):
    """Create a CSV file from stacked array
    Args:
      stacked_arr: Array of stacked parameters
      columns: List of the name of columns for CSV file
      filename: Name of the CSV file
    """

    recorded_data = pd.DataFrame(stacked_arr, columns=columns)
    recorded_data.to_csv(filename, index=False)


def reset_list(*args):
    """Clear the content of all lists."""
    for this_list in args:
        this_list.clear()


def update_list(all_list: list, all_values: list):
    """Update list with newest value
    Args:
      all_list: all list to be updated
      all_values: values to update list
    """
    for (i, this_list) in enumerate(all_list):
        this_list.append(all_values[i])


def draw_text(img,
              text: str,
              font=cv2.FONT_HERSHEY_PLAIN,
              pos: tuple = (0, 0),
              font_size=2,
              font_thickness=2,
              text_color: tuple = (0, 0, 255),
              bg_color: tuple = (255, 255, 255)):
    """Put text to image with color background
    Args:
      img: image to be added with text.
      text: text to put in image.
      font: font type.
      pos: position of text in the image.
      font_size: size of tge font.
      font_thickness: thickness of font.
      text_color: color of the font.
      bg_color: color of the background.
    """
    x_pos, y_pos = pos
    text_size = cv2.getTextSize(text, font, font_size, font_thickness)[0]
    width, height = text_size
    cv2.rectangle(img, (x_pos - 5, y_pos - 5),
                  (x_pos + width + 5, y_pos + height + 5), bg_color, -1)
    cv2.putText(img, text, (int(x_pos), int(y_pos + height + font_size - 1)),
                font, font_size, text_color, font_thickness)


def print_detection_time(count, ir_sns, cmr, start, delay):
    """Print detection time in to terminal/console
    Args:
      count: detection count
      ir_sns: ir sensor detection time
      cmr: camera detection time
      start: start detection time
      delay: delay between detection
    """
    rounded_irs = round((ir_sns - start), 3)
    rounded_cmr = round((cmr - start), 3)
    rounded_dly = round(delay, 3)
    print(f"Detection Session: {count}")
    print(f"IR sensor detection at {rounded_irs} second")
    print(f"Camera detection at {rounded_cmr} second")
    print(f"Delay between detections: {rounded_dly} second")


def print_detected(name, score):
    """Print messages if object is detected
    Args:
      name: name of detected object
      score: probability score
    """
    print(f"The object is {name}")
    print(f"The probability is {score}")


def print_not_detected(count):
    """Print messages if nothing detected
    Args:
      count: detection count
    """
    print(f"Detection Session: {count}")
    print("Object is not detected!")


def print_dimension(dimension, score):
    """print messages about detected dimension
    Args:
      dimension: detected dimension of object
      score: probability score of an object
    """
    print("Dimension:")
    print(f"The height is {dimension[0]} cm")
    print(f"The width is {dimension[1]} cm")
    print(f"The size/area is {dimension[2]} cm2")
    print(f"Probability: {score}")


def print_color(color, score):
    """Print messages if object is detected
    Args:
      color: BGR tuple of detected object
      score: probability score
    """
    print(f"The BGR color is {color}")
    print(f"The probability is {score}")
