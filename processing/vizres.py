"""Functions for deep learning visualization result."""

import cv2
import numpy as np
from tflite_support.task import processor


def visualize(
    image: np.ndarray,
    detection_result: processor.DetectionResult,
    resolution: tuple = (640, 480),
    box_color: tuple = (255, 0, 0),  # blue
) -> np.ndarray:
    """Draws bounding boxes on the input image and return it.

    Args:
      image: The input RGB image.
      detection_result: The list of all "Detection" entities to be visualize.
      resolution: resolution of image frame.
      box_color: Color oh the bounding box.
    Returns:
      Image with bounding boxes.
    """
    # Define viz properties
    width, height = resolution
    size_ratio = np.sqrt(width**2 + height**2) / np.sqrt(640**2 + 480**2)
    margin = 10  # pixels
    row_size = 10  # pixels
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
        class_location = (
            margin + bbox.origin_x + 120,
            margin + row_size + bbox.origin_y,
        )
        score_location = (
            margin + bbox.origin_x + 120,
            margin + row_size + bbox.origin_y + 25,
        )
        cv2.putText(
            image,
            "Class: " + category_name,
            class_location,
            cv2.FONT_HERSHEY_PLAIN,
            font_size,
            box_color,
            font_thickness,
        )
        cv2.putText(
            image,
            "Score: " + str(probability),
            score_location,
            cv2.FONT_HERSHEY_PLAIN,
            font_size,
            box_color,
            font_thickness,
        )
    return image


def show_fps(
    img,
    text: str,
    resolution: tuple = (640, 480),
    font=cv2.FONT_HERSHEY_PLAIN,
    text_color: tuple = (0, 0, 255),
    bg_color: tuple = (255, 255, 255),
):
    """Put text to image with color background.

    Args:
      img: image to be added with text.
      text: text to put in image.
      resolution: resolution of image frame.
      font: font type.
      text_color: color of the font.
      bg_color: color of the background.
    """
    # Define viz properties
    img_width, img_height = resolution
    size_ratio = np.sqrt(img_width**2 + img_height**2) / np.sqrt(
        640**2 + 480**2
    )
    x_pos = 5  # pixels
    y_pos = 5  # pixels
    font_size = 2 * size_ratio
    font_thickness = 3 * int(size_ratio)

    text_size = cv2.getTextSize(text, font, font_size, font_thickness)[0]
    width, height = text_size
    cv2.rectangle(
        img,
        (x_pos - 5, y_pos - 5),
        (x_pos + width + 5, y_pos + height + 5),
        bg_color,
        -1,
    )
    cv2.putText(
        img,
        text,
        (int(x_pos), int(y_pos + height + font_size - 1)),
        font,
        font_size,
        text_color,
        font_thickness,
    )


def localize(detection_result: processor.DetectionResult):
    """Find object bounding box and return it.

    Args:
      detection_result: The list of all "Detection" entities.
    Returns:
      probability: probability score of detection
      index: predicted class id
      start_point: a top-left bounding boxes coordinates
      start_point: a bottom-right bounding boxes coordinates
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


def measure_dim(detection_result: processor.DetectionResult, frame_height: int = 480):
    """Measure the height, width, and size of detected object in centimeters.

    Args:
      detection_result: The list of all "Detection" entities.
      frame_height: The height of frame resolution
    Returns:
      probability: probability score of detection
      index: predicted class id
      dimension: height, width, size of an object in centimeters
    """
    # Define Pixels Per Inch (PPI)
    if frame_height == 600:
        ppi = 99.869
    elif frame_height == 720:
        ppi = 121.638
    elif frame_height == 960:
        ppi = 160.94
    else:
        ppi = 79.327

    # inch to cm conversion
    inch_to_cm = 2.54  # cm

    # pixels to cm conversion
    pixel_to_cm = inch_to_cm / ppi

    for detection in detection_result.detections:
        # Collect object height and width
        bbox = detection.bounding_box
        object_height = bbox.height  # in pixels
        object_width = bbox.width  # in pixels

        # Convert to cm
        true_height = round(object_height * pixel_to_cm, 3)  # in cm
        true_width = round(object_width * pixel_to_cm, 3)  # in cm
        # Calculate size
        true_size = round(true_height * true_width, 3)  # in cm2

        # Collect the score and index
        probability = round(detection.categories[0].score, 3)
        index = detection.categories[0].index

        # Final Dimension
        dimension = (true_height, true_width, true_size)

        return [probability, index, dimension]


def categorize(detection_result: processor.DetectionResult):
    """Categorize detected objects and return its name, score, and index.

    Args:
      detection_result: The list of all "Detection" entities.
    Returns:
      probability: probability score of detection
      index: predicted class id
      name: predicted class name
    """
    for detection in detection_result.detections:
        # Find the category
        category = detection.categories[0]
        name = category.category_name
        index = category.index
        probability = round(category.score, 3)

        return [probability, index, name]


def detect_color(detection_result: processor.DetectionResult):
    """Detect the color of an object and return it.

    Args:
      detection_result: The list of all "Detection" entities.
    Returns:
      probability: probability score of detection
      index: predicted class id
      color: Color of the detected object in BGR Tuple.
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

        return [probability, index, color]
