"""Traditional image processing functions."""

import cv2
import numpy as np
from PIL import Image


def get_limits(
    color: list,
    h_range: list = [10, 10],
    s_range: list = [100, 255],
    v_range: list = [100, 255],
):
    """Get limits for color thresholding.

    Args:
      color: Color in BGR to be thresholded.
      h_range: Range of Hue channels.
      s_range: Range of Saturation channels.
      v_range: Range of Value channels.
    Returns:
      lowerLimit: Lower limit of the threshold in HSV
      upperLimit: Upper limit of the threshold in HSV
    """
    # convert color to HSV space
    bgr = np.uint8([[color]])
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Calculate the limits
    lowerLimit = np.array(
        (hsv[0][0][0] - h_range[0], s_range[0], v_range[0]), dtype=np.uint8
    )
    upperLimit = np.array(
        (hsv[0][0][0] + h_range[1], s_range[1], v_range[1]), dtype=np.uint8
    )

    return lowerLimit, upperLimit


def get_threshold(img):
    """Get threshold for Canny edge detection.

    Args:
      img: image to be processed
    Returns:
      thres1: Lower threshold
      thres2: Upper threshold
    """
    # Calculate median
    med_val = np.median(img)

    # LOWER THRESHOLD IS EITHER 0 OR 70% OF THE MEDIAN, WHICHEVER IS GREATER
    thres1 = int(max(0, 0.7 * med_val))

    # UPPER THRESHOLD IS EITHER 130% OF THE MEDIAN OR 255, WHICHEVER IS SMALLER
    thres2 = int(min(255, 1.3 * med_val))

    # increase the upper threshold
    thres2 = thres2 + 50

    return thres1, thres2


def remove_background(img):
    """Isolate the object from the background.

    Args:
      img: image to be processed
    Returns:
      bbox: bounding box coordinate of object
    """
    # blur and convert image to HSV
    background_img = img.copy()
    blurred = cv2.blur(background_img[:, 185:430], ksize=(5, 5))
    hsvImg = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Apply color thresholding to background (conveyor)
    lower, upper = get_limits([64, 124, 41], v_range=[3, 255], h_range=[15, 15])
    mask = cv2.inRange(hsvImg, lower, upper)
    mask_not = cv2.bitwise_not(mask)

    # Get the bpunding box coordinate from mask
    maskImage = Image.fromarray(mask_not)
    bbox = maskImage.getbbox()

    return bbox


def color_detection(img, bbox):
    """Apply Color Detection.

    Args:
      img: image to be processed
      bbox: bounding box coordinate of object
    Returns:
      detected: A booelan showing whether detection is sucessful or not
      object_id: Index of object class
      object_color: Color of the object in BGR
      object_name: Name of object class
    """
    # Define variables to contain detection
    detected = False
    object_id = 404
    object_color = (0, 0, 0)
    object_name = "-"

    # If bounding box exist and enough pixels are collected, detect it
    if (
        (bbox is not None)
        and (abs(bbox[2] - bbox[0]) >= 110)
        and (abs(bbox[3] - bbox[1]) >= 80)
    ):
        detected = True

        # Collect bbox coordinate
        x1, y1, x2, y2 = bbox

        # Isolate object from the rest of the image
        sampled_img = img.copy()
        duck = sampled_img[y1:y2, (x1 + 185):(x2 + 185)]
        blurred_duck = cv2.blur(duck, ksize=(7, 7))

        # Sample the color
        color_sample = blurred_duck[45:75, 45:75]
        colorHSV = cv2.cvtColor(color_sample, cv2.COLOR_BGR2HSV)

        # Calculate the mean from hue channel
        h_mean = colorHSV[:, :, 0].mean()

        # Classify
        if 17 <= h_mean <= 37:
            object_name = "yellow_duck"
            object_color = (55, 232, 254)
            object_id = 0
        elif 151 <= h_mean <= 171:
            object_name = "pink_duck"
            object_color = (211, 130, 255)
            object_id = 1
        elif 88 <= h_mean <= 108:
            object_name = "blue_duck"
            object_color = (205, 172, 73)
            object_id = 2
        else:
            detected = False

        # Draw label and bounding box
        if bbox is not None:
            x1 += 185
            x2 += 185
            cv2.rectangle(img, (x1, y1), (x2, y2), object_color, 5)
            cv2.putText(
                img,
                "Class: " + object_name,
                (x2 + 10, y1 + 15),
                cv2.FONT_HERSHEY_COMPLEX,
                0.7,
                object_color,
                2,
            )
            cv2.putText(
                img,
                f"Color: {object_color}",
                (x2 + 10, y1 + 40),
                cv2.FONT_HERSHEY_COMPLEX,
                0.7,
                object_color,
                2,
            )

    return detected, object_id, object_color, object_name


def contour_detection(img, bbox, offset: int = 17, targetArea=14000):
    """Apply Contour Detection.

    Args:
      img: image to be processed
      bbox: bounding box coordinate of the object
      offset: offset from bounding box coordinate wrt. original image
      targetArea: the minimum contour area to be drawn
    Returns:
      detected: A booelan showing whether detection is sucessful or not
      object_id: Index of object class
      object_contour: a tuple that contains area and number of corner points
      object_name: Name of object class
    """
    # Define variables to contain detection
    detected = False
    object_id = 404
    object_contour = (0, 0)
    object_name = "-"

    # If bounding box exist
    if bbox is not None:
        # Collect Bounding Box Coordinate
        x1, y1, x2, y2 = bbox
        cnt_offset = (x1 + 150 + offset, y1 - offset)

        # Isolate the object from the rest of image
        sampled_img = img.copy()
        shape = sampled_img[
            (y1 - offset):(y2 + offset), (x1 + 185 - offset):(x2 + 185 + offset)
        ]

        # If the isolated image is not empty, detect contours
        if shape.size != 0:
            # blur it and convert the color to grayspace
            blurred_shape = cv2.blur(shape, ksize=(5, 5))
            gray_shape = cv2.cvtColor(blurred_shape, cv2.COLOR_BGR2GRAY)

            # Apply Edge Detection
            threshold1, threshold2 = get_threshold(gray_shape)
            edges = cv2.Canny(
                image=gray_shape, threshold1=threshold1, threshold2=threshold2
            )

            # Apply Dilation
            kernel = np.ones((3, 3))
            dilation = cv2.dilate(edges, kernel, iterations=1)

            # Find contours
            contours, hierarchy = cv2.findContours(
                dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )

            # draw every detected contour with area larger than the target
            for cnt in contours:
                # area = cv2.contourArea(cnt)

                # Add offset to the contour coordinate
                cnt[:, :, 0] = cnt[:, :, 0] + cnt_offset[0]
                cnt[:, :, 1] = cnt[:, :, 1] + cnt_offset[1]

                # Approximate number of corner points, box coordinate, and area
                perimeter = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
                x, y, w, h = cv2.boundingRect(approx)
                boxArea = w * h

                # If target < box area < 30000 px, count as detected
                if boxArea > targetArea and boxArea < 30000:
                    detected = True

                    # Draw contours on image
                    cv2.drawContours(img, cnt, -1, (255, 0, 255), 3)

                    # Collect the area and the number of corner points
                    object_contour = (int(boxArea), len(approx))

                    # Classify object based on area and number of corner points
                    if (object_contour[0] >= 23000) and (object_contour[1] <= 11):
                        object_id = 0
                        object_name = "duck"
                    elif (
                        (object_contour[0] < 23000)
                        and (object_contour[0] > 18000)
                        and (object_contour[1] >= 9)
                    ):
                        object_id = 1
                        object_name = "cock"
                    elif (object_contour[0] <= 18000) and (object_contour[1] <= 11):
                        object_id = 2
                        object_name = "chick"

                    # Draw box, draw contour, and add text
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 4)
                    cv2.putText(
                        img,
                        "Class: " + object_name,
                        (x + w + 10, y + 15),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
                    cv2.putText(
                        img,
                        f"Points: {object_contour[1]}",
                        (x + w + 10, y + 40),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
                    cv2.putText(
                        img,
                        f"Area: {object_contour[0]}",
                        (x + w + 10, y + 65),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

    return detected, object_id, object_contour, object_name
