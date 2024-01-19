"""Utils fucntion to process detection images and videos."""

import os
import datetime
import shutil

import cv2
import numpy as np


# Define images and parent directory
PARENT_DIR = str(os.getcwd())
IMAGE_DIR = PARENT_DIR + '/' + 'image'
VIDEO_DIR = PARENT_DIR + '/' + 'video'


def save_img(
    image: np.ndarray, predicted_class: str, detection_counter: int, image_counter: int
):
    """Change directory and save image.

    Args:
        image: the input image
        predicted_class: predicted class of the object
        detection_counter: count of detection session
        image_counter: count of images saved
    """
    # Change directory and save images
    os.chdir(IMAGE_DIR)
    current_date = datetime.datetime.now()
    img_filename = (
        current_date.strftime("%d%m%Y")
        + "_"
        + predicted_class
        + "_"
        + str(detection_counter)
        + "_"
        + str(image_counter)
        + ".jpg"
    )
    cv2.imwrite(img_filename, image)
    os.chdir(PARENT_DIR)


def move_video(vid_filename: str):
    """Move video file to another directory.

    Args:
        vid_filename: filename of output video.
    """
    src = PARENT_DIR + "/" + vid_filename
    dst = VIDEO_DIR + "/" + vid_filename
    shutil.move(src, dst)
