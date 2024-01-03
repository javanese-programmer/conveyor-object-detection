"""Utils fucntion to process detection images and videos"""

import os
import datetime
import shutil

import cv2
import numpy as np

from tflite_support.task import processor
from processing import vizres

# Define images and parent directory
IMAGE_DIR = r'/home/pi/Desktop/Object Detection/Raspberry Pi/image'
VIDEO_DIR = r'/home/pi/Desktop/Object Detection/Raspberry Pi/video'
PARENT_DIR = r'/home/pi/Desktop/Object Detection/Raspberry Pi'

def save_img(image: np.ndarray, detection_result: processor.DetectionResult,
             detection_counter: int, image_counter: int):
    """Change directory and save image
    Args:
        image: the input image
        detection_result: The list of all "Detection" entities to be visualize.
        detection_counter: count of detection session
        image_counter: count of images saved
    """
    
    # Change directory and save images
    os.chdir(IMAGE_DIR)
    current_date = datetime.datetime.now()
    img_filename = (current_date.strftime("%d%m%Y") + "_" +
                    vizres.categorize(detection_result)[2] +
                    "_" + str(detection_counter) +
                    "_" + str(image_counter) + ".jpg")
    cv2.imwrite(img_filename, image)
    os.chdir(PARENT_DIR)


def move_video(vid_filename: str):
    """Move video file to another directory.
    Args:
        vid_filename: filename of output video.
    """
    src = PARENT_DIR + '/' + vid_filename
    dst = VIDEO_DIR + '/' + vid_filename
    shutil.move(src, dst)