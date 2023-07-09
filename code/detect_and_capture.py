"""Main script to run the object detection routine."""
import argparse
import os
import sys
import shutil
import time
import datetime

import numpy as np
import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils

import constant

# Define Image and Video Directory
PARENT_DIR = r'/home/pi/examples/lite/examples/object_detection/raspberry_pi'
IMAGE_DIR = r'/home/pi/examples/lite/examples/object_detection/raspberry_pi/image'
VIDEO_DIR = r'/home/pi/examples/lite/examples/object_detection/raspberry_pi/video'


def run(
        is_multiple: bool,
        all_images: bool,
        true_name: str,
        model: str,
        camera_id: int,
        width: int,
        height: int,
        num_threads: int,
        enable_edgetpu: bool,
        detection_threshold: float,
        vid_filename: str):
    """Continuously run inference on images acquired from the camera.
    Args:
      is_multiple: True/False whether there is multiple type of object.
      all_images: True/False whether to collect all images.
      true_name: true label for detected object name.
      model: Name of the TFLite object detection model.
      camera_id: The camera id to be passed to OpenCV.
      width: The width of the frame captured from the camera.
      height: The height of the frame captured from the camera.
      num_threads: The number of CPU threads to run the model.
      enable_edgetpu: True/False whether the model is a EdgeTPU model.
      detection_threshold: Minimum score to be detected by model.
      vid_filename: filename of output video.
    """

    # Variables to calculate FPS
    counter, fps = 0, 0
    start_time = time.time()

    # Variables to collect detection
    detection_counter = 1
    frame_counter = 1
    old_cmr_time = 0
    cmr_time = 0

    # Define flags
    print_message = False

    # Start capturing video input from the camera
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Define the codec and create VideoWriter Object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(vid_filename, fourcc, 2.5, (width, height))

    # Visualization parameters
    size_ratio = np.sqrt(width**2 + height**2)/np.sqrt(640**2 + 480**2)
    x_position = 5  # pixels
    y_position = 5  # pixels
    font_size = 2*size_ratio
    font_thickness = 3*int(size_ratio)
    fps_avg_frame_count = 10

    # Initialize the object detection model
    base_options = core.BaseOptions(
        file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
    detection_options = processor.DetectionOptions(
        max_results=3, score_threshold=detection_threshold)
    options = vision.ObjectDetectorOptions(
        base_options=base_options, detection_options=detection_options)
    detector = vision.ObjectDetector.create_from_options(options)

    # Continuously capture images from the camera and run inference
    print("DETECTION STARTED!")
    print("")
    print(f"Detection Session: {detection_counter}")

    # If there is multiple type of object
    if is_multiple:
        # prompt user to input its true name/label
        true_name = input("True label/name of the object: ")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            sys.exit(
                'ERROR: Unable to read from webcam. Please verify your webcam settings.'
            )

        counter += 1
        image = cv2.flip(image, 1)

        # Convert the image from BGR to RGB as required by the TFLite model.
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create a TensorImage object from the RGB image.
        input_tensor = vision.TensorImage.create_from_array(rgb_image)

        # Run object detection estimation using the model.
        detection_result = detector.detect(input_tensor)

        # If something detected on camera
        if (detection_result.detections and
           (time.time() - old_cmr_time) >= constant.LIMIT_CMR_TIME):
            # Record camera time
            cmr_time = time.time()
            # update old values
            old_cmr_time = cmr_time
            # Update flag
            print_message = True

        # Only print result once every detection session
        if ((time.time() > (old_cmr_time + constant.LIMIT_DET_TIME))
                and (print_message)):
            # print messages
            print("Images of object has been collected.")
            print("")
            # Update counter
            detection_counter = detection_counter + 1
            frame_counter = 1
            print(f"Detection Session: {detection_counter}")

            # If there is multiple type of object
            if is_multiple:
                # prompt user to input its true name/label
                true_name = input("True label/name of the object: ")

            # Update flag
            print_message = False

        # Draw keypoints and edges on input image if detected
        if detection_result.detections:
            image = utils.visualize(image, detection_result, (width, height),
                                    utils.detect_color(detection_result)[2])
            # If all images want to be collected
            if all_images:
                # Change directory and save image
                os.chdir(IMAGE_DIR)
                current_date = datetime.datetime.now()
                img_filename = (
                    current_date.strftime("%d%m%Y") +
                    "_" +
                    utils.categorize(detection_result)[2] +
                    "_" +
                    str(detection_counter) +
                    "_" +
                    str(frame_counter) +
                    ".jpg")
                cv2.imwrite(img_filename, image)
                os.chdir(PARENT_DIR)
                # update counter
                frame_counter = frame_counter + 1
            else:
                # Collect detection that is True Positive
                if utils.categorize(detection_result)[2] == true_name:
                    # Change directory and save image
                    os.chdir(IMAGE_DIR)
                    current_date = datetime.datetime.now()
                    img_filename = (
                        current_date.strftime("%d%m%Y") +
                        "_" +
                        true_name +
                        "_" +
                        str(detection_counter) +
                        "_" +
                        str(frame_counter) +
                        ".jpg")
                    cv2.imwrite(img_filename, image)
                    os.chdir(PARENT_DIR)
                    # update counter
                    frame_counter = frame_counter + 1
        else:
            image = utils.visualize(image, detection_result)

        # Calculate the FPS
        if counter % fps_avg_frame_count == 0:
            end_time = time.time()
            fps = fps_avg_frame_count / (end_time - start_time)
            start_time = time.time()

        # Show the FPS
        fps_text = f"FPS = {round(fps, 1)}"
        text_location = (x_position, y_position)
        utils.draw_text(img=image, text=fps_text, font_size=font_size,
                        font_thickness=int(font_thickness), pos=text_location)

        # Write Image to Videos
        out.write(image)

        # Stop the program if the ESC key is pressed.
        if cv2.waitKey(1) == 27:
            break
        cv2.imshow('object_detector', image)

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("")
    print("DETECTION STOPPED!")


def move_video(src, dst):
    """Move video file to another directory.
    Args:
      src: source path for video files
      dst: source path for video files
    """
    shutil.move(src, dst)


def main():
    """Main function to run detection"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--multipleObject',
        help='Whether to detect single or multiple types of objects.',
        required=False,
        default=False)
    parser.add_argument(
        '--collectAll',
        help='Whether to collect all images or only the True Positive one.',
        required=False,
        default=False)
    parser.add_argument(
        '--trueObject',
        help='Name of the object to be detected if it is a single type.',
        required=False,
        type=str,
        default='yellow_duck')
    parser.add_argument(
        '--model',
        help='Path of the object detection model.',
        required=False,
        default='./model/frogducky2.tflite')
    parser.add_argument(
        '--cameraId',
        help='Id of camera.',
        required=False,
        type=int,
        default=0)
    parser.add_argument(
        '--frameWidth',
        help='Width of frame to capture from camera.',
        required=False,
        type=int,
        default=640)
    parser.add_argument(
        '--frameHeight',
        help='Height of frame to capture from camera.',
        required=False,
        type=int,
        default=480)
    parser.add_argument(
        '--numThreads',
        help='Number of CPU threads to run the model.',
        required=False,
        type=int,
        default=4)
    parser.add_argument(
        '--enableEdgeTPU',
        help='Whether to run the model on EdgeTPU.',
        action='store_true',
        required=False,
        default=False)
    parser.add_argument(
        '--detectionThreshold',
        help='Threshold score to be detected by model.',
        required=False,
        type=float,
        default=0.55)
    parser.add_argument(
        '--videoFilename',
        help='Name of the output video file with .mp4 extension',
        required=False,
        default='Deteksi_Objek.mp4')
    args = parser.parse_args()

    run(bool(args.multipleObject),
        bool(args.collectAll),
        str(args.trueObject),
        args.model,
        int(args.cameraId),
        args.frameWidth,
        args.frameHeight,
        int(args.numThreads),
        bool(args.enableEdgeTPU),
        args.detectionThreshold,
        str(args.videoFilename))
    move_video(PARENT_DIR + '/' + str(args.videoFilename),
               VIDEO_DIR + '/' + str(args.videoFilename))


if __name__ == '__main__':
    main()
