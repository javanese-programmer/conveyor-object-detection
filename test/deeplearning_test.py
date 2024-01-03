"""Script to test detection time on different Raspberry Pi platform"""
import argparse
import os
import time
import sys

import numpy as np
import pandas as pd
import cv2

from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision

# Add project directory to path before importing modules
sys.path.insert(0, '/home/pi/Desktop/Object Detection/Raspberry Pi')
import constant
from utils import vizres, array, plot


def parse_argument():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model',
        help='Path of the object detection model.',
        required=False,
        default='./model/color_detector2.tflite')
    parser.add_argument(
        '--imgPath',
        help='Path of the test images.',
        required=False,
        default='/home/pi/Desktop/Object Detection/Raspberry Pi/test_data')
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
        default=0.50)
    parser.add_argument(
        '--csvFilename',
        help='Name or path of the output CSV file.',
        required=False,
        default='./csv_data/Raspi4_Delay.csv')
    parser.add_argument(
        '--showMean',
        help='Whether to show the mean on detection graph.',
        required=False,
        default=False)
    return parser.parse_args()

def test(
        model: str,
        img_path: str,
        num_threads: int,
        enable_edgetpu: bool,
        detection_threshold: float):
    """Continuously run inference on images acquired from the camera.
    Args:
      model: Name of the TFLite object detection model.
      img_path: Path to the test images.
      num_threads: The number of CPU threads to run the model.
      enable_edgetpu: True/False whether the model is a EdgeTPU model.
      detection_threshold: Minimum score to be detected by model.
    Return:
      delay_list: list of recorded delay_time
    """
    
    delay_list = []
    detection_counter = 0

    # Initialize the object detection model
    base_options = core.BaseOptions(
        file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
    detection_options = processor.DetectionOptions(
        max_results=3, score_threshold=detection_threshold)
    options = vision.ObjectDetectorOptions(
        base_options=base_options, detection_options=detection_options)
    detector = vision.ObjectDetector.create_from_options(options)

    # Continuously capture images from the camera and run inference
    print("TEST DETECTION STARTED!")
    print("")

    for i in range(1, len(os.listdir(img_path))+1):
        
        # Read test image
        image = cv2.imread(img_path + '/' + 'out' + str(i) + '.jpg')
        
        # Start recroding time
        detection_counter += 1
        start_time = time.time()

        # Convert the image from BGR to RGB as required by the TFLite model.
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create a TensorImage object from the RGB image.
        input_tensor = vision.TensorImage.create_from_array(rgb_image)

        # Run object detection estimation using the model.
        detection_result = detector.detect(input_tensor)

        # Draw keypoints and edges on input image
        height, width, _ = rgb_image.shape
        image = vizres.visualize(image, detection_result, (width, height))
        
        # Record the delay
        end_time = time.time()
        current_delay = end_time - start_time
        delay_list.append(current_delay)
        
        # Print results
        print(f"Detection: {detection_counter}")
        print(f"Delay Time: {round(current_delay, 3)} second")
        print("")

        # Show image for 67 miliseconds
        cv2.imshow("Raspi Detection Test", image)
        cv2.waitKey(67)

    cv2.destroyAllWindows()
    print("")
    print("TEST DETECTION STOPPED!")

    # Return list of recorded data
    return delay_list


def main():
    """Main function to run detection"""
    args = parse_argument()
    dly_list = test(args.model, args.imgPath, int(args.numThreads), bool(args.enableEdgeTPU),
                   args.detectionThreshold)
    dly_arr, dt_count,_ = array.collect_data(dly_list)
    recorded_data = pd.DataFrame(dly_arr, columns=["Delay"])
    recorded_data.to_csv(str(args.csvFilename), index=False)
    plot.plot_delay(dly_arr, dt_count, show_mean=bool(args.showMean))


if __name__ == '__main__':
    main()