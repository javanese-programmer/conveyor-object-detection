"""Main script to test detection time"""
import argparse
import os
import time
import numpy as np

import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils

import constant


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

        detection_counter += 1
        
        # Read test image
        image = cv2.imread(img_path + '/' + 'out' + str(i) + '.jpg')

        # Convert the image from BGR to RGB as required by the TFLite model.
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create a TensorImage object from the RGB image.
        input_tensor = vision.TensorImage.create_from_array(rgb_image)

        # Run object detection estimation using the model.
        start_time = time.time()
        detection_result = detector.detect(input_tensor)
        end_time = time.time()
        current_delay = end_time - start_time
        delay_list.append(current_delay)
        
        # Print results
        print(f"Detection: {detection_counter}")
        print(f"Delay Time: {round(current_delay, 3)} second")
        print("")

        # Draw keypoints and edges on input image
        height, width, _ = rgb_image.shape
        image = utils.visualize(image, detection_result, (width, height))

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
        default='./test_data/rubber_duck')
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
    args = parser.parse_args()

    dly_list = test(args.model, args.imgPath, int(args.numThreads), bool(args.enableEdgeTPU),
                   args.detectionThreshold)
    dly_arr, dt_count,_ = utils.collect_data(dly_list)
    utils.create_csv(dly_arr, ["Delay"], str(args.csvFilename))
    utils.plot_line(dly_arr, dt_count,
                    title="Delay Time during Detection",
                    xlabel="Detection Count",
                    ylabel=" Delay Time (Second)",
                    label='Delay',
                    show_mean=bool(args.showMean)
                    )


if __name__ == '__main__':
    main()