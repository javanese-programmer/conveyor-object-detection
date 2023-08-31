"""Main script to run the object detection routine."""
import argparse
import sys
import time
from time import sleep
import numpy as np

import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils

from RPi import GPIO
import constant


def run(
        is_multiple: bool,
        true_dim: str,
        model: str,
        camera_id: int,
        width: int,
        height: int,
        num_threads: int,
        enable_edgetpu: bool,
        detection_threshold: float):
    """Continuously run inference on images acquired from the camera.
    Args:
      is_multiple: True/False whether there is multiple type of object.
      true_dim: true dimension for detected object.
      model: Name of the TFLite object detection model.
      camera_id: The camera id to be passed to OpenCV.
      width: The width of the frame captured from the camera.
      height: The height of the frame captured from the camera.
      num_threads: The number of CPU threads to run the model.
      enable_edgetpu: True/False whether the model is a EdgeTPU model.
      detection_threshold: Minimum score to be detected by model.
    Return:
      delay_list: list of recorded delay_time
      fps_list: list of recorded FPS value
      detected_list: list of bool values of whether object is detected or not
      final_dimension_list: list of predicted dimensions for detection sessions
      true_dim_list: list of true dimensions for detection sessions
    """

    # Variables to calculate FPS
    counter, fps = 0, 0
    fps_list = []
    start_time = time.time()

    # Variable to calculate delay time
    old_cmr_time = 0
    cmr_time = 0
    old_ir_time = 0
    ir_time = 0
    delay_list = []

    # Variable to collect detections
    dimension_list = []
    index_list = []
    score_list = []
    detection_counter = 0

    # Define flags
    print_message = True
    detected_ir = False
    detected_cmr = False

    # List to collect final dimension
    final_dimension_list = []
    true_dim_list = []
    detected_list = []

    # Convert to true dim to tuple
    atuple = tuple(true_dim.strip('()').split(','))
    true_dim = (float(atuple[0]), float(atuple[1]), float(atuple[2]))

    # Start capturing video input from the camera
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Define LED and IR instance
    yellow_led = 18
    red_led = 11
    blue_led = 15
    ir_sensor = 16
    GPIO.setmode(GPIO.BOARD)
    GPIO.setwarnings(False)
    GPIO.setup(ir_sensor, GPIO.IN)
    GPIO.setup(yellow_led, GPIO.OUT)
    GPIO.setup(red_led, GPIO.OUT)
    GPIO.setup(blue_led, GPIO.OUT)

    # Visualization parameters
    size_ratio = np.sqrt(width**2 + height**2) / np.sqrt(640**2 + 480**2)
    x_position = 5  # pixels
    y_position = 5  # pixels
    font_size = 2 * size_ratio
    font_thickness = 3 * int(size_ratio)
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
    detection_start = time.time()

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

        # If object detected by IR sensor, start timer
        if (GPIO.input(ir_sensor) == 0) and (
                (time.time() - old_ir_time) >= constant.LIMIT_IR_TIME):
            ir_time = time.time()
            # Update old values
            old_ir_time = ir_time
            # Update flag
            detected_ir = True
            # Print a space
            print("")

        # If nothing detected, turn off LED
        if not detection_result.detections:
            GPIO.output(yellow_led, False)
            GPIO.output(red_led, False)
            GPIO.output(blue_led, False)
        # If something detected, record the time and detect it
        else:
            if (time.time() - old_cmr_time) >= constant.LIMIT_CMR_TIME:
                # Clear lists
                utils.reset_list(
                    dimension_list,
                    index_list,
                    score_list)
                # Record camera time
                cmr_time = time.time()
                # update old values
                old_cmr_time = cmr_time
                # Update flag
                detected_cmr = True
                print_message = True
            # For several seconds, collect every detections
            if time.time() <= (old_cmr_time + constant.LIMIT_DET_TIME):
                probability, index, dimension = utils.measure_dim(
                    detection_result, height)
                utils.update_list([dimension_list, index_list, score_list],
                                  [dimension, index, probability]
                                  )

        # If object is detected on IR and Camera
        if (detected_ir) and (detected_cmr):
            # Update counter
            detection_counter = detection_counter + 1
            # Calculate delay between ir sensor and first camera detection
            delay_time = cmr_time - ir_time
            # Update detection list
            utils.update_list([detected_list, delay_list, fps_list],
                              [True, round(delay_time, 3), round(fps, 3)])
            # print messages
            utils.print_detection_time(detection_counter, ir_time, cmr_time,
                                       detection_start, delay_time
                                       )
            # Reset flags
            detected_cmr = False
            detected_ir = False
        # Else, if only detected on IR
        elif ((detected_ir) and (not detected_cmr) and
              ((time.time() - old_ir_time) >= constant.LIMIT_IR_TIME)):
            # Update counter
            detection_counter = detection_counter + 1
            # Print messages
            utils.print_not_detected(detection_counter)
            # If there is multiple type of object, promp user to input its true
            # dim
            if is_multiple:
                true_dim = input("True dimension of the object: ")
                atuple = tuple(true_dim.strip('()').split(','))
                true_dim = (
                    float(
                        atuple[0]), float(
                        atuple[1]), float(
                        atuple[2]))
            # Append every lists
            utils.update_list([delay_list,
                              fps_list,
                              final_dimension_list,
                              true_dim_list,
                              detected_list],
                              [0,
                               round(fps, 3),
                               (0, 0, 0),
                               true_dim,
                               False])
            # Reset flag
            detected_ir = False

        # If all object detection has been collected
        # Only print result once every detection session
        if ((score_list) and (time.time() > (
                old_cmr_time + constant.LIMIT_DET_TIME)) and (print_message)):
            # Find object with highest probability score
            object_id = np.argmax(score_list)
            # print messages
            utils.print_dimension(
                dimension_list[object_id],
                score_list[object_id])
            # Turn on LED based on index
            final_index = index_list[object_id]
            if final_index == 0:
                GPIO.output(yellow_led, True)
                sleep(3)
            elif final_index == 1:
                GPIO.output(red_led, True)
                sleep(3)
            else:
                GPIO.output(blue_led, True)
                sleep(3)

            # If there is multiple type of object, promp user to input its true
            # size
            if is_multiple:
                true_dim = input("True dimension of the object: ")
                atuple = tuple(true_dim.strip('()').split(','))
                true_dim = (
                    float(
                        atuple[0]), float(
                        atuple[1]), float(
                        atuple[2]))
            # Update dimension list
            utils.update_list([final_dimension_list, true_dim_list], [
                dimension_list[object_id], true_dim])
            # Update flag
            print_message = False

        # Draw keypoints and edges on input image
        image = utils.visualize(image, detection_result, (width, height))

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

        # Stop the program if the ESC key is pressed.
        if cv2.waitKey(1) == 27:
            break
        cv2.imshow('dimension_detector', image)

    cap.release()
    GPIO.cleanup()
    cv2.destroyAllWindows()
    print("")
    print("DETECTION STOPPED!")

    # Return list of recorded delay time
    return delay_list, fps_list, detected_list, final_dimension_list, true_dim_list


def main():
    """Main function to detect object"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--multipleObject',
        help='Whether to detect single or multiple types of objects.',
        required=False,
        default=False)
    parser.add_argument(
        '--trueDim',
        help='(Height, Width, Size) of the object to be detected if it is a single type.',
        required=False,
        type=str,
        default='(6.4, 3.8, 24.32)')
    parser.add_argument(
        '--model',
        help='Path of the object detection model.',
        required=False,
        default='./model/shape_detector2.tflite')
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
        default=0.50)
    parser.add_argument(
        '--csvFilename',
        help='Name or path of the output CSV file.',
        required=False,
        default='./csv_data/Deteksi_Bentuk.csv')
    parser.add_argument(
        '--showMean',
        help='Whether to show the mean on detection graph.',
        required=False,
        default=False)
    args = parser.parse_args()

    dly_list, fps_list, detected_list, dimension_list, true_dim_list = run(
        bool(
            args.multipleObject), str(
            args.trueDim), args.model, int(
                args.cameraId), args.frameWidth, args.frameHeight, int(
                    args.numThreads), bool(
                        args.enableEdgeTPU), args.detectionThreshold)

    record_arr = utils.stack_array(dly_list[0:len(dly_list)],
                                   fps_list[0:len(dly_list)],
                                   detected_list[0:len(dly_list)],
                                   dimension_list[0:len(dly_list)],
                                   true_dim_list[0:len(dly_list)])
    utils.create_csv(
        record_arr, [
            "Delay", "FPS", "Detected", "Height (Pred)", "Width (Pred)",
            "Size (Pred)", "Height (True)", "Width (True)", "Size (True)"],
        str(args.csvFilename))

    dly_arr, dt_count, _ = utils.collect_data(dly_list)
    fps_arr, fps_count, _ = utils.collect_data(fps_list)
    _, _, dtct_ratio = utils.collect_data(detected_list)
    utils.plot_line(
        dly_arr,
        dt_count,
        title="Delay Time during Detection",
        xlabel="Detection Count",
        ylabel="Time (Second)",
        label='Delay',
        show_mean=bool(args.showMean))
    utils.plot_line(
        fps_arr,
        fps_count,
        title="FPS Change during Detection",
        xlabel="Detection Count",
        ylabel="FPS Level",
        label='FPS',
        show_mean=bool(args.showMean))
    utils.plot_pie(dtct_ratio, det_label=['Yes', 'No'],
                   title='Ratio of Detected Object')


if __name__ == '__main__':
    main()
