"""This module contains Deep Learning Object Detector Class with associated methods"""

import sys
import time
import numpy as np

import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision

from utils import terminal, array, video
from processing import vizres
from constant import LIMIT_CMR_TIME, LIMIT_IR_TIME, LIMIT_DET_TIME
from hardware.sensor import *
from hardware.plc import *
from detection.helper import *

class DeepDetector():
    """Deep Learning Detector Class"""
    
    def __init__(self, is_multiple: bool, model: str, width: int,
                 height: int, num_threads: int, enable_edgetpu: bool):
        """Constructor method
        Args:
            is_multiple: True/False whether there is multiple type of object.
            model: Name of the TFLite object detection model.
            width: The width of the frame captured from the camera.
            height: The height of the frame captured from the camera.
            num_threads: The number of CPU threads to run the model.
            enable_edgetpu: True/False whether the model is a EdgeTPU model.
        """
        # Define detection attributes
        self.is_multiple = is_multiple
        self.model = model
        self.width = width
        self.height = height
        self.num_threads = num_threads
        self.enable_edgetpu = enable_edgetpu
        
        # Define detection option
        base_options = core.BaseOptions(
            file_name=self.model, use_coral=self.enable_edgetpu, num_threads=self.num_threads)
        detection_options = processor.DetectionOptions(max_results=3, score_threshold=0.5)
        self.options = vision.ObjectDetectorOptions(
            base_options=base_options, detection_options=detection_options)
        
        # Attribute to collect final detection
        self.detected_list = []
        self.final_score_list = []
        self.final_pred_list = []
        self.true_label_list = []
        
    
    def detect(self, det_type: str, true_label: str, ip_address: str):
        """Method to run detection using deep learning method.
        Args:
            det_type: Type of detections (color, shape, or category).
            true_label: true label for detected object.
            ip_address: IP address of PLC to be connected.
        Return:
            delay_list: list of recorded delay_time
            fps_list: list of recorded FPS value
            detected_list: list of bool values of whether object is detected or not
            final_score_list: list of probability scores for detection sessions
            final_pred_list: list of predicted values for detection sessions
            true_label_list: list of true label for detection sessions
        """
        # Variable to collect detections
        pred_list, index_list, score_list = [], [], []

        # Convert the true label to tuple
        atuple = tuple(true_label.strip('()').split(','))
        label_length = len(atuple)
        if label_length == 1:
            true_label = atuple[0]
        else:
            true_label = (int(atuple[0]), int(atuple[1]), int(atuple[2]))
        
        # Define LED and IR instance
        prepare_gpio()
        ir = Infrared(right_ir=22, left_ir=16) # We will use left IR
        led = Led(yellow=18, red=11, blue=15)
        
        # Initialize PLC instance and classid-to-bit dict
        plc = PLC(ip_address)
        id_to_bit = {'0': '001', '1': '010', '2': '100'}
        
        # Define Flag and Calculator instance
        flags = Flags()
        calc = Calculator()
        
        # Define camera attributes
        cap = cv2.VideoCapture(0)  # Default camera ID = 0
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        # Initialize the object detection model
        detector = vision.ObjectDetector.create_from_options(self.options)

        # Continuously capture images from the camera and run inference
        print("DETECTION STARTED!")
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                sys.exit(
                    'ERROR: Unable to read from webcam. Please verify your webcam settings.'
                )
            calc.frame_up()
            image = cv2.flip(image, 1)

            # Convert the image from BGR to RGB as required by the TFLite model.
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Create a TensorImage object from the RGB image.
            input_tensor = vision.TensorImage.create_from_array(rgb_image)

            # Run object detection estimation using the model.
            detection_result = detector.detect(input_tensor)

            # If object detected by IR sensor
            if (ir.read_sensor()[1] == 0) and (
                    (time.time() - calc.old_ir_time) >= LIMIT_IR_TIME):
                # Restart timer and reset flag
                calc.restart_ir()
                flags.reverse_ir()
                print("")
                
                # Update cpunter
                calc.det_up()

            # If object is detected, record the time and collect detections
            if detection_result.detections:
                if (time.time() - calc.old_cmr_time) >= LIMIT_CMR_TIME:
                    # Clear lists and restart camera timer
                    array.reset_list(pred_list, index_list, score_list)
                    calc.restart_cmr()
                    # Update flags
                    flags.reverse_cmr()
                    
                if time.time() <= (calc.old_cmr_time + LIMIT_DET_TIME):
                    if det_type == "color":
                        predictions = vizres.detect_color(detection_result)   
                    elif det_type == "shape":
                        predictions = vizres.measure_dim(detection_result, self.height)
                    else:
                        predictions = vizres.categorize(detection_result)
                    array.update_list([score_list, index_list, pred_list], predictions)

            # If object is detected on IR and Camera
            if (flags.detected_ir) and (flags.detected_cmr):
                # Calculate delay and print messages
                calc.calculate_delay()
                calc.print_data()
                
                # Reset flags
                flags.reverse_ir()
                flags.reverse_cmr()
                flags.reverse_msg()
                
            # Else, if only detected on IR 
            elif (flags.detected_ir and (not flags.detected_cmr) and
                  ((time.time() - calc.old_ir_time) >= LIMIT_IR_TIME)):
                # Print message
                terminal.print_undetected(calc.det_count)
                
                # Turn off LED
                led.turn_off()
            
                # If there are multiple type of object, promp user to input true label
                if self.is_multiple:
                    true_label = terminal.prompt_label()
                    
                # Send data to plc
                calc.start_coil()
                plc.write_bits('000')
                calc.calc_coil_latency()
                
                if label_length == 1:
                    array.update_list([self.final_pred_list], ['-'])
                else:
                    array.update_list([self.final_pred_list], [(0,0,0)])
                    calc.start_reg()
                    plc.write_words((0,0,0))
                    calc.calc_reg_latency()
                
                # Update detected result list
                calc.update_data(False)
                array.update_list([self.final_score_list, self.detected_list,
                                   self.true_label_list], [0, False, true_label])
                
                # Reset flag
                flags.reverse_ir()

            # Only print result once every detection session
            if ((score_list) and (time.time() > (calc.old_cmr_time + LIMIT_DET_TIME))
                and (flags.print_message)):
                # Find object with highest probability score
                object_id = np.argmax(score_list)
                
                # print messages
                if det_type == "color":
                    terminal.print_color(pred_list[object_id], score_list[object_id])
                elif det_type == "shape":
                    terminal.print_dimension(pred_list[object_id], score_list[object_id])
                else:
                    terminal.print_detected(pred_list[object_id], score_list[object_id])
                
                # Turn on LED
                led.turn_on(index=index_list[object_id])
                
                # If there is multiple type of object, promp user to input true label
                if self.is_multiple:
                    true_label = terminal.prompt_label()
                
                # Send data to plc based on index
                calc.start_coil()
                plc.write_bits(id_to_bit[str(index_list[object_id])])
                calc.calc_coil_latency()
                
                if label_length != 1:
                    calc.start_reg()
                    plc.write_words(pred_list[object_id])
                    calc.calc_reg_latency()
                    
                # Update detected result list
                calc.update_data(True)
                array.update_list([self.final_pred_list, self.final_score_list,
                                   self.detected_list, self.true_label_list],
                                  [pred_list[object_id], score_list[object_id],
                                   True, true_label])
                
                # Reset flags
                flags.reverse_msg()

            # Draw keypoints and edges on input image based on object color
            try:
                box_color = vizres.detect_color(detection_result)[2]
            except TypeError:
                box_color = (0,0,0) 
            image = vizres.visualize(image, detection_result, (self.width, self.height), box_color)

            # Show the FPS
            calc.calculate_fps()
            fps_text = f"FPS = {round(calc.fps, 1)}"
            vizres.show_fps(img=image, text=fps_text, resolution=(self.width, self.height))

            # Stop the program if the ESC key is pressed.
            if cv2.waitKey(1) == 27:
                break
            cv2.imshow('object_detector', image)

        cap.release()
        led.turn_off()
        plc.write_bits('000')
        plc.write_words((0,0,0))
        clean_gpio()
        cv2.destroyAllWindows()
        print("")
        print("DETECTION STOPPED!")

        # Return list of recorded data
        return (calc.delay_list, calc.fps_list, calc.reg_latency_list,
                calc.coil_latency_list, self.detected_list, self.final_score_list,
                self.final_pred_list, self.true_label_list)
    
    def capture(self, true_label: str, all_images: bool, vid_filename: str):
        """Method to capture detected object
        Args:
            true_label: true label for detected object.
            all_images: True/False whether to collect all images.
            vid_filename: filename of output video.
        """
        # Define Flag and Calculator instance
        flags = Flags()
        calc = Calculator()
        calc.det_up()
        calc.img_up()
        
        # Define camera instance and frame resolution
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        # Define the codec and create VideoWriter Object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(vid_filename, fourcc, 2.5, (self.width, self.height))
        
        # Initizalize object detection model
        detector = vision.ObjectDetector.create_from_options(self.options)
        
        # Continuously capture images from the camera and run inference
        print("DETECTION STARTED!")
        print("")
        print(f"Detection Session: {calc.det_count}")
        # If there is multiple type of object, prompt user to input true label
        if self.is_multiple:
            true_label = input("True label of the object: ")
            
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                sys.exit(
                    'ERROR: Unable to read from webcam. Please verify your webcam settings.'
                )

            calc.frame_up()
            image = cv2.flip(image, 1)

            # Convert the image from BGR to RGB as required by the TFLite model.
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Create a TensorImage object from the RGB image.
            input_tensor = vision.TensorImage.create_from_array(rgb_image)

            # Run object detection estimation using the model.
            detection_result = detector.detect(input_tensor)

            # If something detected on camera, restart timer and flag
            if (detection_result.detections and
               (time.time() - calc.old_cmr_time) >= LIMIT_CMR_TIME):
                calc.restart_cmr()
                flags.reverse_msg()

            # Only print result once every detection session
            if ((time.time() > (calc.old_cmr_time + LIMIT_DET_TIME))
                    and flags.print_message):
                # print messages
                print("Images of object has been collected.")
                print("")
                
                # Update counter and flags
                calc.det_up()
                calc.img_count = 1
                flags.reverse_msg()

                # If there is multiple type of object
                print(f"Detection Session: {calc.det_count}")
                if self.is_multiple:
                    true_label = input("True label of the object: ")
                

            # Draw keypoints and edges on input image if detected
            if detection_result.detections:
                image = vizres.visualize(image, detection_result, (self.width, self.height),
                                         vizres.detect_color(detection_result)[2])
                predicted_class = vizres.categorize(detection_result)[2]
                
                # Either collect all images or image with correct labels
                if all_images:
                    video.save_img(image, predicted_class, calc.det_count, calc.img_count)
                    calc.img_up()
                elif predicted_class  == true_label:
                    video.save_img(image, predicted_class, calc.det_count, calc.img_count)
                    calc.img_up()
            else:
                image = vizres.visualize(image, detection_result)

            # Show the FPS
            calc.calculate_fps()
            fps_text = f"FPS = {round(calc.fps, 1)}"
            vizres.show_fps(img=image, text=fps_text, resolution=(self.width, self.height))
            
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