"""This module contains Traditional Object Detector Class."""

import sys
import time

import cv2
from pyModbusTCP.server import ModbusServer

from utils import terminal, array, video
from processing import process, vizres
from constant import LIMIT_CMR_TIME, LIMIT_IR_TIME
from hardware.sensor import prepare_gpio, clean_gpio
from hardware.sensor import Led, Infrared
from hardware.plc import prepare_server, server_set_di, server_set_ir
from hardware.plc import PLC
from detection.helper import Flags, Calculator


class TraditionalDetector:
    """Traditional Detector Class."""

    def __init__(self, is_multiple: bool, width: int, height: int):
        """Init object instance of the class.

        Args:
            is_multiple: True/False whether there is multiple type of object.
            width: The width of the frame captured from the camera.
            height: The height of the frame captured from the camera.
        """
        # Define detection attributes
        self.is_multiple = is_multiple
        self.width = width
        self.height = height

        # Attribute to collect final detection
        self.detected_list = []
        self.feature_list = []
        self.id_list = []
        self.pred_list = []
        self.true_label_list = []

    def detect(
        self,
        det_type: str,
        true_label: str,
        is_server: bool,
        client_address: str,
        server_address: str,
    ):
        """Run traditional object detection session.

        Args:
            det_type: Type of detections (color OR shape).
            true_label: true label for detected object.
            is_server: flag to decide whether raspberry pi is server or not
            client_address: IP address of client (PLC) to be connected.
            server_address: IP address of Modbus server.
        Return:
            delay_list: recorded delay_times
            fps_list: recorded FPS values
            detected_list: bool values of whether object is detected or not
            feature_list: color/contour features
            pred_list: predicted values for detection sessions
            true_label_list: true labels for detection sessions
        """
        print("DETECTION STARTED!")

        # Define LED and IR instance
        prepare_gpio()
        ir = Infrared(right_ir=22, left_ir=16)  # We will use left IR
        led = Led(yellow=18, red=11, blue=15)

        # Initialize PLC instance and classid-to-bit dict
        id_to_bit = {"0": "001", "1": "010", "2": "100", "404": "000"}
        if is_server:
            server_data, server_id = prepare_server()
            server = ModbusServer(
                server_address,
                5020,
                no_block=True,
                data_bank=server_data,
                device_id=server_id,
            )
            print("Starting server...")
            server.start()
            print("Server is online")
        else:
            plc = PLC(client_address)

        # Define Flag and Calculator instance
        flags = Flags()
        calc = Calculator()

        # Define camera attributes
        cap = cv2.VideoCapture(0)  # Default camera ID = 0
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # Continuously capture images from the camera and run inference
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                sys.exit(
                    """ERROR: Unable to read from webcam.
                    Please verify your webcam settings."""
                )
            calc.frame_up()

            # Remove unwanted background
            bbox = process.remove_background(image)

            # Detect object
            if det_type == "color":
                predictions = process.color_detection(image, bbox)
            elif det_type == "shape":
                predictions = process.contour_detection(image, bbox)

            # If object detected by IR sensor
            if (ir.read_sensor()[1] == 0) and (
                (time.time() - calc.old_ir_time) >= LIMIT_IR_TIME
            ):
                # Update counter
                calc.det_up()

                # restart IR timer and flags
                calc.restart_ir()
                flags.reverse_ir()
                print("")

            # If object is detected on camera, restart CMR timer
            if (
                predictions[0]
                and flags.detected_ir
                and (predictions[3] != "-")
                and ((time.time() - calc.old_cmr_time) >= LIMIT_CMR_TIME)
            ):
                # Restart timer and reset flag
                calc.restart_cmr()
                flags.reverse_cmr()

                # Update detected result list
                array.update_list(
                    [
                        self.detected_list,
                        self.id_list,
                        self.feature_list,
                        self.pred_list,
                    ],
                    list(predictions),
                )

            # If object is detected BOTH on IR and Camera
            if (flags.detected_ir) and (flags.detected_cmr):
                # Calculate delay and print messaage
                calc.calculate_delay()
                calc.print_data()

                # update and reset flags
                flags.reverse_ir()
                flags.reverse_cmr()
                flags.reverse_msg()

            # Only print result once every detection session
            if flags.print_message:
                # print messages
                print(f"The object is {self.pred_list[-1]}")
                if det_type == "color":
                    print(f"The BGR color is {self.feature_list[-1]}")
                elif det_type == "shape":
                    print(f"The area is {self.feature_list[-1][0]} px")
                    print(f"The number of corner is {self.feature_list[-1][1]} points")

                # Turn on LED
                led.turn_on(index=self.id_list[-1])

                # Send data to PLC
                calc.start_coil()
                if is_server:
                    server_set_di(server_data, id_to_bit[str(self.id_list[-1])])
                else:
                    plc.write_bits(id_to_bit[str(self.id_list[-1])])
                calc.calc_coil_latency()

                calc.start_reg()
                if is_server:
                    server_set_ir(server_data, self.feature_list[-1])
                else:
                    plc.write_words(self.feature_list[-1])
                calc.calc_reg_latency()

                # If there are multiple types of object,
                # prompt user to input its true color
                if self.is_multiple:
                    true_label = terminal.prompt_label()

                # Update detected result list and flag
                calc.update_data(True)
                array.update_list([self.true_label_list], [true_label])
                flags.reverse_msg()

            # Show the FPS
            calc.calculate_fps()
            fps_text = f"FPS = {round(calc.fps, 1)}"
            vizres.show_fps(
                img=image, text=fps_text, resolution=(self.width, self.height)
            )

            # Stop the program if the ESC key is pressed.
            if cv2.waitKey(1) == 27:
                print("")
                break

            cv2.imshow("object_detector", image)

        cap.release()
        cv2.destroyAllWindows()

        led.turn_off()
        clean_gpio()

        server_set_di(server_data, "000") if is_server else plc.write_bits("000")
        server_set_ir(server_data, (0, 0, 0)) if is_server else plc.write_words(
            (0, 0, 0)
        )

        print("Shutdown server...")
        server.stop()
        print("Server is offline")
        print("DETECTION STOPPED!")

        # Return list of recorded data
        return (
            calc.delay_list,
            calc.fps_list,
            calc.reg_latency_list,
            calc.coil_latency_list,
            self.detected_list,
            self.feature_list,
            self.pred_list,
            self.true_label_list,
        )

    def capture(
        self, det_type: str, true_label: str, all_images: bool, vid_filename: str
    ):
        """Capture every detected object frame.

        Args:
            det_type: Type of detections (color OR shape).
            true_label: true label for detected object.
            all_images: True/False whether to collect all images.
            vid_filename: filename of output video.
        """
        # Define Flag and Calculator instance
        flags = Flags()
        calc = Calculator()

        # Define camera instance and frame resolution
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # Define the codec and create VideoWriter Object
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(vid_filename, fourcc, 25, (self.width, self.height))

        # Continuously capture images from the camera and run inference
        print("DETECTION STARTED!")
        print("")

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                sys.exit(
                    """ERROR: Unable to read from webcam.
                    Please verify your webcam settings."""
                )
            calc.frame_up()

            # Remove unwanted background
            bbox = process.remove_background(image)

            # Detect object
            if det_type == "color":
                predictions = process.color_detection(image, bbox)
            elif det_type == "shape":
                predictions = process.contour_detection(image, bbox)

            # If something detected on camera, restart timer and flag
            if predictions[0] and (time.time() - calc.old_cmr_time) >= LIMIT_CMR_TIME:
                calc.restart_cmr()
                flags.reverse_msg()

            # Only print result once every detection session
            if flags.print_message:
                # Update counter and flags
                calc.det_up()
                calc.img_count = 1
                flags.reverse_msg()

                # print messages
                print(f"Detection Session: {calc.det_count}")
                print("Images of object has been collected.")

                # If there is multiple type of object
                if self.is_multiple:
                    true_label = input("True label of the object: ")
                print("")

            # If detected
            if predictions[0]:
                # Either collect all images or image with correct labels
                if all_images:
                    video.save_img(
                        image, predictions[3], calc.det_count, calc.img_count
                    )
                    calc.img_up()
                elif predictions[3] == true_label:
                    video.save_img(
                        image, predictions[3], calc.det_count, calc.img_count
                    )
                    calc.img_up()

            # Show the FPS
            calc.calculate_fps()
            fps_text = f"FPS = {round(calc.fps, 1)}"
            vizres.show_fps(
                img=image, text=fps_text, resolution=(self.width, self.height)
            )

            # Write Image to Videos
            out.write(image)

            # Stop the program if the ESC key is pressed.
            if cv2.waitKey(1) == 27:
                break
            cv2.imshow("object_detector", image)

        # Release everything if job is finished
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("DETECTION STOPPED!")
