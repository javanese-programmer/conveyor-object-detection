"""Script to test traditional detection time."""
import argparse
import os
import time
import sys

import cv2
import pandas as pd
from PIL import Image

from utils import vizres, array, plot

# Add project directory to path before importing modules
sys.path.insert(0, "/home/pi/Desktop/Object Detection/Raspberry Pi")


def parse_argument():
    """Parse arguments used in the terminal."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--color",
        help="Color to be detected in BGR",
        required=False,
        type=str,
        default="(55, 232, 254)",
    )
    parser.add_argument(
        "--imgPath",
        help="Path of the test images.",
        required=False,
        default="/home/pi/Desktop/Object Detection/Raspberry Pi/test_data",
    )
    parser.add_argument(
        "--csvFilename",
        help="Name or path of the output CSV file.",
        required=False,
        default="./csv_data/Raspi4_Traditional.csv",
    )
    parser.add_argument(
        "--showMean",
        help="Whether to show the mean on detection graph.",
        required=False,
        default=False,
    )
    return parser.parse_args()


def test(img_path: str, color: str):
    """Continuously run inference on images from directory.

    Args:
      img_path: Path to the test images.
      color: color to be detected.
    Return:
      delay_list: list of recorded delay_time
    """
    delay_list = []
    detection_counter = 0

    target_color = list(color.strip("()").split(","))

    # Continuously capture images from the camera and run inference
    print("TEST DETECTION STARTED!")
    print("")

    for i in range(1, len(os.listdir(img_path)) + 1):
        # Read test image
        image = cv2.imread(img_path + "/" + "out" + str(i) + ".jpg")

        # Run object detection estimation using traditional method
        detection_counter += 1
        start_time = time.time()

        # Blur the image to remove unwanted details
        blurred = cv2.blur(image, ksize=(5, 5))

        # Convert to HSV
        hsv_image = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # Get Upper and Lower limit
        lower, upper = vizres.get_limits(target_color)

        # Apply color thresholding
        mask = cv2.inRange(hsv_image, lower, upper)

        # Get Bounding Box
        maskImage = Image.fromarray(mask)
        bbox = maskImage.getbbox()

        # If detected, draw rectangle around object
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 5)

            # Record the delay
            end_time = time.time()
            current_delay = end_time - start_time
            delay_list.append(current_delay)

            # Print results
            print(f"Detection: {detection_counter}")
            print(f"Delay Time: {round(current_delay, 3)} second")
            print("")

        # Show image for 67 miliseconds
        cv2.imshow("Traditional Method Test", image)
        cv2.waitKey(67)

    cv2.destroyAllWindows()
    print("")
    print("TEST DETECTION STOPPED!")

    # Return list of recorded data
    return delay_list


def main():
    """Run main detection test."""
    args = parse_argument()
    dly_list = test(args.imgPath, args.color)
    dly_arr, dt_count, _ = array.collect_data(dly_list)
    recorded_data = pd.DataFrame(dly_arr, columns=["Delay"])
    recorded_data.to_csv(str(args.csvFilename), index=False)
    plot.plot_delay(dly_arr, dt_count, show_mean=bool(args.showMean))


if __name__ == "__main__":
    main()
