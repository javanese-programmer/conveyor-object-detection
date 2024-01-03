import os
import sys

import numpy as np
import cv2

IMG_ADDR = "/home/pi/Desktop/Object Detection/Raspberry Pi"

counter = 0

cap_port = 0
cap = cv2.VideoCapture(cap_port)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        sys.exit(
            'ERROR: Unable to read from webcam. Please verify your webcam settings.'
        )
    
    counter += 1
    
    #image = cv2.flip(image, 1)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(IMG_ADDR + "/image/Image" + str(counter) + ".png", image)

    # Stop the program if the ESC key is pressed.
    if cv2.waitKey(1) == 27:
        break
    
    cv2.imshow("Captured Image", image)
    
cv2.destroyWindow("Captured Image")