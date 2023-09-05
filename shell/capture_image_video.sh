#! /bin/bash

# Change to tflite virtual environment
source /home/pi/tflite/bin/activate
# Change directory to detection files  dir
cd "/home/pi/Desktop/Object Detection/Raspberry Pi"
# Start detection to save images and videos
python detect_and_capture.py --collectAll True
# Move videos to other folder
mv "./video/Deteksi_Objek.mp4" "/home/pi/Desktop/Detection Result/Videos"
# Move all images to other folder
mv -v ./image/* "/home/pi/Desktop/Detection Result/Images"
