#! /bin/bash

# Change to tflite virtual environment
source /home/pi/tflite/bin/activate
# Change directory to detection files  dir
cd "/home/pi/Desktop/Object Detection/Raspberry Pi"
# Start detection
python detect_category.py --multipleObject True
# Move resulting files to another folder
mv "./csv_data/Deteksi_Kategori.csv" "/home/pi/Desktop/Detection Result/CSV/Multiple Object"
