# Undergraduate Capstone Project: Moving Object Detection on a Conveyor with Traditional Computer Vision and Deep Learning

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white) ![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white) ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black) 
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white) ![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white) ![Raspberry Pi](https://img.shields.io/badge/-RaspberryPi-C51A4A?style=for-the-badge&logo=Raspberry-Pi)

This repository contains my final project as a requirement to obtain my bachelor's degree. In this project, I developed a device to detect objects on a trainer conveyor. This device is expected to be used by students to understand how similar devices work in industry. To support this goal, the device is connected to a programmable logic controller (PLC) through the Modbus protocol. Through this protocol, detection data will be received by the controller so that it can be used for further processing.

In detecting objects, two types of computer vision methods have been implemented. The first method is traditional methods such as color thresholding and contour detection. The second method is detection with an artificial neural network model (deep learning), namely the EfficientDet model. Since the main hardware has limited computational capabilities, the model used is a lightweight variant called EfficientDet-Lite.

The main hardware used in building this tool is Raspberry Pi 4. In addition to Raspberry Pi, IR sensors are also used to detect the presence of objects and calculate the detection delay time. Finally, an LED is added as an indicator of the detected class.

---


## Project Overview

The scope of the capstone design project can be observed in the following Figure

![diagram1](https://github.com/javanese-programmer/conveyor-object-detection/blob/main/image/ProjectOverview.png?raw=true)
