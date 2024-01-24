# Undergraduate Capstone Project: Moving Object Detection on a Conveyor with Traditional Computer Vision and Deep Learning

[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/) [![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](https://jupyter.org/try)

This repository contains my final project as a requirement to obtain my bachelor's degree. In this project, I developed a device to detect objects on a trainer conveyor. This device is expected to be used by students to understand how similar devices work in industry. To support this goal, the device is connected to a programmable logic controller (PLC) through the Modbus protocol. Through this protocol, detection data will be received by the controller so that it can be used for further processing.

In detecting objects, two types of computer vision methods have been implemented. The first method is traditional methods such as color thresholding and contour detection. The second method is detection with an artificial neural network model (deep learning), namely the EfficientDet model. Since the main hardware has limited computational capabilities, the model used is a lightweight variant called EfficientDet-Lite.

The main hardware used in building this tool is Raspberry Pi 4. In addition to Raspberry Pi, IR sensors are also used to detect the presence of objects and calculate the detection delay time. Finally, an LED is added as an indicator of the detected class.

---

## Project Overview

The scope of the capstone design project can be observed in the following Figure

![diagram1](https://github.com/javanese-programmer/conveyor-object-detection/blob/main/image/ProjectOverview.png?raw=true)

---

## License

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
