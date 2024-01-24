# Undergraduate Capstone Project: Moving Object Detection on a Conveyor with Traditional Computer Vision and Deep Learning

[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/) [![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](https://jupyter.org/try)

This repository contains my final project as a requirement to obtain my bachelor's degree. I developed a device to detect objects on a trainer conveyor in this project. This device is expected to be used by students to understand how similar devices work in industry. The device is connected to a programmable logic controller (PLC) through the Modbus protocol to support this goal. Through this protocol, detection data will be received by the controller so that it can be used for further processing.

In detecting objects, two types of computer vision methods have been implemented. The first method is traditional methods such as color thresholding and contour detection. The second method is detection with an artificial neural network model (deep learning), namely the EfficientDet model. Since the main hardware has limited computational capabilities, the model used is a lightweight variant called EfficientDet-Lite.

The main hardware used in building this tool is Raspberry Pi 4. In addition to Raspberry Pi, IR sensors are also used to detect the presence of objects and calculate the detection delay time. Finally, an LED is added as an indicator of the detected class.

---

## Project Overview

The scope of the capstone design project can be observed in the following Figure

![diagram1](https://github.com/javanese-programmer/conveyor-object-detection/blob/main/image/ProjectOverview.png?raw=true)

In preparing the EfficientDet-Lite model, the steps taken consist of image data collection (which is stored in Google Drive), data augmentation process, labeling with Label Studio, and training process. Most of these processes were carried out at Google Collaboratory. The code will then call the resulting model on the Raspberry Pi. Both the dataset and the model can be found in the folder of the same name.

![diagram1](https://github.com/javanese-programmer/conveyor-object-detection/blob/main/image/DepedensiTraining.png?raw=true)

The dependencies of each code in this repository can be observed in the following figure. The main codes to be called are `run.py` and `capture.py`. The first code will run the detection while the second code will record the detected object.

![diagram1](https://github.com/javanese-programmer/conveyor-object-detection/blob/main/image/DepedensiDeployment.png?raw=true)

---

## Dataset

Each dataset contains three classes. The first dataset is _Color Ducky_. The classes in this dataset are "yellow_duck", "pink_duck", and "blue_duck". This dataset contains 181 images. The second dataset is _Ducky Chicken_. This dataset has classes consisting of "duck", "chick", and "cock". This dataset contains 233 images. Meanwhile, the _Ducky Frog_ dataset is only used in testing so it can be ignored. Samples for the two datasets mentioned earlier can be observed below.

![diagram1](https://github.com/javanese-programmer/conveyor-object-detection/blob/main/image/ColorDucky.jpg?raw=true) ![diagram1](https://github.com/javanese-programmer/conveyor-object-detection/blob/main/image/DuckyChicken.jpg?raw=true)

---
## Model

EfficientDet-Lite is a scalable model. By modifying hyperparameters such as input resolution, layer depth, and layer width, EfficientDet-Lite will experience a shift in performance and computational cost. From the model development process, five variants of EfficientDet-Lite have been produced. The variants and their performance comparison can be observed in the following table.

|      **Model**     | **Parameters** | **Size (KB)** | **mAP (color)** | **mAP (shape)** |
|:------------------:|----------------|---------------|-----------------|-----------------|
| EfficientDet-Lite0 |    3.239.711   |     4.342     |      0,315      |      0,320      |
| EfficientDet-Lite1 |    4.241.703   |     5.799     |      0,476      |      0,371      |
| EfficientDet-Lite2 |    5.236.415   |     7.220     |      0,400      |      0,348      |
| EfficientDet-Lite3 |    8.328.991   |     11.456    |      0,420      |      0,438      |
| EfficientDet-Lite4 |   15.110.223   |     20.068    |      0,502      |      0,401      |

---

## License

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
