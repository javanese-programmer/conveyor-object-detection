# Undergraduate Capstone Project: Moving Object Detection on a Conveyor with Traditional Computer Vision and Deep Learning

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1oxaT9i5JsVBN1YmcWmCMnf5iBxpiD9M8?usp=sharing)

This repository contains my final project as a requirement to obtain my bachelor's degree. I developed a device to detect objects on a trainer conveyor in this project. This device is expected to be used by students to understand how similar devices work in industry. The device is connected to a programmable logic controller (PLC) through the Modbus protocol to support this goal. Through this protocol, detection data will be received by the controller so that it can be used for further processing.

In detecting objects, two types of computer vision methods have been implemented. The first method is traditional methods such as color thresholding and contour detection. The second method is detection with an artificial neural network model (deep learning), namely the EfficientDet model. Since the main hardware has limited computational capabilities, the model used is a lightweight variant called EfficientDet-Lite.

The main hardware used in building this tool is Raspberry Pi 4. In addition to Raspberry Pi, IR sensors are also used to detect the presence of objects and calculate the detection delay time. A camera module was also installed to acquire images from the top side of the conveyor. Finally, an LED is added as an indicator of the detected class.

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

## Requirements

[![Generic badge](https://img.shields.io/badge/numpy-1.21.6-blue.svg)](https://shields.io/) [![Generic badge](https://img.shields.io/badge/pandas-1.3.5-purple.svg)](https://pandas.pydata.org/) [![Generic badge](https://img.shields.io/badge/matplotlib-3.5.3-orange.svg)](https://pandas.pydata.org/) [![Generic badge](https://img.shields.io/badge/opencv-4.5.3.56-green.svg)](https://opencv.org/) [![Generic badge](https://img.shields.io/badge/Pillow-9.4.0-red.svg)](https://pillow.readthedocs.io/en/stable/index.html) [![Generic badge](https://img.shields.io/badge/tk-0.1.0-black.svg)](https://docs.python.org/3/library/tkinter.html) 
[![Generic badge](https://img.shields.io/badge/tflite--support-0.4.3-yellow.svg)](https://www.tensorflow.org/lite?hl=id) [![Generic badge](https://img.shields.io/badge/pymodbus-2.5.3-orange.svg)](https://pymodbus.readthedocs.io/en/latest/) [![Generic badge](https://img.shields.io/badge/pyModbusTCP-0.2.1-blue.svg)](https://pymodbustcp.readthedocs.io/en/latest/) 

In running this project, the Python libraries used can be observed in the `requirements.txt` file. However, when installing the `tflite-support` library, errors may occur. Therefore, the library has been provided in the `lib` folder. To access the Raspberry Pi GPIO, the `RPi.GPIO` library is required.

## Setup

Before running the code in the repository, users need to set up a virtual environment with the required libraries. This code needs to be run on Raspberry OS.

First, check the version of Raspberry Pi OS.

```sh
cat /etc/os-release
```

Second, update the packages on Raspberry Pi OS.

```sh
sudo apt-get update
```

Third, check the Python version. Use Python 3.7 and above.

```sh
python3 --version
```

Fourth, install virtualenv and upgrade pip.

```sh
python3 -m pip install --user --upgrade pip
python3 -m pip install --user virtualenv
```

Fifth, create a virtual environment to run the code.

```sh
python3 -m venv ~/tflite-pymodbus
```

When the virtual environment is ready, activate the environment with the following command (this command needs to be called EVERY time you open a new Terminal).

```sh
source ~/tflite-pymodbus/bin/activate
```

Clone the repository from the project and change the active directory to the project directory.

```sh
git clone https://github.com/javanese-programmer/conveyor-object-detection.git
cd conveyor-object-detection
```

Install the dependencies of the project. When everything is installed, the code can be run immediately.

```sh
python3 -m pip install pip --upgrade
python3 -m pip install -r requirements.txt
```

**NOTE**: 
Before running the code, copy all the libraries in the `lib` folder and then paste them at the address `~/tflite-pymodbus/lib/python3.7/site-packages` on the Raspberry Pi OS. This is because the process of installing these libraries with pip can cause errors. 

## Running the Codes

The two main codes that can be used are `run.py` and `capture.py`. The first code will run the detection and establish the Raspberry Pi as a Modbus server that can send object detection data to the PLC. Meanwhile, the second will record the detection process and return both a video file and an image file.

#### Main Code:

```python
python run.py
```
```python
python capture.py
```

To check the arguments that can be added to the code, add `-h` after each command.

```python
python run.py -h
```
```python
python capture.py -h
```

#### Other code

The code below will collect images from the conveyor WITHOUT performing object detection.

```python
python collect.py
```

---

## License

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
