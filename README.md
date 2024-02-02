# Undergraduate Capstone Project: Moving Object Detection on a Conveyor with Traditional Computer Vision and Deep Learning

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1oxaT9i5JsVBN1YmcWmCMnf5iBxpiD9M8?usp=sharing)

This repository contains my final project as a requirement to obtain my bachelor's degree. I developed a device to detect objects on a trainer conveyor in this project. This device is expected to be used by students to understand how similar devices work in industry. The device is connected to a programmable logic controller (PLC) through the Modbus protocol to support this goal. Through this protocol, detection data will be received by the controller so that it can be used for further processing.

In detecting objects, two types of computer vision methods have been implemented. The first method is traditional methods such as color thresholding and contour detection. The second method is detection with an artificial neural network model (deep learning), namely the EfficientDet model. Since the main hardware has limited computational capabilities, the model used is a lightweight variant called EfficientDet-Lite.

The main hardware used in building this tool is Raspberry Pi 4. In addition to Raspberry Pi, IR sensors are also used to detect the presence of objects and calculate the detection delay time. A camera module was also installed to acquire images from the top side of the conveyor. Finally, an LED is added as an indicator of the detected class.

---

## Youtube Video
A demonstration of how the device works can be observed in the following YouTube video. Please click on the thumbnail image to view the video.

[![The Youtube Video](https://github.com/javanese-programmer/conveyor-object-detection/blob/main/image/Thumbnail.jpg?raw=true)](https://youtu.be/NyluVm7MZUU?si=-iUa0rmn3ZUJ-8rz)

---

## Project Overview

The scope of the capstone design project can be observed in the following Figure

![diagram1](https://github.com/javanese-programmer/conveyor-object-detection/blob/main/image/ProjectOverview.png?raw=true)

In preparing the EfficientDet-Lite model, the steps taken consist of image data collection (which is stored in Google Drive), data augmentation process, labeling with Label Studio, and training process. Most of these processes were carried out at Google Collaboratory. The code will then call the resulting model on the Raspberry Pi. Both the dataset and the model can be found in the folder of the same name.

![diagram2](https://github.com/javanese-programmer/conveyor-object-detection/blob/main/image/DepedensiTraining.png?raw=true)

The dependencies of each code in this repository can be observed in the following figure. The main codes to be called are `run.py` and `capture.py`. The first code will run the detection while the second code will record the detected object.

<p align="center">
  <img src="https://github.com/javanese-programmer/conveyor-object-detection/blob/main/image/DepedensiDeployment.png?raw=true" alt="diagram3" width="800" />
</p>

### Detection Scenarios

#### Color Detection

The first detection scenario is to detect objects based on color differences. For this purpose, three rubber duck objects of the same size but with different colors were prepared. When detecting objects based on color, the program will return the predicted class index and a tuple containing the RGB value of the object. These two values will be extracted by the PLC.

<p align="center">
  <img src="https://github.com/javanese-programmer/conveyor-object-detection/blob/main/image/scenario1.jpg?raw=true" width="200" /> <img src="https://github.com/javanese-programmer/conveyor-object-detection/blob/main/image/scenario2.jpg?raw=true" width="200" /> <img src="https://github.com/javanese-programmer/conveyor-object-detection/blob/main/image/scenario3.jpg?raw=true" width="200" />
</p>

<div style="margin-left: auto;
            margin-right: auto;
            width: 30%">
|  **Class**  | **Blue** | **Green** | **Red** |
|:-----------:|:--------:|:---------:|:-------:|
| yellow_duck |    55    |    232    |   254   |
|  blue_duck  |    205   |    172    |    73   |
|  pink_duck  |    211   |    130    |   255   |
</div>

#### Shape Detection

The second detection scenario is to detect objects based on shape differences. Therefore, three toy objects of the same color but different sizes have been prepared. In shape detection, the predicted class index will also be collected. However, in this scenario, the RGB value will be replaced by the size or dimension parameter of the object. For example, the program will return the area of the object in pixels and the approximate corner points of the contour.  

<p align="center">
  <img src="https://github.com/javanese-programmer/conveyor-object-detection/blob/main/image/scenario4.jpg?raw=true" width="200" /> <img src="https://github.com/javanese-programmer/conveyor-object-detection/blob/main/image/scenario5.jpg?raw=true" width="200" /> <img src="https://github.com/javanese-programmer/conveyor-object-detection/blob/main/image/scenario6.jpg?raw=true" width="200" />
</p>

| **Class** | **Height (cm)** | **Width (cm)** | **Size (cm2)** |
|:---------:|:---------------:|:--------------:|:--------------:|
|    duck   |        5        |        5       |       25       |
|    cock   |        6        |        4       |       24       |
|   chick   |        5        |        4       |       20       |


---

## Project Demo

The difference in the detection process for the two scenarios mentioned can be observed below. When an object is detected, the user will be able to observe the object class and the collected features.

<p align="center">
  <img src="https://github.com/javanese-programmer/conveyor-object-detection/blob/main/video/trad_color.gif?raw=true" width="400" /> <img src="https://github.com/javanese-programmer/conveyor-object-detection/blob/main/video/trad_shape.gif?raw=true" width="400" />
</p>

Besides the detection video, other outputs of the program consist of a message on the terminal, detection performance graphs, and a CSV file. The terminal message and the resulting performance graphs can be observed below. The performance visualized here consists of the delay time between camera and IR sensor detection, frame rate, and detected/undetected object ratio.

<p align="center">
  <img src="https://github.com/javanese-programmer/conveyor-object-detection/blob/main/image/Terminal_Message.png?raw=true" width="400" /> <img src="https://github.com/javanese-programmer/conveyor-object-detection/blob/main/image/graph1.jpg?raw=true" width="400" /> <img src="https://github.com/javanese-programmer/conveyor-object-detection/blob/main/image/graph2.jpg?raw=true" width="400" /> <img src="https://github.com/javanese-programmer/conveyor-object-detection/blob/main/image/graph3.jpg?raw=true" width="250" />
</p>

These performance values can also be observed in greater detail within the CSV file. The CSV file allows the performance data to be further analyzed. Examples of CSV files generated by the program can be observed in the `csv_data` folder. The explanation for each column in the CSV data can be observed in the Table below.

| **Column**       | **Description**                                                 | **Unit**                    |
|------------------|-----------------------------------------------------------------|-----------------------------|
| Delay            | Delay time between camera and IR                                | Seconds                     |
| FPS              | Detection frame rate                                            | FPS                         |
| Latency (Regs)   | Latency between Raspberry Pi and PLC registers                  | Seconds                     |
| Latency (Coils)  | Latency between Raspberry Pi and PLC coils                      | Seconds                     |
| Detected         | Boolean value whether the object is detected or not             | -                           |
| Probability      | (Deep Learning) Prediction probability for the object           | -                           |
| Blue, Green, Red | (Traditional) BGR color obtained from object detection          | -                           |
| Area, Points     | (Traditional) Contour parameters obtained from object detection | (Area) Pixels               |
| Prediction       | (Traditional) Predicted object class                            | -                           |
| Label            | (Traditional) True object class                                 | -                           |
| Feature (Pred)   | (Deep Learning) Model-predicted features                        | (Height, Width) centimeters |
| Feature (True)   | (Deep Learning) True feature value                              | (Height, Width) centimeters |

### Programmable Logic Controller

During detection, the resulting data can be accessed by the PLC. For illustration, in this project, the PLC used is PLC M221 from Schneider Electric. This PLC can be programmed with _Ecostruxure Machine Expert - Basic_ software to receive data. Through this software, the detection data will be able to be observed and further processed.

![diagram3](https://github.com/javanese-programmer/conveyor-object-detection/blob/main/image/PLCasClient2Annotated.png?raw=true)
![diagram4](https://github.com/javanese-programmer/conveyor-object-detection/blob/main/image/PLCasClient3Annotated.png?raw=true)

Simple processing can be performed. For example, with predicted class index data, the PLC can be programmed to turn on I/O indicator lights based on class. Programming is done with Ladder Diagram. This change can be observed below. The changes to the PLC can be observed below. More complex processing can be performed as needed.

<p align="center">
  <img src="https://github.com/javanese-programmer/conveyor-object-detection/blob/main/video/PLC.gif?raw=true" width="600" />
</p>

This data transmission is possible because the Raspberry Pi acts as a Modbus Server or Modbus Client. By default, the Raspberry Pi will run _as a server_. Nonetheless, the Modbus communication mode of the Raspberry Pi can be changed through program arguments.

---

## Dataset

Each dataset contains three classes. The first dataset is _Color Ducky_. The classes in this dataset are "yellow_duck", "pink_duck", and "blue_duck". This dataset contains 181 images. The second dataset is _Ducky Chicken_. This dataset has classes consisting of "duck", "chick", and "cock". This dataset contains 233 images. Meanwhile, the _Ducky Frog_ dataset is only used in testing so it can be ignored. Samples for the two datasets mentioned earlier can be observed below.

<p align="center">
  <img src="https://github.com/javanese-programmer/conveyor-object-detection/blob/main/image/ColorDucky.jpg?raw=true" width="300" /> <img src="https://github.com/javanese-programmer/conveyor-object-detection/blob/main/image/DuckyChicken.jpg?raw=true" width="300" />
</p>

All datasets can be obtained from the `dataset` folder. The datasets have been separated for the sake of training the deep learning model (`train`) and validating its performance (`validation`).

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

The trained model can be retrieved from the `model` folder. The model name is based on the usage scenario and its variants. For example, the model `color_detector2.tflite` is the EfficientDet-Lite2 model trained for the color detection scenario.

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

Before running the code, access to port 502 (Modbus port) needs to be granted to the program. This is because the port can only be accessed by the root user. Therefore, run the command to redirect port 502 to a higher port, such as 5020. Run this command every time you want to use the device as a Modbus server.

```sh
sudo iptables -t nat -A PREROUTING -p tcp --dport 502 -j REDIRECT --to-port 5020
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
