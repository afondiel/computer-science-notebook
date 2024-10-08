# Embedded AI - Notes

## Table of Contents (ToC)

  - [Overview](#overview)
  - [Applications](#applications)
  - [Tools \& Frameworks](#tools--frameworks)
  - [Hello World!](#hello-world)
  - [References](#references)


## Overview

Embedded AI involves integrating artificial intelligence algorithms and models into devices with limited resources, enabling real-time inference on the edge.

## Applications

- Edge devices: Smart cameras, IoT sensors, and wearables.
- Automotive: Driver assistance, object recognition.
- Healthcare: Remote patient monitoring, medical imaging.
- Industrial IoT: Predictive maintenance, quality control.
- Consumer Electronics: Voice assistants, image recognition in smartphones.

## Tools & Frameworks

- TensorFlow Lite: Lightweight version for mobile and edge devices.
- PyTorch Mobile: Extension of PyTorch for mobile deployments.
- Edge TPU: Google's Tensor Processing Unit for edge devices.
- CMSIS-NN: ARM's neural network kernel library for microcontroller platforms.
- OpenVINO: Intel's toolkit for optimizing and deploying models on edge devices.
- [Optimum: transformers and Diffusers on hardware](https://github.com/huggingface/optimum)

## Hello World!

```python
# Sample code using TensorFlow Lite
import tflite_runtime.interpreter as tflite

# Load the TFLite model and allocate tensors.
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_tensor = interpreter.tensor(interpreter.get_input_details()[0]['index'])
output = interpreter.tensor(interpreter.get_output_details()[0]['index'])

# Perform inference.
input_tensor()[0] = input_data
interpreter.invoke()
result = output()[0]
print(result)
```

## References

[TensorFlow Lite](https://www.tensorflow.org/lite) 
  - Documentation: 
    - https://www.tensorflow.org/lite/microcontrollers
    - https://www.tensorflow.org/lite/microcontrollers/get_started_low_level

[PyTorch Mobile](https://pytorch.org/mobile/)

[Edge TPU](https://coral.ai/)

[CMSIS-NN](https://arm-software.github.io/CMSIS_5/NN/html/index.html)

[OpenVINO - Intel](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html)

Embedded AI:

- [What Is Embedded AI (EAI)? Why Do We Need EAI? - Huawei](https://info.support.huawei.com/info-finder/encyclopedia/en/EAI.html)
- [Mastering embedded AI - Embedded.com](https://www.embedded.com/mastering-embedded-ai/)
- [Embedded Artificial Intelligence: Intelligence on Devices | IEEE ](https://ieeexplore.ieee.org/document/10224582/)


Arduino: 
- https://docs.arduino.cc/tutorials/nano-33-ble-sense/get-started-with-machine-learning
  
- [Arm CMSIS-NN GitHub Repository](https://github.com/ARM-software/CMSIS-NN)


tinyML: 
- https://www.tinyml.org/

Courses:
- [Introduction to On-Device AI - DLA](https://www.coursera.org/projects/introduction-to-on-device-ai)
- [Computer Vision with Embedded ML](https://www.coursera.org/learn/computer-vision-with-embedded-machine-learning)

