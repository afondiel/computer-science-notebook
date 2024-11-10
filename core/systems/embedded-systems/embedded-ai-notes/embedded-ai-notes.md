# Embedded AI - Notes

## Table of Contents

  - [Overview](#overview)
  - [Applications](#applications)
  - [Embedded AI vs On-Device AI](#embedded-ai-vs-on-device-ai)
  - [Tools & Frameworks](#tools-frameworks)
  - [Hello World!](#hello-world)
  - [References](#references)


## Overview

Embedded AI involves integrating artificial intelligence algorithms and models into devices with limited resources, enabling real-time inference on the edge.

## Applications

- [Edge devices](../../edge-iot-notes/readme.md): Smart cameras, IoT sensors, and wearables.
- Automotive: Driver assistance, object recognition.
- Healthcare: Remote patient monitoring, medical imaging.
- Industrial IoT: Predictive maintenance, quality control.
- Consumer Electronics: Voice assistants, image recognition in smartphones.


## **Embedded AI vs On-Device AI**

**Embedded AI** and **On-Device AI** are related but refer to slightly different concepts.

### Embedded AI
- **Definition**: Embedded AI focuses on integrating AI models within dedicated hardware systems that are usually part of a larger machine or infrastructure. These systems are often constrained in terms of computing resources and are purpose-built for specific applications.
- **Key Characteristics**:
  - Often deployed on **microcontrollers, ASICs (Application-Specific Integrated Circuits)**, or **FPGAs (Field-Programmable Gate Arrays)**.
  - Designed to perform specialized tasks within an embedded system, such as sensor data processing, control systems, or automation.
  - **Examples**: AI in industrial automation systems, smart home devices, robotics, and autonomous vehicles.

### On-Device AI
- **Definition**: On-Device AI refers to deploying AI models directly on end-user devices, like smartphones, tablets, or IoT gadgets, allowing AI to run locally without needing a network connection.
- **Key Characteristics**:
  - Focuses on **real-time processing** and low-latency applications by processing data directly on the device.
  - Primarily designed for consumer-grade hardware, which may include **smartphones, laptops, and other mobile or edge devices**.
  - **Examples**: Facial recognition on phones, voice recognition on smart assistants, real-time language translation apps.

### Key Differences
1. **Purpose and Environment**:
   - Embedded AI: Often serves specific, task-oriented applications within broader systems (e.g., a microcontroller in a car sensor).
   - On-Device AI: Provides more general-purpose AI capabilities on mobile or edge devices intended for individual users.

2. **Hardware Constraints**:
   - Embedded AI: Usually operates within strict resource constraints and is optimized for specific hardware architectures (e.g., microcontrollers).
   - On-Device AI: More flexible, often leveraging device-specific ML frameworks (e.g., TensorFlow Lite or Core ML) on consumer devices with more computational power than embedded systems.

3. **Common Applications**:
   - Embedded AI: Industrial and automotive systems, medical devices, smart sensors, robotics.
   - On-Device AI: Smartphones, tablets, AR/VR headsets, wearable health devices, smart speakers.

### Overlap

There’s a significant overlap since **On-Device AI can be seen as a subset of Embedded AI** when it’s used in mobile or consumer-grade devices with computational capabilities. Both aim to localize AI processing for privacy, real-time response, and reduced dependency on cloud resources.

## Tools & Frameworks

- **TensorFlow Lite**: Lightweight version for mobile and edge devices.
- **PyTorch Mobile**: Extension of PyTorch for mobile deployments.
- **Edge TPU**: Google's Tensor Processing Unit for edge devices.
- **CMSIS-NN**: ARM's neural network kernel library for microcontroller platforms.
- **OpenVINO**: Intel's toolkit for optimizing and deploying models on edge devices.
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

