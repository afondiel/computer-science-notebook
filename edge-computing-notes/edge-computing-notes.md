# Edge Computing - Notes

## Table of Contents (ToC)

- Overview
- Applications
- Tools & Frameworks
- Hello World!
- References

## Overview

Edge Computing is a distributed computing paradigm that brings computation and data storage closer to the source of data generation, reducing latency and optimizing bandwidth usage.

## Applications

- Internet of Things (IoT): Enabling real-time processing and analysis of IoT data at the edge.
- Smart Cities: Implementing edge computing for efficient management of urban infrastructure.
- Healthcare: Facilitating remote patient monitoring and quick decision-making in healthcare applications.
- Industrial IoT (IIoT): Enhancing manufacturing processes with real-time analytics at the edge.
- Autonomous Vehicles: Utilizing edge computing for rapid decision-making in self-driving cars.

## Tools & Frameworks

- AWS IoT Greengrass: Amazon's edge computing service for deploying machine learning models on IoT devices.
- Azure IoT Edge: Microsoft's platform for extending AI and other cloud services to edge devices.
- EdgeX Foundry: Open-source framework for building interoperable edge computing systems.
- TensorFlow Lite for Edge: Lightweight version of TensorFlow optimized for edge devices.


## Hello World!

```python
# Sample code using TensorFlow Lite for edge computing on IoT devices
import tensorflow as tf

# Load the TensorFlow Lite model for edge deployment
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
# Perform inference on edge device data
# (Additional code for data preprocessing and inference goes here)

```

## References
- [Edge Computing: A Comprehensive Guide](https://example.com/edge-computing-guide)
- [AWS IoT Greengrass Documentation](https://aws.amazon.com/greengrass/)
- [Azure IoT Edge Documentation](https://docs.microsoft.com/en-us/azure/iot-edge/)
- [Top 10 Edge Computing Platforms in 2022](https://www.spiceworks.com/tech/edge-computing/articles/best-edge-computing-platforms/)



