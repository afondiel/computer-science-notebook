# Edge Computer Vision - Notes

## Table of Contents (ToC)
  - [Introduction](#introduction)
    - [What's Edge Computer Vision?](#whats-edge-computer-vision)
    - [Key Concepts and Terminology](#key-concepts-and-terminology)
    - [Applications](#applications)
  - [Fundamentals](#fundamentals)
    - [Edge Computer Vision Architecture Pipeline](#edge-computer-vision-architecture-pipeline)
    - [How Edge Computer Vision Works](#how-edge-computer-vision-works)
    - [Advantages and Challenges](#advantages-and-challenges)
    - [Some Hands-On Examples](#some-hands-on-examples)
  - [Tools \& Frameworks](#tools--frameworks)
  - [Hello World!](#hello-world)
  - [Lab: Zero to Hero Projects](#lab-zero-to-hero-projects)
  - [References](#references)


## Introduction
Edge Computer Vision refers to performing computer vision tasks locally on edge devices, close to where the data is generated, reducing the need for cloud processing.

### What's Edge Computer Vision?
- The practice of running computer vision algorithms directly on edge devices, such as IoT sensors, cameras, and embedded systems.
- Aims to minimize latency, bandwidth usage, and data transmission to centralized cloud systems.
- Useful in applications requiring real-time image or video analysis with constrained resources.

### Key Concepts and Terminology
- **Edge Devices**: Computing devices located near the data source (e.g., sensors, cameras, smartphones).
- **Inference at the Edge**: Running machine learning models, like object detection or classification, locally on the edge device.
- **Latency**: The delay between data input (e.g., an image) and the processing output (e.g., object detection).
- **Bandwidth Optimization**: Minimizing data sent to the cloud by processing vision tasks locally, reducing network congestion and costs.

### Applications
- **Smart Cities**: Real-time monitoring and decision-making through smart traffic cameras, parking management, and pedestrian detection.
- **Autonomous Vehicles**: Real-time object recognition and navigation without needing cloud connections.
- **Industrial IoT**: Monitoring product quality or equipment health using vision algorithms at manufacturing plants.
- **Healthcare**: Medical devices analyzing patient data (e.g., X-rays or MRIs) in real-time at the point of care.

## Fundamentals

### Edge Computer Vision Architecture Pipeline
- **Data Acquisition**: Gathering images or video data from local sensors or cameras.
- **Preprocessing**: Basic steps such as resizing, denoising, or filtering images to prepare for analysis.
- **Model Inference**: Running trained models (e.g., neural networks) to analyze visual data in real-time.
- **Decision Making**: Making decisions or triggering actions based on model output, like generating alerts or controlling devices.
- **Communication**: Sending critical data or results to the cloud if necessary for further analysis or storage.

### How Edge Computer Vision Works
- **Local Inference**: Machine learning models are optimized to run on low-power edge devices using techniques like model compression, quantization, or pruning.
- **Optimized Hardware**: Devices like NVIDIA Jetson, Google Coral, and FPGAs (Field-Programmable Gate Arrays) help accelerate vision tasks at the edge.
- **Real-Time Processing**: Minimizes latency by avoiding cloud delays, allowing for immediate decision-making based on processed visual data.
- **Data Privacy**: Keeping sensitive image data on the edge reduces the need to transmit personal or confidential information to the cloud.

### Advantages and Challenges
- **Advantages**:
  - **Reduced Latency**: Immediate response times for time-critical applications (e.g., autonomous driving).
  - **Lower Bandwidth Costs**: Reduces the need to send large volumes of image data to the cloud.
  - **Enhanced Privacy**: Keeps sensitive visual data on the device, improving data security.
  - **Energy Efficiency**: Optimized processing for low-power environments like IoT sensors or drones.

- **Challenges**:
  - **Limited Computational Power**: Edge devices often lack the processing capabilities of cloud servers.
  - **Model Size**: Neural networks need to be compressed to fit into the memory and processing capabilities of edge hardware.
  - **Update Complexity**: Deploying updates to models on many distributed edge devices can be complex.
  - **Hardware Constraints**: Power and cooling requirements for some high-performance edge devices may limit deployment options.

### Some Hands-On Examples
- **Smart Traffic Monitoring**: Deploying object detection models to detect vehicles and pedestrians in real-time on traffic cameras.
- **Edge-Based Facial Recognition**: Implementing face detection on a smart security camera system using edge processing.
- **Industrial Visual Inspection**: Detecting defects or anomalies in a production line using edge vision algorithms.
- **Wildlife Monitoring**: Using edge devices with cameras to identify and track wildlife in their natural habitats without cloud connectivity.

## Tools & Frameworks
- **OpenCV**: A widely-used library for real-time computer vision, with optimizations for edge devices.
- **TensorFlow Lite**: A version of TensorFlow designed for mobile and edge devices, allowing for efficient model inference.
- **NVIDIA Jetson**: A platform that provides powerful GPUs for accelerating deep learning models at the edge.
- **Google Coral**: Hardware and tools for implementing edge AI, including vision applications using TPU (Tensor Processing Unit) accelerators.
- **OpenVINO**: Intel’s toolkit for deploying optimized vision and AI models on edge devices.

## Hello World!
```python
import tensorflow as tf
from tensorflow.keras import layers

# A basic image classifier for edge devices
model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Model summary
model.summary()
```

## Lab: Zero to Hero Projects
- **Edge-Based Smart Traffic Monitoring**: Implement a smart traffic camera system that uses real-time object detection to monitor vehicle flow.
- **Edge AI Security Camera**: Develop a facial recognition system using edge computing to detect intruders.
- **Smart Retail Solutions**: Build a smart camera that tracks foot traffic and customer behavior in a retail environment.
- **Autonomous Drone Navigation**: Use edge computer vision to implement obstacle avoidance and path planning on a drone.

## References
- Satyanarayanan, M. (2017). "The Emergence of Edge Computing."
- Li, J., & Ota, K. (2018). "AI at the Edge: A Vision for Deep Learning in IoT."
- NVIDIA. (2020). "Jetson Xavier NX: Edge AI Development Kit."
- Google Coral. (2020). "Edge TPU: Accelerate ML Inference on Edge Devices."
- Intel. (2020). "OpenVINO Toolkit for Edge AI and Vision Deployment."
- https://xailient.com/blog/what-is-edge-computer-vision-and-how-does-it-work/
- https://www.lacroix-impulse.fr/nos-expertises/computer-vision-edge-ia/
- https://medium.com/@muhammadsamiuddinrafayf18/edge-computer-vision-c5dfd663ab7c
- https://viso.ai/evaluation-guide/edge-ai-for-computer-vision/
- https://www.amd.com/fr/products/system-on-modules/kria/partner-showcase/fanless-edge-ai-computer-vision-system/product-inquiry.html
- https://www.minalogic.com/actualites-adherents/partenariat-entre-neovision-et-dolphin-design/
- Atos: https://atos.net/fr/solutions/atos-computer-vision-platform
- [eurotech: COMPUTER VISION & AI ON EDGE](https://www.eurotech.com/edge-ai/computer-vision/?gad_source=1&gclid=Cj0KCQjw8--2BhCHARIsAF_w1gz0wqEWnTgyNEr8x12ZO2LyFUP0VDO2I3oPju_Xa47HEgqF-zAtZEIaAtTnEALw_wcB)
- [Texas Instruments Edge AI: Advancing intelligence at the edge - Scalable and efficient vision processors bring real-time intelligence to smart camera systems](https://www.ti.com/technologies/edge-ai.html?utm_source=google&utm_medium=cpc&utm_campaign=ti-null-null-58700008391670827_dynamicapplications_edgeai-cpc-pp-google-eu_int&utm_term=&ds_k=DYNAMIC+SEARCH+ADS&DCM=yes&gad_source=1&gclid=Cj0KCQjw8--2BhCHARIsAF_w1gwST1aJaADgu4YR71MJNMxiw4wrJXhd7h7pYz959x3J1Nm9_bIAtCkaAsecEALw_wcB&gclsrc=aw.ds)
- https://www.lacroix-impulse.fr/nos-expertises/computer-vision-edge-ia/

Courses:
- [Introduction to On-Device AI - DLA](https://www.coursera.org/projects/introduction-to-on-device-ai)
- [Computer Vision with Embedded ML](https://www.coursera.org/learn/computer-vision-with-embedded-machine-learning)
