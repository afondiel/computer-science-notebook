# Embedded Vision - Notes

## Table of Contents (ToC)

  - [Introduction](#introduction)
    - [What's Embedded Vision?](#whats-embedded-vision)
    - [Key Concepts and Terminology](#key-concepts-and-terminology)
    - [Applications](#applications)
  - [Fundamentals](#fundamentals)
    - [Embedded Vision Architecture Pipeline](#embedded-vision-architecture-pipeline)
    - [How Embedded Vision Works](#how-embedded-vision-works)
    - [Types of Embedded Vision Systems](#types-of-embedded-vision-systems)
    - [Some Hands-On Examples](#some-hands-on-examples)
  - [Tools \& Frameworks](#tools--frameworks)
  - [Hello World!](#hello-world)
  - [Lab: Zero to Hero Projects](#lab-zero-to-hero-projects)
  - [References](#references)

## Introduction
Embedded Vision refers to the integration of vision capabilities into small, embedded systems for real-time image processing and decision-making.

### What's Embedded Vision?
- The integration of computer vision algorithms into embedded systems, typically with constraints on processing power, memory, and energy consumption.
- Embedded vision systems often operate in real-time, processing visual data and making decisions directly on the device.
- Used in various industries, from consumer electronics to industrial automation.

### Key Concepts and Terminology
- **Edge Computing**: Processing data locally on embedded devices, reducing the need for cloud resources.
- **Inference**: The process of running machine learning models, such as convolutional neural networks (CNNs), on the embedded device.
- **Latency**: The delay between receiving an image input and producing an output or decision.
- **Power Efficiency**: Optimizing vision algorithms to run efficiently on devices with limited power resources, such as battery-operated systems.

### Applications
- **Autonomous Vehicles**: Enabling real-time object detection, lane following, and obstacle avoidance in vehicles.
- **Drones**: Integrating vision capabilities for navigation, object tracking, and environmental monitoring.
- **Smart Cameras**: Embedded vision systems in security cameras for facial recognition, motion detection, and behavior analysis.
- **Medical Devices**: Used in diagnostics and monitoring systems to analyze medical images in real-time.

## Fundamentals

### Embedded Vision Architecture Pipeline
- **Image Acquisition**: Capturing images or video from sensors, such as cameras.
- **Preprocessing**: Basic image processing steps like resizing, normalization, and noise reduction to prepare the data for analysis.
- **Vision Processing**: Using machine learning algorithms like CNNs or traditional image processing techniques to analyze the data.
- **Decision Making**: Generating outputs based on the vision processing, such as classifications or actions.
- **Output Layer**: Providing results or controlling actuators in response to the processed data.

### How Embedded Vision Works
- **Edge Processing**: Vision processing is done directly on the embedded system using efficient algorithms, reducing reliance on external servers.
- **Hardware Acceleration**: Specialized hardware such as GPUs, FPGAs, or dedicated vision chips are used to speed up image processing tasks.
- **Optimized Algorithms**: Vision algorithms are optimized for low-power, low-latency execution, often through techniques like model quantization or pruning.
- **Sensor Integration**: Embedded vision systems work in real-time by closely integrating with sensors like cameras, LiDAR, or infrared sensors.

### Types of Embedded Vision Systems
- **Standalone Embedded Systems**: Devices like smart cameras or drones that process vision data locally without external resources.
- **Embedded Systems with Cloud Support**: Systems that process critical tasks locally but send some data to the cloud for further processing or storage.
- **System-on-Chip (SoC)**: A single chip that integrates sensors, processors, and memory to perform vision tasks efficiently.
- **Embedded AI**: The combination of machine learning algorithms and embedded systems to enable intelligent decision-making based on visual data.

### Some Hands-On Examples
- **Face Recognition**: Implementing a face detection system using an embedded platform like Raspberry Pi.
- **Object Tracking**: Using an embedded vision system to track moving objects in real-time, such as in drones.
- **Lane Detection for Autonomous Vehicles**: Building a lane-following system using a camera and an embedded vision processor.
- **Industrial Quality Control**: Creating an embedded system that checks products for defects using computer vision.

## Tools & Frameworks
- **OpenCV**: A popular open-source library for real-time computer vision, optimized for embedded devices.
- **TensorFlow Lite**: A lightweight version of TensorFlow designed for running machine learning models on embedded systems.
- **NVIDIA Jetson**: A platform with GPU-accelerated hardware for real-time embedded vision and AI applications.
- **OpenVINO**: Intel's toolkit for optimizing and deploying deep learning models on embedded devices with hardware acceleration.
- **YOLO (You Only Look Once)**: A real-time object detection algorithm that is often used in embedded vision applications.
- [Optimum: an extension of 🤗 Transformers and Diffusers, providing a set of optimization tools enabling maximum efficiency to train and run models on targeted hardware](https://github.com/huggingface/optimum)


## Hello World!
```python
import cv2

# Capture video from a camera
cap = cv2.VideoCapture(0)

# Display the video feed
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('Embedded Vision', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Lab: Zero to Hero Projects
- **Smart Camera System**: Develop a smart security camera that can detect and alert for motion using embedded vision.
- **Object Detection on Raspberry Pi**: Implement a real-time object detection system using TensorFlow Lite and Raspberry Pi.
- **Autonomous Drone Navigation**: Build a vision-based navigation system for drones that can follow objects or avoid obstacles.
- **Edge AI with NVIDIA Jetson**: Create a real-time object detection system using the YOLO model on an NVIDIA Jetson platform.

## References
- Bradski, G., & Kaehler, A. (2008). "Learning OpenCV: Computer Vision with the OpenCV Library."
- Szeliski, R. (2010). "Computer Vision: Algorithms and Applications."
- O’Reilly, R. C., & Munakata, Y. (2000). "Computational Explorations in Cognitive Neuroscience."
- NVIDIA. (2020). "Jetson AGX Xavier Developer Kit: Embedded AI Computing for Autonomous Machines."
- Intel. (2020). "OpenVINO Toolkit: Deep Learning Deployment on Hardware."
- https://embeddedvisionsummit.com/
- https://www.edge-ai-vision.com/
- https://www.automate.org/vision/embedded-vision/embedded-vision-what-is-embedded-vision
- https://www.immervision.com/fr/embedded2023/
- https://www.st.com/content/st_com/en/events/embedded-vision-summit-2024.html
- https://www.embeddedvisionsystems.it/
- https://www.adlinktech.com/fr/Machine_Vision_Vision_Systems
- https://www.framos.com/en/resources/what-is-embedded-vision
- https://www.technexion.com/resources/embedded-vision-vs-machine-vision-everything-you-need-to-know/
- https://www.lacroix-impulse.fr/nos-expertises/computer-vision-edge-ia/


Courses:
- [Introduction to On-Device AI - DLA](https://www.coursera.org/projects/introduction-to-on-device-ai)
- [Computer Vision with Embedded ML](https://www.coursera.org/learn/computer-vision-with-embedded-machine-learning)
