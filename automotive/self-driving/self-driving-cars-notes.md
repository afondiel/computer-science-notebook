# Self-Driving Cars - Notes

## Table of Contents (ToC)
- [Introduction](#introduction)
  - [What's Self-Driving Cars?](#whats-self-driving-cars)
  - Key Concepts and Terminology
  - [Applications](#applications)
- [Fundamentals](#fundamentals)
  - [Self-Driving Cars Architecture Pipeline](#self-driving-cars-architecture-pipeline)
  - [How Self-Driving Cars work?](#how-self-driving-cars-work)
  - [Types of Self-Driving Cars](#types-of-self-driving-cars)
  - [Some hands-on examples](#some-hands-on-examples)
- [Tools & Frameworks](#tools--frameworks)
- [Hello World!](#hello-world)
- [Lab: Zero to Hero Projects](#lab-zero-to-hero-projects)
- [References](#references)

## Introduction

Self-driving cars, also known as autonomous vehicles (AVs), are cars capable of sensing their environment and operating without human input.

### What's Self-Driving Cars?
- Vehicles that navigate and control themselves with minimal or no human intervention.
- Use sensors, cameras, and AI to interpret surroundings.
- Aim to reduce accidents and improve traffic flow.

### Key Concepts and Terminology
- **Perception**: Understanding the environment using sensors like LIDAR, RADAR, and cameras.
- **Localization**: Determining the car's position relative to a map or surrounding objects.
- **Planning**: Decision-making algorithms for route planning and obstacle avoidance.
- **Actuation**: Controlling the vehicle’s throttle, brake, and steering.
- **SAE Levels of Automation**: From Level 0 (no automation) to Level 5 (full automation).

### Applications
- Ride-sharing services with autonomous vehicles.
- Last-mile delivery using self-driving delivery robots.
- Reducing traffic congestion and human errors in transportation.
- Autonomous public transport and shuttles.

## Fundamentals

### Self-Driving Cars Architecture Pipeline
- **Perception Layer**: Collects data via sensors (cameras, LIDAR, RADAR).
- **Localization**: Tracks the vehicle's position on a detailed map.
- **Planning**: Chooses the best route and reacts to dynamic changes.
- **Control**: Sends commands to actuate the vehicle's movements.

### How Self-Driving Cars work?
- Sensors gather data about the surrounding environment.
- Algorithms process the data to detect obstacles, vehicles, pedestrians, and lanes.
- Decision-making systems plan the safest and most efficient driving path.
- Actuation systems control steering, braking, and acceleration.

### Types of Self-Driving Cars
- **Level 1**: Driver assistance (e.g., adaptive cruise control).
- **Level 2**: Partial automation (e.g., hands-free lane keeping).
- **Level 3**: Conditional automation (e.g., driver is needed for specific scenarios).
- **Level 4**: High automation (e.g., fully autonomous in controlled environments).
- **Level 5**: Full automation (no human intervention required).

### Some hands-on examples
- Implementing obstacle detection using LIDAR data with ROS.
- Building a lane-following model with OpenCV.
- Simulating a self-driving car in Gazebo or CARLA simulator.

## Tools & Frameworks
- **ROS (Robot Operating System)**: For managing sensor data and control logic.
- **CARLA Simulator**: Open-source platform for autonomous driving research.
- **OpenCV**: For processing camera data and detecting lanes, objects.
- **Autoware**: Autonomous driving software stack.
- **TensorFlow/PyTorch**: For training machine learning models for perception tasks.

## Datasets
- **Waymo Open Dataset**: For self-driving research and development.
- **NuPlan & NuScenes**: For advanced perception tasks and sensor fusion.
- **Argoverse 1 & 2**: For self-driving research and development.
- **KITTI Dataset**: For object detection and scene understanding.
- **Lyft Level 5 Dataset**: For self-driving research and development.
- **MS COCO**: For object detection and segmentation.

## Models & Algorithms
- [YOLO5](https://github.com/ultralytics/yolov5) - Ultralytics
- ...

## Driving Simulators
- [CARLA](https://carla.org/) - Carla.org
- [AirSim](https://microsoft.github.io/AirSim/) - MS
- [NVIDIA DRIVE Sim](https://developer.nvidia.com/drive/simulation) - NVIDIA
- [GAIA-1 world model](https://anthonyhu.github.io/gaia) - Wayne
- [UniSim](https://waabi.ai/introducing-unisim-one-of-the-core-groundbreaking-technologies-powering-waabi-world/) - Waabi

## Self-driving cars Conferences

- [The CVPR Workshop on Autonomous Driving (WAD) - YT Playlist](https://www.youtube.com/@WADatCVPR/playlists)

## Hello World!
```python
import cv2

# Simple object detection using OpenCV
def detect_objects(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    objects = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_car.xml')
    
    detected_objects = objects.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in detected_objects:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    cv2.imshow('Detected Objects', image)
    cv2.waitKey(0)

detect_objects('street_image.jpg')
```

## Lab: Zero to Hero Projects
- Develop an object detection and tracking system using OpenCV and ROS.
- Create a real-time traffic sign recognition model using TensorFlow.
- Build and test a self-driving car simulation in CARLA.
- Implement an obstacle avoidance system with LIDAR and machine learning.

## References
- Miller, J. (2021). *Autonomous Vehicle Technology: A Guide for Policymakers*. RAND Corporation.
- Lin, P. (2022). *The Ethics of Autonomous Cars*. MIT Press.
- CARLA Simulator Documentation: https://carla.org/
- ROS Documentation: https://www.ros.org/
- OpenCV Documentation: https://docs.opencv.org/
