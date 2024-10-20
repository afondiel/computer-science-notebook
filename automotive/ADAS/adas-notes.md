# ADAS (Advanced Driver Assistance Systems) - Notes

## Table of Contents (ToC)
- [Introduction](#introduction)
  - [What's ADAS?](#whats-adas)
  - [Key Concepts and Terminology](#key-concepts-and-terminology)
  - [Applications](#applications)
- [Fundamentals](#fundamentals)
  - [ADAS Architecture Pipeline](#adas-architecture-pipeline)
  - [How ADAS works?](#how-adas-works)
  - [Types of ADAS](#types-of-adas)
  - [ADAS Systems & Taxonomy](#adas-systems--taxonomy)
  - [Some hands-on examples](#some-hands-on-examples)
- [Tools & Frameworks](#tools--frameworks)
- [Hello World!](#hello-world)
- [Lab: Zero to Hero Projects](#lab-zero-to-hero-projects)
- [Safety and Regulatory Considerations](#safety-and-regulatory-considerations)
- [Future Trends](#future-trends)
- [References](#references)

## Introduction
Advanced Driver Assistance Systems (ADAS) are technologies designed to enhance vehicle safety and driving experience by automating, adapting, and enhancing vehicle systems for better driving.

### What's ADAS?
- A set of electronic systems that assist drivers in driving and parking functions.
- Helps reduce human error and improve road safety.
- Includes sensors, cameras, radar, and machine learning algorithms.

### Key Concepts and Terminology
- **LIDAR, RADAR**: Key sensors for object detection.
- **Lane Departure Warning (LDW)**: Alerts when the vehicle drifts out of its lane.
- **Adaptive Cruise Control (ACC)**: Adjusts speed to maintain safe distance from other vehicles.
- **Blind Spot Detection**: Monitors areas not visible to the driver.
- **Computer Vision**: Analyzes camera images for road sign recognition, pedestrian detection, etc.

### Applications
- Enhanced safety features in modern vehicles.
- Reduces the likelihood of accidents through automated interventions.
- Autonomous driving systems for self-driving cars.
- Parking assistance systems, collision avoidance, and emergency braking.

## Fundamentals

### ADAS Architecture Pipeline
- **Sensors**: Collect environmental data (LIDAR, cameras, RADAR).
- **Data Processing**: Fuses sensor inputs using advanced algorithms.
- **Decision Making**: Determines actions like braking or steering.
- **Actuation**: Executes the system's decision (e.g., slowing down the car).

### How ADAS works?
- Detects environmental changes using sensors.
- Processes data using algorithms (computer vision, ML, etc.).
- Issues warnings or automates actions (steering, braking).

### Types of ADAS
- **Level 1**: Driver assistance (e.g., cruise control).
- **Level 2**: Partial automation (e.g., lane keeping with adaptive cruise control).
- **Level 3**: Conditional automation (e.g., hands-off driving under certain conditions).
- **Level 4**: High automation (full driving autonomy in specific areas).
- **Level 5**: Full automation (autonomous driving under all conditions).

### ADAS Systems & Taxonomy Spreadsheet
- [ADAS Systems & Taxonomy Spreadsheet](https://docs.google.com/spreadsheets/d/1-_R2WR6jv2Wxw_fcB8a6t0uluZidPvohOJa3gqDl81Y/edit?usp=sharing)

### Some hands-on examples
- Implementing lane detection using OpenCV.
- Adaptive cruise control with ROS (Robot Operating System).
- Simulating an autonomous parking system with Gazebo and Python.

## Tools & Frameworks
- **OpenCV**: For computer vision tasks like object detection and lane tracking.
- **ROS**: A framework for developing ADAS systems.
- **TensorFlow, PyTorch**: For machine learning and sensor fusion algorithms.
- **Autoware**: An open-source software stack for self-driving technology.
- **Matlab/Simulink**: For modeling and simulating ADAS features.

## Standards
- SAE J3016
- ISO 26262
- MISRA C
- ISO 21434
- ISO 26118
- ISO 9001

## Datasets
- **KITTI Dataset**: For object detection and scene understanding.
- **Waymo Open Dataset**: For self-driving research and development.
- **Nuscenes**: For advanced perception tasks and sensor fusion.

## **ADAS vs Self-Driving Cars**:
- [How a Driverless Car sees the world - Chris Urmson, CEO Aurora](https://youtu.be/tiwVMrTLUWg?si=5zIRHUqHSK1tgK-3)


## Hello World!

```python
import cv2

# Simple lane detection using OpenCV
def detect_lane(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=40, maxLineGap=5)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
    
    cv2.imshow("Lane Detection", image)
    cv2.waitKey(0)

detect_lane('road_image.jpg')
```

## Lab: Zero to Hero Projects
- Develop a lane-keeping assist system using OpenCV and Python.
- Build a pedestrian detection model with TensorFlow.
- Implement adaptive cruise control using ROS and Gazebo.
- Create a collision avoidance system using LIDAR and machine learning.

## References

**Wikipedia**: 
- [Advanced Driver Assistance Systems (ADAS)](https://en.wikipedia.org/wiki/Advanced_driver-assistance_system)


**Books**: 
- John, B. (2020). *Introduction to Autonomous Vehicles*. Pearson.
- Smith, L. (2021). *ADAS and Autonomous Driving: A Comprehensive Guide*. Springer.

**Online Resources**: 
- [OpenCV Documentation](https://docs.opencv.org/)
- [ROS Documentation](https://www.ros.org/)
- [Autoware Documentation](https://www.autoware.org/)


