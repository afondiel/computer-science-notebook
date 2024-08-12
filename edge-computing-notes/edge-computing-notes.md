# Edge Computing - Notes

## Table of Contents (ToC)

  - [Introduction](#introduction)
    - [What's Edge Computing?](#whats-edge-computing)
    - [Key Concepts and Terminology](#key-concepts-and-terminology)
    - [Applications](#applications)
  - [Fundamentals](#fundamentals)
    - [Edge Computing Architecture Pipeline](#edge-computing-architecture-pipeline)
    - [How Edge Computing Works?](#how-edge-computing-works)
    - [Some Hands-on Examples](#some-hands-on-examples)
  - [Tools \& Frameworks](#tools--frameworks)
  - [Hello World!](#hello-world)
  - [References](#references)

## Introduction

`Edge computing` brings computation and data storage closer to the location where it is needed to improve performance and efficiency.

### What's Edge Computing?
- Distributed computing paradigm
- Moves data processing closer to data sources
- Reduces latency and bandwidth use

### Key Concepts and Terminology
- Edge Node: Device that performs data processing at the edge
- Latency: Time delay in communication
- Bandwidth: Data transfer rate

### Applications
- IoT (Internet of Things)
- Smart Cities
- Autonomous Vehicles

## Fundamentals

### Edge Computing Architecture Pipeline
- **Data Source**: Sensors, IoT devices
- **Edge Nodes**: Local servers, gateways
- **Data Processing**: On-site computation
- **Cloud Integration**: Synchronization with cloud resources

### How Edge Computing Works?
- Data is collected from local devices.
- Processed locally on edge nodes.
- Aggregated or synchronized with the cloud as needed.

### Some Hands-on Examples
- Processing video feeds from surveillance cameras locally
- Real-time analytics on sensor data in manufacturing

## Tools & Frameworks
- **AWS Greengrass**: Extends AWS capabilities to edge devices
- **Microsoft Azure IoT Edge**: Runs cloud workloads on edge devices
- **Google Cloud IoT Edge**: Facilitates AI at the edge
- **K3s**: Lightweight Kubernetes distribution for edge deployments

## Hello World!

Hello World I: Infering ML model using Tensoflow 

```python
# Sample code using TensorFlow Lite for edge computing on IoT devices
import tensorflow as tf

# Load the TensorFlow Lite model for edge deployment
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
# Perform inference on edge device data
# (Additional code for data preprocessing and inference goes here)
```

Hello World II: Task Management

```python
# Simple edge computing example using a local Python script
def edge_computing_task(data):
    processed_data = data * 2  # Simulate data processing
    return processed_data

data = 10
result = edge_computing_task(data)
print("Processed Data:", result)
```

## References

- [AWS IoT Greengrass Documentation](https://aws.amazon.com/greengrass/)
- [Azure IoT Edge Documentation](https://docs.microsoft.com/en-us/azure/iot-edge/)
- [Top 10 Edge Computing Platforms in 2022](https://www.spiceworks.com/tech/edge-computing/articles/best-edge-computing-platforms/)
- [Cloud Computing vs. Edge Computing: A Comparative Overview](https://ubiminds.com/en-us/cloud-computing-vs-edge-computing/)
- [NVIDIA - Robotics and Edge Computing Solutions](https://www.nvidia.com/en-us/solutions/robotics-and-edge-computing/)

Lectures & Online Courses:
- [Edge Computing Fundamentals - Coursera by Learn Quest](https://www.coursera.org/learn/security-at-the-edge-first-course-1?utm_medium=sem&utm_source=gg&utm_campaign=B2C_EMEA__coursera_FTCOF_career-academy_pmax-multiple-audiences-country-multi-set2&campaignid=20882109092&adgroupid=&device=c&keyword=&matchtype=&network=x&devicemodel=&adposition=&creativeid=&hide_mobile_promo&gad_source=1&gclid=Cj0KCQjw5ea1BhC6ARIsAEOG5pwDvNtAmQOWXb3ue3QE1_rrH08gy3YCbn-7ExiAnvm4esvTaIxVFA8aAvlfEALw_wcB)
- [Security at the Edge Specialization - Secure your Edge and IoT Devices](https://www.coursera.org/specializations/security-at-the-edge)

Research & Survey:

- [Edge Computing: Concepts, Technologies, and Applications - Satyanand M. & Aruna P. (2021). Springer](#)
- [The Promise of Edge Computing - Shi W., Dustdar S.(2016). IEEE Computer](#)
- [Edge Computing: A Survey  - Zhang, X., & Yang, Z. (2019). IEEE Internet of Things Journal](#)

Books:

- [Books - @afondiel](https://github.com/afondiel/cs-books)

## Related Sources

- [Embedded AI](../embedded-ai-notes/tinyML/tinyML-notes.md)

