# Edge AI Hardware Platforms

## Overview

A list of Edge AI hardware platforms that support AI and machine learning developments for real-world applications.

## Related Notes
- [Edge AI Platforms](https://github.com/afondiel/Edge-AI-Platforms), [Edge-AI Model Zoo](https://github.com/afondiel/Edge-AI-Model-Zoo), [Awesome Smol](https://github.com/afondiel/awesome-smol), [Awesome Edge](https://github.com/afondiel/awesome-edge)

## Table of Contents
- [Overview](#overview)
- [Edge Device Platforms](#edge-device-platforms)
  - [1. System on Chips (SoCs)](#1-system-on-chips-socs)
  - [2. Microcontrollers (MCUs)](#2-microcontrollers-mcus)
  - [3. Field-Programmable Gate Arrays (FPGAs)](#3-field-programmable-gate-arrays-fpgas)
  - [4. Edge AI Boxes and Gateways](#4-edge-ai-boxes-and-gateways)
  - [5. Mobile and Embedded Devices](#5-mobile-and-embedded-devices)
  - [6. AI Development Platforms](#6-ai-development-platforms)
  - [7. Specialized Edge Devices](#7-specialized-edge-devices)
  - [8. Industrial and Custom Edge Devices](#8-industrial-and-custom-edge-devices)
  - [9. Robotics-focused Edge Devices](#9-robotics-focused-edge-devices)
- [Industry Applications](#industry-applications)
  - [1. Industrial Automation](#1-industrial-automation)
  - [2. Healthcare and Medical Devices](#2-healthcare-and-medical-devices)
  - [3. Automotive and Transportation](#3-automotive-and-transportation)
  - [4. Smart Cities and Surveillance](#4-smart-cities-and-surveillance)
  - [5. Retail and E-commerce](#5-retail-and-e-commerce)
  - [6. Robotics](#6-robotics)
  - [7. Agriculture](#7-agriculture)
  - [8. Consumer Electronics and Wearables](#8-consumer-electronics-and-wearables)
  - [9. Edge AI for IoT](#9-edge-ai-for-iot)
- [Edge Devices Categorized by Computing Performance](#edge-devices-categorized-by-computing-performance)
  - [1. Light Performance Applications](#1-light-performance-applications)
  - [2. Mid Performance Applications](#2-mid-performance-applications)
  - [3. High Performance Applications](#3-high-performance-applications)
- [References](#references)

## Edge Device Platforms

[Back to Table of Contents](#table-of-contents)

### **1. System on Chips (SoCs):**
- **NVIDIA Jetson Series:**
  - **Jetson Orin Nano Developer Kit:** Compact AI computer with 67 TOPS for robotics and AI applications.
  - **Jetson Xavier NX:** Up to 21 TOPS for edge AI applications.
  - **Jetson Nano:** Entry-level AI device for robotics and vision tasks.
- **Qualcomm Snapdragon Series:**
  - **Snapdragon 8cx Gen 3:** AI processing for mobile and embedded devices.
  - **Snapdragon 888 AI Accelerator:** Built for high-performance on-device AI.
- **Apple Silicon:**
  - **M1/M2 Series:** Integrated AI capabilities for edge computing.
- **Google Coral Edge TPU:** Specialized for AI inference at the edge.
- **Hailo-8 AI Processor:** High-performance deep learning on edge devices.
- **Rockchip RK3399Pro:** Combines CPU, GPU, and AI processing.
- **HiSilicon Ascend Series:** AI-focused SoCs for industrial and mobile applications.

### **2. Microcontrollers (MCUs):**
- **STMicroelectronics STM32N6 Series:** AI microcontrollers for lightweight edge tasks (image/audio).
- **Texas Instruments SimpleLink™ MCU Series:** AI and ML-ready for IoT and edge computing.
- **Espressif ESP32 Series:** Compact AI-enabled microcontroller for lightweight applications.
- **NXP i.MX RT Series:** Supports AI tasks on ultra-low-power microcontrollers.
- **Renesas RA MCU Series:** Optimized for TinyML workloads in IoT and edge systems.

### **3. Field-Programmable Gate Arrays (FPGAs):**
- **Xilinx Alveo Series:** High-performance AI workloads at the edge.
- **Intel Stratix 10 Series:** Designed for AI inference in resource-intensive tasks.
- **EdgeLLM Accelerator:** Heterogeneous CPU-FPGA for large language models at the edge.

### **4. Edge AI Boxes and Gateways:**
- **Qualcomm Edge AI Box:** Industrial edge AI device for vision and sensor applications.
- **NVIDIA Jetson-based Edge AI Boxes:** Pre-configured AI deployment systems.
- **Advantech Edge AI Box PCs:** Designed for industrial and smart city applications.
- **Intel NUC Kits:** AI-ready edge computing boxes.

### **5. Mobile and Embedded Devices:**
- **AI-enabled Smartphones:**
  - Apple's iPhones with Neural Engine.
  - Android phones with AI-dedicated processors (e.g., Google Tensor).
- **Raspberry Pi with AI Accelerators:**
  - Paired with Google Coral USB Accelerator or Intel Neural Compute Stick.
- **BeagleBone AI-64:** Embedded system for AI development.
- **Khadas VIM3:** Linux-powered AI board for edge applications.
- **Jetson Nano Embedded Systems:** Customized AI systems for robotics and IoT.

### **6. AI Development Platforms:**
- **Edge Impulse:** AI development on microcontrollers and SoCs.
- **AWS SageMaker Edge:** Optimized ML models for edge deployment.
- **TensorFlow Lite:** Framework for deploying ML models on edge devices.
- **PyTorch Mobile:** For edge AI model development and deployment.

### **7. Specialized Edge Devices:**
- **Google Nest Devices:** Smart home devices with AI integration (e.g., Google Nest Cam).
- **Amazon Echo Devices:** AI-powered assistants with on-device processing.
- **Microsoft Azure Percept:** AI-powered platform for edge solutions.
- **AI Surveillance Cameras:** Devices like Hikvision AI-powered security systems.

### **8. Industrial and Custom Edge Devices:**
- **Bosch IoT Suite Devices:** AI-ready industrial IoT systems.
- **Siemens MindSphere Gateways:** Integrated edge AI for industrial automation.
- **Intel Movidius Myriad X:** Vision processing units for industrial AI applications.
- **Advantech Embedded AI Systems:** Specialized for factory automation and logistics.

### **9. Robotics-focused Edge Devices:**
- **Unitree Robotics A1:** Robot with integrated edge AI processing.
- **Boston Dynamics Spot:** AI-powered robot with edge computing capabilities.
- **Open Robotics TurtleBot3:** Affordable robot with AI-enabled edge processing.

### **Industry Applications**

[Back to Table of Contents](#table-of-contents)

#### **1. Industrial Automation**
Edge devices optimized for manufacturing, logistics, and factory automation:  
- **NVIDIA Jetson AGX Xavier:** Robotic arms, smart manufacturing.  
- **Bosch IoT Suite Devices:** Real-time monitoring, predictive maintenance.  
- **Intel Movidius Myriad X:** AI vision for industrial automation.  
- **Advantech Embedded AI Systems:** Factory optimization, process control.  
- **Xilinx Alveo Series:** High-speed, real-time industrial AI.  

#### **2. Healthcare and Medical Devices**
Edge devices used for diagnostics, medical imaging, and remote healthcare:  
- **Qualcomm Snapdragon 8cx Gen 3:** AI in portable medical devices.  
- **Google Coral Edge TPU:** Real-time diagnostics and analysis.  
- **Jetson Orin Nano Developer Kit:** AI in point-of-care devices.  
- **Siemens Healthineers Edge Systems:** Imaging and diagnostic machines.  
- **STMicroelectronics STM32N6 Series:** TinyML in wearable health trackers.  

#### **3. Automotive and Transportation**
Edge solutions for ADAS, autonomous driving, and fleet management:  
- **NVIDIA Jetson Xavier NX:** Autonomous vehicles, ADAS.  
- **Qualcomm Snapdragon Ride Platform:** Automotive AI workloads.  
- **Hailo-8 AI Processor:** Object detection and path planning in vehicles.  
- **Intel NUC Kits:** In-car edge computing for infotainment.  
- **Renesas R-Car Series:** Traffic monitoring, vehicle-to-everything (V2X).  

#### **4. Smart Cities and Surveillance**
AI devices for urban infrastructure, traffic management, and security:  
- **Google Nest Cam:** Real-time AI-enabled surveillance.  
- **Hikvision AI Surveillance Cameras:** Smart security systems.  
- **Advantech Edge AI Boxes:** Urban planning and monitoring.  
- **Jetson Nano Embedded Systems:** Pedestrian and vehicle tracking.  
- **Amazon Echo Devices:** Smart city IoT integration.  

#### **5. Retail and E-commerce**
Devices for inventory management, personalized shopping experiences, and checkout-free stores:  
- **NVIDIA Jetson Nano:** AI-powered checkout systems.  
- **Intel Movidius Myriad X:** Shelf scanning and customer behavior analysis.  
- **Advantech Edge AI Box PCs:** Automated inventory tracking.  
- **Khadas VIM3:** AI in cashier-less stores.  
- **STMicroelectronics STM32 AI Series:** Embedded AI for smart kiosks.  

#### **6. Robotics**
Edge platforms enabling autonomous operation, AI, and smart behaviors in robotics:  
- **NVIDIA Jetson Orin Nano Developer Kit:** Autonomous robots.  
- **Boston Dynamics Spot:** Industrial and field robotics with edge AI capabilities.  
- **NVIDIA Isaac Robot Platform:** AI-driven simulation and deployment for robotics.  
- **Clearpath Robotics Jackal UGV:** Unmanned ground vehicle for autonomous navigation.  
- **Fetch Robotics Freight 500:** AI-driven logistics and warehouse automation.  
- **KUKA iiwa Robot:** AI-enhanced industrial collaborative robots.  
- **Unitree Robotics A1:** Quadruped robot with AI for movement and sensing.  
- **Open Robotics TurtleBot3:** Research and educational robots.  
- **BeagleBone AI-64:** Robotics prototyping with on-device AI.  

#### **7. Agriculture**
AI-enabled devices for precision farming, crop monitoring, and livestock management:  
- **Jetson Xavier NX:** AI in drones for agriculture.  
- **Google Coral Edge TPU:** Crop health assessment via AI.  
- **Raspberry Pi with AI Accelerators:** Soil and weather monitoring.  
- **Hailo-8 AI Processor:** Smart irrigation and yield prediction.  
- **Advantech Embedded Systems:** Automated harvesting systems.
  
#### **8. Consumer Electronics and Wearables**
AI devices for personal use and smart home integration:  
- **Apple Neural Engine (M1/M2):** AI in iPhones, iPads, and wearables.  
- **Amazon Echo Devices:** Smart speakers with on-device AI.  
- **Espressif ESP32 Series:** Home automation with AI.  
- **Google Nest Devices:** Smart home systems.  
- **STMicroelectronics AI-ready MCUs:** Lightweight AI wearables.  

#### **9. Edge AI for IoT**
Devices driving IoT applications across sectors:  
- **[Qualcomm Edge AI Boxes](https://www.qualcomm.com/products/technology/artificial-intelligence/edge-ai-box):** IoT gateways for industrial settings.  
- **AWS SageMaker Edge:** Cloud-enabled IoT applications.  
- **NXP i.MX RT Series:** Smart sensors and IoT edge processing.  
- **TensorFlow Lite-enabled Devices:** ML for IoT sensors and cameras.  
- **PyTorch Mobile on Raspberry Pi:** IoT prototyping with AI models.

## Edge Devices Categorized by Computing Performance

[Back to Table of Contents](#table-of-contents)

### **1. Light Performance Applications**

Devices in this category are suitable for basic tasks, low-power applications, or those with simpler AI/ML models that do not require heavy computation.

| **Device**                        | **Description**                                                                                              | **CPU Type**             | **Memory** | **Storage**   | **Energy**      | **Inference Speed** | **Latency**   | **GPU Support** | **Throughput** | **Use Cases**                      |
|------------------------------------|--------------------------------------------------------------------------------------------------------------|-------------------------|------------|---------------|-------------------|-----------------------|---------------|----------------|----------------|----------------------------------|
| Raspberry Pi 4 (with AI accelerator) | A versatile single-board computer with AI accelerator support for light ML applications.                      | Low-power processors (ARM Cortex-A)   | < 1 GB     | Flash storage  | Low power (1 - 5 W)  | 500 ms - 2 s          | High latency     | Basic GPU         | Low                  | Simple sensors, basic monitoring     |
| Google Coral Dev Board (Edge TPU) | Edge device with Edge TPU for accelerating ML tasks like image recognition at low power.                       | Low-power processors (ARM Cortex-A)   | 1 GB - 4 GB   | Flash storage  | Low power (1 - 5 W)  | 500 ms - 2 s          | High latency     | Basic GPU         | Low                  | IoT, basic AI                           |
| BeagleBone AI-64                   | AI-capable development board offering basic edge AI capabilities.                                             | Low-power processors (ARM Cortex-A)   | 1 GB - 4 GB   | Flash storage  | Low power (1 - 5 W)  | 500 ms - 2 s          | High latency     | Basic GPU         | Low                  | Simple robotics, monitoring systems     |
| ESP32 (with AI capabilities)       | Low-power microcontroller with built-in Wi-Fi and Bluetooth, capable of running basic AI algorithms.           | Low-power processors (low-end RISC)    | < 1 GB     | Flash storage  | Low power (1 - 5 W)  | 500 ms - 2 s          | High latency     | Basic GPU         | Low                  | Simple IoT devices, wearables          |
| Arduino Portenta H7                | High-performance board for edge computing tasks, suitable for moderate AI workloads with low power requirements. | Low-power processors (ARM Cortex-A)   | 1 GB - 4 GB   | Flash storage  | Low power (1 - 5 W)  | 500 ms - 2 s          | High latency     | Basic GPU         | Low                  | Robotics, smart home                  |
| NVIDIA Jetson Nano                 | Entry-level edge AI platform with GPU support for basic AI tasks.                                             | Low-power processors (ARM Cortex-A)   | 1 GB - 4 GB   | Flash storage  | Low power (1 - 5 W)  | 500 ms - 2 s          | High latency     | Basic GPU         | Low                  | Basic AI, robotics                     |
| Unitree Robotics A1                | Lightweight robot designed for basic tasks and low-end AI applications.                                       | Low-power processors (ARM Cortex-A)   | < 1 GB     | Flash storage  | Low power (1 - 5 W)  | 500 ms - 2 s          | High latency     | Basic GPU         | Low                  | Basic robot navigation                   |
| Open Robotics TurtleBot3            | Simple, open-source robot with basic sensors and light AI capabilities.                                       | Low-power processors (ARM Cortex-A)   | < 1 GB     | Flash storage  | Low power (1 - 5 W)  | 500 ms - 2 s          | High latency     | Basic GPU         | Low                  | Educational robotics                   |
| Qualcomm Snapdragon 410e           | Embedded AI solution targeting low-power, mobile applications.                                                | Low-power processors (ARM Cortex-A)   | 1 GB - 4 GB   | Flash storage  | Low power (1 - 5 W)  | 500 ms - 2 s          | High latency     | Basic GPU         | Low                  | Simple embedded AI systems            |

### **2. Mid Performance Applications**
These devices can handle more computationally intensive tasks and are suited for real-time AI inference, edge computing, and moderate machine learning tasks.

| **Device**                        | **Description**                                                                                              | **CPU Type**             | **Memory** | **Storage**   | **Energy**      | **Inference Speed** | **Latency**   | **GPU Support** | **Throughput** | **Use Cases**                      |
|------------------------------------|--------------------------------------------------------------------------------------------------------------|-------------------------|------------|---------------|-------------------|-----------------------|---------------|----------------|----------------|----------------------------------|
| NVIDIA Jetson Xavier NX            | Powerful edge AI device with the ability to handle real-time AI inference.                                    | Mid-range processors (ARM Cortex-A)   | 4 GB - 8 GB  | SSD            | Moderate power (5 - 30 W)  | 50 ms - 500 ms        | Moderate latency    | Integrated GPU | Medium             | Robotics, AI edge inference           |
| NVIDIA Jetson Orin Nano Developer Kit | More advanced edge device for running demanding AI models with low power.                                     | Mid-range processors (ARM Cortex-A)   | 4 GB - 8 GB  | SSD            | Moderate power (5 - 30 W)  | 50 ms - 500 ms        | Moderate latency    | Integrated GPU | Medium             | Robotics, AI processing                 |
| Google Coral Edge TPU              | Edge device designed for running fast, efficient ML models with a dedicated TPU for inference.                | Mid-range processors (ARM Cortex-A)   | 4 GB        | SSD            | Moderate power (5 - 30 W)  | 50 ms - 500 ms        | Moderate latency    | Integrated GPU | Medium             | AI edge, machine vision                   |
| Raspberry Pi 4 (with AI accelerator) | Entry-level AI device with sufficient power for moderate AI inference.                                       | Mid-range processors (ARM Cortex-A)   | 4 GB        | Flash storage  | Moderate power (5 - 30 W)  | 50 ms - 500 ms        | Moderate latency    | Integrated GPU | Medium             | AI for IoT systems                        |
| Intel NUC (Edge AI capabilities)   | Compact PC designed to run AI models at the edge with decent processing power.                                | Mid-range processors (x86-based)   | 4 GB - 8 GB  | SSD            | Moderate power (5 - 30 W)  | 50 ms - 500 ms        | Moderate latency    | Integrated GPU | Medium             | AI applications, robotics                  |
| KUKA iiwa Robot                    | Advanced robotic platform with AI for industrial automation tasks.                                           | Mid-range processors (Intel Atom)   | 8 GB        | SSD            | Moderate power (5 - 30 W)  | 50 ms - 500 ms        | Moderate latency    | Integrated GPU | Medium             | Industrial automation, robots          |
| Clearpath Robotics Jackal UGV      | Autonomous robot designed for outdoor, rugged environments with real-time AI navigation.                    | Mid-range processors (ARM Cortex-A)   | 4 GB        | SSD            | Moderate power (5 - 30 W)  | 50 ms - 500 ms        | Moderate latency    | Integrated GPU | Medium             | Outdoor robotics, AI navigation         |
| Boston Dynamics Spot               | Advanced robot designed for various industrial applications with AI-powered perception and autonomy.         | Mid-range processors (ARM Cortex-A)   | 8 GB        | SSD            | Moderate power (5 - 30 W)  | 50 ms - 500 ms        | Moderate latency    | Integrated GPU | Medium             | Robotics, navigation, AI tasks          |
| Fetch Robotics Freight 500         | Logistics robot using AI for material transport in warehouses.                                               | Mid-range processors (ARM Cortex-A)   | 4 GB        | SSD            | Moderate power (5 - 30 W)  | 50 ms - 500 ms        | Moderate latency    | Integrated GPU | Medium             | Logistics, transport                    |
| NVIDIA Jetson TX2                  | Edge computing platform suitable for AI and robotics applications with moderate processing needs.            | Mid-range processors (ARM Cortex-A)   | 8 GB        | SSD            | Moderate power (5 - 30 W)  | 50 ms - 500 ms        | Moderate latency    | Integrated GPU | Medium             | AI, robotics, automation                   |
| Intel Movidius Neural Compute Stick 2 | USB stick with AI acceleration for running models at the edge.                                                | Low-power processors   | < 1 GB     | Flash storage  | Low power (1 - 5 W)  | 50 ms - 500 ms        | Moderate latency    | Integrated GPU | Medium             | Edge AI inference                        |

### **3. High Performance Applications**

Devices in this category offer the highest computational power and are suitable for complex AI/ML models, heavy real-time processing, and advanced robotics.

| **Device**                        | **Description**                                                                                              | **CPU Type**             | **Memory** | **Storage**   | **Energy**      | **Inference Speed** | **Latency**   | **GPU Support** | **Throughput** | **Use Cases**                      |
|------------------------------------|--------------------------------------------------------------------------------------------------------------|-------------------------|------------|---------------|-------------------|-----------------------|---------------|----------------|----------------|----------------------------------|
| NVIDIA Jetson AGX Orin             | High-performance edge AI device designed for complex machine learning models and real-time AI tasks.            | High-end processors (ARM Neoverse)   | > 8 GB     | NVMe storage  | High power (30 W - 150 W)  | < 50 ms              | Low latency        | Dedicated AI accelerators | High             | Autonomous vehicles, real-time video processing, AI for robotics |
| NVIDIA Jetson Xavier AGX           | Top-tier edge AI platform for heavy-duty AI tasks and robotics applications.                                    | High-end processors (ARM Neoverse) | 16 GB      | SSD            | High power (30 W - 150 W)  | < 50 ms              | Low latency        | Dedicated GPU | High             | AI in autonomous systems, robotics    |
| Intel Xeon Scalable (with AI accelerator) | High-end edge computing platform for advanced AI applications, typically used in data centers.              | High-end processors (Intel Xeon)   | > 8 GB     | SSD            | High power (30 W - 150 W)  | < 50 ms              | Low latency        | Dedicated AI accelerators | High             | Cloud and edge AI, machine learning, AI applications in robotics  |
| Google TPU (Cloud-based AI inference)  | Cloud-based solution for running complex machine learning models at scale with low latency.                   | High-end processors (custom TPUs)     | > 16 GB    | SSD/NVMe       | High power (30 W - 150 W)  | < 50 ms              | Low latency        | High throughput  | High             | Large-scale AI, autonomous vehicles    |
| IBM Power9 with AI Acceleration   | High-performance computing platform optimized for AI and deep learning tasks.                                | High-end processors (Power9)   | > 32 GB    | SSD/NVMe       | High power (30 W - 150 W)  | < 50 ms              | Low latency        | High throughput  | High             | AI in advanced robotics and industrial applications |


## References

[Back to Table of Contents](#table-of-contents)

- [NVIDIA Jetson Orin Nano announcement](https://blogs.nvidia.com/blog/jetson-generative-ai-supercomputer/)
- [Edge AI Platforms: From entry-level to high performance, how to select your AI devices?](https://www.axiomtek.com/Default.aspx?MenuId=Solutions&FunctionId=SolutionView&ItemId=2807&Title=Edge+AI+Platforms)
- [SparkFun Edge Development Board - Apollo3 Blue](https://www.sparkfun.com/products/15170)
- [Edge TPU live demo: Coral Dev Board & Microcontrollers (TF Dev Summit '19)](https://www.youtube.com/watch?v=CukaWaWbBHY)





