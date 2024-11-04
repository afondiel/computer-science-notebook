# On-Device AI - Notes

## Table of Contents

- [Introduction](#introduction)
- [Key Concepts](#key-concepts)
- [Applications](#applications)
- [Architecture Pipeline](#architecture-pipeline)
- [Description](#description)
- [Framework / Key Theories or Models](#framework--key-theories-or-models)
- [How On-Device AI Works](#how-on-device-ai-works)
- [Methods, Types & Variations](#methods-types--variations)
- [Self-Practice / Hands-On Examples](#self-practice--hands-on-examples)
- [Pitfalls & Challenges](#pitfalls--challenges)
- [Feedback & Evaluation](#feedback--evaluation)
- [Tools, Libraries & Frameworks](#tools-libraries--frameworks)
- [Hello World! (Practical Example)](#hello-world-practical-example)
- [Advanced Exploration](#advanced-exploration)
- [Zero to Hero Lab Projects](#zero-to-hero-lab-projects)
- [Continuous Learning Strategy](#continuous-learning-strategy)
- [References](#references)

## Introduction

On-Device AI allows machine learning models to run directly on a device (such as a smartphone or embedded system) rather than relying on cloud computing, enhancing privacy and enabling real-time processing.

### Key Concepts
- **Edge Computing**: Processing data at or near the source rather than in centralized cloud servers.
- **Latency**: The delay before data processing begins; on-device AI minimizes latency by processing data locally.
- **Model Compression**: Techniques like quantization and pruning reduce model size to fit the device's memory.
- **Privacy**: By keeping data on-device, privacy is improved as data doesn’t need to be sent to the cloud.
- **Resource Constraints**: Understanding the device’s limitations (CPU, GPU, memory) is crucial for optimizing models.

> *Common Misconceptions*: 
> - On-device AI is often mistakenly thought to match cloud-level capabilities exactly, but it generally involves trade-offs in accuracy or complexity due to hardware constraints.
> - On-device AI is more commonly associated with `consumer-facing applications` directly on devices that a user interacts with daily, while, Embedded AI can cover a broad range of embedded applications within complex systems.

### Applications
- **Voice Assistants**: AI processing enables real-time responses on devices without a network connection.
- **Smart Cameras**: On-device AI is used in facial recognition or motion detection for real-time monitoring.
- **Augmented Reality (AR)**: Processes environmental data on the device to overlay digital information seamlessly.
- **Health Wearables**: Fitness trackers and health monitors analyze data directly on the device, increasing responsiveness.
- **Automotive Sensors**: In self-driving cars, AI analyzes sensor data on the car’s system to ensure timely decision-making.

## Architecture Pipeline

```mermaid
graph LR
    A[Data Collection] --> B[Data Preprocessing]
    B --> C[Inference Model]
    C --> D[Output Processing]
```

### Description
- **Data Collection**: Capture data locally.
- **Data Preprocessing**: Reduce and filter data to fit memory constraints.
- **Inference Model**: Uses optimized AI models (often smaller or less complex) for efficient processing.
- **Output Processing**: Provides results directly to the device user or as system feedback.

## Framework / Key Theories or Models
- **TinyML**: Focuses on running ML models on microcontrollers.
- **Quantization**: Reduces model size by lowering the precision of weights and activations.
- **Pruning**: Removes unnecessary parts of a neural network to improve efficiency.
- **Knowledge Distillation**: Transfers knowledge from a larger model to a smaller, faster one.
- **Edge AI Models**: Frameworks like MobileNet and Edge TPU specialize in compact, efficient on-device models.

## How On-Device AI Works
- **Data processing** occurs locally, often in an optimized model designed for real-time inference.
- **Model compression** techniques ensure that the model fits within device constraints.
- **Output results** (like alerts, recommendations) are displayed or acted upon directly, enabling faster interaction without relying on cloud-based responses.

## Methods, Types & Variations
- **Model Optimization**: Techniques like pruning and quantization.
- **Specialized Architectures**: MobileNet, SqueezeNet, etc., are created to run well on low-resource devices.
- **Edge Devices Variations**: Different implementations for IoT, smartphones, wearables, etc.

## Self-Practice / Hands-On Examples
1. Create a TinyML model for image classification on a microcontroller.
2. Implement voice recognition using TensorFlow Lite on a mobile device.
3. Deploy an AR application using an edge AI model.

## Pitfalls & Challenges
- **Limited Processing Power**: On-device AI models often have to be significantly simplified.
- **Battery Constraints**: AI tasks consume power, so balancing efficiency with battery life is essential.
- **Maintenance**: Models may need frequent updates as the device operates in a dynamic real-world environment.

## Feedback & Evaluation
- **Self-explanation Test**: Describe why certain models are more suitable for on-device applications.
- **Real-world Simulation**: Test on multiple devices to gauge performance variability.
- **Peer Review**: Discuss model trade-offs with peers for optimization insights.

## Tools, Libraries & Frameworks
- **TensorFlow Lite**: TensorFlow's solution for on-device machine learning.
- **Core ML**: Apple’s framework for running models on iOS devices.
- **OpenCV**: Widely used for image processing; optimized for performance on devices.
- **NVIDIA Jetson Platform**: Specialized in running deep learning models on embedded devices.
- **ML Kit by Firebase**: Google’s mobile SDK for ML.

### Comparison of Tools
- **TensorFlow Lite**: Excellent for custom ML model deployment.
- **Core ML**: Optimized for iOS; integrates well with Apple hardware.
- **OpenCV**: Broad support for image processing; integrates with multiple hardware platforms.

## Hello World! (Practical Example)

```python
import tensorflow as tf
import tensorflow.lite as tflite

# Load and convert a model to TensorFlow Lite for on-device deployment
model = tf.keras.models.load_model('my_model.h5')
converter = tflite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the converted model to use on an edge device
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

## Advanced Exploration
1. **Edge AI Whitepapers**: Papers from NVIDIA and ARM on optimizing for edge devices.
2. **Courses**: TinyML course by HarvardX for microcontroller-based ML.
3. **Research**: Review papers on federated learning for on-device data security.

## Zero to Hero Lab Projects
- **Home Security**: Build a smart camera using Raspberry Pi and TensorFlow Lite to detect motion or intruders.
- **Gesture Recognition**: Implement gesture recognition on a mobile app using Core ML and model compression.
- **Health Monitoring**: Create a health tracker on a wearable device using optimized, compressed AI models.

## Continuous Learning Strategy
- **Next Steps**: Explore federated learning for on-device training; investigate device-specific model optimizations.
- **Related Topics**: Embedded AI, Model Compression, IoT Security.

## References
- **Harvard TinyML Course**: [Link to course](https://www.edx.org/course/tiny-machine-learning)
- **NVIDIA AI on Edge Whitepaper**: [Link to whitepaper](https://developer.nvidia.com)
- **TensorFlow Lite Documentation**: [Link to docs](https://www.tensorflow.org/lite) 

