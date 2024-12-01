# End-to-End Edge-AI Deployment Pipeline

![](https://github.com/afondiel/Introduction-to-On-Device-AI-DLAI/blob/main/lab/chapters/slides/01_Why_on_device/why_on_device_6.png?raw=true)

(Source: [Link](https://github.com/afondiel/Introduction-to-On-Device-AI-DLAI/blob/main/lab/chapters/slides/01_Why_on_device/why_on_device_6.png))

## Table of Contents (ToC)  
1. [Overview](#overview)
2. [Choose the Right Edge Device](#1-choose-the-right-edge-device)
3. [Optimize Your Model](#2-optimize-your-model)
4. [Convert Your Model](#3-convert-your-model)
5. [Deploy the Model](#4-deploy-the-model)
6. [Test and Validate](#5-test-and-validate)
7. [Monitor and Update](#6-monitor-and-update)
8. [Tools and Frameworks](#tools-and-frameworks)
9. [Conclusion](#conclusion)
10. [References](#references)


## Overview

Edge deployment of AI models involves running machine learning algorithms on local hardware devices rather than in the cloud.

This approach offers benefits like:

- **reduced latency**,
    - instead of waiting for a cloud server to process data, the data is processed locally on the edge device.
- **improved privacy**,
    - the data is not sent to a remote server, so it is not exposed to the public internet.
- **offline functionality**,
    - the device can operate without an internet connection.
- **real-time processing**,
    - the data is processed immediately on the device **without the need for a cloud server**.

---

**Here's a step-by-step guide to deploying AI models on edge devices:**
```mermaid
graph LR
    A[Choose the Right Edge Device] --> B[Optimize Your Model]
    B --> C[Convert Your Model]
    C --> D[Deploy the Model]
    D --> E[Test and Validate]
    E --> F[Monitor and Update]
```


## 1. Choose the Right Edge Device

Select an edge device based on your project requirements:

- **Single-board computers**: Raspberry Pi, NVIDIA Jetson Nano
- **Microcontrollers**: Arduino, ESP32
- **Specialized AI hardware**: Google Coral, Intel Neural Compute Stick
- **Mobile devices**: Smartphones, tablets

Consider factors like **processing power, memory, power consumption**, and supported frameworks.

## 2. Optimize Your Model

Edge devices often have limited resources, so model optimization is crucial:

- **Quantization**: Reduce model precision (e.g., from 32-bit to 8-bit)
- **Pruning**: Remove unnecessary connections in neural networks
- **Knowledge Distillation**(teacher-student approach): Create a smaller model that mimics a larger one
- **Model Compression**: Use techniques like `weight sharing` or `huffman coding`

## 3. Convert Your Model

Convert your model to a format suitable for edge deployment:

- **TensorFlow Lite** for TensorFlow models
- **ONNX** (Open Neural Network Exchange) for cross-framework compatibility
- **CoreML** for iOS devices
- **OpenVINO** for Intel hardware
- [**ML Kit**](https://developers.google.com/ml-kit)

Example using TensorFlow Lite:

```	python
import tensorflow as tf
# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()
# Save the model
with open('model.tflite', 'wb') as f:
f.write(tflite_model)
```

## 4. Set Up the Edge Device

Prepare your edge device:

1. Install the necessary operating system (e.g., Raspbian for Raspberry Pi)
2. Set up the development environment
3. Install required libraries and drivers

## 5. Deploy the Model

Transfer the optimized model to the edge device and set up the inference pipeline:

1. Copy the model file to the device
2. Install the inference framework (e.g., TensorFlow Lite, OpenVINO)
3. Write code to load the model and perform inference

```python
import tflite_runtime.interpreter as tflite
# Load the model
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# Perform inference
input_data = ... # Prepare your input data
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
```

## 6. Optimize Performance

Fine-tune your deployment for better performance:

- Use hardware acceleration if available (e.g., EdgeTPU, Neural Compute Stick)
- Implement efficient data preprocessing on the device
- Optimize the inference pipeline for your specific use case

## 7. Ensure Reliability and Maintenance

Implement strategies for reliable operation:

- Error handling and logging
- Over-the-air (OTA) updates for model and software
- Monitoring and telemetry for performance tracking

## 8. Consider Security

Implement security measures:

- Encrypt the model and sensitive data
- Secure communication channels
- Implement access controls and authentication

## Tools and Frameworks

Several tools can help with edge deployment:

- **Edge Impulse**: Enables development and deployment of TinyML models
- **TensorFlow Lite**: Optimized version of TensorFlow for mobile and embedded devices
- **OpenVINO**: Intel's toolkit for optimizing and deploying AI models
- **NVIDIA TensorRT**: SDK for high-performance deep learning inference
- **Apache TVM**: Open-source machine learning compiler framework
- **Onnx**: Open Neural Network Exchange for cross-platform compatibility.
- **[Qualcomm AI Hub](https://aihub.qualcomm.com/)**: The platform for on-device AI

## Conclusion

Deploying AI models on edge devices requires careful consideration of hardware limitations, model optimization, and efficient implementation. 

By following this guide and leveraging appropriate tools, you can successfully run AI models on edge devices, enabling a wide range of applications in IoT, robotics, and embedded systems.

## References

- [OpenVINO](https://docs.openvino.ai/latest/index.html)
- [TensorFlow Lite](https://www.tensorflow.org/lite)
- [Edge Impulse](https://www.edgeimpulse.com/)
- [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt)
- [Apache TVM](https://tvm.apache.org/)
- [Onnx](https://onnx.ai/)

Vision platforms:

- Roboflow
- [viso.ai](http://viso.ai/)
- Encord
- Hugging Face
- https://modelzoo.co/
- https://www.customvision.ai/

**Top 10 Companies developing AI Edge Solutions**:

- NVIDIA => Jetson
- Google => Coral, ML Kit
- Apple => CoreML:
    - coreML: https://developer.apple.com/videos/play/wwdc2024/10161/
    - Apple Intelligence: https://datasciencedojo.com/blog/apple-intelligence/
- Amazon =>
    - AWS edge
    - AWS iot:
        - Bittle Robot Dog AWS IoT Showcase at Latency Conference 2023 - Perth, Australia: https://www.petoi.com/blogs/blog/bittle-robot-dog-aws-iot-showcase-at-latency-conference-2023-perth-australia?srsltid=AfmBOoo4AElmruyMlmWIIarJtCym2Aglg8kUcb8AI37-wMvET3GsxyD2
    - Inferentia
- Infineon => Imagimob AI
- Edge Impulse =>
- MS => Azure => AI Edge => ONNX
- tinyML =>
- Facebooks => Meta => AI Edge => PyTorch Mobile
- alibaba => AI Edge => X-brain
- Baidu => EdgeBoard/DeepBench
- IBM => Edge platform

**SW Tools:**
- OpenVINO
- TensorFlow Lite
- PyTorch
- ONNX
- Apache TVM (compiler)
- NVIDIA TensorRT
- Edge Impulse/tinyML

**HW tools:**
- Raspberry Pi 4 Model B
- NVIDIA Jetson Nano
- Arduino
- ESP32
- Google Coral
- Intel Neural Compute Stick
- NVIDIA Jetson Nano
- AURIX

edges companies:
https://stlpartners.com/articles/edge-computing/edge-computing-companies-2024/

On device AI/Embedded AI/Edge AI:

- https://www.n-ix.com/on-device-ai/
- https://www.forbes.com/consent/ketch/?toURL=https://www.forbes.com/councils/forbestechcouncil/2024/04/17/on-device-generative-ai-unlocks-true-smartphone-and-pc-value/

Why on-device AI Is the future of consumer and enterprise applications:
https://www.computerweekly.com/opinion/Why-On-Device-AI-Is-the-future-of-consumer-and-enterprise-applications

26 Innovative Edge AI Companies You Should Know in 2024:
https://www.omdena.com/blog/top-edge-ai-companies

