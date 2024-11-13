# Model Compression - Notes

## Table of Contents
- [Introduction](#introduction)
- [Key Concepts](#key-concepts)
- [Applications](#applications)
- [Model Compression Techniques](#model-compression-techniques)
- [Compression Process & Pipelines](#model-compression-process--pipelines)
- [Key Models and Frameworks](#key-models-and-frameworks)
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
- `Model compression` is a set of techniques used to reduce the size and computational demands of machine learning models without significantly compromising performance, enabling deployment on resource-constrained devices like smartphones, IoT, or embedded systems.

### Key Concepts
- **Quantization**: Reducing the precision of numbers in the model (e.g., 32-bit floating points to 8-bit integers).
- **Pruning**: Removing less significant weights or neurons that contribute minimally to the model's output.
- **Knowledge Distillation**: Training a smaller model (student) to mimic the outputs of a larger model (teacher).
- **Neural Architecture Search (NAS)**: Optimizing the architecture to balance size and performance.
- **Common Misconception**: Smaller models necessarily lose accuracy—effective compression can retain most of a model’s original accuracy.

### Applications
- **Mobile Applications**: Compressing models for on-device AI, like image recognition or speech processing.
- **IoT & Edge Computing**: Enabling real-time inference on constrained devices for tasks such as anomaly detection, security, and monitoring.
- **Autonomous Vehicles**: Reducing computational loads in embedded systems for object detection or navigation.
- **Healthcare**: Deploying models for diagnostics on portable medical devices where memory and power are limited.
- **Environmental Monitoring**: Low-power ML applications on drones or sensors for real-time analysis in remote areas.

## Model Compression Techniques
- **Quantization**: Converts high-precision weights and activations to lower precision, reducing model size and computation.
- **Pruning**: Eliminates redundant parameters, often by removing nodes or filters with minimal effect.
- **Knowledge Distillation**: Trains a small model to mimic the "knowledge" or outputs of a larger model.
- **Weight Sharing**: Limits the diversity of weights in the model by sharing parameters among similar layers.
- **Neural Architecture Search (NAS)**: Optimizes model structure specifically for reduced size or processing requirements.

### Description
- **Quantization**:
   - Example: Transforming weights from 32-bit floats to 8-bit integers can reduce model size by 75%.
- **Pruning**:
   - Step-by-step: (1) Identify less important parameters, (2) remove or zero them out, (3) retrain model to recover lost accuracy.
- **Distillation**:
   - Teacher-student training: A smaller “student” network learns from the predictions of a larger, pre-trained “teacher” network.

## Model Compression Process & Pipelines
1. **Select a Model**: Choose an initial model architecture.
2. **Quantize or Prune**: Apply quantization or pruning to reduce model complexity.
3. **Distill Knowledge**: If applicable, train a compact model to emulate a larger one’s behavior.
4. **Optimize**: Fine-tune to regain any lost accuracy.
5. **Validate & Deploy**: Test the model on real-world data to ensure it meets requirements.

### Example Pipeline

```mermaid
graph LR;
    Start(Original Model) --> Quantize[Quantization] --> Prune[Pruning] --> Distill[Knowledge Distillation]
    Distill --> FineTune(Fine-Tuning) --> Validate[Validate on Dataset] --> Deploy[Deploy Model]
```

## Key Models and Frameworks
- **MobileNet**: Designed specifically for mobile and edge applications with minimal parameters.
- **EfficientNet**: Uses compound scaling to balance width, depth, and resolution.
- **YOLO-Nano**: Lightweight version of YOLO for real-time object detection on limited hardware.
- **TensorFlow Lite** and **ONNX Runtime**: Frameworks optimized for deploying compressed models on edge devices.
- **TinyML Models**: Specialized models for ultra-low-power applications, such as in microcontrollers.

## Self-Practice / Hands-On Examples
1. **Quantization Exercise**: Apply quantization to a neural network model in TensorFlow Lite and observe the impact on size and accuracy.
2. **Pruning Experiment**: Use PyTorch to prune a model layer-by-layer and monitor changes in performance.
3. **Knowledge Distillation**: Implement a teacher-student model and compare the student’s performance against the original large model.
4. **NAS with MobileNet**: Use NAS tools to create a smaller variant of MobileNet tailored for a specific edge device.
5. **Benchmarking**: Compare the inference times of compressed models versus the original on an edge device like Raspberry Pi.

## Pitfalls & Challenges
- **Loss of Accuracy**: Over-compression can lead to significant drops in model performance.
- **Inference Latency**: Compressed models may still suffer from high latency depending on the device.
- **Compatibility Issues**: Different devices and frameworks may not support all compression techniques.
- **Training Complexity**: Techniques like knowledge distillation require additional training stages.

## Feedback & Evaluation
- **Feynman Test**: Explain model compression techniques as if teaching someone with no AI experience.
- **Real-World Simulation**: Deploy on a constrained device and measure performance metrics.
- **Benchmark Testing**: Use model accuracy, latency, and memory usage to evaluate effectiveness.

## Tools, Libraries & Frameworks
1. **TensorFlow Lite**: Great for quantization and deployment on mobile/edge devices.
2. **ONNX Runtime**: Supports model optimization for edge environments.
3. **PyTorch Quantization Toolkit**: Offers quantization and pruning utilities.
4. **Distiller by Neural Network Compression Framework**: Open-source tool for pruning and quantization in PyTorch.
5. **Apache TVM**: Compiles models for optimized edge inference across diverse hardware.

## Hello World! (Practical Example)
- **Quantization Example** in TensorFlow Lite:
  ```python
  import tensorflow as tf

  # Convert model to TFLite format with quantization
  converter = tf.lite.TFLiteConverter.from_saved_model("model_path")
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  tflite_model = converter.convert()

  # Save and deploy tflite_model
  with open("compressed_model.tflite", "wb") as f:
      f.write(tflite_model)
  ```

## Advanced Exploration
- **Model Compression Survey**: Dive into recent research papers on model compression trends.
- **Distillation Techniques for Vision Models**: Advanced articles on distilling complex vision models.
- **Quantization Aware Training**: Learn techniques that incorporate quantization during training to retain accuracy.

## Zero to Hero Lab Projects
- **Project 1**: Build and compress a model for real-time object detection on Raspberry Pi.
- **Project 2**: Develop a distillation pipeline for a large language model using PyTorch.
- **Project 3**: Use pruning and quantization to optimize a healthcare diagnostic model for a portable device.

## Continuous Learning Strategy
- **Next Steps**: Explore hardware-specific optimizations for models (like NVIDIA TensorRT).
- **Related Topics**: Dive into Edge AI, TinyML, and Embedded Systems for further specialization.

## References
- *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks* by Mingxing Tan and Quoc V. Le
- *Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding* by Song Han et al.
- *Knowledge Distillation* by Geoffrey Hinton, Oriol Vinyals, and Jeff Dean
