# Convolutional Neural Network (CNN) - Notes

## Table of Contents (ToC)

  - [Introduction](#introduction)
    - [What's a Convolutional Neural Network (CNN)?](#whats-a-convolutional-neural-network-cnn)
    - [Key Concepts and Terminology](#key-concepts-and-terminology)
    - [Applications](#applications)
  - [Fundamentals](#fundamentals)
    - [CNN Architecture Pipeline](#cnn-architecture-pipeline)
    - [How CNNs Work](#how-cnns-work)
    - [Types of CNN Architectures](#types-of-cnn-architectures)
    - [Some Hands-On Examples](#some-hands-on-examples)
  - [Tools \& Frameworks](#tools--frameworks)
  - [Hello World!](#hello-world)
  - [Lab: Zero to Hero Projects](#lab-zero-to-hero-projects)
  - [References](#references)


## Introduction
A Convolutional Neural Network (CNN) is a deep learning model particularly effective for image and spatial data processing.

### What's a Convolutional Neural Network (CNN)?
- A type of deep neural network designed to process structured grid data such as images.
- Utilizes convolutional layers to automatically detect spatial hierarchies in data.
- Primarily used in image recognition, object detection, and related fields.

### Key Concepts and Terminology
- **Convolutional Layer**: Applies filters to input data to create feature maps.
- **Pooling Layer**: Reduces the spatial dimensions of feature maps, retaining essential information.
- **ReLU (Rectified Linear Unit)**: Activation function used to introduce non-linearity.
- **Fully Connected Layer**: Connects neurons from one layer to another, typically used before the output layer.

### Applications
- **Image Classification**: Identifying objects or subjects in images.
- **Object Detection**: Locating and identifying objects within images.
- **Facial Recognition**: Matching faces in images to identities.
- **Medical Imaging**: Analyzing and diagnosing conditions from medical images (e.g., X-rays, MRIs).

## Fundamentals

### CNN Architecture Pipeline
- **Input Layer**: Accepts image data typically in 2D (height x width x channels).
- **Convolutional Layer**: Applies filters to detect features like edges, textures, and patterns.
- **Activation Function (ReLU)**: Applies non-linearity after each convolution operation.
- **Pooling Layer**: Downsamples feature maps to reduce dimensionality and computation.
- **Fully Connected Layer**: Combines features for final classification or regression tasks.
- **Output Layer**: Provides the final predictions, often using softmax for classification.

### How CNNs Work
- **Convolution Operation**: Uses filters (kernels) to slide over input data and compute dot products, generating feature maps.
- **Feature Extraction**: Early layers capture low-level features (edges), while deeper layers capture high-level features (shapes, objects).
- **Pooling**: Reduces spatial dimensions to keep computation efficient and to prevent overfitting.
- **Flattening**: Converts 2D feature maps into a 1D vector for input into fully connected layers.
- **Classification/Regression**: The final fully connected layers output predictions based on extracted features.

### Types of CNN Architectures
- **LeNet-5**: Early CNN architecture used for digit recognition.
- **AlexNet**: Introduced deeper networks with more filters and layers, won the 2012 ImageNet competition.
- **VGGNet**: Uses very small (3x3) convolution filters but with a deep network architecture.
- **ResNet (Residual Networks)**: Introduced skip connections to allow for very deep networks, solving the vanishing gradient problem.
- **Inception Network (GoogLeNet)**: Combines convolutions of different sizes in parallel to capture varying spatial features.

### Some Hands-On Examples
- **Image Classification**: Using CNNs to classify CIFAR-10 or MNIST datasets.
- **Object Detection**: Implementing YOLO or Faster R-CNN for detecting objects in images.
- **Facial Recognition**: Building a CNN model to recognize faces from a dataset.
- **Medical Image Analysis**: Applying CNNs to classify diseases in medical scans.

## Tools & Frameworks
- **TensorFlow**: Popular framework for building and training CNN models.
- **Keras**: High-level API for easy implementation of CNNs, built on top of TensorFlow.
- **PyTorch**: Widely used for flexible CNN development with dynamic computation graphs.
- **OpenCV**: Library for image processing, often used alongside CNNs for pre-processing.

## Hello World!
```python
from tensorflow.keras import layers, models

# Build a simple CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Example model summary
model.summary()
```

## Lab: Zero to Hero Projects
- **Digit Recognizer**: Build a CNN to classify handwritten digits using the MNIST dataset.
- **Dog vs Cat Classifier**: Train a CNN to distinguish between images of dogs and cats.
- **Real-Time Object Detection**: Implement a CNN-based object detection system using YOLO.
- **Medical Image Classification**: Develop a CNN to diagnose diseases from X-ray images.

## References
- LeCun, Y., et al. (1998). "Gradient-based learning applied to document recognition."
- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). "ImageNet Classification with Deep Convolutional Neural Networks."
- Simonyan, K., & Zisserman, A. (2014). "Very Deep Convolutional Networks for Large-Scale Image Recognition."
- He, K., et al. (2015). "Deep Residual Learning for Image Recognition."
- Szegedy, C., et al. (2015). "Going Deeper with Convolutions."
