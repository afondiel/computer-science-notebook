# Torchvision - Notes

## Table of Contents (ToC)

  - [Introduction](#introduction)
    - [What's Torchvision?](#whats-torchvision)
    - [Key Concepts and Terminology](#key-concepts-and-terminology)
    - [Applications](#applications)
  - [Fundamentals](#fundamentals)
    - [Torchvision Architecture Pipeline](#torchvision-architecture-pipeline)
    - [How Torchvision Works?](#how-torchvision-works)
    - [Torchvision Techniques](#torchvision-techniques)
    - [Some Hands-on Examples](#some-hands-on-examples)
  - [Tools \& Frameworks](#tools--frameworks)
  - [Hello World!](#hello-world)
  - [Lab: Zero to Hero Projects](#lab-zero-to-hero-projects)
  - [References](#references)


## Introduction
Torchvision is a library that provides tools and utilities for computer vision tasks in PyTorch.

### What's Torchvision?
- A PyTorch library for computer vision.
- Includes datasets, model architectures, and image transformations.
- Facilitates building and training deep learning models for image data.

### Key Concepts and Terminology
- **Dataset**: Collections of images for training and evaluation.
- **Transforms**: Functions for preprocessing and augmenting images.
- **Models**: Pretrained architectures for image classification, detection, and segmentation.
- **DataLoader**: Utility for loading datasets efficiently.

### Applications
- Image classification with pretrained models.
- Object detection and segmentation.
- Data augmentation and preprocessing.
- Building custom computer vision pipelines.

## Fundamentals

### Torchvision Architecture Pipeline
- Loading and preprocessing datasets.
- Applying image transformations and augmentations.
- Selecting and initializing pretrained models.
- Training and evaluating models on datasets.

### How Torchvision Works?
- Utilizing built-in datasets and DataLoader for data handling.
- Applying transformations using the `torchvision.transforms` module.
- Fine-tuning pretrained models available in `torchvision.models`.
- Customizing models and training loops with PyTorch.

### Torchvision Techniques
- **Data Augmentation**: Enhancing dataset variety with transformations like flips, rotations, and color jitter.
- **Transfer Learning**: Fine-tuning pretrained models for specific tasks.
- **Feature Extraction**: Using pretrained models to extract features from images.
- **Model Zoo**: Accessing a variety of pretrained models for different tasks.

### Some Hands-on Examples
- **Image Classification**: Using a pretrained ResNet model.
- **Object Detection**: Implementing a Faster R-CNN model.
- **Image Augmentation**: Applying transformations to a dataset.
- **Feature Extraction**: Extracting features with a pretrained network.

## Tools & Frameworks
- PyTorch
- Torchvision Datasets
- Torchvision Transforms
- Torchvision Models

## Hello World!
```python
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

# Load and preprocess the image
image_path = "path_to_image.jpg"
input_image = Image.open(image_path)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

# Load the pretrained model
model = models.resnet18(pretrained=True)
model.eval()

# Perform inference
with torch.no_grad():
    output = model(input_batch)
print(output.argmax().item())
```

## Lab: Zero to Hero Projects
- **Project 1**: Image classification with custom dataset and pretrained models.
- **Project 2**: Object detection in real-time with a webcam feed.
- **Project 3**: Building an image segmentation pipeline.
- **Project 4**: Developing a data augmentation tool for image datasets.

## References
- [PyTorch documentation](https://pytorch.org/docs/stable/index.html)
- [Torchvision documentation](https://pytorch.org/vision/stable/index.)
- [Pretrained models](https://pytorch.org/vision/stable/models.html) 
- [Tutorials and examples](https://pytorch.org/tutorials/) 