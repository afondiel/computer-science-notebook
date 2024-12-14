# Large Vision Models (LVMs) - Notes

## Table of Contents (ToC)
  - [Introduction](#introduction)
    - [What's Large Vision Models (LVMs)?](#whats-large-vision-models-lvms)
    - [Key Concepts and Terminology](#key-concepts-and-terminology)
    - [Applications](#applications)
  - [Fundamentals](#fundamentals)
    - [LVM Architecture Pipeline](#lvm-architecture-pipeline)
    - [How LVMs Work?](#how-lvms-work)
    - [Types of LVMs](#types-of-lvms)
    - [Some Hands-on Examples](#some-hands-on-examples)
  - [Tools \& Frameworks](#tools--frameworks)
  - [Hello World!](#hello-world)
  - [Lab: Zero to Hero Projects](#lab-zero-to-hero-projects)
  - [References](#references)

## Introduction

`Large Vision Models (LVMs)` are deep learning models designed to handle large-scale visual tasks by leveraging extensive data and advanced architectures.

### What's Large Vision Models (LVMs)?
- High-capacity models trained on vast visual datasets.
- Designed to perform complex vision tasks with high accuracy.
- Examples include Vision Transformers (ViTs) and large-scale Convolutional Neural Networks (CNNs).

### Key Concepts and Terminology
- **Scale**: Refers to the massive amount of data and parameters used in LVMs.
- **Vision Transformer (ViT)**: A model that uses transformer architecture for image tasks.
- **Pre-training and Fine-tuning**: LVMs are often pre-trained on large datasets and then fine-tuned for specific tasks.

### Applications
- Image classification, object detection, and segmentation.
- Autonomous driving, medical imaging, and surveillance.
- Generative tasks like image synthesis and style transfer.

## Fundamentals

### LVM Architecture Pipeline
- Data Collection and Preprocessing at scale.
- Core model architecture: Vision Transformers, CNNs, or hybrid models.
- Training pipeline: Pre-training on large datasets followed by fine-tuning.

### How LVMs Work?
- Utilize massive datasets (e.g., ImageNet, JFT-300M) for pre-training.
- Leverage attention mechanisms (in ViTs) or deep convolutional layers (in CNNs).
- Transfer learning is applied to adapt the model for specific downstream tasks.

### Types of LVMs

- **Vision Transformers (ViTs):**
  - Uses self-attention mechanisms to process images.
  - Scales well with larger datasets and model sizes.
  - Excels in image classification tasks.

- **Convolutional Neural Networks (CNNs):**
  - Deep hierarchical models that use convolutional layers to extract features.
  - Popular for tasks like image classification, detection, and segmentation.
  - Examples include ResNet, Inception, and EfficientNet.

- **Hybrid Models:**
  - Combines CNNs with transformers to leverage the strengths of both architectures.
  - Example: Swin Transformer, which integrates CNN-like patch processing with attention mechanisms.
  - Useful for tasks requiring both fine-grained and contextual understanding.

### Some Hands-on Examples
- Image classification using a pre-trained Vision Transformer.
- Fine-tuning a large CNN on a custom dataset.
- Applying LVMs for object detection in autonomous driving datasets.

## Tools & Frameworks
- TensorFlow and PyTorch for implementing LVMs.
- Hugging Face Transformers for Vision Transformers.
- Google’s TPU and NVIDIA’s GPU for large-scale training.

## Hello World!

```python
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import requests

model_name = "google/vit-base-patch16-224"
model = ViTForImageClassification.from_pretrained(model_name)
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = feature_extractor(images=image, return_tensors="pt")

outputs = model(**inputs)
logits = outputs.logits

# Print the predicted class
predicted_class = logits.argmax(-1).item()
print("Predicted class:", predicted_class)
```

## Lab: Zero to Hero Projects
- Training a Vision Transformer from scratch on a custom dataset.
- Implementing object detection using a large-scale CNN model.
- Exploring generative capabilities using LVMs for style transfer.

## References

- [Hugging Face Vision Documentation](https://huggingface.co/docs/transformers/model_doc/vit)
- [A New Era of Large Vision Models (LVMs) after the LLMs epoch: approach, examples, use cases - @springs_apps](https://medium.com/@springs_apps/a-new-era-of-large-vision-models-lvms-after-the-llms-epoch-approach-examples-use-cases-7c41f1aaf5cd)
- Dosovitskiy, Alexey, et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." (2020).
- He, Kaiming, et al. "Deep Residual Learning for Image Recognition." (2016).
