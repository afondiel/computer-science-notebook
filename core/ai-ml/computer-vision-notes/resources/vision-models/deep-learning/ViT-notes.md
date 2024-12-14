# Vision Transformer (ViT) - Notes

## Table of Contents (ToC)
  - [Introduction](#introduction)
    - [What's ViT?](#whats-vit)
    - [Key Concepts and Terminology](#key-concepts-and-terminology)
    - [Applications](#applications)
  - [Fundamentals](#fundamentals)
    - [ViT Architecture Pipeline](#vit-architecture-pipeline)
    - [How ViT works?](#how-vit-works)
    - [Some hands-on examples](#some-hands-on-examples)
  - [Tools \& Frameworks](#tools--frameworks)
  - [Hello World!](#hello-world)
  - [Lab: Zero to Hero Projects](#lab-zero-to-hero-projects)
  - [References](#references)


## Introduction
ViT, or Vision Transformer, is a deep learning model that applies Transformer architecture to image classification tasks.

### What's ViT?
- Vision Transformer (ViT) adapts the Transformer model, originally designed for natural language processing, to the domain of computer vision.
- It divides images into patches, treats them as sequences, and processes these sequences using Transformer encoders.

### Key Concepts and Terminology
- **Patch Embeddings:** Splitting an image into fixed-size patches and embedding them into a vector space.
- **Transformer Encoder:** The core component that processes the sequence of image patches.
- **Positional Encoding:** Adding position information to patches since the Transformer lacks intrinsic knowledge of order.
- **Self-Attention Mechanism:** A key feature of Transformers that allows the model to focus on different parts of the image.

### Applications
- Image classification in various domains such as medical imaging, autonomous driving, and security.
- Fine-grained image recognition tasks.
- Transfer learning for specialized vision tasks.

## Fundamentals

### ViT Architecture Pipeline
- Input image is divided into fixed-size patches.
- Patches are linearly embedded and combined with positional encodings.
- Embedded patches are processed through a stack of Transformer encoders.
- The output of the Transformer is fed to a classification head for prediction.

### How ViT works?
- **Patch Splitting:** Divide the image into non-overlapping patches.
- **Linear Embedding:** Flatten each patch and linearly project it into a lower-dimensional space.
- **Positional Encoding:** Add positional information to each patch embedding.
- **Transformer Encoding:** Process the sequence of patches through multiple layers of Transformer encoders.
- **Classification Head:** Use the final output of the Transformer for classification tasks.

### Some hands-on examples
- Image classification on the CIFAR-10 dataset.
- Transfer learning using pre-trained ViT on ImageNet.
- Fine-tuning ViT for custom datasets.

## Tools & Frameworks
- **TensorFlow:** Implementations of ViT using TensorFlow.
- **PyTorch:** PyTorch libraries and models for ViT.
- **Hugging Face Transformers:** Pre-trained ViT models and fine-tuning tools.
- **JAX/Flax:** JAX and Flax implementations for high-performance training.

## Hello World!
```python
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import requests

# Load the model and feature extractor
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

# Load an image from the web
url = 'https://example.com/path/to/your/image.jpg'
image = Image.open(requests.get(url, stream=True).raw)

# Preprocess the image
inputs = feature_extractor(images=image, return_tensors="pt")

# Perform inference
outputs = model(**inputs)
logits = outputs.logits

# Get the predicted class
predicted_class = logits.argmax(-1).item()
print("Predicted class:", predicted_class)
```

## Lab: Zero to Hero Projects
- **Basic Image Classification:** Train a ViT model from scratch on the MNIST dataset.
- **Transfer Learning Project:** Fine-tune a pre-trained ViT model on a medical imaging dataset.
- **Custom Dataset Project:** Use ViT for image classification on a custom dataset, including data preprocessing, training, and evaluation.

## References
- ["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." - Dosovitskiy, A., et al. (2020).](https://arxiv.org/pdf/2010.11929)
- [Official ViT implementation in TensorFlow and PyTorch](https://github.com/google-research/vision_transformer)
- [Hugging Face Transformers documentation](https://huggingface.co/transformers/model_doc/vit.html)


