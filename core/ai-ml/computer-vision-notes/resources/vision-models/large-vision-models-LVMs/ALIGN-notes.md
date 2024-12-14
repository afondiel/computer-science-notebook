# ALIGN (A Large-scale ImaGe and Noisy-text embedding) - Notes

## Table of Contents (ToC)
  - [Introduction](#introduction)
    - [What's ALIGN?](#whats-align)
    - [Key Concepts and Terminology](#key-concepts-and-terminology)
    - [Applications](#applications)
  - [Fundamentals](#fundamentals)
    - [ALIGN Architecture Pipeline](#align-architecture-pipeline)
    - [How ALIGN Works?](#how-align-works)
    - [Some Hands-on Examples](#some-hands-on-examples)
  - [Tools \& Frameworks](#tools--frameworks)
  - [Hello World!](#hello-world)
  - [Lab: Zero to Hero Projects](#lab-zero-to-hero-projects)
  - [References](#references)

## Introduction
`ALIGN (A Large-scale ImaGe and Noisy-text embedding)` is a model developed by Google Research to align image and text representations on a massive scale.

### What's ALIGN?
- Developed by Google Research, introduced in 2021.
- Trains on billions of noisy image-text pairs.
- Focuses on scaling up vision-language models.

### Key Concepts and Terminology
- **Noisy Text Data**: Large-scale text data that may include noise but is used for robust training.
- **Scaling**: Training models on vast datasets for improved performance.
- **Image-Text Embedding**: Joint representation of images and text in the same embedding space.

### Applications
- Image-text retrieval and search.
- Zero-shot classification.
- Transfer learning for vision-language tasks.

## Fundamentals

### ALIGN Architecture Pipeline
- EfficientNet as the visual encoder.
- BERT as the text encoder.
- Contrastive loss to align image and text embeddings.

### How ALIGN Works?
- Pre-trains on over a billion noisy image-text pairs.
- Learns to align embeddings by contrasting positive pairs against random negatives.
- Achieves state-of-the-art results in various benchmarks, including zero-shot classification.

### Some Hands-on Examples
- Zero-shot classification using ALIGN embeddings.
- Image and text retrieval tasks.
- Fine-tuning on domain-specific datasets.

## Tools & Frameworks
- TensorFlow and JAX for implementing ALIGN.
- Hugging Face Transformers for related models.
- Google’s TensorFlow Hub for pretrained models.

## Hello World!

```python
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image

model = hub.load("https://tfhub.dev/google/align/bert_efficientnet_b7/1")

def preprocess_image(image_path):
    image = Image.open(image_path).resize((224, 224))
    image = np.array(image) / 255.0
    return image

image_path = "path_to_your_image.jpg"
image = preprocess_image(image_path)
image_embeddings = model.signatures["image_embedding"](tf.convert_to_tensor([image]))

print(image_embeddings)
```

## Lab: Zero to Hero Projects
- Implementing zero-shot classification with ALIGN.
- Building a multimodal search engine using ALIGN embeddings.
- Exploring transfer learning techniques with ALIGN.

## References

Papers:
- [Jia, Chao, et al. "Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision." (2021).](https://arxiv.org/pdf/2102.05918)

Docs:
- [ALIGN Documentation - Hugging Face](https://huggingface.co/docs/transformers/model_doc/align)

Article & Release:
- [ALIGN: Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision - Google Reasearch, 2021](https://research.google/blog/align-scaling-up-visual-and-vision-language-representation-learning-with-noisy-text-supervision/)
