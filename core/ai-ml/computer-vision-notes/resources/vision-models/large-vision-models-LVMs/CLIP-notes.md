# CLIP (Contrastive Language–Image Pre-training) - Notes

## Table of Contents (ToC)

  - [Introduction](#introduction)
    - [What's CLIP?](#whats-clip)
    - [Key Concepts and Terminology](#key-concepts-and-terminology)
    - [Applications](#applications)
  - [Fundamentals](#fundamentals)
    - [CLIP Architecture Pipeline](#clip-architecture-pipeline)
    - [How CLIP Works?](#how-clip-works)
    - [Some Hands-on Examples](#some-hands-on-examples)
  - [Tools \& Frameworks](#tools--frameworks)
  - [Hello World!](#hello-world)
  - [Lab: Zero to Hero Projects](#lab-zero-to-hero-projects)
  - [References](#references)

## Introduction
`CLIP (Contrastive Language–Image Pre-training)` is a model designed to connect vision and language by learning visual concepts from natural language supervision.

### What's CLIP?
- Developed by [OpenAI](https://openai.com), introduced in 2021.
- Combines `vision and language models` to perform various tasks.
- Uses natural language prompts for `zero-shot learning`.

### Key Concepts and Terminology
- **Contrastive Learning**: Technique to learn representations by comparing positive and negative pairs.
- **Zero-Shot Learning**: Ability to generalize to unseen tasks using natural language descriptions.
- **Pre-training**: Training on large-scale datasets to learn general features before fine-tuning on specific tasks.

### Applications
- Zero-shot image classification.
- Text-to-image and image-to-text retrieval.
- Image generation guidance.

## Fundamentals

### CLIP Architecture Pipeline
- Vision Transformer (ViT) or ResNet as the visual encoder.
- Text Transformer as the language encoder.
- Contrastive loss aligns image and text representations.

### How CLIP Works?
- Pre-trains on 400 million (image, text) pairs.
- Learns to match images with corresponding text descriptions.
- Enables zero-shot classification by comparing image features with text prompts.

### Some Hands-on Examples
- Zero-shot classification with natural language prompts.
- Image-text retrieval using CLIP embeddings.
- Fine-tuning for specific downstream tasks.

## Tools & Frameworks
- Hugging Face Transformers library.
- OpenAI's CLIP GitHub repository.
- PyTorch and TensorFlow for custom implementations.

## Hello World!

```python
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image 
probs = logits_per_image.softmax(dim=1) 

print(probs) 
```

## Lab: Zero to Hero Projects
- Implementing zero-shot classification with CLIP.
- Creating a multimodal search engine using CLIP embeddings.
- Fine-tuning CLIP on custom datasets.

## References

Paper:
- [Learning Transferable Visual Models From Natural Language Supervision - Radford, Alec, et al. (2021)](https://arxiv.org/pdf/2103.00020) 
- [Model Card: CLIP](https://github.com/openai/CLIP/blob/main/model-card.md)

Docs: 

- [CLIP Documentation - Hugging Face](https://huggingface.co/docs/transformers/model_doc/clip)

Articles & Release:
- [CLIP: Connecting text and images - Jan 2021](https://openai.com/index/clip/)


