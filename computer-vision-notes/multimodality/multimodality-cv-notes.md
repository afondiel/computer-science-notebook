# Deep Multimodal Learning for Computer Vision - Notes

## Table of Contents (ToC)

- [Introduction](#introduction)
  - [What's Deep Multimodal Learning?](#whats-deep-multimodal-learning)
  - [Key Concepts and Terminology](#key-concepts-and-terminology)
  - [Applications](#applications)
- [Fundamentals](#fundamentals)
  - [Deep Multimodal Learning Architecture Pipeline](#deep-multimodal-learning-architecture-pipeline)
  - [How Deep Multimodal Learning works?](#how-deep-multimodal-learning-works)
  - [Types of Multimodal Learning](#types-of-multimodal-learning)
  - [Some hands-on examples](#some-hands-on-examples)
- [Tools & Frameworks](#tools--frameworks)
- [Hello World!](#hello-world)
- [Lab: Zero to Hero Projects](#lab-zero-to-hero-projects)
- [References](#references)

## Introduction
Deep multimodal learning integrates information from multiple data modalities, such as images, text, audio, and video, to enhance machine perception and decision-making.

### What's Deep Multimodal Learning?
- A deep learning approach that processes and fuses multiple input types (e.g., image + text).
- Enhances the model’s understanding by utilizing complementary information from different sources.
- Widely used in tasks requiring a combination of vision, language, and audio data.

### Key Concepts and Terminology
- **Modality**: Different types of data sources (e.g., image, text, audio).
- **Feature Fusion**: Combining features from multiple modalities to form a unified representation.
- **Cross-modal Learning**: Learning shared representations across different modalities.
- **Multimodal Embeddings**: Representations that combine multiple modalities into a single feature space.
- **Alignment**: Ensuring consistency between modalities (e.g., aligning text with corresponding image regions).

### Applications
- Image captioning: Generating textual descriptions from images.
- Visual question answering (VQA): Answering questions about visual content using text and image data.
- Speech-to-vision translation: Matching spoken language with visual content.
- Multimodal sentiment analysis: Combining visual, textual, and audio cues to detect sentiment.
- Medical imaging: Fusing modalities like MRI, CT scans, and textual reports for better diagnosis.

## Fundamentals

### Deep Multimodal Learning Architecture Pipeline
- **Data Input**: Multiple modalities are fed into the model (e.g., images, text, or audio).
- **Feature Extraction**: Each modality goes through a feature extractor, such as CNNs for images and RNNs/transformers for text.
- **Fusion Layer**: Combines extracted features from different modalities (early, middle, or late fusion).
- **Multimodal Learning**: Trains the model to predict based on the combined information.
- **Output**: A decision or prediction is made based on the fused multimodal data.

### How Deep Multimodal Learning works?
- Individual feature extractors process each modality independently.
- Feature fusion layers integrate features from different modalities.
- Cross-modal interactions help the model learn associations between data types.
- The fused representation is used for downstream tasks like classification or generation.

### Types of Multimodal Learning
- **Early Fusion**: Combines raw data from multiple modalities before learning representations.
- **Late Fusion**: Processes each modality separately and combines their representations at the decision-making stage.
- **Intermediate Fusion**: Combines modality-specific features after partial processing in intermediate layers.
- **Cross-modal Learning**: Models that learn shared latent representations for different modalities.

### Some hands-on examples
- Building a multimodal image captioning system with image and text.
- Implementing visual question answering using pre-trained vision and language models.
- Creating a deep learning model that uses video and audio data for action recognition.
- Developing a speech-to-image retrieval system that finds images based on spoken queries.

## Tools & Frameworks
- **Transformers (Hugging Face)**: For multimodal models like Vision-Language Transformers (VLTs).
- **OpenAI CLIP**: Combines vision and language embeddings for cross-modal retrieval tasks.
- **Deep Multimodal Alignment (DMA)**: For aligning image and text data for multimodal applications.
- **VL-BERT**: A vision-and-language BERT model for tasks like image captioning and VQA.
- **PyTorch**: General-purpose deep learning library with support for multimodal architectures.
- **TensorFlow**: Another deep learning framework with modules for multimodal learning.

## Hello World!
```python
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image

# Load the pre-trained CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

# Example image and text
image = Image.open("sample_image.jpg")
text = ["a photo of a dog", "a photo of a cat"]

# Preprocess image and text
inputs = processor(text=text, images=image, return_tensors="pt", padding=True)

# Compute the similarity scores between image and text
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)

print("Similarity scores:", probs)
```

## Lab: Zero to Hero Projects
- Build a multimodal image-captioning model using OpenAI CLIP and Hugging Face transformers.
- Develop a visual question answering (VQA) system using VL-BERT or CLIP.
- Create a multimodal sentiment analysis tool using images, text, and audio data.
- Implement a speech-to-vision retrieval system using multimodal embeddings.
- Develop a medical imaging fusion system that combines MRI and CT scans with clinical notes.

## References
- Baltrusaitis, T., Ahuja, C., & Morency, L. P. (2019). *Multimodal Machine Learning: A Survey and Taxonomy*. IEEE Transactions on Pattern Analysis and Machine Intelligence.
- Radford, A., et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision*. arXiv preprint arXiv:2103.00020.
- OpenAI CLIP Documentation: https://github.com/openai/CLIP
- Hugging Face Transformers Documentation: https://huggingface.co/transformers/
- PyTorch Documentation: https://pytorch.org/
