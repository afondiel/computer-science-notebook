# Transformer Models - Notes

## Table of Contents (ToC)

  - [Introduction](#introduction)
    - [What's the Transformer Model?](#whats-the-transformer-model)
    - [Key Concepts and Terminology](#key-concepts-and-terminology)
    - [Applications](#applications)
  - [Fundamentals](#fundamentals)
    - [Transformer Architecture Pipeline](#transformer-architecture-pipeline)
    - [How the Transformer Models Work](#how-the-transformer-models-work)
    - [Types of Transformer Models](#types-of-transformer-models)
    - [Some Hands-On Examples](#some-hands-on-examples)
  - [Tools \& Frameworks](#tools--frameworks)
  - [Hello World!](#hello-world)
  - [Lab: Zero to Hero Projects](#lab-zero-to-hero-projects)
  - [References](#references)


## Introduction
The `Transformer model` is a deep learning architecture that has revolutionized [natural language processing (NLP)](https://github.com/afondiel/computer-science-notes/blob/master/ai/nlp-notes/nlp-notes.md) and other AI fields.

### What's the Transformer Model?
- Introduced in the paper ["Attention is All You Need" (2017)](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf).
- Utilizes a self-attention mechanism to weigh the importance of different input tokens.
- Designed to handle sequential data but can process all input simultaneously.

### Key Concepts and Terminology
- **Attention Mechanism**: Focuses on relevant parts of the input data.
- **Encoder-Decoder Architecture**: Standard setup for many transformer, used for tasks like translation.
- **Self-Attention**: Enables the model to weigh the importance of each part of the input relative to every other part.
- **Positional Encoding**: Adds information about the position of tokens in the sequence.

### Applications
- **Natural Language Processing (NLP)**: Machine translation, text summarization, sentiment analysis.
- **Vision Transformers**: Image classification and object detection.
- **Speech Recognition**: Enhanced performance in automatic speech recognition tasks.
- **Generative Models**: Used in models like GPT for text generation.

## Fundamentals
### Transformer Architecture Pipeline

<img width="680" height="400" src="../docs/transformer-architecture.png">

- **Input Embedding**: Converts tokens into dense vectors.
- **Positional Encoding**: Adds sequence information to embeddings.
- **Multi-Head Attention**: Computes attention in multiple subspaces.
- **Feed-Forward Network**: Applies dense layers with activation functions.
- **Output**: Transformed data used for various tasks like classification or generation.

### How the Transformer Models Work

<img width="520" height="280" src="../docs/attention-mechanisms.png">

- **Input Representation**: Converts words/tokens into vector embeddings.
- **Self-Attention Calculation**: Measures relationships between tokens in the input.
- **Stacking Layers**: Multiple layers of self-attention and feed-forward networks.
- **Output Generation**: Final output depends on the specific task (e.g., translation, classification).

### Types of Transformer Models
- **BERT (Bidirectional Encoder Representations from Transformer)**: Pre-trained model for NLP tasks.
- **GPT (Generative Pre-trained Transformer)**: Focused on text generation.
- **T5 (Text-To-Text Transfer Transformer)**: Converts all NLP tasks into a text-to-text format.
- **Vision Transformers (ViT)**: Adaptation for image classification.

### Some Hands-On Examples
- **Text Classification**: Fine-tuning BERT for sentiment analysis.
- **Text Generation**: Using GPT for generating coherent paragraphs.
- **Image Classification**: Implementing Vision Transformers for image datasets.

## Tools & Frameworks
- **Hugging Face Transformers**: Popular library for using pre-trained transformer models.
- **TensorFlow & PyTorch**: Frameworks supporting the implementation of transformer.
- [**Transformers-Interpret**: Library for model interpretability in transformer.](https://github.com/cdpierse/transformers-interpret)
- **AllenNLP**: Toolkit for building and evaluating transformer models.

## Hello World!
```python
from transformers import pipeline

# Load pre-trained model and tokenizer
classifier = pipeline("sentiment-analysis")

# Example usage
result = classifier("Transformer models are amazing!")
print(result)
```
Output: 
```
[{'label': 'POSITIVE', 'score': 0.9998762607574463}]
```

## Lab: Zero to Hero Projects
- **Text Summarization Tool**: Build an app that summarizes articles.
- **Chatbot Using GPT**: Create an interactive chatbot.
- **Image Classification with ViT**: Train a Vision Transformer on a custom image dataset.
- **Custom Sentiment Analysis**: Fine-tune BERT on your own sentiment dataset.

## References

Documentation: 
- [HuggingFace Transformers Framework](https://huggingface.co/docs/transformers/index)

Papers:
- [Vaswani, A., et al. (2017). "Attention Is All You Need."](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
- [Devlin, J., et al. (2018). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding."](https://arxiv.org/pdf/1810.04805)
- [Radford, A., et al. (2019). "Language Models are Unsupervised Multitask Learners."](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Dosovitskiy, A., et al. (2020). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale."](https://arxiv.org/pdf/2010.11929)

