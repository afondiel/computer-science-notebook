# Recurrent Neural Network (RNN) - Notes

## Table of Contents (ToC)

  - [Introduction](#introduction)
    - [What's a Recurrent Neural Network (RNN)?](#whats-a-recurrent-neural-network-rnn)
    - [Key Concepts and Terminology](#key-concepts-and-terminology)
    - [Applications](#applications)
  - [Fundamentals](#fundamentals)
    - [RNN Architecture Pipeline](#rnn-architecture-pipeline)
    - [How RNNs Work](#how-rnns-work)
    - [Types of RNN Architectures](#types-of-rnn-architectures)
    - [Some Hands-On Examples](#some-hands-on-examples)
  - [Tools \& Frameworks](#tools--frameworks)
  - [Hello World!](#hello-world)
  - [Lab: Zero to Hero Projects](#lab-zero-to-hero-projects)
  - [References](#references)

## Introduction
A Recurrent Neural Network (RNN) is a type of neural network designed for processing sequential data by maintaining a memory of previous inputs.

### What's a Recurrent Neural Network (RNN)?
- An RNN is a neural network that processes data sequentially, with each step dependent on previous ones.
- Utilizes loops within its architecture to maintain an internal state (memory).
- Effective for tasks where the order of input data matters, such as time series and language modeling.

### Key Concepts and Terminology
- **Hidden State**: The memory of the network, which is updated at each time step.
- **Sequence Data**: Data that has an order or sequence, such as text or time series.
- **Vanishing Gradient Problem**: Difficulty in training due to gradients becoming too small in long sequences.
- **Long Short-Term Memory (LSTM)**: A type of RNN designed to handle long-term dependencies and mitigate the vanishing gradient problem.
- **Gated Recurrent Unit (GRU)**: A simplified variant of LSTM that also addresses the vanishing gradient issue.

### Applications
- **Language Modeling**: Predicting the next word in a sentence or generating text.
- **Speech Recognition**: Converting spoken language into text.
- **Time Series Prediction**: Forecasting future values based on historical data.
- **Machine Translation**: Translating text from one language to another while considering word order.

## Fundamentals

### RNN Architecture Pipeline
- **Input Layer**: Accepts sequential data (e.g., sequences of words or time series data).
- **Recurrent Layer**: Processes input while maintaining a hidden state that captures past information.
- **Activation Function**: Typically uses tanh or ReLU to introduce non-linearity.
- **Output Layer**: Produces the final prediction for each time step, often with a softmax function for classification.

### How RNNs Work
- **Sequential Processing**: RNNs process data one step at a time, passing information forward through a hidden state.
- **Hidden State Update**: At each time step, the hidden state is updated based on the current input and the previous hidden state.
- **Backpropagation Through Time (BPTT)**: The training algorithm for RNNs that calculates gradients across all time steps.
- **Handling Long Sequences**: RNNs struggle with long sequences due to the vanishing gradient problem, which LSTMs and GRUs help address.

### Types of RNN Architectures
- **Vanilla RNN**: The basic RNN with a single hidden state passed through time steps.
- **LSTM (Long Short-Term Memory)**: Includes gates (input, forget, and output) to control the flow of information, allowing it to remember or forget information over long periods.
- **GRU (Gated Recurrent Unit)**: A simplified version of LSTM with fewer gates, reducing computational complexity.
- **Bidirectional RNN**: Processes the sequence in both forward and backward directions to capture information from both past and future contexts.
- **Deep RNN**: Stacks multiple RNN layers to increase model capacity and learning potential.

### Some Hands-On Examples
- **Text Generation**: Using RNNs to generate text one character or word at a time.
- **Stock Price Prediction**: Forecasting future stock prices based on historical data.
- **Language Translation**: Implementing an RNN-based translator for converting text from one language to another.
- **Sentiment Analysis**: Classifying the sentiment of text data (e.g., positive or negative reviews).

## Tools & Frameworks
- **TensorFlow**: Provides support for RNNs, LSTMs, and GRUs with high flexibility.
- **Keras**: A high-level API within TensorFlow that simplifies building and training RNN models.
- **PyTorch**: Offers dynamic computation graphs and built-in modules for RNNs, LSTMs, and GRUs.
- **Theano**: An older deep learning library that supports RNNs, used for research purposes.

## Hello World!
```python
import tensorflow as tf
from tensorflow.keras import layers

# Build a simple RNN model
model = tf.keras.Sequential([
    layers.SimpleRNN(50, input_shape=(None, 1), activation='tanh'),
    layers.Dense(1)
])

# Example model summary
model.summary()
```

## Lab: Zero to Hero Projects
- **Text Generator**: Build an RNN that generates text character by character.
- **Sentiment Classifier**: Develop an RNN-based model to classify the sentiment of movie reviews.
- **Stock Price Predictor**: Implement an RNN to forecast future stock prices using historical data.
- **Speech-to-Text System**: Create an RNN-based model that converts spoken language into text.

## References
- Elman, J. L. (1990). "Finding Structure in Time."
- Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory."
- Cho, K., et al. (2014). "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation."
- Graves, A. (2013). "Generating Sequences With Recurrent Neural Networks."
