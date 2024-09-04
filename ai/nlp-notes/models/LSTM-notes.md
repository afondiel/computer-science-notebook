# Long Short-Term Memory (LSTM) - Notes

## Table of Contents (ToC)
  - [Introduction](#introduction)
    - [What's a Long Short-Term Memory (LSTM)?](#whats-a-long-short-term-memory-lstm)
    - [Key Concepts and Terminology](#key-concepts-and-terminology)
    - [Applications](#applications)
  - [Fundamentals](#fundamentals)
    - [LSTM Architecture Pipeline](#lstm-architecture-pipeline)
    - [How LSTMs Work](#how-lstms-work)
    - [Comparison to GRU](#comparison-to-gru)
    - [Some Hands-On Examples](#some-hands-on-examples)
  - [Tools \& Frameworks](#tools--frameworks)
  - [Hello World!](#hello-world)
  - [Lab: Zero to Hero Projects](#lab-zero-to-hero-projects)
  - [References](#references)

## Introduction
Long Short-Term Memory (LSTM) is a type of recurrent neural network (RNN) designed to effectively learn long-term dependencies in sequential data.

### What's a Long Short-Term Memory (LSTM)?
- A specialized RNN architecture that can capture long-range dependencies in sequential data.
- Uses gating mechanisms to control the flow of information, mitigating issues like the vanishing gradient problem.
- Widely used in tasks involving time series, natural language processing, and sequence prediction.

### Key Concepts and Terminology
- **Cell State**: The memory of the network, which carries long-term information through sequences.
- **Forget Gate**: Determines which information in the cell state should be discarded.
- **Input Gate**: Controls which new information is added to the cell state.
- **Output Gate**: Decides what information from the cell state should be output to the next time step.
- **Vanishing Gradient Problem**: A challenge in training RNNs where gradients diminish over time, leading to difficulty in learning long-term dependencies.

### Applications
- **Text Generation**: Creating coherent sequences of text by predicting the next word or character.
- **Speech Recognition**: Converting spoken language into text by understanding sequences of sounds.
- **Time Series Prediction**: Forecasting future values based on historical data.
- **Machine Translation**: Translating text from one language to another while maintaining the correct word order.

## Fundamentals

### LSTM Architecture Pipeline
- **Input Layer**: Accepts sequential data (e.g., sequences of words, time series data).
- **LSTM Cell**: Contains input, forget, and output gates to manage the flow of information through the cell state.
- **Recurrent Layer**: Processes sequences and updates the cell state and hidden state at each time step.
- **Output Layer**: Produces the final predictions based on the processed sequences.

### How LSTMs Work
- **Forget Gate**: Decides which information in the cell state should be forgotten or retained.
- **Input Gate**: Determines which new information should be added to the cell state based on the current input and previous hidden state.
- **Cell State Update**: The cell state is updated by combining the retained information (from the forget gate) and the new information (from the input gate).
- **Output Gate**: Controls what part of the cell state is output as the hidden state for the next time step, influencing subsequent predictions.
- **Long-Term Dependency Handling**: LSTMs excel at capturing long-term dependencies due to their ability to maintain and update the cell state effectively.

### Comparison to GRU
- **Complexity**: LSTMs are more complex than GRUs, featuring three gates (input, forget, and output) compared to GRU's two (reset and update).
- **Memory Management**: LSTMs manage long-term dependencies more explicitly with a dedicated cell state, while GRUs combine states for simplicity.
- **Performance**: Both LSTMs and GRUs perform well in practice, but LSTMs may have an edge in tasks requiring fine-grained memory control.
- **Training Speed**: GRUs are generally faster to train due to their simpler architecture, though LSTMs are more flexible for certain applications.

### Some Hands-On Examples
- **Text Generation**: Using LSTMs to generate text, one word or character at a time.
- **Stock Price Prediction**: Forecasting future stock prices based on historical data using LSTMs.
- **Language Translation**: Implementing an LSTM-based model for translating text between languages.
- **Sentiment Analysis**: Building an LSTM model to classify the sentiment of a given text.

## Tools & Frameworks
- **TensorFlow**: Offers extensive support for LSTM layers through its Keras API.
- **Keras**: Simplifies the implementation of LSTM networks with high-level abstractions.
- **PyTorch**: Provides dynamic and flexible LSTM implementations for advanced users.
- **Theano**: A research-oriented library that also supports LSTM networks, though it's less commonly used today.

## Hello World!
```python
import tensorflow as tf
from tensorflow.keras import layers

# Build a simple LSTM model
model = tf.keras.Sequential([
    layers.LSTM(50, input_shape=(None, 1), activation='tanh'),
    layers.Dense(1)
])

# Example model summary
model.summary()
```

## Lab: Zero to Hero Projects
- **Text Generator**: Build an LSTM model to generate text one character at a time.
- **Stock Price Predictor**: Develop an LSTM-based model to predict future stock prices.
- **Language Translator**: Create an LSTM model for translating text from one language to another.
- **Sentiment Analyzer**: Implement an LSTM model to classify the sentiment of reviews or social media posts.

## References
- Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory."
- Graves, A., & Schmidhuber, J. (2005). "Framewise phoneme classification with bidirectional LSTM and other neural network architectures."
- Gers, F. A., Schmidhuber, J., & Cummins, F. (2000). "Learning to Forget: Continual Prediction with LSTM."
- Olah, C. (2015). "Understanding LSTM Networks."
