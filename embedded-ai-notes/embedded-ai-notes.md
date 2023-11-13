# Embedded-AI - Notes

## Table of Contents (ToC)

- Overview
- Applications
- Tools & Frameworks
- Hello World!
- References

## Overview

Embedded-AI is a general-purpose framework system for AI functions that is built into network devices and provides common model management, data obtaining, and data preprocessing functions for AI algorithm-based functions for these devices¹.

## Applications

Some of the applications of embedded-AI are:

- Face-ID: Using facial recognition to authorize access to machine controls on a factory floor².
- Anomaly Detection: Detecting abnormal patterns in data streams, such as network traffic, sensor readings, or machine logs³.
- Voice Control: Using natural language processing to enable voice-based interactions with devices, such as smart speakers, voice assistants, or robots.

## Tools & Frameworks

Some of the tools and frameworks that can be used to develop embedded-AI applications are:

- TensorFlow: An open-source platform for machine learning that supports various types of neural networks and algorithms.
- CEVA-XM: A family of low-power and high-performance DSP cores that are optimized for embedded-AI applications, such as vision, audio, and sensor fusion.
- Huawei EAI: A system that integrates multiple AI algorithms and provides model management, data obtaining, and data preprocessing functions for AI functions on network devices¹.

## Hello World!

Here is a simple example of how to use TensorFlow to create a neural network that can classify handwritten digits:

```python
# Import TensorFlow and Keras
import tensorflow as tf
from tensorflow import keras

# Load the MNIST dataset of handwritten digits
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the input images
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define the neural network model
model = keras.models.Sequential([
  keras.layers.Flatten(input_shape=(28, 28)), # Flatten the 28x28 images into 784-element vectors
  keras.layers.Dense(128, activation='relu'), # Add a hidden layer with 128 neurons and ReLU activation
  keras.layers.Dropout(0.2), # Add a dropout layer to prevent overfitting
  keras.layers.Dense(10, activation='softmax') # Add an output layer with 10 neurons and softmax activation
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5)

# Evaluate the model
model.evaluate(x_test, y_test, verbose=2)
```
Src code: [github.com/jraman](https://github.com/jraman/ml-tryouts/tree/862a66d9c35ed717bd620ead9672a2e024ebb55b/mnist_beginner.py) 

## References

- [What Is Embedded AI (EAI)? Why Do We Need EAI? - Huawei](https://info.support.huawei.com/info-finder/encyclopedia/en/EAI.html)
- [Mastering embedded AI - Embedded.com](https://www.embedded.com/mastering-embedded-ai/)
- [Embedded Artificial Intelligence: Intelligence on Devices | IEEE ](https://ieeexplore.ieee.org/document/10224582/)


