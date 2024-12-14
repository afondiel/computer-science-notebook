## Generative Adversarial Networks (GANs) - Notes

## Table of Contents

- [Introduction](#introduction)
- [What are Generative Adversarial Networks (GANs)?](#what-are-generative-adversarial-networks-gans)
- [Key Concepts and Terminology](#key-concepts-and-terminology)
- [Applications](#applications)
- [Fundamentals](#fundamentals)
  - [GAN Architecture Pipeline](#gan-architecture-pipeline)
  - [How GANs Work](#how-gans-work)
  - [Hands-on Examples](#hands-on-examples)
- [Tools \& Frameworks](#tools--frameworks)
- [Hello World!](#hello-world)
- [Lab: Zero to Hero Projects](#lab-zero-to-hero-projects)
- [References](#references)


## Introduction

Generative Adversarial Networks (GANs) are a type of machine learning model that can generate new data.

## What are Generative Adversarial Networks (GANs)?

* GANs consist of two neural networks: a generator and a discriminator.
* The generator learns to create new data instances that resemble the training data.
* The discriminator learns to distinguish between real data and the generated data.
* Through an adversarial process, both networks improve over time.

## Key Concepts and Terminology

* Generative Model: A model that learns to create new data.
* Discriminative Model: A model that learns to classify data.
* Adversarial Training: A training process where two models compete with each other.
* Loss Function: A function that measures the error between the model's output and the desired output.

## Applications

* Generating realistic images and videos
* Creating new data for training other machine learning models
* Image editing and style transfer
* Drug discovery and material science

## Fundamentals

### GAN Architecture Pipeline

![GAN architecture pipeline](https://miro.medium.com/v2/resize:fit:1400/1*DvjKI7AyJPaPuBLV86MEpA.jpeg)
(Src: [Leo Pauly - @Medium](https://leopauly.medium.com/generative-ai-return-of-the-gans-e73b83904bee))

### How GANs Work

* The generator takes random noise as input and generates new data.
* The discriminator evaluates the generated data and determines if it is real or fake.
* The generator is trained to minimize the discriminator's ability to correctly classify the generated data as fake.
* The discriminator is trained to maximize its ability to distinguish between real and generated data.

### Hands-on Examples

* Training a GAN to generate images of cats
* Using a GAN to translate images from one style to another (e.g., photo to painting)
* Creating new music samples with a GAN

## Tools & Frameworks

* TensorFlow
* PyTorch
* Generative Adversarial Networks Library (GANlib)

## Hello World!

```python
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Flatten

# Load the MNIST dataset
(x_train, _), (_, _) = mnist.load_data()

# Rescale the data
x_train = x_train.astype('float32')
x_train = (x_train - 127.5) / 127.5

# Define the generator model
def define_generator():
  model = Sequential()
  model.add(Dense(128 * 7 * 7, activation='relu', input_dim=100))
  model.add(Reshape((7, 7, 128)))
  model.add(Dense(256 * 7 * 7, activation='relu'))
  model.add(Reshape((7, 7, 256)))
  model.add(Dense(1 * 28 * 28, activation='tanh'))
  model.add(Reshape((28, 28, 1)))
  return model

# Define the discriminator model
def define_discriminator():
  model = Sequential()
  model.add(Flatten(input_shape=(28, 28, 1)))
  model.add(Dense(128, activation='leakrelu'))
  model.add(Dense(1, activation='sigmoid'))
  return model

# Create the GAN model
generator = define_generator()
discriminator = define_discriminator()

# Combine the models
discriminator.compile(loss='binary_crossentropy', optimizer='adam')
gan_model = Sequential()
gan_model.add(generator)
gan_model.add(discriminator)
discriminator.trainable = False
gan_model.compile(loss='binary_crossentropy', optimizer='adam')

# Train the model
for epoch in range(epochs):
  # Train the discriminator
  # ...
  # Train the generator
  # ...

# Generate new images
noise = np.random.rand(batch_size, 100)
generated_images = generator.predict(noise)
```

## Lab: Zero to Hero Projects

* Train a GAN to generate images of your favorite celebrity.
* Develop a GAN-based application for image editing or style transfer.
* Explore the use of GANs for music or text generation.

## References

*