# Deep Learning (DL) -  Notes

## Table of Contents

  - [Overview](#overview)
  - [Introduction](#introduction)
    - [What's Deep Learning?](#whats-deep-learning)
    - [Key Concepts and Terminology](#key-concepts-and-terminology)
    - [Applications](#applications)
  - [Fundamentals](#fundamentals)
    - [Deep Learning Architecture Pipeline](#deep-learning-architecture-pipeline)
    - [How Deep Learning works?](#how-deep-learning-works)
    - [Some hands-on examples](#some-hands-on-examples)
  - [What's Artificial Neural Networks (Neural Nets)?](#whats-artificial-neural-networks-neural-nets)
  - [Biological Neuron Vs Artificial Neuron](#biological-neuron-vs-artificial-neuron)
  - [Applications](#applications-1)
  - [The perceptron (A Linear Unit)](#the-perceptron-a-linear-unit)
  - [Deep Neural Nets](#deep-neural-nets)
    - [Deep learning algorithms/model](#deep-learning-algorithmsmodel)
    - [Deep Learning Modeling Pipeline](#deep-learning-modeling-pipeline)
    - [Data Collection](#data-collection)
    - [Modeling](#modeling)
    - [Prediction](#prediction)
    - [Test \& Model Update (Weights and Biases)](#test--model-update-weights-and-biases)
    - [Model configuration / Parameters](#model-configuration--parameters)
    - [Model Evaluation \& fitting Metrics](#model-evaluation--fitting-metrics)
  - [Tools \& frameworks](#tools--frameworks)
  - [DL Glossary](#dl-glossary)
  - [Hello World!](#hello-world)
  - [Lab: Zero to Hero Projects](#lab-zero-to-hero-projects)
  - [References](#references)


## Overview

Deep learning is a [machine learning](../ml-notes/ml-notes.md) subset based on artificial **neural networks** that performs sophisticated computations on large amounts of data.

## Introduction
Deep learning is a subset of machine learning that utilizes neural networks with multiple layers to model and understand complex patterns in data.

### What's Deep Learning?
- A branch of machine learning focused on neural networks.
- Employs layers of algorithms to interpret data.
- Mimics human brain structure for pattern recognition.

### Key Concepts and Terminology
- **Neural Networks**: The foundational structure consisting of layers.
- **Layers**: Different levels in a neural network, such as input, hidden, and output layers.
- **Activation Functions**: Functions that determine the output of a neural network node.
- **Backpropagation**: The process used to train neural networks.

### Applications
- Image and speech recognition.
- Natural language processing.
- Autonomous vehicles.
- Healthcare diagnostics.
- Computer vision 
- Object detection 
- nlp 
- Virtual assistant 
- Face recognition 
- Chatbots
- Sound addition to silent films
- colorization of black and white images ...
 ...
- eveywhere since data is everywhere in modern world and taking part of daily life ...


## Fundamentals

### Deep Learning Architecture Pipeline
- Data Collection
- Data Preprocessing
- Model Building
- Model Training
- Model Evaluation
- Deployment

### How Deep Learning works?
- Data is fed into the input layer.
- The data is processed through hidden layers using weights and biases.
- Activation functions determine the output at each node.
- Output layer produces the final prediction or classification.
- Backpropagation adjusts weights to minimize error.

### Some hands-on examples
- Building a simple neural network for image classification.
- Implementing a speech recognition model.
- Developing a text classification system using recurrent neural networks (RNNs).


## What's Artificial Neural Networks (Neural Nets)?

Neural Nets (NN) are a stack of algorithms that simulates the way humans learn. Neural Nets are composed of artificial neurons inspired biological neuron. 
  
```mermaid
  graph LR;
    A(synapse of previous neuron)-->B(dendrites);
    B-->C((cell body));
    C-->D(axon);    
    D-->E(synapse of current neuron);
    style A fill:#fff,stroke:#f66,stroke-width:2px,stroke-dasharray: 5, 5;
```


## Biological Neuron Vs Artificial Neuron  

    (neuron)               (perceptron)
    --------------------+--------------------+
    | electrical signals|   data samples     |
    --------------------+---------------------
    |   synapse(node)   | input node(x, w, b)|
    --------------------+---------------------
    |     dendrite      |      summation     |
    |-------------------+---------------------
    |  cell-body(soma)  |      activation    |
    --------------------+---------------------
    |       axon        |    output node     |
    --------------------+--------------------+

*New connexion : axon + synapse => next dendrite 
## The perceptron (A Linear Unit)

- The perceptron: first model invented by Mc Culloch-Pitts 
  - `1 input layer`
    - input nodes : `x1*w1 + x2*w2 ...+ b`
      - where : 
        - x : input signals
        - b : bias 
        - w : weights
    - Neuron 
        - Summation function
        - activation function (step : 0/1) 
  - `1 output layer` 

- The multilayer perceptron forms fully connected NN 
    - 1 input layer 
    - multiple hidden layers
    - Neuron 
        - Summation function
        - activation functions (step, tanh, sigmoid, RELU...)

    - 1 output layer
     
## Deep Neural Nets


### Deep learning algorithms/model

- Perceptron = biological neuron 
- Deep Neural Network 
- MNIST Image Recognition 
- Recurrent Neural Network (RNN) 
- Transformer 
- BERT 
- Convolution Neural Network (CNN)
- Long Short Term Memory Networks (LSTMs)
- Recurrent Neural Networks (RNNs)
- Generative Adversarial Networks (GANs)
- Radial Basis Function Networks (RBFNs)
- Multilayer Perceptrons (MLPs)
- Self Organizing Maps (SOMs)
- Deep Belief Networks (DBNs)
- Restricted Boltzmann Machines(RBMs)

Check the [Neural Networks architecture notes](neural-nets-architecture-notes.mdneura) for more.


### High Level Deep Learning Pipeline

```mermaid
graph LR;
    A(Data Collection)-->B(Model training);
    B-->C(Model Evaluation and iteration);
    C-->D(Deployment)
    D-->E(Monitoring)    
    E-->A;
```

### Modeling

WHAT? (the problem)
- A "loss function" that measures how good the network's predictions are.

HOW? (to solve it)
  - An "optimizer" that can tell the network how to change its weights.

**Dataset**

Dataset: 
- Dataset is split into:
  - `Training set`:  new dataset (generalization)
  - `Validation set`: evaluate the performance of the model based on diffrent 
  - `Test set`: final evaluation

**training Process**
- Error/cost function: 
    - R^2

- Backpropagation: allows NeuralNet figure out patterns that are convoluted(complex) for human to extract

- Optimizer functions: 
  - gradient descent ... 
 
- Prediction 
  - activation function:      
    - Heaveside
    - Sigmoid       
    - SoftMAx
    - Tanh
    - relu
    - GLU
    - SwiGLU

More: https://deepgram.com/ai-glossary/activation-functions

**Metrics**
- Error
  - training
  - test 
- Accuracy(prediction w/ test data) 


**Model Tuning/Configuration**

Hyperparameters: are used to control the behavior of the learning algorithm
- Learning Rate (LR): the speed the model learning
- Epoch: when the model is trained on the entire dataset (forward + backpropagation)  
- Batch size: process of splitting the dataset into small chuncks
- iteration: number of batch size in entire dataset

We can configure the the capacity (complexity) of the model by tuning its hyperparameters 
- learning rate
- number of layers 
- numbers of hidden layers
- the depth of NeuralNet

### Model Evaluation
<img width="400" height="380" src="./docs/model-train-test.png" >

Evaluation: 

- Condition of good model: `test error > train error` 
  - `underfitting` : small model high capacity (larger dataset)  - `overfitting` : big model low capacity (smalller dataset)

||Underfitting | Overfitting|
|--|--|--|
|Bias|high|low|
|Variance ( $\sigma^2$ ) |low|high|
|Train data|bad|good|
|Unseen data|bad|bad|
|Accuracy|train + val/test: Bad|train: OK , Val/test: NOK|
|Cause|less data|noisy data|
|Solution|more data can't help| more data can help|


Error vs Capacity 
- capacity: the complexity of the model
  - the deeper the NeuralNet the higher the capacity to learn
- error: ? 
  - train error:? 
  - test error: 
- Generalization gap: the gap btw train error and test error 
- Bias
- Variance
- Early Stopping (optimal solution)
  - it's not bad if the accuracy ok (train + test) => bias + variance: ok
  - Otherwise [Regularization](https://www.ibm.com/topics/regularization#:~:text=Regularization%20is%20a%20set%20of,overfitting%20in%20machine%20learning%20models) is the way to go!!!


## Model Fine-tuning
- @TODO

## Tools & Frameworks

| Low-Level Library | High-Level Framework |
|---|---|
| [PyTorch (Facebook)](https://pytorch.org/) | [Fastai](https://www.fast.ai/) |
| [TensorFlow (Google)](https://www.tensorflow.org/) | [Keras](https://keras.io/) |
| [CNTK (Microsoft Cognitive Toolkit)](https://github.com/microsoft/CNTK) | [Keras](https://keras.io/) |
| [Jax](https://jax.readthedocs.io/en/latest/quickstart.html) | [Haiku](https://github.com/google-deepmind/dm-haiku) |
| [MXNet (Apache)](https://mxnet.apache.org/versions/1.9.1/) | [Gluon](https://mxnet.apache.org/versions/1.3.1/gluon/index.html) |
| [PaddlePaddle](https://www.paddlepaddle.org.cn/en) | [Paddle2.0 ](https://pgl.readthedocs.io/en/latest/)|
| [Theano (LISA Lab - Mila institute - University of Montreal)](https://github.com/Theano/Theano) | [Lasagne](https://github.com/Lasagne/Lasagne), [Keras](https://keras.io/) |
| [Chainer](https://chainer.org/) | [ChainerCV](https://github.com/chainer/chainercv) |
| [Caffe (Berkeley Artificial Intelligence Research (BAIR))](https://caffe.berkeleyvision.org/) | - |

## Hello World! 

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Build a simple neural network
model = Sequential([
    Dense(32, activation='relu', input_shape=(784,)),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Summary of the model
model.summary()
```

## Lab: Zero to Hero Projects

- **Project 1**: Image Classification with Convolutional Neural Networks (CNNs).
- **Project 2**: Sentiment Analysis using Long Short-Term Memory networks (LSTMs).
- **Project 3**: Developing a Chatbot with Deep Learning techniques.
- **Project 4**: Time Series Forecasting with Recurrent Neural Networks (RNNs).

## References
Wikipedia : 
- [Deep Learning (DL)](https://en.wikipedia.org/wiki/Deep_learning)
  - [Artificial Neural Network (NeuralNets)](https://en.wikipedia.org/wiki/Artificial_neural_network)
- [Supervised Learning](https://en.wikipedia.org/wiki/Supervised_learning)
- [Semi-supervised Learning](https://en.wikipedia.org/wiki/Weak_supervision#Semi-supervised_learning)
- [Unsupervised Learning](https://en.wikipedia.org/wiki/Unsupervised_learning)
- [Reinforcement Learning](https://en.wikipedia.org/wiki/Reinforcement_learning)
  
Lectures & Tutorials:
- [Neural Networks: Zero to Hero - karpathy.ai](https://karpathy.ai/zero-to-hero.html) 
- [Hacker's guide to Neural Networks -  karpathy.ai](https://karpathy.github.io/neuralnets/)
- [Deep Learning Crash Course for Beginners - FreebootCamp](https://www.youtube.com/watch?v=VyWAvY2CF9c)
- [Neural Network(3Blue1Brown)](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [How machine learns](https://www.youtube.com/watch?v=IHZwWFHWa-w) 

- [NVIDIA Deep Learning Institute](https://www.nvidia.com/en-us/training/)
- [Learn to use Deep Learning, Computer Vision and Machine Learning techniques to Build an Autonomous Car with Python](https://www.udemy.com/course/applied-deep-learningtm-the-complete-self-driving-car-course/learn/lecture/11294836#overview)

- IBM
  - [What is regularization?](https://www.ibm.com/topics/regularization#:~:text=Regularization%20is%20a%20set%20of,overfitting%20in%20machine%20learning%20models)

Forums & Discussions:
- [Neural Networks: Zero to Hero - YC News](https://news.ycombinator.com/item?id=35459031)


Research Paper/Works
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
- LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature, 521(7553), 436-444.
- Chollet, F. (2017). Deep Learning with Python. Manning Publications.