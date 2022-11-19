# Neural Network (NN) Architecture - Notes

Agenda
- [PERCEPTRON (P)](#perceptron-p)
- [FEED FORWARD (FF)](#feed-forward-ff)
- [RADIAL BASIS NETWORK (RBF)](#radial-basis-network-rbf)
- [DEEP FEED FORWARD (DFF)](#deep-feed-forward-dff)
- [RECURRENT NEURAL NETWORK(RNN)](#recurrent-neural-networkrnn)
- [LONG SHORT-TERM MEMORY (LSTM)](#long-short-term-memory-lstm)
- [CONVOLUTIONAL NEURAL NETWORK (CNN)](#convolutional-neural-network-cnn)
- [Application](#application)
- [DEEP CONVOLUTIONAL NETWORK (DCN)](#deep-convolutional-network-dcn)
- [HOPFIELD NETWORK](#hopfield-network)
- [BOLTZMANN MACHINE](#boltzmann-machine)
- [TRANFORMER](#tranformer)
- [GPT-3](#gpt-3)


## PERCEPTRON (P)
- Link1 : 
  
[Perceptron Keras](https://github.com/afondiel/research-notes/blob/master/ai/ml-notes/deep-learning-notes/neural-nets/perpectron-model-keras.ipynb)
## FEED FORWARD (FF)
Link :
- [FF](https://github.com/afondiel/research-notes/blob/master/ai/ml-notes/deep-learning-notes/neural-nets/perpectron-model-keras.ipynb)
## RADIAL BASIS NETWORK (RBF)
Link : 
- [](#) 
## DEEP FEED FORWARD (DFF)
Link : 
- [DFF](https://github.com/afondiel/research-notes/blob/master/ai/ml-notes/deep-learning-notes/neural-nets/deep-neural-network-keras.ipynb) 
## RECURRENT NEURAL NETWORK(RNN)
Link : 
- [RNN](https://github.com/afondiel/research-notes/blob/master/ai/ml-notes/deep-learning-notes/neural-nets/recurrent_neural_network_LSTM_notes.ipynb) 
## LONG SHORT-TERM MEMORY (LSTM) 
Link : 
- [LSTM](https://github.com/afondiel/research-notes/blob/master/ai/ml-notes/deep-learning-notes/neural-nets/recurrent_neural_network_LSTM_notes.ipynb)
## MODIFIED/MIXED NATIONAL INSTITUTE OF STANDARD TECHNOLOGY (MNIST)
**Overview:** Multiclass dataset aim to classify 10 classes (1,2 ..10)
- sees data as images
- uses Deep NN for training
  
**Application:** sees data as images

**Architecture:**
- input layer size : 28x28 = 784 pixels
- hidden layer size : ??
- ouput layer size : 10 classes
  
Link : 
- [MNIST](https://github.com/afondiel/research-notes/blob/master/ai/ml-notes/deep-learning-notes/neural-nets/MNIST%20Image%20Recognition.ipynb)
## CONVOLUTIONAL NEURAL NETWORK (CNN)
Overview : 
- extract features from image
- Ideal for image classification
- for complex images/informations ? 
  
Application :
- Face recognition
- object detection
- sign traffic/self-driving cars
- correlation problems : cross correlation/auto correlation issues

Architecture

- CNN process:


        Input => Extraction => Fully Connected(multilayer perceptron) => activation => Output

- CNN train :

        Input => CNN => pooling => ... => FC => FC => activation => Output

Where : 
1. Input : img (2D-Grayscale or 3D-RGB)
   - 2D-Grayscale : 1 x kernel => 28x28=784 px 
   - 3D-RGB: 3 x kernel => 72x72=(5184 * 3) = 15552px
2. Extraction : conv + pool
   
- *Convolution* : extract and learn some specific feature
  - convolution filter (kernel matrix 3x3) : slides  the matrix along the img pixel(stride)
  - convolution operation : `img pixel * kernel matrix` = feature of map/interest? (ready to be pooled) 
    - The more feature the bigger the accuracy 

- *Pooling* : help to avoid overfitting in convoluted feature map (by reducing the number of params)
  - functions used for pooling : *sum*, *average*, *Max* (get the max value of 2x2 matrix neighbor)
- the deeper the resulting map the more features we extract and improving network
3. Fully Connected (fc) : Multilayer perceptron
1. Activation : Softmax (linear), ReLU (Non linear) ...
2. Output : class

Link : 
- [Implementation CNN](https://github.com/afondiel/research-notes/blob/master/ai/ml-notes/deep-learning-notes/neural-nets/convolutional-neural-network.ipynb) 
## DEEP CONVOLUTIONAL NETWORK (DCN)
Link : 
- [DCN](#)
## HOPFIELD NETWORK
Link :
- [Hopfield Network](https://en.wikipedia.org/wiki/Hopfield_network)
## BOLTZMANN MACHINE
Link : 
-  [BOLTZMANN](#)
## TRANSFORMER
Link : 
- [Transformer](https://github.com/afondiel/research-notes/blob/master/ai/research-papers/Attention%20is%20all%20you%20need%20-%20Google%20Research%20(2017).pdf)
## GPT-3
Link : 
- [GPT-3](https://en.wikipedia.org/wiki/GPT-3)

