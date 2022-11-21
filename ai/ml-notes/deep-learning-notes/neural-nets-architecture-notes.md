# Neural Network (NN) Architecture - Notes

Agenda
- [PERCEPTRON (P)](#perceptron-p)
- [FEED FORWARD (FF)](#feed-forward-ff)
- [RADIAL BASIS NETWORK (RBF)](#radial-basis-network-rbf)
- [DEEP FEED FORWARD (DFF)](#deep-feed-forward-dff)
- [RECURRENT NEURAL NETWORK(RNN)](#recurrent-neural-networkrnn)
- [LONG SHORT-TERM MEMORY (LSTM)](#long-short-term-memory-lstm)
- [CONVOLUTIONAL NEURAL NETWORK (CNN)](#convolutional-neural-network-cnn)
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
**Learning Process**
- by tuning the hyperparameters we can configure the the complexity of the model
- the deeper the NN the higher the capacity to learn
  
Link : 
- [MNIST](https://github.com/afondiel/research-notes/blob/master/ai/ml-notes/deep-learning-notes/neural-nets/MNIST%20Image%20Recognition.ipynb)
## CONVOLUTIONAL NEURAL NETWORK (CNN)
Overview : 
- extract features from image
- fast computation comparing to a regular NN
  - No limitation and high accuracy
- Ideal for image classification
- for complex images

  
Application :
- Face recognition
- object detection
- NLP
- sign traffic/self-driving cars
- correlation problems : cross correlation/auto correlation issues
- used as linear filter ? 

### Architecture

- CNN process:


        Input => Extraction => Fully Connected(multilayer perceptron) => activation => Output

- CNN train :

        Input => CNN => pooling => ... => Flattening FC => FC => activation => Output

Where : 
1. Input : img (2D-Grayscale or 3D-RGB)
   - 2D-Grayscale : 1 x kernel => 28x28=784 px 
   - 3D-RGB: 3 x kernel => 72x72=(5184 * 3) = 15552px
2. Extraction operation: conv + pool
   
   - *Convolution* : extract and learn some specific feature
     - convolution operation :  `img pixel matrix * filter(kernel matrix)` => (feature map)
       - stride: the action of sliding the matrix kernel along the img pixel
         - high stride => low feature (decrese pixels)
       - padding : the action of adding pixels around the input image (based on kernel size?)
         - the more feature => the higher the accuracy 
     feature od interest : special feature repeated
     - the deeper the resulting map the more features we extract and improving network
   - *Pooling* : is the process of merging the feature map
     - help to avoid overfitting in convoluted feature map (by reducing the number of params)
     - the result of pooling a matrix size of (N x N ) ?  
     - functions used for pooling : 
       - *sum* : get the avg value of 2x2 matrix neighbor
       - *average* : get the avg value of 2x2 matrix neighbor
       - *Max* :  get the max value of 2x2 matrix neighbor
3. Flattening : converting the data(from pooling) into a 1-dimensional array for inputting it to the next layer(fully connected layer)
4. Fully Connected (fc) : Multilayer perceptron
   - Activation : Softmax (linear), ReLU (Non linear) ...
5. Output : class

### Learning Process
- @TODO



Links : 
- [Implementation CNN - @afondiel](https://github.com/afondiel/research-notes/blob/master/ai/ml-notes/deep-learning-notes/neural-nets/convolutional-neural-network.ipynb) 
- [pytorch-artificial-intelligence-book](https://github.com/afondiel/research-notes/blob/master/books/ai/pytorch-artificial-intelligence-fundamentals-2020-Jibin-Mathew.pdf)
- [Building Model with Pytorch](https://pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html)
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

