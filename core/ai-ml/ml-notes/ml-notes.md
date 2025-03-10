# Machine Learning (ML) - Notes

![](https://miro.medium.com/v2/resize:fit:720/format:webp/1*iWJWMeOQmI5kmPMGZ3tovA.jpeg)

Src : [towardsdatascience - Hunter Heidenreich](https://towardsdatascience.com/the-future-with-reinforcement-learning-877a17187d54)

## Agenda
- [Agenda](#agenda)
- [Intro](#intro)
- [ML vs Traditional coding ](#ml-vs-traditional-coding)
- [Training the Model (CLassifier)](#training-the-model-classifier)
- [ML algorithms](#ml-algorithms)
- [ML Frameworks/tools ](#ml-frameworkstools)
- [ML Problem solving in 7 steps](#ml-problem-solving-in-7-steps)
- [References](#references)

## Intro

ML is a field of AI algorithms that learn from EXAMPLES and EXPERIENCES instead of traditional HARDCODE and RULES

Ex : Apple and oranges detection

## ML vs Traditional coding  

- Tradictional coding  : too much rules 
- ML : Model / Classifier and train it to generate the RULES instead of writing them

- classifier (as function) : takes a data as input (FEATURES) and signs LABEL to it as output(LABELS) 
  - Fruit(apple or orange?) => Classifier => apple (if apple chose)
  - email (spam/mail_ok) => Classifier => spam (if mail_nok)

## Training the Model (CLassifier) 

To Train the Classifier we use  : 

- Supervised Learning(SL) : it learns from examples/experiences
- Unsuspervised Learning(USL) :  it learns from events ?
- Reinforcement Learning (RL): Conceptually similar to human learning processes 
  - ex: a robot learning to walk
  - Strategy games : Go, Chess etc
	 
The more the training data exists => the better the classifier Will be 

ML train 

	|--------|			|-----------| 		|-----------|
	|Collect |          |Train 	 	|       |Make 		|
	|Training|    =>    |Classifier |	=>  |Predictions|
	|	Data |          |			|       |			|
	|--------|          |-----------|       |-----------|


## ML algorithms
- `Supervised` : 
  - **Regression** : Predicting a continuous-valued attribute associated with an object
    - Multiple Linear Regression(MLR)
    - Polynomial Regression (PR)
  - **Classification** : Identifying which category an object belongs to
	- k-Nearest Neighbor
	- Decision Trees(ID3, C4.5, C5.0)
	- logistic regression
	- Naïve Bayes
	- Linear Discriminant Analysis
	- Neural Networks
	- Support Vector Machines (SVM)
	- Random Forest(RF)

- `Unsupervised` : 
	- **Clustering** : Automatic grouping of similar objects into sets.
		- k-Means
		- Mean-shift
		- Hierarchical Clustering (HC)
		- Density-based Clustering (DBSCAN)
		- Gaussian Mixture Models(GMM)
- `Reinforcement` : 
	- Q-Learning
	- Deep Q-Network (DQN)
	- A3C

## ML Frameworks/tools 
- TensorFlow
- PyTorch
- Scikit-learn
- Spark ML
- Torch
- Huggingface
- Keras

## ML Problem solving in 7 steps

1. GATHERING / COLLECTING DATA : The more data we collect the more accurate will the model.
- We collect datas to train the model of the system we want to deploy

2. DATA PREPARATION : 

	- Features ? : the input of the system
	- Labels ? : the output of the system 
	```
		----------------------------------
		| 		Features 	    | Labels|	
		----------------------------------
		|	x1	|	x2	|	xn	|	y1	|
		----------------------------------
		|	..	|	..	|	..	|	..	|
		----------------------------------
		| xn,m |		|		|	yn,m|
		----------------------------------
	```		
	- visualization of datas
	- balances, relationship between datas
	- split : training/evaluation(performance of the model)


3. Choosing the MODEL : There are already alot of model created by DataScientists : 
	- For : Music, image, number, text, text based data, linear model (y=ax+b)
	

4. TRAINING (the model : Y = mx+b) :

	- Y : output
	- m : SLope (many m possible, as many features)
	- X : input
	- b : Y-intercept
	```
				[m1,1 m1,2] 
		Weight = [m2,1 m2,2] 
				[m3,1 m3,2]
				
				[b1,1 b1,2] 
		biases = [b2,1 b2,2] 
				[b3,1 b3,2] 
	```		 
	Training process
	```
		|--------|			|-----------| 		|-----------|
		|        |          |   Model 	|       | 		    |
		|Training|    =>    |   (W,b)   |	=>  |Prediction |
		|	Data |          |			|       |			|
		|--------|          |-----------|       |-----------|

							|-----------|			||
			/\				| 		    |			\/
			||		<=		|Test/update|	<=
							| (W,b)	    |	
							|-----------|	
	```

Each iteration it's called, a training steps.


/!\ #Residus = Biais ( θ ^ ) ≡ E [ θ ^ ] − θ 
	#Définition —  Si θ ^ est l'estimateur de θ 
	

5. EVALUATION : after the model is good time to evaluate

```
		|----------|		 |-----------| 		 |------------|
		|          |         |   Model 	 |       | 		      |
		|EVALUATION|  =>     |   (W,b)   |	=>   | Prediction |
		|	Data   |         |			 |       |			  |
		|----------|         |-----------|       |------------|

							|-----------|			||
			/\				| 		    |			\/
			||		<=		|Test       |	<=
							| (W,b)	    |	
							|-----------|	
```

This metric allows the model to see the data that has not yet seen.
This is to test how the model might act in the real world


6. PARAMETER TUNING 

	- To improve the training 
	- Repeat the training data several time to increase the accuracy
	- Learning rate : limit of the train / how far we shift the line between two input datas
	- initial conditions : for complexes models (value = 0 ...)
	- Hyperparameters.
	
/!\ : it's important to choose the good parameters to be changed

7. PREDICTIONS : ML uses datas to answer questions

Input : features
Output : Labels 


## References

Scikit learn :
https://scikit-learn.org/stable/#

TensorFlow : 
https://www.tensorflow.org/resources/learn-ml

Spark ML  : 
https://spark.apache.org/docs/latest/ml-guide.html

PyTorch : 
https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html
https://docs.microsoft.com/en-us/learn/paths/pytorch-fundamentals/

Google course/ Josh gordon : 
https://www.youtube.com/watch?v=cKxRvEZd3Mw&list=PLOU2XLYxmsIIuiBfYad6rFYQU_jL2ryal

Google Cloud Plateform / Yufeng Guo: 
https://www.youtube.com/watch?v=nKW8Ndu7Mjw

IBM Cloud : 
https://www.ibm.com/cloud/blog/supervised-vs-unsupervised-learning


