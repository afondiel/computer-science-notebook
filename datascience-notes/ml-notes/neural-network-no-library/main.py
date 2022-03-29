#------------------------------------------------------------
#	- 
# 	- Hello world, ML
# 	- 28-07-2020
# 	- aD
#-------------------------------------------------------------
"""
Siraj Raval : https://www.youtube.com/watch?v=h3l4qz76JhQ

1. Build IT
2. train IT
3. Test IT


Three layers neural network  Model  : 

2 inputs layers
5 hidden layers
2 outputs layers

##### GARBAGEEEEE #######

"""
import numpy as np  #scientif computing in python

#sigmoid  : 0< x < 1
def sig(x, deriv=False):
	if(deriv == True):
		return x*(1 - x)
	return 1/(1 + np.exp(-x))

#input data
X = np.array([[0,0,1],
			 [0,1,1],
			 [1,0,1],
			 [1,1,1]])
		
#output data
y = np.array([[0],
			  [1],
			  [1],
			  [0]])

#seed function : to generate same starting point/
#to have the same sequence of generate number every program run		  
np.random.seed(1)

#synapses (x,W)
#random numer for L1
syn0 = 2*np.random.random((3,4)) - 1 
#random numer for L2 
syn1 = 2*np.random.random((4,1)) - 1  

#training step
for j in range(60000):
	l0 = X
	l1 = sig(np.dot(l0, syn0))
	l2 = sig(np.dot(l1, syn1))
	
	#biaises//error
	l2_error = y - l2
	if(j % 10000) == 0:
		print ("Error:" + str(np.mean(np.abs(l2_error))))
	l2_delta = l2_error * sig(l2, deriv = True)
	
	l1_error = l2_delta.dot(syn1.T)
	l1_delta = l1_error*sig(l1, deriv=True)
	
	#update weight : gradient descent
	syn1 += l1.T.dot(l2_delta)
	syn0 += l0.T.dot(l1_delta)
	
print("Output after training")
print(l2)








