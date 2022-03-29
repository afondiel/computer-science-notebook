#------------------------------------------------------------
#	- 
# 	- Hello world, ML
# 	- 27-07-2020
# 	- aD
#-------------------------------------------------------------
#!/usr/bin/env python3
from sklearn import tree 


#test
print ("My first Neural Network\n")

""" ======== SUPERVISED LEARNING STRUCTURE =============
|--------|			|-----------| 		|-----------|
|Collect |          |Train 	 	|       |Make 		|
|Training|    =>    |Classifier |	=>  |Predictions|
|	Data |          |			|       |			|
|--------|          |-----------|       |-----------|

"""

""" Colleting input datas """
#Initial values
#features = [[140,"smooth"], [130,"smooth"], [150,"bumpy"], [170,"bumpy"]]
#labels = ["apple", "apple", "orange", "orange"]

"""scikit-learn uses real-valued features """
#smooth : 1
#bumpy  : 0
features = [[140, 1], [130, 1], [150, 0], [170, 0]]
#apple : 0
#orange :1
labels = [0, 0, 1, 1]

""" Train Classifier (Box of RULES) """

""" Decison Tree

	 weight > 150? 
		 / \
	yes /   \ No
Texture == bumby  ?
	 / \
yes /   \ No
 Orange apple
 
"""

# creating a model
clf = tree.DecisionTreeClassifier()

# training
#fit : find pattern in data
clf = clf.fit(features, labels) 

# prediction
print (clf.predict([[150, 0]]))



