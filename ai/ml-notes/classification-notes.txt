//////// Classification Notes //////////

- Predictive modeling approximates a mapping function which predicts the class or category far a given observation. 

# Linear Model
                +-----------+        
X(xi,xi') ====> |    Model  | ====> Classified(xi, xi')  
                +-----------+        features   

- where : 
    =>xi : classe1/label1
    =>xi' : classe2/label2

Ex:  Diabet classification model
- high bloods sugar levels  <=> old age
        |                           |
        v                           v
- insulin resistance        <=> less exercise 
        |                           |
        v                           v
- diabet task               <=> diabetic risk 

- labels to classify : no disease, disease 

## Prediction(scoring)
- if new person (classified input) is below the line (model) then we estmated it its "no disease" class according to previous data locationn


## Type of classification models

- Binary classification
- Multi-Class classification
- Multi-Label classification
- Imbalanced classification


# References
https://en.wikipedia.org/wiki/Statistical_classification
https://machinelearningmastery.com/types-of-classification-in-machine-learning/
https://towardsdatascience.com/top-machine-learning-algorithms-for-classification-2197870ff501?gi=560bc277d5fd