# Classification Notes

## Overview
- Predictive modeling approximates a mapping function which predicts the class or category far a given observation. 

## Linear Model

```
                +-----------+        
X(x1,x2) ====> |    Model  | ====> Classified(x1, x2)  
                +-----------+        features   
```
- where : 
  - x1 : classe1/label1/target1
  - x2 : classe2/label2/target2

Ex:  Diabet classification model

```
- high bloods sugar levels  <=> old age
        |                           |
        v                           v
- insulin resistance        <=> less exercise 
        |                           |
        v                           v
- diabet task               <=> diabetic risk 

```
- labels to classify : [no disease, disease] 

**Prediction(scoring)**
- if new person (classified input) is below the line (model) then we estimated it its `no disease` class according to previous data locationn


## Type of classification models

- Binary classification
- Multi-Class classification
- Multi-Label classification
- Imbalanced classification


# References

Notes From : [The Complete Self-Driving Car Course Applied Deep-Learning - Udemy - Notes](https://github.com/afondiel/The-Complete-Self-Driving-Car-Course-Udemy/blob/main/self-driving-cars-dl-notes.md)

- [Classification - wikipedia](https://en.wikipedia.org/wiki/Statistical_classification)
- [Types of classification in machine-learning](https://machinelearningmastery.com/types-of-classification-in-machine-learning/)
- [Top machine-learning algorithms for classification](https://towardsdatascience.com/top-machine-learning-algorithms-for-classification-2197870ff501?gi=560bc277d5fd)

