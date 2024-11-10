# Regression Notes

## Overview 

- Predicts values based on line which best models the relationship between the independent(X) and dependent variables(Y)
- Predict values of a `continues` spectrum rather than discret classes
- use continue and quantitative variables
- Invented by Francis Galton (1886)
- Goal: 
  - Study correlation between prices
  - Magnetude of relationship (between variables, features of a system ..)


## Applications : 
- medecines (medical models)
- Business (consummer behavior, firm productivity, competiveness of public/private sector)

## Regression Modeling 

**Linear Regression**

$$
Y = mX + b
$$

where : 
- Y : Response/dependent variable to explain/predict based on value of the independent (X)
- X : explanatory/independent variable/preditor
- m : weight
- b : bias
  
Ex: This model is often used to predict the price of any sized house based on where the value falls on the regression line 
```
(price) 
^  
|  /
| /
|/
+++++-> (surface/size)
```



- Adjusted Response

$$
\displaystyle \hat{Y} = m\hat{X}
$$

- Residual error

$$
\displaystyle e = Y  - \hat{Y}
$$

- deviation & Reajustement : $R^2$

$$
\displaystyle Dev(Y) = 
\sum_{i=1}^{n} (Yi - \bar{Yi})^2 
$$

The coeffiecient of determination  :  $R^2$

$$
\displaystyle R^2 =
\frac{Dev(\hat{Y})}{Dev(Y)} 
$$


**Cost Fonction for linear regression**
                  
Mean squared error (MSE)

$$
\displaystyle MSE = 
\frac {1}{n} \sum_{i=1}^{n} (Yi - \hat{Yi})^2 
$$


**Model Optimization**
- Least Squares
  - small dataset
- Gradien Descent : 1st order partial derivative to find the minimum value i.e reducing error of the cost function
  - larger dataset 

Equation @TODO : 

**Least Squares**
@TODO
**Gradien Descent**

$$
\displaystyle f(m,b) = 
\begin{bmatrix} 
\frac{df}{dm} \\ 
\frac{df}{db} 
\end{bmatrix}
= 
\begin{bmatrix} 
? \\ 
? 
\end{bmatrix}
$$ 


## Model evaluation metrics
- MAE
- sMAPE
- MAPE
- MASE
- MSPE
- RMS
- RMSE/RMSD
- R2
- MDA
- MAD

## Interpolation vs Extrapolation

|Interpolation|Extrapolation|
|--|--|
|The reading of values between two points in a data set| Estimating a value that's outside the data set |
|Primarily used to identify missing past values| Plays a major role in forecasting|
The estimated record is more likely to be correct.|The estimated values are only probabilities, so they may not be entirely correct.|

- In interpolation however, the number of sampling of the function shall be greater than the numbers of parameters


# References 
Wiki : 
- [Linear Regression](https://en.wikipedia.org/wiki/Linear_regression)
- [Least_squares](https://en.wikipedia.org/wiki/Least_squares)
- [Gradient descent](https://en.wikipedia.org/wiki/Gradient_descent)

[Extrapolation and interpolation](https://www.techtarget.com/whatis/definition/extrapolation-and-interpolation)

Glossary 
  - **Bias** : is a weighted factor on the interpretation of the results in a more or less hidden way. When we speak of a biased analysis, the results are mathematically correct but their interpretation is distorted by the bias.
  - **Error**: generates false results. Basically, if the data is wrong the results are wrong, if the calculation formulas are wrong or inadequate the results are wrong.
  - **residuals**  : represent the portion of **variability** not explained by the model.





