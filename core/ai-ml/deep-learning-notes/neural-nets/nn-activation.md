# Comprehensive Technical Guide to Activation Functions in Deep Learning

Activation functions introduce non-linearity into neural networks, enabling them to model complex relationships. The choice of activation function impacts learning dynamics, convergence, and the ability to solve specific tasks. Below is a detailed guide covering the most widely used activation functions, their mathematical formulations, use cases, and code examples.

---

## **Table of Contents**

- [1. Linear (Identity) Activation](#1-linear-identity-activation)
- [2. Binary Step Function](#2-binary-step-function)
- [3. Sigmoid (Logistic)](#3-sigmoid-logistic)
- [4. Hyperbolic Tangent (Tanh)](#4-hyperbolic-tangent-tanh)
- [5. Rectified Linear Unit (ReLU)](#5-rectified-linear-unit-relu)
- [6. Leaky ReLU](#6-leaky-relu)
- [7. Parametric ReLU (PReLU)](#7-parametric-relu-prelu)
- [8. Exponential Linear Unit (ELU)](#8-exponential-linear-unit-elu)
- [9. Scaled Exponential Linear Unit (SELU)](#9-scaled-exponential-linear-unit-selu)
- [10. Softplus](#10-softplus)
- [11. Swish](#11-swish)
- [12. Softmax](#12-softmax)
- [Choosing the Right Activation Function](#choosing-the-right-activation-function)
- [References and Further Reading](#references-and-further-reading)
- [Citations](#citations)

## **1. Linear (Identity) Activation**

- **Formula:** $$ f(x) = x $$
- **Range:** $$(-\infty, +\infty)$$
- **Use Case:** Output layer for regression problems, where the target is a real value.
- **Pros:** Simple, preserves input.
- **Cons:** No non-linearity; stacking layers with linear activations collapses to a single linear transformation.

**Python Example:**
```python
def linear(x):
    return x
```
---

## **2. Binary Step Function**

- **Formula:** $$ f(x) = \begin{cases} 1 & \text{if } x \geq 0 \\ 0 & \text{if } x  0, x, alpha * x)

---

## **7. Parametric ReLU (PReLU)**

- **Formula:** Like Leaky ReLU, but $$\alpha$$ is learned during training.
- **Use Case:** Similar to Leaky ReLU, but potentially better performance.
- **Pros:** Adaptively learns the negative slope.
- **Cons:** Adds parameters to the model[2].

**Python Example:**
```python
def prelu(x, alpha):
    return np.where(x > 0, x, alpha * x)  # alpha is learned
```
---

## **8. Exponential Linear Unit (ELU)**

- **Formula:** $$ f(x) = \begin{cases} x & \text{if } x \geq 0 \\ \alpha (e^x - 1) & \text{if } x = 0, x, alpha * (np.exp(x) - 1))

---

## **9. Scaled Exponential Linear Unit (SELU)**

- **Formula:** $$ f(x) = \lambda \begin{cases} x & \text{if } x > 0 \\ \alpha (e^x - 1) & \text{if } x \leq 0 \end{cases} $$
- **Use Case:** Self-normalizing neural networks.
- **Pros:** Induces self-normalizing properties, stabilizes training.
- **Cons:** Requires specific initialization and architecture[2].

**Python Example:**
```python
def selu(x, lambda_=1.0507, alpha=1.67326):
    return lambda_ * np.where(x > 0, x, alpha * (np.exp(x) - 1))
```
---

## **10. Softplus**

- **Formula:** $$ f(x) = \ln(1 + e^x) $$
- **Range:** $$(0, +\infty)$$
- **Use Case:** Smooth approximation of ReLU.
- **Pros:** Differentiable everywhere, no dead neurons.
- **Cons:** Computationally expensive[2].

**Python Example:**
```python
def softplus(x):
    return np.log(1 + np.exp(x))
```
---

## **11. Swish**

- **Formula:** $$ f(x) = x \cdot \text{sigmoid}(x) $$
- **Range:** $$(-\infty, +\infty)$$
- **Use Case:** Hidden layers, especially in deep networks.
- **Pros:** Smooth, non-monotonic, often outperforms ReLU.
- **Cons:** Slightly more computationally expensive[2].

**Python Example:**
```python
def swish(x):
    return x * sigmoid(x)
```
---

## **12. Softmax**

- **Formula:** $$ f(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}} $$
- **Range:** $$(0, 1)$$, sum to 1
- **Use Case:** Output layer for multi-class classification.
- **Pros:** Converts logits to class probabilities.
- **Cons:** Not used in hidden layers[3][5].

**Python Example:**
```python
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)
```
---

## **Choosing the Right Activation Function**

| Use Case                        | Hidden Layers         | Output Layer (Regression) | Output Layer (Binary) | Output Layer (Multi-class) |
|----------------------------------|----------------------|--------------------------|-----------------------|----------------------------|
| Regression                      | ReLU, Leaky ReLU     | Linear                   | -                     | -                          |
| Binary Classification           | ReLU, Tanh           | -                        | Sigmoid               | -                          |
| Multi-class Classification      | ReLU, Tanh           | -                        | -                     | Softmax                    |
| Deep/Residual Networks          | ReLU, Swish, ELU     | -                        | -                     | -                          |
| Self-Normalizing Networks       | SELU                 | -                        | -                     | -                          |
| GANs (Generator/Discriminator)  | Leaky ReLU, Tanh     | -                        | Sigmoid (Discriminator) | -                        |

---

## **References and Further Reading**

- [V7 Labs: Activation Functions in Neural Networks][1]
- [DataCamp: Introduction to Activation Functions][3]
- [Number Analytics: Practical Guide][4]
- [Turing: How to Choose Activation Functions][5]

---

**Summary:**  
Activation functions are essential for deep learning, enabling non-linear modeling. The most common choices are ReLU for hidden layers, sigmoid for binary classification outputs, and softmax for multi-class outputs. Advanced functions like Leaky ReLU, ELU, SELU, and Swish address specific limitations and can further improve performance based on the network architecture and task requirements[1][2][3][4][5][6].

## Citations
- [1] https://www.v7labs.com/blog/neural-networks-activation-functions
- [2] https://dergipark.org.tr/tr/download/article-file/2034482
- [3] https://www.datacamp.com/tutorial/introduction-to-activation-functions-in-neural-networks
- [4] https://www.numberanalytics.com/blog/practical-guide-activation-functions-deep-learning
- [5] https://www.turing.com/kb/how-to-choose-an-activation-function-for-deep-learning
- [6] https://www.nbshare.io/notebook/751082217/Activation-Functions-In-Python/
- [7] https://apxml.com/courses/pytorch-for-tensorflow-developers/chapter-2-pytorch-nn-module-for-keras-users/activation-functions-pytorch-tf
- [8] https://en.wikipedia.org/wiki/Activation_function
- [9] https://www.exxactcorp.com/blog/Deep-Learning/activation-functions-and-optimizers-for-deep-learning-models
- [10] https://encord.com/blog/activation-functions-neural-networks/
- [11] https://www.machinelearningmastery.com/choose-an-activation-function-for-deep-learning/
- [12] https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial3/Activation_Functions.html
- [13] https://github.com/xbeat/Machine-Learning/blob/main/Deep%20Learning%20Slideshow%20with%20TensorFlow%20and%20PyTorch.md
- [14] https://arxiv.org/abs/2109.14545
- [15] https://www.linkedin.com/pulse/top-10-activation-functions-deep-learning-suresh-beekhani-vbisf
- [16] https://developers.google.com/machine-learning/crash-course/neural-networks/activation-functions
- [17] https://stats.stackexchange.com/questions/115258/comprehensive-list-of-activation-functions-in-neural-networks-with-pros-cons
- [18] https://paperswithcode.com/methods/category/activation-functions
- [19] https://www.doc.ic.ac.uk/~bkainz/teaching/DL/L03_DL_activation_and_losses.pdf
- [20] https://stackoverflow.com/questions/37947558/neural-network-composed-of-multiple-activation-functions
- [21] https://www.askpython.com/python/examples/activation-functions-python
- [22] https://github.com/siebenrock/activation-functions
- [23] https://keras.io/api/layers/activations/
- [24] https://machinelearningmastery.com/using-activation-functions-in-deep-learning-models/
- [25] https://www.reddit.com/r/MachineLearning/comments/fikvm7/d_is_there_ever_a_reason_to_use_multiple/
- [26] https://www.digitalocean.com/community/tutorials/sigmoid-activation-function-python

---

Source (Perplexity)