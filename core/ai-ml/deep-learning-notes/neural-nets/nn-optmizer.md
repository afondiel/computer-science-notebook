# Comprehensive Technical Guide to Deep Learning Optimizers

Optimizers are algorithms or methods used to update neural network weights iteratively to minimize the loss function. The choice of optimizer can significantly affect model convergence speed, stability, and final performance. Below is a technical overview of the most widely used optimizers, their mathematical formulations, use cases, and code examples.

---


## **Table of Contents**

- [1. Stochastic Gradient Descent (SGD)](#1-stochastic-gradient-descent-sgd)
- [2. SGD with Momentum](#2-sgd-with-momentum)
- [3. Nesterov Accelerated Gradient (NAG)](#3-nesterov-accelerated-gradient-nag)
- [4. RMSProp](#4-rmsprop)
- [5. AdaGrad](#5-adagrad)
- [6. AdaDelta](#6-adadelta)
- [7. Adam (Adaptive Moment Estimation)](#7-adam-adaptive-moment-estimation)
- [8. Adamax](#8-adamax)
- [9. Nadam (Nesterov-accelerated Adam)](#9-nadam-nesterov-accelerated-adam)
- [10. FTRL (Follow The Regularized Leader)](#10-ftrl-follow-the-regularized-leader)
- [Optimizer Selection and Practical Considerations](#optimizer-selection-and-practical-considerations)
- [Summary Table](#summary-table)
- [Key Takeaways](#key-takeaways)
- [References](#references)
- [Citations](#citations)

## **1. Stochastic Gradient Descent (SGD)**

**Formula:**  
$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)
$$
where $$\eta$$ is the learning rate.

**Use Case:**  
- Baseline optimizer for all tasks (classification, regression, vision, NLP).
- Works well with large datasets and simple models.
- Often used with learning rate schedules and momentum for better performance.

**Code Example (PyTorch):**
```python
import torch.optim as optim
optimizer = optim.SGD(model.parameters(), lr=0.01)
```

---

## **2. SGD with Momentum**

**Formula:**  
$$
v_{t+1} = \gamma v_t + \eta \nabla_\theta \mathcal{L}(\theta_t)
$$
$$
\theta_{t+1} = \theta_t - v_{t+1}
$$
where $$\gamma$$ is the momentum term (e.g., 0.9).

**Use Case:**  
- Deep networks (CNNs, RNNs) to accelerate convergence and escape shallow minima.
- Reduces oscillation in ravines.

**Code Example (TensorFlow):**
```python
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
```

---

## **3. Nesterov Accelerated Gradient (NAG)**

**Formula:**  
$$
v_{t+1} = \gamma v_t + \eta \nabla_\theta \mathcal{L}(\theta_t - \gamma v_t)
$$
$$
\theta_{t+1} = \theta_t - v_{t+1}
$$

**Use Case:**  
- Faster and more stable convergence than vanilla momentum.
- Used in deep vision models (e.g., ResNet, VGG).

**Code Example (PyTorch):**
```python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
```

---

## **4. RMSProp**

**Formula:**  
$$
E[g^2]_t = \rho E[g^2]_{t-1} + (1-\rho)g_t^2
$$
$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} g_t
$$

**Use Case:**  
- RNNs and models with non-stationary objectives.
- Good for online and mini-batch learning.

**Code Example (TensorFlow):**
```python
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
```

---

## **5. AdaGrad**

**Formula:**  
$$
G_t = G_{t-1} + g_t^2
$$
$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} g_t
$$

**Use Case:**  
- Sparse data (NLP, recommendation systems).
- Automatically adapts learning rate for each parameter.

**Code Example (PyTorch):**
```python
optimizer = optim.Adagrad(model.parameters(), lr=0.01)
```

---

## **6. AdaDelta**

**Formula:**  
$$
E[g^2]_t = \rho E[g^2]_{t-1} + (1-\rho)g_t^2
$$
$$
\Delta\theta_t = - \frac{\sqrt{E[\Delta\theta^2]_{t-1} + \epsilon}}{\sqrt{E[g^2]_t + \epsilon}} g_t
$$
$$
E[\Delta\theta^2]_t = \rho E[\Delta\theta^2]_{t-1} + (1-\rho)(\Delta\theta_t)^2
$$

**Use Case:**  
- Like AdaGrad but mitigates rapid learning rate decay.
- Robust for deep architectures and noisy gradients.

**Code Example (TensorFlow):**
```python
optimizer = tf.keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95)
```

---

## **7. Adam (Adaptive Moment Estimation)**

**Formula:**  
$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t
$$
$$
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2
$$
$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t},\quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}
$$
$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
$$

**Use Case:**  
- Default optimizer for most deep learning tasks (vision, NLP, GANs, transformers).
- Fast convergence and robust to hyperparameters.

**Code Example (PyTorch):**
```python
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
```

---

## **8. Adamax**

**Formula:**  
Variant of Adam using the infinity norm.

**Use Case:**  
- Large-scale models with sparse gradients.
- More stable than Adam in some cases.

**Code Example (TensorFlow):**
```python
optimizer = tf.keras.optimizers.Adamax(learning_rate=0.002)
```

---

## **9. Nadam (Nesterov-accelerated Adam)**

**Formula:**  
Adam optimizer with Nesterov momentum.

**Use Case:**  
- Combines fast convergence of Adam and Nesterov.
- Used in image and language models for improved performance[3].

**Code Example (PyTorch):**
```python
optimizer = optim.NAdam(model.parameters(), lr=0.002)
```

---

## **10. FTRL (Follow The Regularized Leader)**

**Formula:**  
Designed for online learning and large-scale problems.

**Use Case:**  
- Large-scale linear models, click prediction, recommendation systems.
- Used in Google’s ad systems.

**Code Example (TensorFlow):**
```python
optimizer = tf.keras.optimizers.Ftrl(learning_rate=0.001)
```

---

## **Optimizer Selection and Practical Considerations**

- **SGD** is a strong baseline and often yields the best generalization when tuned well, especially for vision models[2][3].
- **Momentum/Nesterov** accelerates convergence, especially for deep CNNs.
- **RMSProp/AdaGrad/AdaDelta** are preferred for RNNs and sparse data.
- **Adam/Nadam** are the default for most modern deep learning tasks (transformers, GANs, etc.) due to adaptive learning rates and fast convergence[2][3][4][5].
- **Hyperparameter tuning** (especially learning rate) is critical for all optimizers and can change their relative performance[3].
- **Advanced optimizers** (like AdamW, LAMB, Lookahead, etc.) are used for very large models (BERT, GPT) and distributed training but follow similar principles.

---

## **Summary Table**

| Optimizer   | Main Use Case                    | Strengths                         | Common Models/Tasks         |
|-------------|----------------------------------|-----------------------------------|-----------------------------|
| SGD         | All-purpose, baseline            | Simplicity, generalization        | CNNs, basic RNNs            |
| Momentum    | Deep nets, vision                | Faster, less oscillation          | ResNet, VGG                 |
| Nesterov    | Deep nets, vision                | Lookahead, stable                 | ResNet, VGG                 |
| RMSProp     | RNNs, online learning            | Handles non-stationarity          | LSTM, GRU                   |
| AdaGrad     | Sparse data, NLP                 | Adaptive per-parameter rates      | NLP, recommender systems    |
| AdaDelta    | Deep nets, noisy gradients       | No manual learning rate tuning    | Deep CNNs, RNNs             |
| Adam        | Default for most tasks           | Fast, robust, adaptive            | Transformers, GANs, BERT    |
| Adamax      | Large-scale, sparse gradients    | Stable, robust                    | Large NLP models            |
| Nadam       | Fast convergence, stable         | Combines Adam & Nesterov          | Vision, NLP                 |
| FTRL        | Online, large-scale, sparse      | Efficient, scalable               | Ads, recommendations        |

---

## **Key Takeaways**

- **Start with Adam** for most tasks, especially if unsure.
- **Tune learning rates** and other hyperparameters for best results[3].
- **Switch to SGD/Momentum** for better generalization if overfitting occurs.
- **Use RMSProp/AdaGrad/AdaDelta** for RNNs or sparse data.
- **Advanced optimizers** (AdamW, LAMB) are used in large-scale distributed training.

---

**References:**  
- [1] arXiv: A Comparison of Optimization Algorithms for Deep Learning  
- [2] Neptune.ai: Deep Learning Optimization Algorithms  
- [3] arXiv: On Empirical Comparisons of Optimizers for Deep Learning  
- [4] ADCAIJ: Comparative Analysis of Optimization Algorithms  
- [5] Comet.ml: Empirical Comparison of Optimizers  
- [6] World Scientific: A Comparison of Optimization Algorithms for Deep Learning

This guide covers the most widely used optimizers, their technical details, practical scenarios, and code examples for both PyTorch and TensorFlow.

## Citations:
- [1] https://arxiv.org/abs/2007.14166
- [2] https://neptune.ai/blog/deep-learning-optimization-algorithms
- [3] https://arxiv.org/pdf/1910.05446.pdf
- [4] https://revistas.usal.es/cinco/index.php/2255-2863/article/view/ADCAIJ2020927990
- [5] https://heartbeat.comet.ml/an-empirical-comparison-of-optimizers-for-machine-learning-models-b86f29957050
- [6] https://www.worldscientific.com/doi/10.1142/S0218001420520138

---

Source (Perplexity)