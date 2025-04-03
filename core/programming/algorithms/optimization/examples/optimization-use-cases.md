# Optimization algorithms in Computer Science and Beyond: Hands-On Use Cases

## Table of Contents
1. [Computer Science (General)](#1-computer-science-general)
    - [Context](#context)
    - [Key Algorithms](#key-algorithms)
    - [Application](#application)
    - [C++ Example (Greedy: Dijkstra’s Algorithm)](#c-example-greedy-dijkstras-algorithm)
2. [Embedded Systems](#2-embedded-systems)
    - [Context](#context-1)
    - [Key Algorithms](#key-algorithms-1)
    - [Application](#application-1)
    - [C++ Example (Hill Climbing for Duty Cycle)](#c-example-hill-climbing-for-duty-cycle)
3. [Computer Computing (High-Performance Computing - HPC)](#3-computer-computing-high-performance-computing---hpc)
    - [Context](#context-2)
    - [Key Algorithms](#key-algorithms-2)
    - [Application](#application-2)
    - [C++ Example (Conjugate Gradient - Simplified)](#c-example-conjugate-gradient---simplified)
4. [Computer Vision](#4-computer-vision)
    - [Context](#context-3)
    - [Key Algorithms](#key-algorithms-3)
    - [Application](#application-3)
    - [C++ Example (Levenberg-Marquardt - Simplified)](#c-example-levenberg-marquardt---simplified)
5. [Edge AI](#5-edge-ai)
    - [Context](#context-4)
    - [Key Algorithms](#key-algorithms-4)
    - [Application](#application-4)
    - [C++ Example (PSO for Layer Size)](#c-example-pso-for-layer-size)
6. [Machine Learning (ML)](#6-machine-learning-ml)
    - [Context](#context-5)
    - [Key Algorithms](#key-algorithms-5)
    - [Application](#application-5)
    - [C++ Example (SGD for Logistic Regression)](#c-example-sgd-for-logistic-regression)
7. [Deep Learning (DL)](#7-deep-learning-dl)
    - [Context](#context-6)
    - [Key Algorithms](#key-algorithms-6)
    - [Application](#application-6)
    - [C++ Example (Adam Optimizer)](#c-example-adam-optimizer)
8. [Summary by Domain](#summary-by-domain)

## 1. **Computer Science (General)**
## Context
Optimization in general computer science spans algorithms for resource allocation, scheduling, graph problems, and combinatorial tasks.

### Key Algorithms
- **Greedy Algorithms**: E.g., Dijkstra’s algorithm for shortest paths.
- **Dynamic Programming**: E.g., Knapsack problem for resource optimization.
- **Simulated Annealing**: For combinatorial optimization like Traveling Salesman Problem (TSP).
- **Genetic Algorithms**: For search and optimization in large discrete spaces.

### Application
Optimizing process scheduling in operating systems.

### C++ Example (Greedy: Dijkstra’s Algorithm)
```cpp
#include <iostream>
#include <vector>
#include <queue>
#include <limits>

using Graph = std::vector<std::vector<std::pair<int, int>>>; // {node, weight}

std::vector<int> dijkstra(const Graph& graph, int src) {
    int n = graph.size();
    std::vector<int> dist(n, std::numeric_limits<int>::max());
    dist[src] = 0;
    std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<>> pq;
    pq.push({0, src});

    while (!pq.empty()) {
        int u = pq.top().second;
        pq.pop();
        for (const auto& [v, weight] : graph[u]) {
            if (dist[u] + weight < dist[v]) {
                dist[v] = dist[u] + weight;
                pq.push({dist[v], v});
            }
        }
    }
    return dist;
}

int main() {
    Graph graph(4);
    graph[0].push_back({1, 4});
    graph[0].push_back({2, 8});
    graph[1].push_back({2, 4});
    graph[2].push_back({3, 2});
    auto dist = dijkstra(graph, 0);
    for (int i = 0; i < 4; ++i) std::cout << "Distance to " << i << ": " << dist[i] << "\n";
    return 0;
}
```
**Use Case**: Network routing or task scheduling.

---

## 2. **Embedded Systems**
### Context
Optimization focuses on minimizing power, memory, and latency under strict hardware constraints.

### Key Algorithms
- **Hill Climbing**: Local search for parameter tuning.
- **Genetic Algorithms**: Optimizing firmware configurations.
- **Branch and Bound**: Resource allocation with constraints.
- **Simplex Method**: Linear programming for power optimization.

### Application
Optimizing duty cycles in a sensor node to extend battery life.

### C++ Example (Hill Climbing for Duty Cycle)
```cpp
#include <iostream>
#include <random>

double power_usage(double duty_cycle) {
    return duty_cycle * 10 + (1 - duty_cycle) * 2; // Simplified model
}

double hill_climbing(double init, double step, int max_iter) {
    double current = init;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-step, step);

    for (int i = 0; i < max_iter; ++i) {
        double neighbor = current + dis(gen);
        if (neighbor < 0 || neighbor > 1) continue; // Bound to [0, 1]
        if (power_usage(neighbor) < power_usage(current)) current = neighbor;
    }
    return current;
}

int main() {
    double best_duty = hill_climbing(0.5, 0.1, 100);
    std::cout << "Optimal duty cycle: " << best_duty << ", Power: " << power_usage(best_duty) << "\n";
    return 0;
}
```
**Use Case**: Tuning sleep/wake cycles in IoT devices.

---

## 3. **Computer Computing (High-Performance Computing - HPC)**
### Context
Optimization targets parallel efficiency, load balancing, and scalability in large-scale computations.

### Key Algorithms
- **Conjugate Gradient**: Solving large sparse systems.
- **Particle Swarm Optimization (PSO)**: Parallel parameter tuning.
- **Simulated Annealing**: Load balancing across nodes.
- **Dynamic Programming**: Optimizing task partitioning.

### Application
Minimizing execution time in matrix multiplication on a cluster.

### C++ Example (Conjugate Gradient - Simplified)
```cpp
#include <iostream>
#include <vector>
#include <numeric>

double dot(const std::vector<double>& a, const std::vector<double>& b) {
    return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

std::vector<double> conjugate_gradient(const std::vector<std::vector<double>>& A, const std::vector<double>& b, int max_iter) {
    int n = b.size();
    std::vector<double> x(n, 0.0), r = b, p = b;
    double r_norm = dot(r, r);

    for (int i = 0; i < max_iter && r_norm > 1e-10; ++i) {
        std::vector<double> Ap(n, 0.0);
        for (int j = 0; j < n; ++j) for (int k = 0; k < n; ++k) Ap[j] += A[j][k] * p[k];
        double alpha = r_norm / dot(p, Ap);
        for (int j = 0; j < n; ++j) x[j] += alpha * p[j];
        std::vector<double> r_new(n);
        for (int j = 0; j < n; ++j) r_new[j] = r[j] - alpha * Ap[j];
        double r_norm_new = dot(r_new, r_new);
        double beta = r_norm_new / r_norm;
        for (int j = 0; j < n; ++j) p[j] = r_new[j] + beta * p[j];
        r = r_new;
        r_norm = r_norm_new;
    }
    return x;
}

int main() {
    std::vector<std::vector<double>> A = {{4, 1}, {1, 3}};
    std::vector<double> b = {1, 2};
    auto x = conjugate_gradient(A, b, 10);
    std::cout << "Solution: x = " << x[0] << ", y = " << x[1] << "\n";
    return 0;
}
```
**Use Case**: Solving PDEs in scientific simulations.

---

## 4. **Computer Vision**
### Context
Optimization enhances image processing, feature detection, and model fitting.

### Key Algorithms
- **Gradient Descent**: Bundle adjustment in 3D reconstruction.
- **Levenberg-Marquardt**: Non-linear least squares for camera calibration.
- **Simulated Annealing**: Image segmentation.
- **Genetic Algorithms**: Feature selection.

### Application
Fitting a circle to edge points in an image.

### C++ Example (Levenberg-Marquardt - Simplified)
```cpp
#include <iostream>
#include <vector>
#include <cmath>

struct Circle { double x, y, r; };

double error(const Circle& c, const std::vector<std::pair<double, double>>& points) {
    double sum = 0;
    for (const auto& p : points) {
        double dist = std::sqrt((p.first - c.x) * (p.first - c.x) + (p.second - c.y) * (p.second - c.y));
        sum += (dist - c.r) * (dist - c.r);
    }
    return sum;
}

Circle lm_fit(const std::vector<std::pair<double, double>>& points, Circle init, int max_iter) {
    Circle c = init;
    double lambda = 0.001;
    for (int i = 0; i < max_iter; ++i) {
        double e = error(c, points);
        // Simplified update (numerical approximation)
        Circle c_dx = {c.x + 0.01, c.y, c.r}, c_dy = {c.x, c.y + 0.01, c.r}, c_dr = {c.x, c.y, c.r + 0.01};
        double dx = (error(c_dx, points) - e) / 0.01;
        double dy = (error(c_dy, points) - e) / 0.01;
        double dr = (error(c_dr, points) - e) / 0.01;
        c.x -= lambda * dx;
        c.y -= lambda * dy;
        c.r -= lambda * dr;
        if (i % 10 == 0) std::cout << "Error: " << e << "\n";
    }
    return c;
}

int main() {
    std::vector<std::pair<double, double>> points = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
    Circle init{0, 0, 1};
    Circle result = lm_fit(points, init, 50);
    std::cout << "Circle: x = " << result.x << ", y = " << result.y << ", r = " << result.r << "\n";
    return 0;
}
```
**Use Case**: Hough transform refinement.

---

## 5. **Edge AI**
### Context
Optimization balances accuracy, latency, and power on resource-constrained devices.

### Key Algorithms
- **Particle Swarm Optimization**: Model compression.
- **Bayesian Optimization**: Hyperparameter tuning for tiny models.
- **Gradient Descent Variants**: Lightweight neural network training.
- **Simulated Annealing**: Scheduling inference tasks.

### Application
Optimizing a neural network’s layer sizes for an IoT device.

### C++ Example (PSO for Layer Size)
```cpp
#include <iostream>
#include <vector>
#include <random>

double evaluate_size(int size) {
    return size * 0.1 + 1000.0 / size; // Latency vs. accuracy trade-off
}

int pso(int min_size, int max_size, int n_particles, int max_iter) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(min_size, max_size);
    std::uniform_real_distribution<> dis_v(-1, 1);

    std::vector<int> positions(n_particles), velocities(n_particles, 0);
    std::vector<double> p_best(n_particles);
    int g_best = min_size;

    for (int i = 0; i < n_particles; ++i) {
        positions[i] = dis(gen);
        p_best[i] = evaluate_size(positions[i]);
        if (p_best[i] < evaluate_size(g_best)) g_best = positions[i];
    }

    for (int iter = 0; iter < max_iter; ++iter) {
        for (int i = 0; i < n_particles; ++i) {
            velocities[i] += static_cast<int>(1 * dis_v(gen) * (p_best[i] - positions[i]) + 2 * dis_v(gen) * (g_best - positions[i]));
            positions[i] = std::max(min_size, std::min(max_size, positions[i] + velocities[i]));
            double fitness = evaluate_size(positions[i]);
            if (fitness < p_best[i]) p_best[i] = fitness;
            if (fitness < evaluate_size(g_best)) g_best = positions[i];
        }
    }
    return g_best;
}

int main() {
    int best_size = pso(10, 100, 20, 50);
    std::cout << "Optimal layer size: " << best_size << ", Fitness: " << evaluate_size(best_size) << "\n";
    return 0;
}
```
**Use Case**: Pruning TinyML models.

---

## 6. **Machine Learning (ML)**
### Context
Optimization drives model training, hyperparameter tuning, and feature selection.

### Key Algorithms
- **Stochastic Gradient Descent (SGD)**: Model training.
- **Bayesian Optimization**: Hyperparameter search.
- **Genetic Algorithms**: Feature selection.
- **Coordinate Descent**: LASSO regression.

### Application
Training a logistic regression model.

### C++ Example (SGD for Logistic Regression)
```cpp
#include <iostream>
#include <vector>
#include <random>

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

std::vector<double> sgd(const std::vector<std::vector<double>>& X, const std::vector<int>& y, int max_iter) {
    int n = X.size(), d = X[0].size();
    std::vector<double> w(d, 0.0);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, n - 1);
    double lr = 0.01;

    for (int iter = 0; iter < max_iter; ++iter) {
        int i = dis(gen);
        double pred = 0;
        for (int j = 0; j < d; ++j) pred += w[j] * X[i][j];
        pred = sigmoid(pred);
        double error = y[i] - pred;
        for (int j = 0; j < d; ++j) w[j] += lr * error * X[i][j];
    }
    return w;
}

int main() {
    std::vector<std::vector<double>> X = {{1, 2}, {2, 3}, {3, 1}};
    std::vector<int> y = {0, 1, 1};
    auto w = sgd(X, y, 1000);
    std::cout << "Weights: " << w[0] << ", " << w[1] << "\n";
    return 0;
}
```
**Use Case**: Binary classification.

---

## 7. **Deep Learning (DL)**
### Context
Optimization focuses on training large neural networks with high-dimensional data.

### Key Algorithms
- **Adam**: Adaptive training.
- **RMSProp**: Gradient scaling.
- **Momentum SGD**: Faster convergence.
- **L-BFGS**: Second-order optimization for small networks.

### Application
Training a simple feedforward network.

### C++ Example (Adam Optimizer)
```cpp
#include <iostream>
#include <vector>
#include <cmath>

double loss(const std::vector<double>& w, const std::vector<std::vector<double>>& X, const std::vector<double>& y) {
    double sum = 0;
    for (size_t i = 0; i < X.size(); ++i) {
        double pred = w[0] * X[i][0] + w[1] * X[i][1];
        sum += (pred - y[i]) * (pred - y[i]);
    }
    return sum / X.size();
}

std::vector<double> adam(const std::vector<std::vector<double>>& X, const std::vector<double>& y, int max_iter) {
    int d = X[0].size();
    std::vector<double> w(d, 0.0), m(d, 0.0), v(d, 0.0);
    double lr = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8;

    for (int t = 1; t <= max_iter; ++t) {
        std::vector<double> grad(d, 0.0);
        for (size_t i = 0; i < X.size(); ++i) {
            double pred = w[0] * X[i][0] + w[1] * X[i][1];
            double error = pred - y[i];
            grad[0] += 2 * error * X[i][0];
            grad[1] += 2 * error * X[i][1];
        }
        for (int j = 0; j < d; ++j) {
            grad[j] /= X.size();
            m[j] = beta1 * m[j] + (1 - beta1) * grad[j];
            v[j] = beta2 * v[j] + (1 - beta2) * grad[j] * grad[j];
            double m_hat = m[j] / (1 - std::pow(beta1, t));
            double v_hat = v[j] / (1 - std::pow(beta2, t));
            w[j] -= lr * m_hat / (std::sqrt(v_hat) + epsilon);
        }
        if (t % 100 == 0) std::cout << "Loss: " << loss(w, X, y) << "\n";
    }
    return w;
}

int main() {
    std::vector<std::vector<double>> X = {{1, 1}, {2, 2}, {3, 3}};
    std::vector<double> y = {2, 4, 6};
    auto w = adam(X, y, 1000);
    std::cout << "Weights: " << w[0] << ", " << w[1] << "\n";
    return 0;
}
```
**Use Case**: Regression in small neural networks.

---

## Summary by Domain
- **Computer Science**: Broad, combinatorial focus (e.g., Dijkstra’s).
- **Embedded Systems**: Lightweight, constraint-driven (e.g., Hill Climbing).
- **HPC**: Parallel, scalable (e.g., Conjugate Gradient).
- **Computer Vision**: Geometric, non-linear (e.g., Levenberg-Marquardt).
- **Edge AI**: Efficient, compact (e.g., PSO).
- **ML**: Model training, tuning (e.g., SGD).
- **DL**: Gradient-based, adaptive (e.g., Adam).

These implementations can be enhanced with libraries like Eigen (linear algebra), OpenMP (parallelism), or frameworks like TensorFlow C++ for DL.