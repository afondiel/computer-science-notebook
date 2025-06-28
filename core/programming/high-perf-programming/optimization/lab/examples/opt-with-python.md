# Core optimization algorithms in Python

## Overview

These span the general comprehensive list, advanced examples, domain-specific examples (by field), and industry-specific examples. 

It keeps the structure consistent, providing concise implementations with examples where applicable. Python’s simplicity and libraries (e.g., NumPy) will streamline some operations compared to C++.

---
## Table of Contents

- [Overview](#overview)
- [Notes](#notes)
- [General Comprehensive List](#general-comprehensive-list)
    - [Gradient Descent](#1-gradient-descent)
    - [Genetic Algorithm](#2-genetic-algorithm)
- [Advanced Examples (from Second Response)](#advanced-examples-from-second-response)
    - [Adam Optimizer](#1-adam-optimizer)
    - [CMA-ES](#2-cma-es)
    - [Bayesian Optimization](#3-bayesian-optimization)
    - [Ant Colony Optimization (ACO)](#4-ant-colony-optimization-aco)
- [By Domain (from Fourth Response)](#by-domain-from-fourth-response)
    - [Computer Science - Dijkstra’s Algorithm](#1-computer-science---dijkstras-algorithm)
    - [Embedded Systems - Hill Climbing](#2-embedded-systems---hill-climbing)
    - [HPC - Conjugate Gradient](#3-hpc---conjugate-gradient)
    - [Computer Vision - Levenberg-Marquardt](#4-computer-vision---levenberg-marquardt)
    - [Edge AI - PSO](#5-edge-ai---pso)
    - [ML - SGD](#6-ml---sgd)
    - [DL - Adam](#7-dl---adam)
- [By Industry (from Fifth Response)](#by-industry-from-fifth-response)
    - [Autonomous Systems - A*](#1-autonomous-systems---a)
    - [Agriculture - Linear Programming](#2-agriculture---linear-programming)
    - [Aerospace/Defense - PSO](#3-aerospacedefense---pso)
    - [Healthcare - Simulated Annealing](#4-healthcare---simulated-annealing)
    - [Smart Cities - ACO](#5-smart-cities---aco)
    - [Retail - Dynamic Programming](#6-retail---dynamic-programming)
    - [Manufacturing - Tabu Search](#7-manufacturing---tabu-search)
---

## Notes
- Python implementations leverage NumPy for efficiency where applicable (e.g., Adam, CMA-ES).
- Randomness uses Python’s `random` or `np.random` instead of C++’s `<random>`.
- Some algorithms (e.g., A*, ACO) are simplified but retain core logic.
- For production, libraries like SciPy, PyTorch, or NetworkX could enhance these implementations.

This covers all algorithms from previous responses. Let me know if you need refinements or additional Python examples!

### General Comprehensive List
#### 1. Gradient Descent
```python
def objective_function(x):
    return x**2 + 2*x + 1

def gradient(x):
    return 2*x + 2

def gradient_descent(x_init, learning_rate=0.1, max_iter=10):
    x = x_init
    for i in range(max_iter):
        x -= learning_rate * gradient(x)
        print(f"Iteration {i}: x = {x:.4f}, f(x) = {objective_function(x):.4f}")
    return x

x_init = 5.0
result = gradient_descent(x_init)
print(f"Optimized x: {result:.4f}")
```

#### 2. Genetic Algorithm
```python
import random

def fitness(x):
    return -x**2 + 10*x

def genetic_algorithm(pop_size=20, max_gen=50, mutation_rate=0.1):
    population = [random.randint(0, 10) for _ in range(pop_size)]
    for g in range(max_gen):
        fitness_values = [fitness(ind) for ind in population]
        new_population = []
        total_fitness = sum(fitness_values)
        for _ in range(pop_size):
            r = random.uniform(0, total_fitness)
            s = 0
            for i, f in enumerate(fitness_values):
                s += f
                if s >= r:
                    new_population.append(population[i])
                    break
        for i in range(0, pop_size, 2):
            if i + 1 < pop_size and random.random() < 0.8:
                p1, p2 = new_population[i], new_population[i + 1]
                new_population[i] = (p1 + p2) // 2
                new_population[i + 1] = p1 + p2 - new_population[i]
        for i in range(pop_size):
            if random.random() < mutation_rate:
                new_population[i] = random.randint(0, 10)
        population = new_population
        best = max(population, key=fitness)
        print(f"Generation {g}: Best x = {best}, Fitness = {fitness(best)}")
    return max(population, key=fitness)

best = genetic_algorithm()
print(f"Best solution: x = {best}, Fitness = {fitness(best)}")
```

---

### Advanced Examples (from Second Response)
#### 1. Adam Optimizer
```python
import numpy as np

def objective_function(v):
    return v[0]**2 + v[1]**2

def gradient(v):
    return np.array([2*v[0], 2*v[1]])

def adam_optimizer(init, lr=0.001, beta1=0.9, beta2=0.999, max_iter=100):
    params = np.array(init)
    m, v = np.zeros_like(params), np.zeros_like(params)
    epsilon = 1e-8
    for t in range(1, max_iter + 1):
        g = gradient(params)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g**2)
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        params -= lr * m_hat / (np.sqrt(v_hat) + epsilon)
        if t % 10 == 0:
            print(f"Iteration {t}: x = {params[0]:.4f}, y = {params[1]:.4f}")
    return params

init = [5.0, 5.0]
result = adam_optimizer(init)
print(f"Optimized: x = {result[0]:.4f}, y = {result[1]:.4f}, f(x, y) = {objective_function(result):.4f}")
```

#### 2. CMA-ES
```python
import numpy as np

def sphere(x):
    return np.sum(x**2)

def cma_es(initial_mean, sigma=0.5, pop_size=10, max_iter=100):
    n = len(initial_mean)
    mean = np.array(initial_mean)
    cov = np.eye(n) * sigma**2
    for iteration in range(max_iter):
        population = np.random.multivariate_normal(mean, cov, pop_size)
        fitness = [sphere(ind) for ind in population]
        elite_idx = np.argsort(fitness)[:pop_size // 2]
        elite = population[elite_idx]
        mean = np.mean(elite, axis=0)
        cov = np.cov(elite.T) + 1e-6 * np.eye(n)
        print(f"Iteration {iteration}: Best fitness = {min(fitness):.4f}")
    return mean

initial_mean = [5.0, 5.0]
best_solution = cma_es(initial_mean)
print(f"Optimized solution: {best_solution}")
```

#### 3. Bayesian Optimization
```python
import numpy as np
from scipy.stats import norm

def func(x):
    return (x[0] - 2)**2

def bayesian_optimization(objective_func, bounds, n_iter=10):
    X = np.random.uniform(bounds[0, 0], bounds[0, 1], (5, 1))
    Y = [objective_func(x) for x in X]
    for _ in range(n_iter):
        mu, sigma = np.mean(Y), np.std(Y) + 1e-6
        x_new = np.random.uniform(bounds[0, 0], bounds[0, 1], 1)
        z = (mu - objective_func(x_new)) / sigma
        ei = (mu - objective_func(x_new)) * norm.cdf(z) + sigma * norm.pdf(z)
        X = np.vstack([X, x_new])
        Y.append(objective_func(x_new))
        print(f"Best value so far: {min(Y):.4f}")
    best_idx = np.argmin(Y)
    return X[best_idx]

bounds = np.array([[0.0, 5.0]])
best_x = bayesian_optimization(func, bounds)
print(f"Optimized x: {best_x[0]:.4f}")
```

#### 4. Ant Colony Optimization (ACO)
```python
import numpy as np

def aco_tsp(distances, n_ants=10, n_iter=50, alpha=1, beta=2, evaporation_rate=0.5):
    n_cities = len(distances)
    pheromones = np.ones((n_cities, n_cities)) * 0.1
    best_tour, best_length = None, float('inf')
    for _ in range(n_iter):
        tours = []
        for _ in range(n_ants):
            tour = [0]
            unvisited = set(range(1, n_cities))
            while unvisited:
                current = tour[-1]
                probs = [(pheromones[current, j]**alpha) * ((1 / distances[current, j])**beta) for j in unvisited]
                probs /= np.sum(probs)
                next_city = np.random.choice(list(unvisited), p=probs)
                tour.append(next_city)
                unvisited.remove(next_city)
            tours.append(tour)
        pheromones *= (1 - evaporation_rate)
        for tour in tours:
            length = sum(distances[tour[i], tour[i+1]] for i in range(n_cities-1)) + distances[tour[-1], tour[0]]
            if length < best_length:
                best_tour, best_length = tour, length
            for i in range(n_cities-1):
                pheromones[tour[i], tour[i+1]] += 1 / length
        print(f"Iteration {_}: Best length = {best_length:.2f}")
    return best_tour, best_length

distances = np.array([[0, 10, 15, 20], [10, 0, 35, 25], [15, 35, 0, 30], [20, 25, 30, 0]])
tour, length = aco_tsp(distances)
print(f"Best tour: {tour}, Length: {length}")
```

---

### By Domain (from Fourth Response)
#### 1. Computer Science - Dijkstra’s Algorithm
```python
import heapq

def dijkstra(graph, src):
    n = len(graph)
    dist = [float('inf')] * n
    dist[src] = 0
    pq = [(0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, weight in graph[u]:
            if dist[u] + weight < dist[v]:
                dist[v] = dist[u] + weight
                heapq.heappush(pq, (dist[v], v))
    return dist

graph = [[(1, 4), (2, 8)], [(2, 4)], [(3, 2)], []]
dist = dijkstra(graph, 0)
for i, d in enumerate(dist):
    print(f"Distance to {i}: {d}")
```

#### 2. Embedded Systems - Hill Climbing
```python
import random

def power_usage(duty_cycle):
    return duty_cycle * 10 + (1 - duty_cycle) * 2

def hill_climbing(init, step=0.1, max_iter=100):
    current = init
    for _ in range(max_iter):
        neighbor = current + random.uniform(-step, step)
        if 0 <= neighbor <= 1 and power_usage(neighbor) < power_usage(current):
            current = neighbor
    return current

best_duty = hill_climbing(0.5)
print(f"Optimal duty cycle: {best_duty:.4f}, Power: {power_usage(best_duty):.4f}")
```

#### 3. HPC - Conjugate Gradient
```python
import numpy as np

def conjugate_gradient(A, b, max_iter=10):
    n = len(b)
    x = np.zeros(n)
    r = b.copy()
    p = r.copy()
    r_norm = np.dot(r, r)
    for i in range(max_iter):
        if r_norm < 1e-10:
            break
        Ap = np.dot(A, p)
        alpha = r_norm / np.dot(p, Ap)
        x += alpha * p
        r_new = r - alpha * Ap
        r_norm_new = np.dot(r_new, r_new)
        beta = r_norm_new / r_norm
        p = r_new + beta * p
        r = r_new
        r_norm = r_norm_new
    return x

A = np.array([[4, 1], [1, 3]])
b = np.array([1, 2])
x = conjugate_gradient(A, b)
print(f"Solution: x = {x[0]:.4f}, y = {x[1]:.4f}")
```

#### 4. Computer Vision - Levenberg-Marquardt
```python
def error(c, points):
    x, y, r = c
    return sum((np.sqrt((p[0] - x)**2 + (p[1] - y)**2) - r)**2 for p in points)

def lm_fit(points, init, max_iter=50):
    c = init.copy()
    lambda_ = 0.001
    for i in range(max_iter):
        e = error(c, points)
        for j in range(3):
            c_new = c.copy()
            c_new[j] += 0.01
            de = (error(c_new, points) - e) / 0.01
            c[j] -= lambda_ * de
        if i % 10 == 0:
            print(f"Error: {e:.4f}")
    return c

points = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
init = [0, 0, 1]
result = lm_fit(points, init)
print(f"Circle: x = {result[0]:.4f}, y = {result[1]:.4f}, r = {result[2]:.4f}")
```

#### 5. Edge AI - PSO
```python
import random

def evaluate_size(size):
    return size * 0.1 + 1000 / size

def pso(min_size=10, max_size=100, n_particles=20, max_iter=50):
    positions = [random.uniform(min_size, max_size) for _ in range(n_particles)]
    velocities = [0] * n_particles
    p_best = positions.copy()
    g_best = min(positions, key=evaluate_size)
    for _ in range(max_iter):
        for i in range(n_particles):
            velocities[i] += random.uniform(-1, 1) * (p_best[i] - positions[i]) + 2 * random.uniform(-1, 1) * (g_best - positions[i])
            positions[i] = max(min_size, min(max_size, positions[i] + velocities[i]))
            fitness = evaluate_size(positions[i])
            if fitness < evaluate_size(p_best[i]):
                p_best[i] = positions[i]
            if fitness < evaluate_size(g_best):
                g_best = positions[i]
    return g_best

best_size = pso()
print(f"Optimal layer size: {best_size:.4f}, Fitness: {evaluate_size(best_size):.4f}")
```

#### 6. ML - SGD
```python
import random

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sgd(X, y, max_iter=1000):
    n, d = len(X), len(X[0])
    w = [0] * d
    lr = 0.01
    for _ in range(max_iter):
        i = random.randint(0, n-1)
        pred = sigmoid(sum(w[j] * X[i][j] for j in range(d)))
        error = y[i] - pred
        for j in range(d):
            w[j] += lr * error * X[i][j]
    return w

X = [[1, 2], [2, 3], [3, 1]]
y = [0, 1, 1]
w = sgd(X, y)
print(f"Weights: {w}")
```

#### 7. DL - Adam
```python
import numpy as np

def loss(w, X, y):
    return np.mean([(w[0] * x[0] + w[1] * x[1] - yi)**2 for x, yi in zip(X, y)])

def adam(X, y, max_iter=1000):
    w = np.zeros(2)
    m, v = np.zeros(2), np.zeros(2)
    lr, beta1, beta2, epsilon = 0.001, 0.9, 0.999, 1e-8
    for t in range(1, max_iter + 1):
        grad = np.array([np.mean([2 * (w[0] * x[0] + w[1] * x[1] - yi) * x[0] for x, yi in zip(X, y)]),
                         np.mean([2 * (w[0] * x[0] + w[1] * x[1] - yi) * x[1] for x, yi in zip(X, y)])])
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad**2)
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        w -= lr * m_hat / (np.sqrt(v_hat) + epsilon)
        if t % 100 == 0:
            print(f"Loss: {loss(w, X, y):.4f}")
    return w

X = [[1, 1], [2, 2], [3, 3]]
y = [2, 4, 6]
w = adam(X, y)
print(f"Weights: {w}")
```

---

### By Industry (from Fifth Response)
#### 1. Autonomous Systems - A*
```python
import heapq

def a_star(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    visited = set()
    pq = [(0, start)]
    path = []
    while pq:
        f, (x, y) = heapq.heappop(pq)
        if (x, y) == goal:
            path.append((x, y))
            break
        if (x, y) in visited:
            continue
        visited.add((x, y))
        path.append((x, y))
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and not grid[nx][ny] and (nx, ny) not in visited:
                h = ((nx - goal[0])**2 + (ny - goal[1])**2)**0.5
                heapq.heappush(pq, (f + 1 + h, (nx, ny)))
    return path

grid = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
path = a_star(grid, (0, 0), (2, 2))
for node in path:
    print(node)
```

#### 2. Agriculture - Linear Programming
```python
def maximize_yield(water, yield_per_unit):
    return sum(w * y for w, y in zip(water, yield_per_unit))

def optimize_water(total_water, yield_per_unit, max_iter=100):
    n = len(yield_per_unit)
    water = [total_water / n] * n
    step = 0.1
    for _ in range(max_iter):
        for i in range(n):
            new_water = water[i] + step
            old_water = water[i]
            water[i] = new_water
            if sum(water) > total_water:
                water[i] = old_water
                continue
            if maximize_yield(water, yield_per_unit) < maximize_yield(water[:i] + water[i+1:], yield_per_unit[:i] + yield_per_unit[i+1:]):
                water[i] = old_water
    return water

yield_per_unit = [5, 3, 4]
total_water = 10.0
water_dist = optimize_water(total_water, yield_per_unit)
print(f"Water distribution: {water_dist}")
print(f"Total Yield: {maximize_yield(water_dist, yield_per_unit):.4f}")
```

#### 3. Aerospace/Defense - PSO
```python
import random

def fuel_cost(altitude):
    return altitude * 0.1 + 1000 / altitude

def pso_orbit(min_alt=100, max_alt=1000, n_particles=20, max_iter=50):
    positions = [random.uniform(min_alt, max_alt) for _ in range(n_particles)]
    velocities = [0] * n_particles
    p_best = positions.copy()
    g_best = min(positions, key=fuel_cost)
    for _ in range(max_iter):
        for i in range(n_particles):
            velocities[i] += random.uniform(-1, 1) * (p_best[i] - positions[i]) + 2 * random.uniform(-1, 1) * (g_best - positions[i])
            positions[i] = max(min_alt, min(max_alt, positions[i] + velocities[i]))
            fitness = fuel_cost(positions[i])
            if fitness < fuel_cost(p_best[i]):
                p_best[i] = positions[i]
            if fitness < fuel_cost(g_best):
                g_best = positions[i]
    return g_best

best_altitude = pso_orbit()
print(f"Optimal altitude: {best_altitude:.4f}, Fuel cost: {fuel_cost(best_altitude):.4f}")
```

#### 4. Healthcare - Simulated Annealing
```python
import random
import math

def treatment_effect(dose):
    return -dose**2 + 20*dose - 50

def simulated_annealing(init_dose, max_dose, max_iter=1000):
    current = init_dose
    best = current
    temp = 1000
    for i in range(max_iter):
        next_dose = current + random.uniform(-1, 1)
        if 0 <= next_dose <= max_dose:
            delta = treatment_effect(next_dose) - treatment_effect(current)
            if delta > 0 or math.exp(delta / temp) > random.random():
                current = next_dose
            if treatment_effect(current) > treatment_effect(best):
                best = current
        temp *= 0.99
    return best

best_dose = simulated_annealing(5, 20)
print(f"Optimal dose: {best_dose:.4f}, Effect: {treatment_effect(best_dose):.4f}")
```

#### 5. Smart Cities - ACO
```python
import numpy as np

def traffic_cost(path, costs):
    return sum(costs[path[i]][path[i+1]] for i in range(len(path)-1)) + costs[path[-1]][path[0]]

def aco_traffic(costs, n_ants=10, max_iter=50):
    n = len(costs)
    pheromones = np.ones((n, n)) * 1.0
    best_path, best_cost = None, float('inf')
    for _ in range(max_iter):
        tours = []
        for _ in range(n_ants):
            path = [0]
            unvisited = set(range(1, n))
            while unvisited:
                current = path[-1]
                probs = [pheromones[current, j] / costs[current, j] for j in unvisited]
                probs /= sum(probs)
                next_city = np.random.choice(list(unvisited), p=probs)
                path.append(next_city)
                unvisited.remove(next_city)
            tours.append(path)
        pheromones *= 0.5
        for tour in tours:
            cost = traffic_cost(tour, costs)
            if cost < best_cost:
                best_cost, best_path = cost, tour
            for i in range(len(tour)-1):
                pheromones[tour[i]][tour[i+1]] += 1 / cost
    return best_path

costs = np.array([[0, 4, 8], [4, 0, 2], [8, 2, 0]])
path = aco_traffic(costs)
print(f"Best path: {path}, Cost: {traffic_cost(path, costs)}")
```

#### 6. Retail - Dynamic Programming
```python
def inventory_cost(stock, demand):
    return 10 * (demand - stock) if stock < demand else 2 * (stock - demand)

def dp_inventory(demands, max_stock):
    n = len(demands)
    dp = [[float('inf')] * (max_stock + 1) for _ in range(n + 1)]
    policy = [0] * n
    for i in range(n + 1):
        dp[i][0] = 0 if i == n else dp[i][0]
    for i in range(n-1, -1, -1):
        for s in range(max_stock + 1):
            dp[i][s] = inventory_cost(s, demands[i]) + (dp[i+1][s] if i+1 < n else 0)
            for order in range(max_stock - s + 1):
                next_stock = s + order
                cost = inventory_cost(next_stock, demands[i]) + (dp[i+1][next_stock] if i+1 < n else 0)
                if cost < dp[i][s]:
                    dp[i][s] = cost
                    if i == 0:
                        policy[i] = next_stock
    return policy

demands = [5, 3, 7]
policy = dp_inventory(demands, 10)
print(f"Stock policy: {policy}")
```

#### 7. Manufacturing - Tabu Search
```python
import random

def completion_time(schedule, times):
    n, m = len(schedule), len(times[0])
    machine_end = [0] * m
    for job in schedule:
        start = 0
        for i in range(m):
            start = max(start, machine_end[i])
            machine_end[i] = start + times[job][i]
    return max(machine_end)

def tabu_search(times, max_iter=100):
    n = len(times)
    current = list(range(n))
    best = current.copy()
    tabu_list = []
    for _ in range(max_iter):
        i, j = random.sample(range(n), 2)
        current[i], current[j] = current[j], current[i]
        if current not in tabu_list:
            cost = completion_time(current, times)
            if cost < completion_time(best, times):
                best = current.copy()
            tabu_list.append(current.copy())
            if len(tabu_list) > 10:
                tabu_list.pop(0)
        else:
            current[i], current[j] = current[j], current[i]  # Revert
    return best

times = [[2, 3], [1, 2], [3, 1]]
schedule = tabu_search(times)
print(f"Schedule: {schedule}, Completion time: {completion_time(schedule, times)}")
```

---

