# Core optimization algorithms in C

## Overview

The following implementations cover key examples. Since C lacks the high-level abstractions of C++ (e.g., STL containers like `vector`, `priority_queue`) and Python (e.g., NumPy), We use basic arrays, manual memory management, and simpler constructs. This maintain functionality while adapting to C’s constraints, focusing on clarity and correctness.

---
## Table of Contents

- [Overview](#overview)
- [General Comprehensive List](#general-comprehensive-list)
    - [Gradient Descent](#1-gradient-descent)
    - [Genetic Algorithm](#2-genetic-algorithm)
- [Advanced Examples](#advanced-examples)
    - [Adam Optimizer](#1-adam-optimizer)
    - [CMA-ES (Simplified)](#2-cma-es-simplified)
- [By Domain](#by-domain)
    - [Computer Science - Dijkstra’s Algorithm](#1-computer-science---dijkstras-algorithm)
    - [Embedded Systems - Hill Climbing](#2-embedded-systems---hill-climbing)
    - [Computer Vision - Levenberg-Marquardt (Simplified)](#3-computer-vision---levenberg-marquardt-simplified)
- [By Industry](#by-industry)
    - [Autonomous Systems - A*](#1-autonomous-systems---a)
    - [Healthcare - Simulated Annealing](#2-healthcare---simulated-annealing)
    - [Manufacturing - Tabu Search](#3-manufacturing---tabu-search)

---

### Notes
- **Randomness**: Uses `rand()` with `srand(time(NULL))` for simplicity; a better RNG could be implemented.
- **Memory**: Static arrays replace dynamic allocation for brevity; real applications would use `malloc`/`free`.
- **Libraries**: C lacks linear algebra libraries like Eigen, so matrix operations (e.g., CMA-ES) are simplified.
- **Limitations**: Some algorithms (e.g., ACO, Bayesian Optimization) are omitted due to complexity in pure C without libraries; they’d require custom data structures.

---

### General Comprehensive List
#### 1. Gradient Descent
```c
#include <stdio.h>
#include <math.h>

double objective_function(double x) {
    return x * x + 2 * x + 1;
}

double gradient(double x) {
    return 2 * x + 2;
}

double gradient_descent(double x_init, double learning_rate, int max_iter) {
    double x = x_init;
    for (int i = 0; i < max_iter; i++) {
        x -= learning_rate * gradient(x);
        printf("Iteration %d: x = %.4f, f(x) = %.4f\n", i, x, objective_function(x));
    }
    return x;
}

int main() {
    double x_init = 5.0;
    double learning_rate = 0.1;
    int max_iter = 10;
    double result = gradient_descent(x_init, learning_rate, max_iter);
    printf("Optimized x: %.4f\n", result);
    return 0;
}
```

#### 2. Genetic Algorithm
```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

double fitness(int x) {
    return -x * x + 10 * x;
}

void genetic_algorithm(int pop_size, int max_gen, double mutation_rate, int* best) {
    int population[pop_size];
    double fitness_values[pop_size];
    int new_population[pop_size];
    srand(time(NULL));

    for (int i = 0; i < pop_size; i++) population[i] = rand() % 11;

    for (int g = 0; g < max_gen; g++) {
        double total_fitness = 0;
        for (int i = 0; i < pop_size; i++) {
            fitness_values[i] = fitness(population[i]);
            total_fitness += fitness_values[i];
        }
        for (int i = 0; i < pop_size; i++) {
            double r = (double)rand() / RAND_MAX * total_fitness;
            double s = 0;
            for (int j = 0; j < pop_size; j++) {
                s += fitness_values[j];
                if (s >= r) {
                    new_population[i] = population[j];
                    break;
                }
            }
        }
        for (int i = 0; i < pop_size - 1; i += 2) {
            if ((double)rand() / RAND_MAX < 0.8) {
                int p1 = new_population[i], p2 = new_population[i + 1];
                new_population[i] = (p1 + p2) / 2;
                new_population[i + 1] = p1 + p2 - new_population[i];
            }
        }
        for (int i = 0; i < pop_size; i++) {
            if ((double)rand() / RAND_MAX < mutation_rate) new_population[i] = rand() % 11;
            population[i] = new_population[i];
        }
        int max_idx = 0;
        for (int i = 1; i < pop_size; i++) if (fitness(population[i]) > fitness(population[max_idx])) max_idx = i;
        printf("Generation %d: Best x = %d, Fitness = %.2f\n", g, population[max_idx], fitness(population[max_idx]));
        *best = population[max_idx];
    }
}

int main() {
    int best;
    genetic_algorithm(20, 50, 0.1, &best);
    printf("Best solution: x = %d, Fitness = %.2f\n", best, fitness(best));
    return 0;
}
```

---

### Advanced Examples
#### 1. Adam Optimizer
```c
#include <stdio.h>
#include <math.h>

void gradient(double* v, double* g) {
    g[0] = 2 * v[0];
    g[1] = 2 * v[1];
}

double objective_function(double* v) {
    return v[0] * v[0] + v[1] * v[1];
}

void adam_optimizer(double* init, double lr, double beta1, double beta2, int max_iter, double* result) {
    double params[2] = {init[0], init[1]};
    double m[2] = {0, 0}, v[2] = {0, 0}, g[2];
    double epsilon = 1e-8;
    for (int t = 1; t <= max_iter; t++) {
        gradient(params, g);
        for (int i = 0; i < 2; i++) {
            m[i] = beta1 * m[i] + (1 - beta1) * g[i];
            v[i] = beta2 * v[i] + (1 - beta2) * g[i] * g[i];
            double m_hat = m[i] / (1 - pow(beta1, t));
            double v_hat = v[i] / (1 - pow(beta2, t));
            params[i] -= lr * m_hat / (sqrt(v_hat) + epsilon);
        }
        if (t % 10 == 0) printf("Iteration %d: x = %.4f, y = %.4f\n", t, params[0], params[1]);
    }
    result[0] = params[0];
    result[1] = params[1];
}

int main() {
    double init[2] = {5.0, 5.0}, result[2];
    adam_optimizer(init, 0.001, 0.9, 0.999, 100, result);
    printf("Optimized: x = %.4f, y = %.4f, f(x, y) = %.4f\n", result[0], result[1], objective_function(result));
    return 0;
}
```

#### 2. CMA-ES (Simplified)
```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

double sphere(double* x, int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) sum += x[i] * x[i];
    return sum;
}

void cma_es(double* initial_mean, double sigma, int pop_size, int max_iter, int n, double* result) {
    double mean[n], population[pop_size][n], fitness[pop_size];
    double cov[n][n];
    srand(time(NULL));
    for (int i = 0; i < n; i++) mean[i] = initial_mean[i];
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) cov[i][j] = (i == j) ? sigma * sigma : 0;

    for (int iter = 0; iter < max_iter; iter++) {
        for (int i = 0; i < pop_size; i++) {
            for (int j = 0; j < n; j++) population[i][j] = mean[j] + ((double)rand() / RAND_MAX - 0.5) * sigma;
            fitness[i] = sphere(population[i], n);
        }
        int elite_size = pop_size / 2;
        for (int i = 0; i < n; i++) {
            mean[i] = 0;
            for (int j = 0; j < elite_size; j++) mean[i] += population[j][i];
            mean[i] /= elite_size;
        }
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) cov[i][j] = 0;
        for (int k = 0; k < elite_size; k++) for (int i = 0; i < n; i++) for (int j = 0; j < n; j++)
            cov[i][j] += (population[k][i] - mean[i]) * (population[k][j] - mean[j]) / elite_size;
        printf("Iteration %d: Best fitness = %.4f\n", iter, fitness[0]);
    }
    for (int i = 0; i < n; i++) result[i] = mean[i];
}

int main() {
    double initial_mean[2] = {5.0, 5.0}, result[2];
    cma_es(initial_mean, 0.5, 10, 100, 2, result);
    printf("Optimized solution: [%.4f, %.4f]\n", result[0], result[1]);
    return 0;
}
```

---

### By Domain
#### 1. Computer Science - Dijkstra’s Algorithm
```c
#include <stdio.h>
#include <limits.h>

#define V 4

void dijkstra(int graph[V][V], int src, int* dist) {
    int visited[V] = {0};
    for (int i = 0; i < V; i++) dist[i] = INT_MAX;
    dist[src] = 0;

    for (int count = 0; count < V - 1; count++) {
        int min = INT_MAX, u;
        for (int v = 0; v < V; v++) if (!visited[v] && dist[v] <= min) min = dist[v], u = v;
        visited[u] = 1;
        for (int v = 0; v < V; v++) if (!visited[v] && graph[u][v] && dist[u] != INT_MAX && dist[u] + graph[u][v] < dist[v])
            dist[v] = dist[u] + graph[u][v];
    }
}

int main() {
    int graph[V][V] = {{0, 4, 8, 0}, {4, 0, 4, 0}, {8, 4, 0, 2}, {0, 0, 2, 0}};
    int dist[V];
    dijkstra(graph, 0, dist);
    for (int i = 0; i < V; i++) printf("Distance to %d: %d\n", i, dist[i]);
    return 0;
}
```

#### 2. Embedded Systems - Hill Climbing
```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

double power_usage(double duty_cycle) {
    return duty_cycle * 10 + (1 - duty_cycle) * 2;
}

double hill_climbing(double init, double step, int max_iter) {
    double current = init;
    srand(time(NULL));
    for (int i = 0; i < max_iter; i++) {
        double neighbor = current + ((double)rand() / RAND_MAX * 2 - 1) * step;
        if (neighbor >= 0 && neighbor <= 1 && power_usage(neighbor) < power_usage(current)) current = neighbor;
    }
    return current;
}

int main() {
    double best_duty = hill_climbing(0.5, 0.1, 100);
    printf("Optimal duty cycle: %.4f, Power: %.4f\n", best_duty, power_usage(best_duty));
    return 0;
}
```

#### 3. Computer Vision - Levenberg-Marquardt (Simplified)
```c
#include <stdio.h>
#include <math.h>

struct Circle { double x, y, r; };

double error(struct Circle c, double points[][2], int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) {
        double dist = sqrt((points[i][0] - c.x) * (points[i][0] - c.x) + (points[i][1] - c.y) * (points[i][1] - c.y));
        sum += (dist - c.r) * (dist - c.r);
    }
    return sum;
}

struct Circle lm_fit(double points[][2], int n, struct Circle init, int max_iter) {
    struct Circle c = init;
    double lambda = 0.001;
    for (int i = 0; i < max_iter; i++) {
        double e = error(c, points, n);
        struct Circle c_dx = {c.x + 0.01, c.y, c.r}, c_dy = {c.x, c.y + 0.01, c.r}, c_dr = {c.x, c.y, c.r + 0.01};
        double dx = (error(c_dx, points, n) - e) / 0.01;
        double dy = (error(c_dy, points, n) - e) / 0.01;
        double dr = (error(c_dr, points, n) - e) / 0.01;
        c.x -= lambda * dx;
        c.y -= lambda * dy;
        c.r -= lambda * dr;
        if (i % 10 == 0) printf("Error: %.4f\n", e);
    }
    return c;
}

int main() {
    double points[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
    struct Circle init = {0, 0, 1};
    struct Circle result = lm_fit(points, 4, init, 50);
    printf("Circle: x = %.4f, y = %.4f, r = %.4f\n", result.x, result.y, result.r);
    return 0;
}
```

---

### By Industry
#### 1. Autonomous Systems - A*
```c
#include <stdio.h>
#include <math.h>

struct Node {
    int x, y;
    double g, h;
};

double heuristic(int x1, int y1, int x2, int y2) {
    return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}

void a_star(int grid[3][3], int start_x, int start_y, int goal_x, int goal_y, int* path_x, int* path_y, int* path_len) {
    int visited[3][3] = {0};
    struct Node pq[100];
    int pq_size = 1, path_size = 0;
    pq[0] = (struct Node){start_x, start_y, 0, heuristic(start_x, start_y, goal_x, goal_y)};

    while (pq_size > 0) {
        struct Node current = pq[0];
        for (int i = 1; i < pq_size; i++) pq[i-1] = pq[i];
        pq_size--;
        if (current.x == goal_x && current.y == goal_y) {
            path_x[path_size] = current.x;
            path_y[path_size] = current.y;
            *path_len = ++path_size;
            break;
        }
        if (visited[current.x][current.y]) continue;
        visited[current.x][current.y] = 1;
        path_x[path_size] = current.x;
        path_y[path_size] = current.y;
        path_size++;

        int dx[] = {0, 1, 0, -1}, dy[] = {1, 0, -1, 0};
        for (int i = 0; i < 4; i++) {
            int nx = current.x + dx[i], ny = current.y + dy[i];
            if (nx >= 0 && nx < 3 && ny >= 0 && ny < 3 && !grid[nx][ny] && !visited[nx][ny]) {
                double h = heuristic(nx, ny, goal_x, goal_y);
                pq[pq_size++] = (struct Node){nx, ny, current.g + 1, h};
            }
        }
    }
}

int main() {
    int grid[3][3] = {{0, 0, 0}, {0, 1, 0}, {0, 0, 0}};
    int path_x[10], path_y[10], path_len;
    a_star(grid, 0, 0, 2, 2, path_x, path_y, &path_len);
    for (int i = 0; i < path_len; i++) printf("(%d, %d)\n", path_x[i], path_y[i]);
    return 0;
}
```

#### 2. Healthcare - Simulated Annealing
```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

double treatment_effect(double dose) {
    return -dose * dose + 20 * dose - 50;
}

double simulated_annealing(double init_dose, double max_dose, int max_iter) {
    double current = init_dose, best = current, temp = 1000;
    srand(time(NULL));
    for (int i = 0; i < max_iter; i++) {
        double next = current + ((double)rand() / RAND_MAX * 2 - 1);
        if (next >= 0 && next <= max_dose) {
            double delta = treatment_effect(next) - treatment_effect(current);
            if (delta > 0 || exp(delta / temp) > (double)rand() / RAND_MAX) current = next;
            if (treatment_effect(current) > treatment_effect(best)) best = current;
        }
        temp *= 0.99;
    }
    return best;
}

int main() {
    double best_dose = simulated_annealing(5, 20, 1000);
    printf("Optimal dose: %.4f, Effect: %.4f\n", best_dose, treatment_effect(best_dose));
    return 0;
}
```

#### 3. Manufacturing - Tabu Search
```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int completion_time(int* schedule, int times[][2], int n) {
    int machine_end[2] = {0, 0};
    for (int i = 0; i < n; i++) {
        int start = 0;
        for (int j = 0; j < 2; j++) {
            start = (start > machine_end[j]) ? start : machine_end[j];
            machine_end[j] = start + times[schedule[i]][j];
        }
    }
    return (machine_end[0] > machine_end[1]) ? machine_end[0] : machine_end[1];
}

void tabu_search(int times[][2], int n, int max_iter, int* best) {
    int current[n], tabu_list[10][3];
    for (int i = 0; i < n; i++) current[i] = i;
    for (int i = 0; i < n; i++) best[i] = current[i];
    int tabu_size = 0;
    srand(time(NULL));

    for (int iter = 0; iter < max_iter; iter++) {
        int i = rand() % n, j = rand() % n;
        if (i != j) {
            int temp = current[i];
            current[i] = current[j];
            current[j] = temp;
            int in_tabu = 0;
            for (int k = 0; k < tabu_size; k++) {
                in_tabu = 1;
                for (int l = 0; l < n; l++) if (current[l] != tabu_list[k][l]) { in_tabu = 0; break; }
                if (in_tabu) break;
            }
            if (!in_tabu) {
                int cost = completion_time(current, times, n);
                if (cost < completion_time(best, times, n)) for (int k = 0; k < n; k++) best[k] = current[k];
                if (tabu_size < 10) {
                    for (int k = 0; k < n; k++) tabu_list[tabu_size][k] = current[k];
                    tabu_size++;
                }
            } else {
                current[j] = current[i];
                current[i] = temp;
            }
        }
    }
}

int main() {
    int times[3][2] = {{2, 3}, {1, 2}, {3, 1}};
    int best[3];
    tabu_search(times, 3, 100, best);
    printf("Schedule: ");
    for (int i = 0; i < 3; i++) printf("%d ", best[i]);
    printf("\nCompletion time: %d\n", completion_time(best, times, 3));
    return 0;
}
```

---


