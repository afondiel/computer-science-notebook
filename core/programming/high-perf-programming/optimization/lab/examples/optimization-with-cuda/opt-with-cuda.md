# Optimization Algorithms using CUDA

## Overview

NVIDIA’s parallel computing platform. CUDA is tailored for NVIDIA GPUs, offering high performance through massive parallelism. 

We’ll adapt a subset of the previously discussed algorithms—Gradient Descent, Genetic Algorithm, Adam Optimizer, and Simulated Annealing—to CUDA, including host code (in C) and kernel code (in CUDA C). 

These examples will demonstrate how to leverage CUDA’s thread hierarchy (grids, blocks, threads) for optimization tasks.

---
## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [CUDA Optimization Techniques](#cuda-optimization-techniques)
- [Quick Notes](#quick-notes)
- [1. Gradient Descent with CUDA](#1-gradient-descent-with-cuda)
    - [Overview](#overview-1)
    - [Application](#application)
    - [Code (CUDA C)](#code-cuda-c)
    - [Explanation](#explanation)
- [2. Genetic Algorithm with CUDA](#2-genetic-algorithm-with-cuda)
    - [Overview](#overview-2)
    - [Application](#application-1)
    - [Code (CUDA C)](#code-cuda-c-1)
    - [Explanation](#explanation-1)
- [3. Adam Optimizer with CUDA](#3-adam-optimizer-with-cuda)
    - [Overview](#overview-3)
    - [Application](#application-2)
    - [Code (CUDA C)](#code-cuda-c-2)
    - [Explanation](#explanation-2)
- [4. Simulated Annealing with CUDA](#4-simulated-annealing-with-cuda)
    - [Overview](#overview-4)
    - [Application](#application-3)
    - [Code (CUDA C)](#code-cuda-c-3)
    - [Explanation](#explanation-3)
---

### Prerequisites
- **CUDA Setup**: Ensure an NVIDIA GPU and CUDA Toolkit are installed.
- **Compilation**: Adjust for your architecture (e.g., `-arch=sm_75` for Turing GPUs).

To compile these, you’ll need the NVIDIA CUDA Toolkit. Use a command like:
```bash
nvcc -o program program.cu
```

---
### CUDA Optimization Techniques
1. **Thread Hierarchy**: Use `threads_per_block = 256` or 512 (multiples of 32 for warp size); adjust `blocks` dynamically.
2. **Shared Memory**: Use `__shared__` memory for local data (e.g., in GA crossover) to reduce global memory access.
3. **Coalesced Access**: Ensure consecutive threads access consecutive memory (e.g., `x[idx]`).
4. **Stream Concurrency**: Use CUDA streams for overlapping kernel execution and data transfer (not shown here).
5. **Warp Divergence**: Minimize branching in kernels (e.g., `if` conditions) to avoid thread divergence.

---

### Quick Notes
- **Scalability**: Examples use 1D grids; extend to 2D/3D for larger problems (e.g., image-based optimization).
- **Error Handling**: Omitted for brevity; use `cudaGetLastError()` and `cudaDeviceSynchronize()` checks in production.
- **Randomness**: `cuRAND` provides per-thread randomness, more efficient than host-seeded approaches.
- **Performance**: Optimal for large `n` (e.g., 1024+); tune block sizes via profiling (e.g., NVIDIA Nsight).
---

### 1. Gradient Descent with CUDA
#### Overview
Parallelize gradient computation and parameter updates across multiple initial points.

#### Application
Minimizing \( f(x) = x^2 \) for multiple starting points.

#### Code (CUDA C)
```c
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void gradient_descent_kernel(float* x, float lr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float grad = 2.0f * x[idx]; // Gradient of x^2
    x[idx] -= lr * grad;
}

int main() {
    int n = 1024; // Number of points
    float *x, *d_x;
    float lr = 0.01f;

    // Allocate host memory
    x = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) x[i] = (float)i / 100.0f - 5.0f; // Range [-5, 5]

    // Allocate device memory
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);

    // Kernel launch
    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    for (int iter = 0; iter < 10; iter++) {
        gradient_descent_kernel<<<blocks, threads_per_block>>>(d_x, lr);
        cudaDeviceSynchronize();
        cudaMemcpy(x, d_x, n * sizeof(float), cudaMemcpyDeviceToHost);
        printf("Iteration %d: x[0] = %.4f\n", iter, x[0]);
    }

    // Cleanup
    cudaFree(d_x);
    free(x);
    return 0;
}
```

#### Explanation
- **Kernel**: Each thread updates one `x` value.
- **Host**: Manages memory and iterates kernel execution.
- **Optimization**: Parallelizes across `n` points, utilizing GPU threads.

---

### 2. Genetic Algorithm with CUDA
#### Overview
Parallelize fitness evaluation and crossover/mutation across a population.

#### Application
Maximizing \( f(x) = -x^2 + 10x \) over \( x \in [0, 10] \).

#### Code (CUDA C)
```c
#include <stdio.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void setup_rng(curandState* state, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &state[idx]);
}

__global__ void evaluate_fitness_kernel(int* population, float* fitness, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int x = population[idx];
        fitness[idx] = -x * x + 10.0f * x;
    }
}

__global__ void crossover_mutation_kernel(int* population, float* fitness, curandState* state, float mutation_rate, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n - 1 && idx % 2 == 0) {
        curandState local_state = state[idx];
        float r = curand_uniform(&local_state);
        int p1 = population[idx], p2 = population[idx + 1];
        if (r < 0.8f) {
            population[idx] = (p1 + p2) / 2;
            population[idx + 1] = p1 + p2 - population[idx];
        }
        r = curand_uniform(&local_state);
        if (r < mutation_rate) population[idx] = (int)(curand_uniform(&local_state) * 11);
        state[idx] = local_state;
    }
}

int main() {
    int pop_size = 1024;
    int *population, *d_population;
    float *fitness, *d_fitness;
    curandState *d_state;

    // Allocate host memory
    population = (int*)malloc(pop_size * sizeof(int));
    fitness = (float*)malloc(pop_size * sizeof(float));
    for (int i = 0; i < pop_size; i++) population[i] = rand() % 11;

    // Allocate device memory
    cudaMalloc(&d_population, pop_size * sizeof(int));
    cudaMalloc(&d_fitness, pop_size * sizeof(float));
    cudaMalloc(&d_state, pop_size * sizeof(curandState));
    cudaMemcpy(d_population, population, pop_size * sizeof(int), cudaMemcpyHostToDevice);

    // Setup RNG
    int threads_per_block = 256;
    int blocks = (pop_size + threads_per_block - 1) / threads_per_block;
    setup_rng<<<blocks, threads_per_block>>>(d_state, time(NULL));
    cudaDeviceSynchronize();

    // Evolution loop
    float mutation_rate = 0.1f;
    for (int g = 0; g < 50; g++) {
        evaluate_fitness_kernel<<<blocks, threads_per_block>>>(d_population, d_fitness, pop_size);
        crossover_mutation_kernel<<<blocks, threads_per_block>>>(d_population, d_fitness, d_state, mutation_rate, pop_size);
        cudaDeviceSynchronize();
        cudaMemcpy(population, d_population, pop_size * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(fitness, d_fitness, pop_size * sizeof(float), cudaMemcpyDeviceToHost);
        int best_idx = 0;
        for (int i = 1; i < pop_size; i++) if (fitness[i] > fitness[best_idx]) best_idx = i;
        printf("Generation %d: Best x = %d, Fitness = %.2f\n", g, population[best_idx], fitness[best_idx]);
    }

    // Cleanup
    cudaFree(d_population);
    cudaFree(d_fitness);
    cudaFree(d_state);
    free(population);
    free(fitness);
    return 0;
}
```

#### Explanation
- **Kernels**: `evaluate_fitness_kernel` computes fitness; `crossover_mutation_kernel` performs crossover and mutation using `cuRAND`.
- **Host**: Manages evolution, with selection on CPU for simplicity.
- **Optimization**: Fitness evaluation is fully parallelized.

---

### 3. Adam Optimizer with CUDA
#### Overview
Parallelize parameter updates across dimensions or data points.

#### Application
Minimizing \( f(x, y) = x^2 + y^2 \) for a 2D vector.

#### Code (CUDA C)
```c
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void adam_kernel(float* params, float* m, float* v, float lr, float beta1, float beta2, int t) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 2) {
        float grad = 2.0f * params[idx];
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * grad;
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * grad * grad;
        float m_hat = m[idx] / (1.0f - powf(beta1, t));
        float v_hat = v[idx] / (1.0f - powf(beta2, t));
        params[idx] -= lr * m_hat / (sqrtf(v_hat) + 1e-8f);
    }
}

int main() {
    float params[2] = {5.0f, 5.0f};
    float m[2] = {0.0f, 0.0f}, v[2] = {0.0f, 0.0f};
    float *d_params, *d_m, *d_v;

    // Allocate device memory
    cudaMalloc(&d_params, 2 * sizeof(float));
    cudaMalloc(&d_m, 2 * sizeof(float));
    cudaMalloc(&d_v, 2 * sizeof(float));
    cudaMemcpy(d_params, params, 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, m, 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, 2 * sizeof(float), cudaMemcpyHostToDevice);

    // Kernel launch
    float lr = 0.001f, beta1 = 0.9f, beta2 = 0.999f;
    int threads_per_block = 2;
    int blocks = 1;
    for (int t = 1; t <= 100; t++) {
        adam_kernel<<<blocks, threads_per_block>>>(d_params, d_m, d_v, lr, beta1, beta2, t);
        cudaDeviceSynchronize();
        cudaMemcpy(params, d_params, 2 * sizeof(float), cudaMemcpyDeviceToHost);
        if (t % 10 == 0) printf("Iteration %d: x = %.4f, y = %.4f\n", t, params[0], params[1]);
    }

    // Cleanup
    cudaFree(d_params);
    cudaFree(d_m);
    cudaFree(d_v);
    return 0;
}
```

#### Explanation
- **Kernel**: Updates parameters, momentum, and velocity for each dimension.
- **Host**: Manages small-scale data (2D); scale up for larger problems.
- **Optimization**: Parallelizes updates, efficient for high-dimensional optimization.

---

### 4. Simulated Annealing with CUDA
#### Overview
Parallelize exploration of multiple candidate solutions.

#### Application
Maximizing \( f(x) = -x^2 + 20x - 50 \) over \( x \in [0, 20] \).

#### Code (CUDA C)
```c
#include <stdio.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void setup_rng(curandState* state, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &state[idx]);
}

__global__ void sa_kernel(float* solutions, float* fitness, curandState* state, float temp, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curandState local_state = state[idx];
        float r = curand_uniform(&local_state);
        float next = solutions[idx] + (r * 2.0f - 1.0f);
        if (next >= 0.0f && next <= 20.0f) {
            float current_fit = -solutions[idx] * solutions[idx] + 20.0f * solutions[idx] - 50.0f;
            float next_fit = -next * next + 20.0f * next - 50.0f;
            float delta = next_fit - current_fit;
            if (delta > 0.0f || expf(delta / temp) > curand_uniform(&local_state)) {
                solutions[idx] = next;
                fitness[idx] = next_fit;
            } else {
                fitness[idx] = current_fit;
            }
        }
        state[idx] = local_state;
    }
}

int main() {
    int n = 1024;
    float *solutions, *fitness, *d_solutions, *d_fitness;
    curandState *d_state;

    // Allocate host memory
    solutions = (float*)malloc(n * sizeof(float));
    fitness = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) solutions[i] = (float)(rand() % 21);

    // Allocate device memory
    cudaMalloc(&d_solutions, n * sizeof(float));
    cudaMalloc(&d_fitness, n * sizeof(float));
    cudaMalloc(&d_state, n * sizeof(curandState));
    cudaMemcpy(d_solutions, solutions, n * sizeof(float), cudaMemcpyHostToDevice);

    // Setup RNG
    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    setup_rng<<<blocks, threads_per_block>>>(d_state, time(NULL));
    cudaDeviceSynchronize();

    // SA loop
    float temp = 1000.0f;
    for (int iter = 0; iter < 1000; iter++) {
        sa_kernel<<<blocks, threads_per_block>>>(d_solutions, d_fitness, d_state, temp, n);
        cudaDeviceSynchronize();
        cudaMemcpy(solutions, d_solutions, n * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(fitness, d_fitness, n * sizeof(float), cudaMemcpyDeviceToHost);
        int best_idx = 0;
        for (int i = 1; i < n; i++) if (fitness[i] > fitness[best_idx]) best_idx = i;
        if (iter % 100 == 0) printf("Iteration %d: Best x = %.4f, Fitness = %.4f\n", iter, solutions[best_idx], fitness[best_idx]);
        temp *= 0.99f;
    }

    // Cleanup
    cudaFree(d_solutions);
    cudaFree(d_fitness);
    cudaFree(d_state);
    free(solutions);
    free(fitness);
    return 0;
}
```

#### Explanation
- **Kernel**: Each thread explores a solution using `cuRAND` for randomness.
- **Host**: Manages cooling and selects the best solution.
- **Optimization**: Parallelizes solution evaluation across `n` threads.

---

