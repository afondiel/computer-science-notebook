# Optimization algorithms using OpenCL

## Overview

These implementations showcase OpenCL’s power for optimization tasks, focusing on parallelizing them to leverage GPU or multi-core CPU capabilities. OpenCL (Open Computing Language) is designed for heterogeneous computing, allowing algorithms to run across diverse hardware like GPUs, CPUs, and FPGAs. We’ll adapt a subset of the previously discussed algorithms—Gradient Descent, Genetic Algorithm, Adam Optimizer, and Simulated Annealing—demonstrating how to offload computation to OpenCL kernels. These examples will include host code (in C) and kernel code (in OpenCL C), with explanations tailored to optimization tasks.

To compile these, you’ll need an OpenCL SDK (e.g., from Intel, AMD, or NVIDIA) and link against the OpenCL library (`-lOpenCL`). The host code manages memory and kernel execution, while kernels perform parallel computations.

---
## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [OpenCL Optimization Techniques](#opencl-optimization-techniques)
- [Quick Notes](#quick-notes)
- [1. Gradient Descent with OpenCL](#1-gradient-descent-with-opencl)
    - [Overview](#overview-1)
    - [Application](#application)
    - [Host Code (C)](#host-code-c)
    - [Explanation](#explanation)
- [2. Genetic Algorithm with OpenCL](#2-genetic-algorithm-with-opencl)
    - [Overview](#overview-2)
    - [Application](#application-1)
    - [Host Code (C)](#host-code-c-1)
    - [Explanation](#explanation-1)
- [3. Adam Optimizer with OpenCL](#3-adam-optimizer-with-opencl)
    - [Overview](#overview-3)
    - [Application](#application-2)
    - [Host Code (C)](#host-code-c-2)
    - [Explanation](#explanation-2)
- [4. Simulated Annealing with OpenCL](#4-simulated-annealing-with-opencl)
    - [Overview](#overview-4)
    - [Application](#application-3)
    - [Host Code (C)](#host-code-c-3)
    - [Explanation](#explanation-3)
---

### Prerequisites
- **OpenCL Setup**: Ensure an OpenCL-compatible device and driver are installed.
- **Compilation**: Use a command like `gcc -o program program.c -lOpenCL` (adjust for your platform).
---

### OpenCL Optimization Techniques
1. **Work-Group Size**: Adjust `global_size` and use `local_size` (e.g., 64 or 256) to match device capabilities. Query `CL_KERNEL_WORK_GROUP_SIZE` with `clGetKernelWorkGroupInfo`.
2. **Memory Coalescing**: Ensure consecutive work-items access consecutive memory (e.g., `x[gid]`), reducing memory latency.
3. **Local Memory**: Use `__local` memory for shared data within work-groups (e.g., in GA crossover), minimizing global memory access.
4. **Vectorization**: Use `float4` or `int4` for SIMD operations on compatible devices (e.g., Intel GPUs).
5. **Asynchronous Execution**: Overlap computation and data transfer using events (not shown here for simplicity).

---

### Quick Notes
- **Scalability**: These examples use 1D NDRanges for simplicity; extend to 2D/3D for larger problems (e.g., image processing).
- **Error Handling**: Omitted for brevity; add checks (e.g., `if (err != CL_SUCCESS)`) in production code.
- **Randomness**: Host-seeded randomness is used; for true parallelism, use OpenCL’s random number extensions or pass a seed array.
- **Performance**: GPUs excel with large `n` (e.g., 1024+); CPUs may benefit from smaller work-groups.

---

### 1. Gradient Descent with OpenCL
#### Overview
Gradient Descent updates parameters iteratively to minimize a function. In OpenCL, we parallelize the gradient computation across multiple data points.

#### Application
Minimizing \( f(x) = x^2 \) for multiple initial points.

#### Host Code (C)
```c
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

const char* kernel_source = 
    "__kernel void gradient_descent(__global float* x, __global float* grad, float lr) {\n"
    "    int gid = get_global_id(0);\n"
    "    grad[gid] = 2.0f * x[gid]; // Gradient of x^2\n"
    "    x[gid] -= lr * grad[gid];\n"
    "}\n";

int main() {
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem x_buf, grad_buf;
    cl_int err;

    // Initialize OpenCL
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    queue = clCreateCommandQueue(context, device, 0, &err);
    program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, &err);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    kernel = clCreateKernel(program, "gradient_descent", &err);

    // Data setup
    int n = 1024; // Number of points
    float x[n], grad[n];
    for (int i = 0; i < n; i++) x[i] = (float)i / 100.0f - 5.0f; // Range [-5, 5]

    // Create buffers
    x_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * n, x, &err);
    grad_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * n, NULL, &err);

    // Set kernel arguments
    float lr = 0.01f;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &x_buf);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &grad_buf);
    clSetKernelArg(kernel, 2, sizeof(float), &lr);

    // Execute kernel
    size_t global_size = n;
    for (int iter = 0; iter < 10; iter++) {
        clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
        clFinish(queue);
        clEnqueueReadBuffer(queue, x_buf, CL_TRUE, 0, sizeof(float) * n, x, 0, NULL, NULL);
        printf("Iteration %d: x[0] = %.4f\n", iter, x[0]);
    }

    // Cleanup
    clReleaseMemObject(x_buf);
    clReleaseMemObject(grad_buf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 0;
}
```

#### Explanation
- **Kernel**: Each work-item computes the gradient and updates one `x` value in parallel.
- **Host**: Manages buffers and iterates the kernel execution.
- **Optimization**: Parallelizes across `n` points, leveraging GPU threads.

---

### 2. Genetic Algorithm with OpenCL
#### Overview
Genetic Algorithms evolve a population. OpenCL parallelizes fitness evaluation and crossover/mutation.

#### Application
Maximizing \( f(x) = -x^2 + 10x \) over \( x \in [0, 10] \).

#### Host Code (C)
```c
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <time.h>

const char* kernel_source = 
    "__kernel void evaluate_fitness(__global int* population, __global float* fitness) {\n"
    "    int gid = get_global_id(0);\n"
    "    int x = population[gid];\n"
    "    fitness[gid] = -x * x + 10.0f * x;\n"
    "}\n"
    "__kernel void crossover_mutation(__global int* population, __global float* fitness, float mutation_rate, uint seed) {\n"
    "    int gid = get_global_id(0);\n"
    "    if (gid % 2 == 0 && gid + 1 < get_global_size(0)) {\n"
    "        int p1 = population[gid], p2 = population[gid + 1];\n"
    "        if ((float)(seed + gid) / 4294967295.0f < 0.8f) {\n"
    "            population[gid] = (p1 + p2) / 2;\n"
    "            population[gid + 1] = p1 + p2 - population[gid];\n"
    "        }\n"
    "        if ((float)(seed + gid + 1) / 4294967295.0f < mutation_rate) {\n"
    "            population[gid] = gid % 11;\n"
    "        }\n"
    "    }\n"
    "}\n";

int main() {
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel eval_kernel, cm_kernel;
    cl_mem pop_buf, fit_buf;
    cl_int err;

    // Initialize OpenCL
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    queue = clCreateCommandQueue(context, device, 0, &err);
    program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, &err);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    eval_kernel = clCreateKernel(program, "evaluate_fitness", &err);
    cm_kernel = clCreateKernel(program, "crossover_mutation", &err);

    // Data setup
    int pop_size = 1024;
    int population[pop_size];
    float fitness[pop_size];
    srand(time(NULL));
    for (int i = 0; i < pop_size; i++) population[i] = rand() % 11;

    // Create buffers
    pop_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * pop_size, population, &err);
    fit_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * pop_size, NULL, &err);

    // Set kernel arguments
    clSetKernelArg(eval_kernel, 0, sizeof(cl_mem), &pop_buf);
    clSetKernelArg(eval_kernel, 1, sizeof(cl_mem), &fit_buf);
    float mutation_rate = 0.1f;
    clSetKernelArg(cm_kernel, 0, sizeof(cl_mem), &pop_buf);
    clSetKernelArg(cm_kernel, 1, sizeof(cl_mem), &fit_buf);
    clSetKernelArg(cm_kernel, 2, sizeof(float), &mutation_rate);

    // Execute kernels
    size_t global_size = pop_size;
    for (int g = 0; g < 50; g++) {
        clSetKernelArg(cm_kernel, 3, sizeof(unsigned int), &(unsigned int){rand()});
        clEnqueueNDRangeKernel(queue, eval_kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
        clEnqueueNDRangeKernel(queue, cm_kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
        clFinish(queue);
        clEnqueueReadBuffer(queue, pop_buf, CL_TRUE, 0, sizeof(int) * pop_size, population, 0, NULL, NULL);
        clEnqueueReadBuffer(queue, fit_buf, CL_TRUE, 0, sizeof(float) * pop_size, fitness, 0, NULL, NULL);
        int best_idx = 0;
        for (int i = 1; i < pop_size; i++) if (fitness[i] > fitness[best_idx]) best_idx = i;
        printf("Generation %d: Best x = %d, Fitness = %.2f\n", g, population[best_idx], fitness[best_idx]);
    }

    // Cleanup
    clReleaseMemObject(pop_buf);
    clReleaseMemObject(fit_buf);
    clReleaseKernel(eval_kernel);
    clReleaseKernel(cm_kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 0;
}
```

#### Explanation
- **Kernels**: `evaluate_fitness` computes fitness in parallel; `crossover_mutation` performs crossover and mutation.
- **Host**: Manages evolution loop, with selection done on CPU for simplicity.
- **Optimization**: Fitness evaluation is parallelized across population size.

---

### 3. Adam Optimizer with OpenCL
#### Overview
Adam adapts learning rates using moment estimates. OpenCL parallelizes updates across dimensions or data points.

#### Application
Minimizing \( f(x, y) = x^2 + y^2 \) for a 2D vector.

#### Host Code (C)
```c
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

const char* kernel_source = 
    "__kernel void adam_update(__global float* params, __global float* m, __global float* v, float lr, float beta1, float beta2, int t) {\n"
    "    int gid = get_global_id(0);\n"
    "    float grad = 2.0f * params[gid]; // Gradient of x^2 + y^2\n"
    "    m[gid] = beta1 * m[gid] + (1.0f - beta1) * grad;\n"
    "    v[gid] = beta2 * v[gid] + (1.0f - beta2) * grad * grad;\n"
    "    float m_hat = m[gid] / (1.0f - pow(beta1, t));\n"
    "    float v_hat = v[gid] / (1.0f - pow(beta2, t));\n"
    "    params[gid] -= lr * m_hat / (sqrt(v_hat) + 1e-8f);\n"
    "}\n";

int main() {
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem params_buf, m_buf, v_buf;
    cl_int err;

    // Initialize OpenCL
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    queue = clCreateCommandQueue(context, device, 0, &err);
    program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, &err);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    kernel = clCreateKernel(program, "adam_update", &err);

    // Data setup
    float params[2] = {5.0f, 5.0f};
    float m[2] = {0.0f, 0.0f}, v[2] = {0.0f, 0.0f};

    // Create buffers
    params_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * 2, params, &err);
    m_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * 2, m, &err);
    v_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * 2, v, &err);

    // Set kernel arguments
    float lr = 0.001f, beta1 = 0.9f, beta2 = 0.999f;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &params_buf);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &m_buf);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &v_buf);
    clSetKernelArg(kernel, 3, sizeof(float), &lr);
    clSetKernelArg(kernel, 4, sizeof(float), &beta1);
    clSetKernelArg(kernel, 5, sizeof(float), &beta2);

    // Execute kernel
    size_t global_size = 2;
    for (int t = 1; t <= 100; t++) {
        clSetKernelArg(kernel, 6, sizeof(int), &t);
        clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
        clFinish(queue);
        clEnqueueReadBuffer(queue, params_buf, CL_TRUE, 0, sizeof(float) * 2, params, 0, NULL, NULL);
        if (t % 10 == 0) printf("Iteration %d: x = %.4f, y = %.4f\n", t, params[0], params[1]);
    }

    // Cleanup
    clReleaseMemObject(params_buf);
    clReleaseMemObject(m_buf);
    clReleaseMemObject(v_buf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 0;
}
```

#### Explanation
- **Kernel**: Updates parameters, momentum, and velocity in parallel for each dimension.
- **Host**: Iterates and manages small-scale data (2D here; scale up for larger problems).
- **Optimization**: Parallelizes updates, efficient for high-dimensional problems on GPUs.

---

### 4. Simulated Annealing with OpenCL
#### Overview
Simulated Annealing explores a solution space. OpenCL parallelizes evaluation of multiple candidate solutions.

#### Application
Maximizing \( f(x) = -x^2 + 20x - 50 \) over \( x \in [0, 20] \).

#### Host Code (C)
```c
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <time.h>

const char* kernel_source = 
    "__kernel void simulated_annealing(__global float* solutions, __global float* fitness, float temp, uint seed) {\n"
    "    int gid = get_global_id(0);\n"
    "    float r = (float)(seed + gid) / 4294967295.0f;\n"
    "    float next = solutions[gid] + (r * 2.0f - 1.0f);\n"
    "    if (next >= 0.0f && next <= 20.0f) {\n"
    "        float current_fit = -solutions[gid] * solutions[gid] + 20.0f * solutions[gid] - 50.0f;\n"
    "        float next_fit = -next * next + 20.0f * next - 50.0f;\n"
    "        float delta = next_fit - current_fit;\n"
    "        if (delta > 0.0f || exp(delta / temp) > r) {\n"
    "            solutions[gid] = next;\n"
    "            fitness[gid] = next_fit;\n"
    "        } else {\n"
    "            fitness[gid] = current_fit;\n"
    "        }\n"
    "    }\n"
    "}\n";

int main() {
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem sol_buf, fit_buf;
    cl_int err;

    // Initialize OpenCL
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    queue = clCreateCommandQueue(context, device, 0, &err);
    program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, &err);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    kernel = clCreateKernel(program, "simulated_annealing", &err);

    // Data setup
    int n = 1024;
    float solutions[n], fitness[n];
    srand(time(NULL));
    for (int i = 0; i < n; i++) solutions[i] = (float)(rand() % 21);

    // Create buffers
    sol_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * n, solutions, &err);
    fit_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * n, NULL, &err);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &sol_buf);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &fit_buf);

    // Execute kernel
    size_t global_size = n;
    float temp = 1000.0f;
    for (int iter = 0; iter < 1000; iter++) {
        clSetKernelArg(kernel, 2, sizeof(float), &temp);
        clSetKernelArg(kernel, 3, sizeof(unsigned int), &(unsigned int){rand()});
        clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
        clFinish(queue);
        clEnqueueReadBuffer(queue, sol_buf, CL_TRUE, 0, sizeof(float) * n, solutions, 0, NULL, NULL);
        clEnqueueReadBuffer(queue, fit_buf, CL_TRUE, 0, sizeof(float) * n, fitness, 0, NULL, NULL);
        int best_idx = 0;
        for (int i = 1; i < n; i++) if (fitness[i] > fitness[best_idx]) best_idx = i;
        if (iter % 100 == 0) printf("Iteration %d: Best x = %.4f, Fitness = %.4f\n", iter, solutions[best_idx], fitness[best_idx]);
        temp *= 0.99f;
    }

    // Cleanup
    clReleaseMemObject(sol_buf);
    clReleaseMemObject(fit_buf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 0;
}
```

#### Explanation
- **Kernel**: Each work-item explores a solution, accepting worse solutions probabilistically.
- **Host**: Manages cooling schedule and selects the best solution.
- **Optimization**: Parallelizes solution exploration, ideal for large search spaces.

