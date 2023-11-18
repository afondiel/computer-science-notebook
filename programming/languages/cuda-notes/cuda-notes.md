# CUDA - Notes

## Table of Contents (ToC)
- [Overview](#overview)
- [Applications](#applications)
- [Tools \& Frameworks](#tools--frameworks)
- [Hello World!](#hello-world)
- [References](#references)


## Overview

CUDA (Compute Unified Device Architecture) is a parallel computing platform and programming model developed by NVIDIA, enabling the use of GPUs for general-purpose computing.

## Applications

- GPU Acceleration: Utilizing CUDA for parallel processing to accelerate computationally intensive tasks.
- Deep Learning: Training and inference of neural networks benefit from CUDA-enabled GPUs.
- Scientific Computing: Performing complex simulations and numerical computations with GPU acceleration.
- Image and Signal Processing: Speeding up image and signal processing algorithms using CUDA.
- High-Performance Computing (HPC): Enhancing performance in various scientific and engineering applications.

## Tools & Frameworks

- NVIDIA CUDA Toolkit: Software development kit for building GPU-accelerated applications.
- cuDNN (CUDA Deep Neural Network library): Optimized GPU-accelerated library for deep neural networks.
- Nsight: NVIDIA's suite of debugging and profiling tools for CUDA development.
- PyCUDA: Python wrapper for CUDA, allowing GPU programming in Python.
- TensorFlow with GPU support: Deep learning framework leveraging CUDA for accelerated training.
  
## Hello World!

```python
# Sample code using PyCUDA for a basic CUDA program
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# CUDA kernel
cuda_kernel = SourceModule("""
__global__ void hello_cuda() {
    printf("Hello from CUDA!\\n");
}
""")

# Run the CUDA kernel
hello_cuda = cuda_kernel.get_function("hello_cuda")
hello_cuda(block=(1, 1, 1), grid=(1, 1))
```

## References

Documentation

- [NVIDIA CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [CUDA C++ Programming Guide Release 12.3 - NVIDIA](https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf)
- [PyCUDA Documentation](https://documen.tician.de/pycuda/)

- [CUDA Wikipedia](https://en.wikipedia.org/wiki/CUDA)

Lectures & Tutorials: 

- [CUDA Tutorial](https://cuda-tutorial.readthedocs.io/en/latest/)
- [geeksforgeeks.org](https://www.geeksforgeeks.org/introduction-to-cuda-programming/)
- [Intro to CUDA (part 1): High Level Concepts](https://www.youtube.com/watch?v=4APkMJdiudU&list=PLC6u37oFvF40BAm7gwVP7uDdzmW83yHPe)
- [CUDA Crash Course: Vector Addition - Nick CoffeeBeforeArch](https://www.youtube.com/watch?v=2NgpYFdsduY&list=PLxNPSjHT5qvtYRVdNN1yDcdSl39uHV_sU)
- [NVIDIA Developer - CUDA training and updates](https://www.youtube.com/watch?v=Iuy_RAvguBM&list=PL5B692fm6--vScfBaxgY89IRWFzDt0Khm)

NVIDIA DLI Online courses: 

- [An Even Easier Introduction to CUDA - NVIDIA DLI](https://courses.nvidia.com/courses/course-v1:DLI+T-AC-01+V1/)
  - [Blog post - An Easy Introduction to CUDA C and C++ - 2012](https://developer.nvidia.com/blog/easy-introduction-cuda-c-and-c/)
  - [Blog post - An Even Easier Introduction to CUDA 2017](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)
  - [Notebook](./lab/notebook/An_Even_Easier_Introduction_to_CUDA.ipynb)
- [Getting Started with Accelerated Computing in CUDA C/C++](https://courses.nvidia.com/courses/course-v1:DLI+S-AC-04+V1/)
  - [Blog post - CUDA Refresher - Getting started with CUDA - 2020](https://developer.nvidia.com/blog/cuda-refresher-getting-started-with-cuda/)
  - [Blog post - GPU Accelerated Computing with C and C++ - 2017](https://developer.nvidia.com/how-to-cuda-c-cpp)
  - [Notebook - @TODO](#)
- [Fundamentals of Accelerated Computing with CUDA C/C++ NVIDIA DLI - INSTRUCTOR-LED ](https://courses.nvidia.com/courses/course-v1:DLI+C-AC-01+V1/)
  - [Blog post - GPU Accelerated Computing with C and C++ - 2017](https://developer.nvidia.com/how-to-cuda-c-cpp)  - [Notebook](#)
- [Fundamentals of Accelerated Computing with CUDA Python NVIDIA DLI](https://courses.nvidia.com/courses/course-v1:DLI+C-AC-02+V1/)
  - [Blog post - @TODO](#)
  - [Notebook - TODO](#)
- [Accelerating CUDA C++ Applications with Concurrent Streams NVIDIA DLI](#)
  - [Blog post - @TODO](#)
  - [Notebook - @TODO](#)
- [Scaling Workloads Across Multiple GPUs with CUDA C++ NVIDIA DLI](#)
  - [Blog post - @]()
  - [Notebook - @TODO](#)

Udacity: 

- [Udacity CS344: Intro to Parallel Programming](https://developer.nvidia.com/udacity-cs344-intro-parallel-programming)

Books: 

- [CUDA - cs-books](https://github.com/afondiel/cs-books/tree/main/computer-science/programming/cuda)
- [CUDA Books archive - NVIDIA](https://developer.nvidia.com/cuda-books-archive)

