# Parallel Computing (Parallelization) Frameworks

Parallel Computing (Parallelization) frameworks provide tools and infrastructure to execute computational tasks concurrently across multiple processors or systems. Here are some widely used frameworks:

## **Popular Frameworks**

1. **OpenMP**:
   - Designed for shared-memory systems.
   - Uses compiler directives to parallelize code for multicore CPUs.
   - Ideal for scientific computing and numerical simulations.

2. **MPI (Message Passing Interface)**:
   - Focuses on distributed-memory systems.
   - Enables communication between nodes in clusters or grids.
   - Commonly used in high-performance computing.

3. **CUDA**:
   - NVIDIA-specific framework for GPU programming.
   - Optimized for parallelizing tasks on NVIDIA GPUs.
   - Used in AI, deep learning, and graphics-intensive applications.

4. **OpenCL**:
   - Cross-platform framework for heterogeneous systems (CPUs, GPUs, FPGAs).
   - Facilitates parallel execution across diverse hardware.

5. **Intel TBB (Threading Building Blocks)**:
   - Task-based parallelism for multicore CPUs.
   - Provides abstractions like `parallel_for` for dynamic scheduling.

6. **Chapel**:
   - High-level language designed for parallel programming.
   - Supports shared-memory and distributed systems.

7. **RaftLib**:
   - C++ library for stream and dataflow parallel computation.
   - Useful for real-time processing tasks.

8. **IPython Parallel**:
   - Python-based framework supporting multi-core, distributed, and GPU computing.
   - Provides tools for dynamic task creation and load balancing.

## **Comparison Table**

| Framework        | Memory Model       | Hardware Support       | Language Support         | Use Case                     |
|------------------|--------------------|------------------------|--------------------------|------------------------------|
| OpenMP           | Shared-memory      | Multicore CPUs         | C/C++, Fortran           | Scientific computing         |
| MPI              | Distributed-memory | Clusters, Grids        | C/C++, Python            | High-performance computing   |
| **CUDA**             | Shared-memory      | NVIDIA GPUs            | C/C++, Python            | GPU-intensive tasks          |
| **OpenCL**           | Heterogeneous      | CPUs, GPUs, FPGAs      | C/C++, Python            | Cross-platform parallelism   |
| Intel TBB        | Shared-memory      | Multicore CPUs         | C++                      | Dynamic task scheduling      |
| Chapel           | Both               | CPUs, Clusters         | Chapel                   | High-level parallelism       |
| RaftLib          | Shared-memory      | Multicore CPUs         | C++                      | Real-time data processing    |


These frameworks cater to different architectures and use cases, making them suitable for various parallel computing needs.

## Reference
- [1] https://www.heavy.ai/technical-glossary/parallel-computing
- [2] https://www.run.ai/guides/distributed-computing/parallel-computing-with-python
- [3] https://www.alooba.com/skills/concepts/software-engineering/parallel-computing-framework/
- [4] https://github.com/taskflow/awesome-parallel-computing
- [5] https://en.wikipedia.org/wiki/Parallel_computing
- [6] https://docs.frib.msu.edu/daq/newsite/nscldaq-11.4/c10958.html
- [7] https://en.wikipedia.org/wiki/Parallel_programming_model
- [8] https://learn.microsoft.com/en-us/dotnet/standard/parallel-programming/
- [9] https://hpc.llnl.gov/documentation/tutorials/introduction-parallel-computing-tutorial

---

- (Source: [Perplexity](pplx.ai/share))