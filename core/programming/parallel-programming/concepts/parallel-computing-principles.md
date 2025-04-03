# Parallel Programming vs Parallel Computing vs Supercomputing principles   

A structured breakdown of key laws, theorems, and principles governing parallel programming, parallel computing, and supercomputing.

## Table of Contents

1. [Amdahl's Law](#1-amdahls-law)  
2. [Gustafson's Law (Scaled Speedup)](#2-gustafsons-law-scaled-speedup)  
3. [Sun and Ni's Memory-Bounded Speedup](#3-sun-and-nis-memory-bounded-speedup)  
4. [Work and Span Laws (Brent’s Theorem)](#4-work-and-span-laws-brents-theorem)  
5. [Speedup and Efficiency Metrics](#5-speedup-and-efficiency-metrics)  
6. [Slackness](#6-slackness)  
7. [Task vs. Data Parallelism](#7-task-vs-data-parallelism)  
8. [Massively Parallel Processing (MPP)](#8-massively-parallel-processing-mpp)  
9. [Load Balancing Algorithms](#9-load-balancing-algorithms)  
10. [Fault Tolerance via Lockstep Systems](#10-fault-tolerance-via-lockstep-systems)  
11. [Summary Table](#summary-table)  
12. [References](#references)  

---

### **1. Amdahl's Law**  
**Formula**:  
$$ S = \frac{1}{(1 - p) + \frac{p}{s}} $$  
- **Application**: Predicts maximum speedup achievable by parallelizing a fraction $$ p $$ of a task across $$ s $$ processors. Highlights diminishing returns due to sequential bottlenecks ($$ 1 - p $$)[3][7][8].  
- **Context**: Fundamental for evaluating parallel computing efficiency and identifying scalability limits in algorithms.  

---

### **2. Gustafson's Law (Scaled Speedup)**  
**Formula**:  
$$ S = s + (1 - s)N $$  
- **Application**: Addresses Amdahl's limitations by scaling workload size with processors ($$ N $$), enabling higher practical speedup for large problems[4].  
- **Context**: Explains why supercomputers achieve speedups by increasing problem size (e.g., climate modeling, molecular dynamics)[4][5].  

---

### **3. Sun and Ni's Memory-Bounded Speedup**  
**Principle**: Extends Amdahl/Gustafson by incorporating memory constraints. Speedup depends on both parallelism and memory availability.  
- **Application**: Critical for optimizing supercomputing workloads where data size exceeds individual node memory[4].  

---

### **4. Work and Span Laws (Brent’s Theorem)**  
- **Work Law**: $$ p \cdot T_p \geq T_1 $$ (parallel runtime × processors ≥ sequential work).  
- **Span Law**: $$ T_p \geq T_\infty $$ (parallel runtime ≥ critical path length).  
- **Application**: Guides parallel algorithm design by balancing workload distribution and minimizing critical path delays[1].  

---

### **5. Speedup and Efficiency Metrics**  
- **Speedup**: $$ S = \frac{T_1}{T_p} $$ (sequential time / parallel time).  
- **Efficiency**: $$ \text{Efficiency} = \frac{S}{p} $$ (speedup per processor).  
- **Parallelism**: $$ \frac{T_1}{T_\infty} $$ (theoretical maximum speedup)[1][7].  

---

### **6. Slackness**  
**Formula**:  
$$ \text{Slackness} = \frac{T_1}{p \cdot T_\infty} $$  
- **Application**: Determines if perfect linear speedup is achievable. Slackness < 1 implies scalability limits[1].  

---

### **7. Task vs. Data Parallelism**  
- **Task Parallelism**: Splits tasks into subtasks (e.g., different operations on same/different data).  
- **Data Parallelism**: Applies same operation to multiple data chunks (e.g., matrix operations).  
- **Context**: Core models in parallel programming (e.g., MPI for task, CUDA for data)[2][5].  

---

### **8. Massively Parallel Processing (MPP)**  
**Principle**: Divides problems into thousands of subproblems for concurrent execution across distributed systems.  
- **Application**: Supercomputing architecture (e.g., weather simulations, cryptography)[5][6].  

---

### **9. Load Balancing Algorithms**  
- **Longest/Shortest Execution Time First**: Prioritizes tasks by runtime to minimize idle processors.  
- **Probabilistic Scheduling**: Uses randomization to optimize resource allocation (e.g., Algorithm 3 in[6]).  
- **Context**: Ensures efficient utilization in supercomputers[6].  

---

### **10. Fault Tolerance via Lockstep Systems**  
**Principle**: Runs identical tasks in parallel for redundancy, enabling error detection/correction.  
- **Application**: Critical for reliability in high-performance computing[2].  

---

### **Summary Table**  

| **Concept**          | **Key Principles**                              | **Primary Use Case**                    |  
|-----------------------|------------------------------------------------|-----------------------------------------|  
| **Parallel Programming** | Work/Span Laws, Task/Data Parallelism         | Algorithm design, code optimization     |  
| **Parallel Computing**   | Amdahl/Gustafson Laws, Speedup Metrics        | Scalability analysis, system architecture |  
| **Supercomputing**       | MPP, Memory-Bounded Speedup, Load Balancing   | Large-scale simulations, real-time HPC   |  

These principles form the theoretical backbone for **analyzing** and **optimizing performance** across parallel systems.

## References

- [1] https://en.wikipedia.org/wiki/Analysis_of_parallel_algorithms
- [2] https://en.wikipedia.org/wiki/Parallel_computing
- [3] https://deviq.com/laws/amdahls-law/
- [4] https://jcst.ict.ac.cn/fileup/1000-9000/PDF/JCST-2023-1-5-2950-80.pdf
- [5] https://www.spiceworks.com/tech/tech-101/articles/what-is-supercomputer/
- [6] https://pmc.ncbi.nlm.nih.gov/articles/PMC9512182/
- [7] https://w3.cs.jmu.edu/kirkpams/OpenCSF/Books/csf/html/Scaling.html
- [8] https://en.wikipedia.org/wiki/Amdahl's_law
- [9] https://www.designgurus.io/answers/detail/what-are-problem-solving-techniques-in-computers
- [10] https://baylor.ai/?p=3322
- [11] https://www.linkedin.com/pulse/law-keeps-mind-while-designing-parallel-concurrent-systems-goyal
- [12] https://en.wikipedia.org/wiki/Parallel_algorithm
- [13] https://blog.2600hz.com/amdahls-law-and-parallel-computing
- [14] https://www.khanacademy.org/computing/ap-computer-science-principles/algorithms-101/x2d2f703b37b450a3:parallel-and-distributed-computing/a/parallel-computing
- [15] https://www.splunk.com/en_us/blog/learn/amdahls-law.html
- [16] https://users.ece.cmu.edu/~koopman/titan/rules.html
- [17] https://www.fujitsu.com/global/about/businesspolicy/tech/k/column/
- [18] https://publishing.cdlib.org/ucpressebooks/view?docId=ft0f59n73z&chunk.id=d0e6438&toc.depth=1&- toc.id=d0e6438&brand=ucpress
- [19] https://en.wikipedia.org/wiki/Moore's_law
- [20] http://www.cs.cmu.edu/~guyb/papers/BM04.pdf
- [21] https://www.learnpdc.org/PDCBeginners/0-introduction/4.strategies.html
- [22] https://www3.nd.edu/~zxu2/acms60212-40212-S16/Lec-05-S16.pdf
- [23] https://www.mathworks.com/company/technical-articles/improving-optimization-performance-with-parallel-computing.html
- [24] https://www.ibm.com/think/topics/parallel-computing
- [25] https://bcalabs.org/subject/parallel-processing-architecture-approaches-and-laws