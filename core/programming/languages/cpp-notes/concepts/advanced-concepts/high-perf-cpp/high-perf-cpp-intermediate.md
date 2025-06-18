# High Performance C++ Technical Notes
<!-- A rectangular diagram depicting an intermediate-level high-performance C++ pipeline, showing a C++ program processing data (e.g., matrices, streams) using advanced techniques (e.g., SIMD, multi-threading, cache optimization), leveraging hardware features (e.g., CPU vector units, multi-core), and producing optimized outputs (e.g., high-throughput computations), with annotations for profiling and concurrency. -->

## Quick Reference
- **Definition**: Intermediate high-performance C++ involves writing C++ programs optimized for speed and scalability, using modern C++ features (e.g., C++17/20), SIMD, multi-threading, and cache-aware design to maximize hardware utilization.
- **Key Use Cases**: Real-time data processing, parallel numerical computations, game engine components, and performance-sensitive libraries requiring high throughput.
- **Prerequisites**: Familiarity with C++ (e.g., templates, smart pointers, STL), basic performance concepts (e.g., cache, loop optimization), and experience with compilers like `g++`.

## Table of Contents
1. Introduction
2. Core Concepts
3. Implementation Details
4. Real-World Applications
5. Tools & Resources
6. References
7. Appendix

## Introduction
- **What**: Intermediate high-performance C++ uses advanced C++ features, SIMD instructions, multi-threading, and cache-aware programming to achieve high throughput and low latency in performance-critical applications.
- **Why**: C++’s blend of high-level abstractions and low-level control enables intermediate users to exploit modern hardware (e.g., multi-core CPUs, vector units) while maintaining code safety and readability.
- **Where**: Used in scientific simulations, multimedia processing, real-time systems, and high-performance libraries on platforms like Linux, Windows, or macOS.

## Core Concepts
### Fundamental Understanding
- **Basic Principles**:
  - **Performance Goals**: Maximize throughput and minimize latency by optimizing CPU, memory, and parallel execution.
  - **C++’s Role**: Provides modern features (e.g., `std::thread`, `std::simd`), templates for type-safe optimizations, and low-level access for hardware control.
  - **Hardware Utilization**: Leverages CPU vector units (SIMD), multi-core parallelism, and cache hierarchies.
- **Key Components**:
  - **SIMD Programming**:
    - Use intrinsics (e.g., SSE, AVX) or `std::simd` (C++23, experimental) for parallel data operations.
    - Example: Add eight floats with `_mm256_add_ps`.
  - **Multi-Threading**:
    - Use `std::thread` or `std::async` for task parallelism.
    - Manage synchronization with `std::mutex`, `std::atomic`, or condition variables.
  - **Cache Optimization**:
    - **Data Locality**: Use contiguous containers (`std::vector`) and structure-of-arrays (SoA) layouts.
    - **Prefetching**: Manually prefetch data or rely on compiler optimizations.
  - **Memory Management**:
    - Use `std::unique_ptr`/`std::shared_ptr` for RAII-based memory safety.
    - Allocate aligned memory with `std::aligned_alloc` for SIMD/cache efficiency.
  - **Modern C++ Features**:
    - **Templates**: Enable generic, compile-time optimized code.
    - **Move Semantics**: Reduce copying with `std::move` and rvalue references.
    - **Constexpr**: Evaluate computations at compile time.
  - **Compiler Optimizations**:
    - Flags like `-O3`, `-march=native` enable vectorization and CPU-specific instructions.
    - Use `[[likely]]` or `[[unlikely]]` (C++20) for branch prediction hints.
  - **Profiling**:
    - Use `perf` or `valgrind` to analyze cache misses, branch mispredictions, or thread contention.
    - Measure with `std::chrono` for high-resolution timing.
- **Common Misconceptions**:
  - **Misconception**: Multi-threading always improves performance.
    - **Reality**: Thread overhead and synchronization costs require careful design.
  - **Misconception**: C++ abstractions degrade performance.
    - **Reality**: Zero-cost abstractions (e.g., templates, `constexpr`) match C’s performance when used correctly.

### Visual Architecture
```mermaid
graph TD
    A[Data Input <br> (Matrix, Stream)] --> B[C++ Program <br> (g++, SIMD, threads)]
    B --> C[Processing <br> (Parallel, Cache, Vector)]
    C --> D[Output <br> (High-Throughput Results)]
```
- **System Overview**: The diagram shows data processed by a C++ program, optimized with SIMD, threading, and cache techniques, producing high-throughput results.
- **Component Relationships**: Input is processed in parallel, leveraging hardware for efficient output.

## Implementation Details
### Intermediate Patterns
```cpp
// Example: Parallel matrix addition with SSE intrinsics
#include <iostream>
#include <vector>
#include <thread>
#include <immintrin.h> // SSE
#include <chrono>

constexpr size_t N = 1024;
constexpr size_t THREADS = 4;
constexpr size_t ALIGNMENT = 16; // SSE alignment

void add_matrix(const std::vector<float>& a, const std::vector<float>& b,
                std::vector<float>& c, size_t start, size_t end) {
    // Process 4 elements with SSE
    for (size_t i = start; i < end - 3; i += 4) {
        __m128 va = _mm_load_ps(&a[i]);
        __m128 vb = _mm_load_ps(&b[i]);
        __m128 vc = _mm_add_ps(va, vb);
        _mm_store_ps(&c[i], vc);
    }
    // Handle remainder
    for (size_t i = end - (end % 4); i < end; ++i) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    // Allocate aligned vectors
    std::vector<float, std::allocator<float>> a(N * N, 0.0f), b(N * N, 0.0f), c(N * N, 0.0f);
    a.reserve(N * N);
    b.reserve(N * N);
    c.reserve(N * N);

    // Initialize matrices
    for (size_t i = 0; i < N * N; ++i) {
        a[i] = static_cast<float>(i) / 1000.0f;
        b[i] = static_cast<float>(i) / 2000.0f;
    }

    // Measure time
    auto start = std::chrono::high_resolution_clock::now();

    // Create threads
    std::vector<std::thread> threads;
    size_t chunk = N * N / THREADS;
    for (size_t i = 0; i < THREADS; ++i) {
        size_t thread_start = i * chunk;
        size_t thread_end = (i == THREADS - 1) ? N * N : (i + 1) * chunk;
        threads.emplace_back(add_matrix, std::ref(a), std::ref(b), std::ref(c),
                            thread_start, thread_end);
    }

    // Join threads
    for (auto& t : threads) {
        t.join();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Verify result
    std::cout << "Sample: c[0] = " << c[0] << "\n";
    std::cout << "Time: " << duration.count() / 1e6 << " seconds\n";

    return 0;
}
```
- **Step-by-Step Setup** (Linux):
  1. **Install Tools**:
     - Install `g++`: `sudo apt install g++` (Ubuntu/Debian) or `sudo dnf install gcc-c++` (Fedora).
     - Verify: `g++ --version`.
  2. **Save Code**: Save as `matrix_add.cpp`.
  3. **Compile**: Run `g++ -O3 -msse2 -std=c++17 matrix_add.cpp -o matrix_add` (`-O3` for optimizations, `-msse2` for SSE, `-std=c++17` for modern C++).
  4. **Run**: Execute `./matrix_add`.
- **Code Walkthrough**:
  - Uses `std::vector` with `reserve` for contiguous, cache-friendly memory.
  - Implements SSE intrinsics (`_mm_load_ps`, `_mm_add_ps`, `_mm_store_ps`) to add four floats per instruction.
  - Divides work across four threads with `std::thread`, each processing a matrix chunk.
  - Measures time with `std::chrono` for high precision.
  - Uses `std::ref` to pass vectors by reference to threads, avoiding copies.
  - Includes a remainder loop for non-SIMD elements.
- **Common Pitfalls**:
  - **Alignment Errors**: Ensure `std::vector` memory is 16-byte aligned for SSE (typically guaranteed for `float`).
  - **Thread Safety**: Avoid data races by assigning distinct ranges to threads.
  - **Compiler Flags**: Missing `-msse2` or `-O3` reduces performance.
  - **Thread Overhead**: Too many threads for small datasets can degrade performance (tune `THREADS`).

## Real-World Applications
### Industry Examples
- **Use Case**: Real-time graphics in game engines.
  - Optimize vector transformations with SIMD and threading.
  - **Implementation**: Use SSE for vertex calculations, threads for parallel rendering.
  - **Metrics**: >60 FPS, low CPU usage.
- **Use Case**: Signal processing in IoT devices.
  - Process streaming data with low latency.
  - **Implementation**: Cache-aware buffers, SIMD for filtering.
  - **Metrics**: <10ms latency, minimal memory footprint.

### Hands-On Project
- **Project Goals**: Perform parallel matrix addition with SIMD optimizations.
- **Implementation Steps**:
  1. Install `g++` and save the example code.
  2. Compile with `g++ -O3 -msse2 -std=c++17 matrix_add.cpp -o matrix_add`.
  3. Run and note execution time.
  4. Experiment by disabling SSE (use scalar addition) or reducing threads (`THREADS=1`) and compare times.
  5. Verify results with sample checks (e.g., `c[0] = a[0] + b[0]`).
- **Validation Methods**: Confirm speedup with SSE and threading; ensure correct results via spot checks.

## Tools & Resources
### Essential Tools
- **Development Environment**: `g++`, IDE (e.g., CLion, VS Code).
- **Key Tools**:
  - `g++`: Compiler with SIMD and threading support.
  - `perf`: Performance profiling (`perf stat`).
  - `valgrind`: Cache/memory analysis (`--tool=cachegrind`).
  - `gprof`: Function-level profiling.
- **Testing Tools**: `time` command, `gdb` for debugging.

### Learning Resources
- **Documentation**:
  - C++ Reference: https://en.cppreference.com/w/cpp
  - GCC: https://gcc.gnu.org/onlinedocs/
  - Intel Intrinsics Guide: https://software.intel.com/sites/landingpage/IntrinsicsGuide/
- **Tutorials**:
  - SIMD in C++: https://www.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top.html
  - Multi-threading: https://en.cppreference.com/w/cpp/thread
- **Communities**: Stack Overflow, r/cpp, C++ Slack (https://cpp-slack.herokuapp.com/).

## References
- C++ Reference: https://en.cppreference.com/w/cpp
- GCC documentation: https://gcc.gnu.org/onlinedocs/
- Intel Intrinsics Guide: https://software.intel.com/sites/landingpage/IntrinsicsGuide/
- Cache optimization: https://lwn.net/Articles/255364/
- Optimization guide: https://www.agner.org/optimize/

## Appendix
- **Glossary**:
  - **SIMD**: Single Instruction, Multiple Data for parallel operations.
  - **Cache Miss**: Failure to find data in CPU cache, causing latency.
  - **RAII**: Resource Acquisition Is Initialization for resource management.
- **Setup Guides**:
  - Install `g++` (Ubuntu): `sudo apt install g++`.
  - Compile with SIMD: `g++ -O3 -msse2 -std=c++17 file.cpp -o file`.
- **Code Templates**:
  - SIMD addition: `__m128 v = _mm_add_ps(_mm_load_ps(a), _mm_load_ps(b));`
  - Thread creation: `std::thread t(func, std::ref(data));`