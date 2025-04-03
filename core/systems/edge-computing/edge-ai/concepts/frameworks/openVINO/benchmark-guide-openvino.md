# **Benchmark Tools for Small Models**

This document provides a practical guide to benchmark small machine learning models using existing tools.

Small models are optimized for edge devices, mobile platforms, or low-latency applications.

Efficient benchmarking helps assess their speed, memory usage, and accuracy in real-world scenarios.

---

## **Quick Guide: Benchmarking with OpenVINO**

**OpenVINO** is a highly efficient tool for benchmarking vision models, especially for `Intel hardware`. Follow these steps to benchmark your model:

### **Step 1: Install OpenVINO Toolkit**
```bash
pip install openvino-dev
```

### **Step 2: Convert Your Model to IR Format (Optional)**
If your model is not already in OpenVINO's Intermediate Representation (IR) format, convert it:
```bash
mo --input_model path/to/your/model.onnx --output_dir path/to/output
```

### **Step 3: Run the Benchmark Tool**
Use OpenVINO's Benchmark Tool for performance evaluation:
```bash
benchmark_app -m path/to/your/model.xml -d CPU -niter 100
```

### **Step 4: Analyze the Output**
- **Throughput**: Number of inferences per second.  
- **Latency**: Average time taken for a single inference.  

### **Tips for Optimization**
- Add `-d GPU` for GPU acceleration.  
- Use quantized models for better performance with `-q`.

---

## **Other Benchmarking Tools**

| **Tool**            | **Purpose**                         | **Best For**                 | **Platform**         |
|----------------------|-------------------------------------|------------------------------|----------------------|
| **ONNX Runtime**     | Cross-platform model benchmarking  | Small ONNX models            | CPU, GPU, ARM        |
| **TensorFlow Lite**  | Mobile-specific benchmarking       | Android, iOS edge models     | Mobile, Edge         |
| **PyTorch Benchmark**| Granular PyTorch model profiling   | Development-phase models      | Desktop              |
| **Edge Impulse**     | Embedded ML benchmarking           | IoT and edge devices          | ARM Cortex, NVIDIA   |
| **DeepSparse**       | CPU inference optimization         | Quantized ONNX models         | CPU                  |
| **TensorRT**         | GPU-optimized inference            | NVIDIA Jetson/GPUs            | GPU                  |
| **Hugging Face Evaluate** | Multimodal task evaluation    | Vision-language tasks         | Cloud, Local         |
| **MLPerf Tiny**      | Standardized tiny model benchmarks | Embedded ML systems           | Multi-platform       |

---

## **Why Benchmark Small Models?**
- Ensure **low-latency performance** for real-time applications.  
- Optimize for deployment on **resource-constrained devices**.  
- Compare different tools and hardware to find the best fit for your use case.  

Next phases: benchmarking automating
