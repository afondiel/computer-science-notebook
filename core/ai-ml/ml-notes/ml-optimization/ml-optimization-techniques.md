# ML Optimization Techniques Notes

## Overview

Optimization strategies used in deep learning, especially for training large language models. 

**Quantization:**
* Converting parameters from FP32 to FP16
* **LLM.int8()** for quantizing large models to 8-bit integers
* **GPTQ** for finding an optimal quantized weight matrix

**Parameter-Efficient Fine-Tuning (PEFT):**
* **LoRA (Low-Rank Adaptation)**:  training low-rank adapters instead of full fine-tuning
* **QLoRA**: combining LoRA with 4-bit quantization for reduced memory usage

**Other techniques:**
* **Flash Attention**: reordering attention computation for speed and reduced memory usage
* **Gradient Accumulation**: training on larger effective batch sizes by accumulating gradients
* **8-bit Optimizers**: quantizing optimizer states to 8-bit
* **Sequence Packing**: concatenating training sequences to maximize GPU utilization
* **torch.compile()**: JIT-compiling PyTorch code for faster execution
* **Multi-query Attention (MQA)** and **Grouped-query Attention (GQA)**: modifying multihead attention for efficiency

**Distributed Training Techniques:**
* **Data Parallelism (DP)**: replicating model parameters on multiple devices and processing data subsets concurrently
* **Model Parallelism**: splitting the model across multiple devices
    * Naive Model Parallelism
    * **Pipeline Parallelism (PP)**:  processing micro-batches in a pipeline across GPUs
    * **Tensor Parallelism**: distributing tensor computations across multiple GPUs
* **Fully Sharded Data Parallel (FSDP)**: sharding model parameters, optimizer states, and gradients across devices 


## References
- [Efficient Deep Learning: A Comprehensive Overview of Optimization Techniques - HF](https://huggingface.co/blog/Isayoften/optimization-rush)