# Edge AI Optimization and resource efficiency Laws

## Table of Contents

1. [Laws, Rules, and Quotations on Optimization and Resource Efficiency](#laws-rules-and-quotations-on-optimization-and-resource-efficiency)
    - [1. Amdahl’s Law](#1-amdahls-law)
    - [2. Gustafson’s Law](#2-gustafsons-law)
    - [3. Moore’s Law](#3-moores-law)
    - [4. Dennard Scaling](#4-dennard-scaling)
    - [5. Jevons Paradox](#5-jevons-paradox)
    - [6. Pareto Principle (80/20 Rule)](#6-pareto-principle-8020-rule)
    - [7. Pareto Front (Pareto Efficiency)](#7-pareto-front-pareto-efficiency)
    - [8. Power Law](#8-power-law)
    - [9. Scaling Laws for Neural Networks](#9-scaling-laws-for-neural-networks)
    - [10. Little’s Law](#10-littles-law)
    - [11. Roofline Model](#11-roofline-model)
    - [12. Knuth’s Optimization Rule](#12-knuths-optimization-rule)
    - [13. Wirth’s Law](#13-wirths-law)
    - [14. Law of Diminishing Returns](#14-law-of-diminishing-returns)
    - [15. Brooks’ Law](#15-brooks-law)
    - [16. Grosch’s Law](#16-groschs-law)
    - [17. Rent’s Rule](#17-rents-rule)
2. [Relationships and Insights](#relationships-and-insights)
3. [Edge AI Vision Context (Medical Anomaly Detector)](#edge-ai-vision-context-medical-anomaly-detector)

## Laws, Rules, and Quotations on Optimization and Resource Efficiency

### 1. Amdahl’s Law
- **Statement**: Speedup from parallelization is limited by the serial fraction of a task: \( S = \frac{1}{(1 - P) + \frac{P}{N}} \), where \( P \) is the parallelizable portion and \( N \) is the number of processors.
- **Relevance**: In Edge AI vision (e.g., Jetson Nano), parallelizing convolutions (e.g., 90% of compute) is key, but serial bottlenecks like data loading limit gains—optimization must target the serial part.
- **Relationship**: Complements scaling laws—more cores (N) hit diminishing returns unless \( P \) nears 1.

### 2. Gustafson’s Law
- **Statement**: Speedup scales with problem size: \( S = N + (1 - N) \cdot S_{\text{serial}} \), where larger tasks make serial fractions less impactful.
- **Relevance**: For vision tasks, increasing batch size (e.g., 16 to 64 on Jetson) leverages GPU parallelism, boosting efficiency—optimize for larger, parallel-friendly workloads.
- **Relationship**: Counters Amdahl’s pessimism, aligns with scaling laws for AI training.

### 3. Moore’s Law
- **Statement**: Transistor count doubles every ~18–24 months, historically boosting compute power.
- **Quotation**: “The number of transistors in an integrated circuit doubles approximately every two years.” – Gordon Moore (1965).
- **Relevance**: Drove Edge AI hardware (e.g., Jetson’s Maxwell GPU), but slowing since ~2015 shifts focus to software optimization (e.g., INT8 on Arduino).
- **Relationship**: Ties to Dennard Scaling—when it slowed, efficiency became king.

### 4. Dennard Scaling
- **Statement**: As transistors shrink, power density stays constant—voltage scales with size.
- **Relevance**: Its breakdown (~2006) forced Edge AI to prioritize power efficiency (e.g., FP16 on Jetson) over clock speed—optimization now balances compute and watts.
- **Relationship**: Linked to Moore’s Law; its failure spurred multi-core and quantization trends.

### 5. Jevons Paradox
- **Statement**: Increased efficiency in resource use can increase total consumption due to demand growth.
- **Quotation**: “It is wholly a confusion of ideas to suppose that the economical use of fuel is equivalent to a diminished consumption. The very contrary is the truth.” – William Stanley Jevons (1865).
- **Relevance**: In Edge AI, efficient models (e.g., quantized MobileNet) might lead to more devices deployed (e.g., smart cameras), raising overall power use—optimize with caps (e.g., power limits).
- **Relationship**: Challenges efficiency assumptions, intersects with scaling laws as demand scales.

### 6. Pareto Principle (80/20 Rule)
- **Statement**: 80% of effects come from 20% of causes.
- **Relevance**: In vision optimization, 80% of latency might stem from 20% of code (e.g., inference)—focus pruning/quantization there (e.g., Jetson’s hot paths).
- **Relationship**: Ties to Pareto Front—optimizing the vital few yields efficient trade-offs.

### 7. Pareto Front (Pareto Efficiency)
- **Statement**: A set of solutions where improving one objective (e.g., speed) worsens another (e.g., accuracy) without a better alternative.
- **Relevance**: In Edge AI, balancing FPS vs. accuracy (e.g., Jetson 15 FPS/90% vs. Pi 5 FPS/85%)—optimization seeks the front.
- **Relationship**: Extends Pareto Principle to multi-objective optimization, guides resource trade-offs.

### 8. Power Law
- **Statement**: A relationship where \( y = kx^a \) (e.g., small inputs yield disproportionate outputs).
- **Relevance**: In vision, a few layers (e.g., convolutions) dominate compute—optimize these for massive gains (e.g., TensorRT on Jetson).
- **Relationship**: Underpins Pareto Principle and scaling laws—small changes scale nonlinearly.

### 9. Scaling Laws for Neural Networks
- **Statement**: Performance scales with model size, data, and compute: \( \text{Loss} \propto N^{-\alpha}, D^{-\beta}, C^{-\gamma} \) (N=parameters, D=data, C=compute).
- **Relevance**: Larger models improve accuracy but strain edge devices—optimize by shrinking \( N \) (e.g., MobileNet on Pi).
- **Relationship**: Ties to Jevons Paradox—efficiency gains may increase compute demand.

### 10. Little’s Law
- **Statement**: \( L = \lambda \cdot W \) (L=items in system, \(\lambda\)=arrival rate, W=wait time).
- **Relevance**: In vision pipelines, reducing latency (W) or increasing FPS (\(\lambda\)) optimizes throughput—e.g., Jetson’s CUDA streams cut W.
- **Relationship**: Aligns with efficiency—faster pipelines reduce resource queues.

### 11. Roofline Model
- **Statement**: Performance is bound by peak compute or memory bandwidth: \( \text{Perf} = \min(\text{FLOPS}, \text{Bandwidth} \cdot \text{AI}) \).
- **Relevance**: On Jetson (472 GFLOPS, 5GB/s), vision tasks (high AI) are compute-bound—optimize kernels, not bandwidth.
- **Relationship**: Guides hardware-aware optimization, links to power efficiency.

### 12. Knuth’s Optimization Rule
- **Quotation**: “Premature optimization is the root of all evil.” – Donald Knuth.
- **Relevance**: In Edge AI, don’t tweak loops before profiling (e.g., `nvprof` on Jetson)—focus on bottlenecks first.
- **Relationship**: Balances effort vs. gain, echoes Pareto’s 80/20 focus.

### 13. Wirth’s Law
- **Statement**: “Software gets slower faster than hardware gets faster.” – Niklaus Wirth.
- **Relevance**: Bloated vision code (e.g., unoptimized OpenCV) negates hardware gains—optimize leanly (e.g., TFLite on Pi).
- **Relationship**: Counters Moore’s Law, stresses software efficiency.

### 14. Law of Diminishing Returns
- **Statement**: Additional effort yields smaller gains past a point.
- **Relevance**: In Edge AI, pushing FPS beyond 20 on Jetson may spike power with little user benefit—optimize within limits.
- **Relationship**: Ties to Jevons Paradox—efficiency gains plateau, demand may not.

### 15. Brooks’ Law
- **Quotation**: “Adding manpower to a late software project makes it later.” – Fred Brooks.
- **Relevance**: Throwing more resources at optimization (e.g., extra cores on Pi) can complicate—simplify first.
- **Relationship**: Warns against over-scaling, aligns with cost-effectiveness.

### 16. Grosch’s Law
- **Statement**: Performance increases as the square of cost.
- **Relevance**: Historically true, but Edge AI seeks linear gains (e.g., INT8 on Arduino)—optimize for cost-efficiency.
- **Relationship**: Challenges Moore’s Law as hardware costs plateau.

### 17. Rent’s Rule
- **Statement**: I/O pins scale as \( P = k \cdot G^r \) (G=gates, \( r \approx 0.5 \)).
- **Relevance**: In embedded vision, camera bandwidth limits throughput—optimize data flow (e.g., zero-copy on Jetson).
- **Relationship**: Links to Roofline—bandwidth constrains efficiency.

---

## Relationships and Insights
- **Jevons Paradox vs. Efficiency**: Efficiency (e.g., INT8 on Jetson) may increase device proliferation, raising total resource use—counter with power caps or sparse models.
- **Scaling Laws vs. Edge Constraints**: Bigger models (Scaling Laws) clash with edge limits (e.g., Pi’s 1GB)—optimization shrinks \( N \) (parameters) to fit.
- **Pareto Front vs. Power Laws**: Optimizing the “vital few” (Power Law) finds the Pareto Front—e.g., tuning inference vs. accuracy on Jetson.
- **Amdahl’s Law vs. Roofline**: Serial bottlenecks (Amdahl) often tie to compute/memory limits (Roofline)—e.g., Jetson’s convolution vs. bandwidth trade-off.
- **Moore’s Law vs. Wirth’s Law**: Hardware gains (Moore) are offset by software bloat (Wirth)—e.g., optimize vision pipelines to reclaim efficiency.

---

## Edge AI Vision Context (Medical Anomaly Detector)
- **Cost-Effective**: Pruning (Pareto Principle) and INT8 (Jevons-aware) keep Jetson/Pi/Arduino local, avoiding cloud costs.
- **Efficient**: Roofline-guided TensorRT (Jetson) and CMSIS-NN (Arduino) maximize FPS/watt—e.g., 15 FPS/3W vs. 2 FPS/0.5W.
- **Private & Secure**: Static allocation (Rent’s Rule) and low-res inputs (Scaling Laws) keep data on-device—e.g., 32x32 on Arduino.
- **Personalized**: Incremental learning (Little’s Law) adapts models locally—e.g., Pi fine-tunes in <5min.
