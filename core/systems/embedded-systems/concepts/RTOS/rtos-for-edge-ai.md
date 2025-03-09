# RTOS for Edge AI Applications and Benchmarks

RTOSes suitable for edge computing with AI workloads, such as real-time inference on devices with limited resources (e.g., ARM Cortex, RISC-V, or hardware accelerators). Includes supported AI frameworks and latency benchmarks where available.

| RTOS Name       | Developer                   | License         | Certification Standards | Supported AI Frameworks                  | Latency Benchmarks (Approx.)         | Notes                                      |
|-----------------|-----------------------------|-----------------|-------------------------|------------------------------------------|--------------------------------------|--------------------------------------------|
| **FreeRTOS**    | Amazon (Real Time Engineers) | MIT            | IEC 61508 (via SafeRTOS)| TensorFlow Lite Micro, AWS IoT AI        | ~10-50 µs (context switch on Cortex-M4) | Lightweight, integrates with AWS for edge AI inference (e.g., smart cameras). |
| **Zephyr**      | Linux Foundation            | Apache 2.0      | -                       | TensorFlow Lite Micro, ONNX Runtime      | ~15-60 µs (preemption on Cortex-M4)   | Open-source, supports AI on constrained devices (e.g., wearables, sensors). |
| **NuttX**       | Apache Foundation           | Apache 2.0      | -                       | TensorFlow Lite Micro, custom frameworks | ~20-70 µs (task switch on Cortex-M4)  | POSIX-compliant, used in edge AI nodes (e.g., IoT gateways). |
| **ThreadX**     | Microsoft (via Express Logic) | Proprietary   | IEC 61508, ISO 26262    | TensorFlow Lite Micro, Azure RTOS AI     | ~5-30 µs (context switch on Cortex-M4) | Low latency, used in AI-enabled edge devices (e.g., industrial sensors). |
| **QNX**         | BlackBerry                  | Proprietary     | ISO 26262, IEC 61508    | TensorFlow Lite, Caffe, custom frameworks| ~10-40 µs (interrupt latency on ARM)  | High performance for edge AI in automotive (e.g., ADAS inference). |

### Supported AI Frameworks

- **FreeRTOS**: Integrates with TensorFlow Lite Micro for lightweight inference and AWS IoT for cloud-backed AI.
- **Zephyr**: Officially supports TensorFlow Lite Micro; extensible for ONNX Runtime via community efforts.
- **NuttX**: Supports TensorFlow Lite Micro; flexible for custom frameworks due to POSIX compatibility.
- **ThreadX**: Optimized for TensorFlow Lite Micro; ties into Azure ecosystem for AI at the edge.
- **QNX**: Supports TensorFlow Lite, Caffe, and proprietary frameworks, leveraging its POSIX environment.

### Latency Benchmarks

Approximate values from Thread Metric benchmarks (e.g., Beningo Embedded 2024 Report) on STM32 Cortex-M4 at 80 MHz, showing context switch or preemption times. Exact latency depends on hardware and configuration.

### Use Cases

Edge AI examples include:
- Real-time object detection (FreeRTOS, ThreadX)
- Sensor data classification (Zephyr, NuttX)
- Automotive perception systems (QNX).

## AI Applications

RTOSes capable of supporting broader AI applications, including real-time processing for machine learning, computer vision, or robotics (often requiring POSIX support or advanced hardware integration).

| RTOS Name       | Developer                   | License         | Certification Standards | Supported AI Frameworks                  | Latency Benchmarks (Approx.)         | Notes                                      |
|-----------------|-----------------------------|-----------------|-------------------------|------------------------------------------|--------------------------------------|--------------------------------------------|
| **QNX**         | BlackBerry                  | Proprietary     | ISO 26262, IEC 61508    | TensorFlow, Caffe, PyTorch, ROS2         | ~10-40 µs (interrupt latency on ARM)  | POSIX-compliant, used in AI-driven robotics and autonomous systems. |
| **VxWorks**     | Wind River                  | Proprietary     | DO-178C, ISO 26262      | TensorFlow, OpenCV, NVIDIA CUDA          | ~5-25 µs (context switch on ARM)      | Supports AI in embedded systems (e.g., vision, drones). |
| **Fuchsia (Zircon)** | Google                  | BSD/MIT/Apache 2.0 | -                    | TensorFlow, MLKit, custom frameworks     | ~20-80 µs (task switch on ARM64)      | Modern microkernel, experimental for AI workloads (e.g., smart devices). |
| **RTEMS**       | RTEMS Project               | BSD             | -                       | TensorFlow Lite, custom frameworks       | ~15-50 µs (preemption on Cortex-M4)   | POSIX support, adaptable for AI in research (e.g., space robotics). |
| **LynxOS**      | Lynx Software Technologies  | Proprietary     | DO-178B                 | TensorFlow, OpenCV, custom frameworks    | ~10-35 µs (interrupt latency on ARM)  | POSIX-compliant, suitable for AI in avionics (e.g., synthetic vision). |

- **Supported AI Frameworks**: 
  - **QNX**: Broad POSIX support enables TensorFlow, Caffe, PyTorch, and ROS2 for robotics/AI.
  - **VxWorks**: Integrates with TensorFlow, OpenCV, and NVIDIA CUDA for GPU-accelerated AI.
  - **Fuchsia (Zircon)**: Experimental support for TensorFlow, MLKit; extensible via Google’s ecosystem.
  - **RTEMS**: Supports TensorFlow Lite and custom frameworks; used in academic AI projects.
  - **LynxOS**: POSIX enables TensorFlow, OpenCV; tailored for avionics AI tasks.
- **Latency Benchmarks**: Approximate values from typical RTOS benchmarks (e.g., Thread Metric suite) on ARM platforms, showing interrupt latency or context switch times. VxWorks and QNX excel in low-latency scenarios.
- **Use Cases**: AI applications include autonomous navigation (QNX, VxWorks), computer vision (LynxOS), experimental AI devices (Fuchsia), and space robotics (RTEMS).

