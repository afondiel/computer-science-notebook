# Spiking Neural Networks (SNNs) Technical Notes
<!-- A rectangular diagram depicting the spiking neural network (SNN) pipeline, illustrating input data (e.g., time-series sensor readings) encoded into spike trains, processed through a layered network of spiking neurons (e.g., Leaky Integrate-and-Fire), with synaptic weights and temporal dynamics, producing output spikes for classification or control, annotated with spike timing and learning rules. -->

## Quick Reference
- **Definition**: Spiking Neural Networks (SNNs) are bio-inspired neural networks that process information using time-dependent spikes, enabling efficient, event-driven computation.
- **Key Use Cases**: Real-time signal processing, neuromorphic hardware applications, and modeling temporal dynamics in neuroscience.
- **Prerequisites**: Familiarity with neural networks, basic programming (e.g., Python), and understanding of time-series data.

## Table of Contents
1. Introduction
2. Core Concepts
3. Implementation Details
4. Real-World Applications
5. Tools & Resources
6. References
7. Appendix

## Introduction
- **What**: SNNs are neural networks that use discrete spikes to transmit information, mimicking biological neurons with temporal dynamics and event-driven processing.
- **Why**: They offer energy-efficient computation for neuromorphic systems and excel at processing temporal or event-based data, unlike traditional neural networks.
- **Where**: Applied in neuromorphic chips (e.g., Intel Loihi), robotics, and research into brain-inspired AI and sensory processing.

## Core Concepts
### Fundamental Understanding
- **Basic Principles**:
  - SNNs operate on spike trains, where the timing and frequency of spikes encode information.
  - Neurons integrate inputs over time, firing spikes when a membrane potential threshold is reached, often using models like Leaky Integrate-and-Fire (LIF).
  - Learning in SNNs often involves spike-timing-dependent plasticity (STDP), adjusting synaptic weights based on spike timing.
- **Key Components**:
  - **Neuron Model**: Defines spiking behavior (e.g., LIF or Izhikevich models).
  - **Synaptic Weights**: Modulate the strength of spike transmission between neurons.
  - **Spike Encoding**: Converts continuous or discrete data into spike trains (e.g., rate or temporal coding).
- **Common Misconceptions**:
  - Misconception: SNNs are just a variant of traditional neural networks.
    - Reality: Their event-driven, temporal nature makes them fundamentally different, suited for dynamic data.
  - Misconception: SNNs are only for neuromorphic hardware.
    - Reality: They can be simulated on standard hardware for research or prototyping.

### Visual Architecture
```mermaid
graph TD
    A[Input Data <br> (e.g., Time-Series)] --> B[Spike Encoder <br> (Rate/Temporal Coding)]
    B --> C[Input Layer <br> (Spiking Neurons)]
    C -->|Weighted Synapses| D[Hidden Layer <br> (LIF Neurons)]
    D -->|Weighted Synapses| E[Output Layer]
    E --> F[Output Spikes <br> (e.g., Classification)]
    G[STDP Learning] -->|Adjust Weights| C
    G -->|Adjust Weights| D
```
- **System Overview**: The diagram shows input data encoded as spikes, processed through layered spiking neurons with weighted synapses, and producing output spikes, with STDP adjusting weights.
- **Component Relationships**: The encoder generates spikes, neurons process them temporally, and STDP refines connections for learning.

## Implementation Details
### Intermediate Patterns
```python
# Example: Simple SNN with LIF neurons and STDP in Python using Brian2
from brian2 import *

# Simulation parameters
duration = 100*ms
num_inputs = 2
num_neurons = 1

# LIF neuron model
eqs = '''
dv/dt = (-v + I)/tau : volt
I : volt
tau : second
'''
threshold = 'v > 20*mV'
reset = 'v = 0*mV'

# Create neurons
inputs = PoissonGroup(num_inputs, rates=50*Hz)  # Spike inputs
neurons = NeuronGroup(num_neurons, eqs, threshold=threshold, reset=reset, method='euler')
neurons.tau = 10*ms

# Synapses with STDP
synapses = Synapses(inputs, neurons, model='w : 1', on_pre='I += w*10*mV')
synapses.connect()  # Connect all inputs to neuron
synapses.w = 'rand()*0.5'  # Random initial weights

# STDP learning rule
stdp = Synapses(inputs, neurons, 
                model='''
                w : 1
                dApre/dt = -Apre/taupre : 1 (event-driven)
                dApost/dt = -Apost/taupost : 1 (event-driven)
                ''',
                on_pre='''
                Apre += 0.01
                w = clip(w + Apost, 0, 1)
                I += w*10*mV
                ''',
                on_post='''
                Apost += -0.01
                w = clip(w + Apre, 0, 1)
                ''')
stdp.connect()
stdp.w = synapses.w

# Record spikes and weights
spike_monitor = SpikeMonitor(neurons)
weight_monitor = StateMonitor(stdp, 'w', record=True)

# Run simulation
run(duration)

# Plot results
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.plot(spike_monitor.t/ms, spike_monitor.i, '.k')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index')
plt.title('Spike Raster')
plt.subplot(122)
for i in range(num_inputs):
    plt.plot(weight_monitor.t/ms, weight_monitor.w[i], label=f'Synapse {i}')
plt.xlabel('Time (ms)')
plt.ylabel('Weight')
plt.title('Synaptic Weights')
plt.legend()
plt.tight_layout()
plt.show()
```
- **Design Patterns**:
  - **Event-Driven Processing**: Use spike-based computation for efficiency.
  - **Temporal Coding**: Encode data in spike timing for richer representations.
  - **STDP Learning**: Implement bio-inspired learning to adapt synaptic weights.
- **Best Practices**:
  - Choose appropriate neuron models (e.g., LIF for simplicity, Izhikevich for realism).
  - Tune time constants (e.g., `tau`) to match input dynamics.
  - Validate spike rates and weight changes to ensure learning stability.
- **Performance Considerations**:
  - Optimize simulation step size (e.g., `dt`) for accuracy vs. speed.
  - Use sparse connectivity to reduce memory usage in large networks.
  - Profile simulation time for scalability with more neurons or synapses.

## Real-World Applications
### Industry Examples
- **Use Case**: Event-based vision for autonomous drones.
  - An SNN processes spikes from an event camera to detect obstacles in real-time.
- **Implementation Patterns**: Encode camera events as spikes, use a small SNN for classification, and deploy on neuromorphic hardware.
- **Success Metrics**: Low power usage (<1W) and millisecond-latency detection.

### Hands-On Project
- **Project Goals**: Build an SNN to classify temporal patterns from simulated sensor data.
- **Implementation Steps**:
  1. Use the above Brian2 code to create an SNN with 2 input neurons and 1 output neuron.
  2. Generate two input patterns: high-frequency (100Hz) and low-frequency (20Hz) Poisson spikes.
  3. Train the SNN with STDP to distinguish patterns based on output spike rates.
  4. Test classification by counting output spikes over 100ms.
- **Validation Methods**: Verify higher spike rates for high-frequency inputs; ensure weights stabilize after training.

## Tools & Resources
### Essential Tools
- **Development Environment**: Python, Jupyter for interactive simulations.
- **Key Frameworks**: Brian2 for SNN simulation, PyNN for hardware compatibility.
- **Testing Tools**: Matplotlib for spike visualization, NumPy for data processing.

### Learning Resources
- **Documentation**: Brian2 (https://brian2.readthedocs.io), PyNN (http://neuralensemble.org/docs/PyNN).
- **Tutorials**: Blogs on SNNs, Coursera neuroscience courses.
- **Community Resources**: r/neuroAI, Stack Overflow for Brian2 questions.

## References
- SNN fundamentals: https://www.frontiersin.org/articles/10.3389/fnins.2018.00774
- STDP learning: https://en.wikipedia.org/wiki/Spike-timing-dependent_plasticity
- Intel Loihi: https://www.intel.com/content/www/us/en/research/neuromorphic-computing.html
- Brian2 guide: https://brian2.readthedocs.io/en/stable/user/introduction.html

## Appendix
- **Glossary**:
  - **STDP**: Spike-Timing-Dependent Plasticity, a learning rule based on spike timing.
  - **LIF**: Leaky Integrate-and-Fire, a simple spiking neuron model.
  - **Spike Train**: A sequence of spikes representing neural activity.
- **Setup Guides**:
  - Install Brian2: `pip install brian2`.
  - Install Matplotlib: `pip install matplotlib`.
- **Code Templates**:
  - Temporal coding: Encode input as precise spike times instead of Poisson rates.
  - Multi-layer SNN: Extend the code with additional hidden layers.