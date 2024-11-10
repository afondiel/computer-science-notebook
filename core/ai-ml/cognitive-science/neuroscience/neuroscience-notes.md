# Neuroscience - Notes

## Table of Contents
- [Overview](#overview)
- [Applications](#applications)
- [Tools & Frameworks](#tools-&-frameworks)
- [Hello World!](#hello-world!)
- [References](#references)

## Overview
Neuroscience is the scientific study of the nervous system (the brain, spinal cord, and peripheral nervous system), its functions and disorders.

## Applications
- Neuroscience can help us understand the biological basis of learning, memory, behavior, perception, and consciousness.
- Neuroscience can also inform the diagnosis and treatment of various neurological and psychiatric disorders, such as stroke, Alzheimer's disease, Parkinson's disease, epilepsy, schizophrenia, depression, and autism.
- Neuroscience can also contribute to other fields of study, such as economics, education, artificial intelligence, and philosophy.

## Tools & Frameworks
- Neuroscience uses a variety of techniques and methods to study the nervous system at different levels of analysis, from molecules to cells to circuits to systems to behavior.
- Some of the tools and frameworks used by neuroscientists include:
    - **Molecular and cellular biology**: to investigate the structure and function of genes, proteins, and signaling molecules involved in neural development, plasticity, and communication.
    - **Anatomy and histology**: to examine the gross and microscopic organization of the nervous system and its components.
    - **Physiology and electrophysiology**: to measure the electrical and chemical activity of neurons and neural networks.
    - **Pharmacology and neurochemistry**: to study the effects of drugs and neurotransmitters on neural function and behavior.
    - **Imaging and optogenetics**: to visualize and manipulate the structure and activity of the brain in vivo using techniques such as magnetic resonance imaging (MRI), positron emission tomography (PET), functional near-infrared spectroscopy (fNIRS), electroencephalography (EEG), magnetoencephalography (MEG), and light-sensitive proteins.
    - **Computational and mathematical modeling**: to simulate and analyze the dynamics and information processing of neural systems using mathematical equations, algorithms, and computer programs.
    - **Psychology and neuropsychology**: to assess the cognitive, emotional, and behavioral functions of the brain and their impairments.
    - **Ethology and evolutionary biology**: to compare the nervous systems and behaviors of different species and understand their evolutionary origins and adaptations. ¹

## Hello World!

Here is a simple Python code that simulates a leaky integrate-and-fire neuron model, which is one of the simplest models of a neuron's electrical activity. 

```python
import numpy as np
import matplotlib.pyplot as plt

# Define parameters
dt = 0.1 # time step (ms)
tmax = 100 # simulation time (ms)
Vrest = -65 # resting membrane potential (mV)
Vth = -50 # spike threshold (mV)
Vreset = -70 # reset potential after spike (mV)
Rm = 10 # membrane resistance (MOhm)
tau = 10 # membrane time constant (ms)
Ie = 1.5 # constant input current (nA)

# Initialize variables
t = np.arange(0, tmax+dt, dt) # time array
V = np.zeros(len(t)) # membrane potential array
V[0] = Vrest # initial potential
spikes = [] # list of spike times

# Simulate the neuron
for i in range(1, len(t)):
    # Update the membrane potential
    dV = (Vrest - V[i-1] + Rm * Ie) / tau # change in potential
    V[i] = V[i-1] + dV * dt # new potential
    
    # Check for spike
    if V[i] >= Vth:
        V[i] = Vreset # reset potential
        spikes.append(t[i]) # record spike time

# Plot the results
plt.figure()
plt.plot(t, V, label="Membrane potential")
plt.plot(spikes, np.ones(len(spikes))*Vth, 'ro', label="Spikes")
plt.xlabel("Time (ms)")
plt.ylabel("Potential (mV)")
plt.legend()
plt.show()
```

## References

- (1) Neuroscience - Wikipedia. https://en.wikipedia.org/wiki/Neuroscience.

- (2) Neuroscience | Psychology Today. https://www.psychologytoday.com/us/basics/neuroscience.

- (3) Neurosciences — Wikipédia. https://fr.wikipedia.org/wiki/Neurosciences.
  - [Leaky integrate-and-fire - Wikipedia](https://en.wikipedia.org/wiki/Leaky_integrate-and-fire).

- (4) Neuroscience - Wikipedia. https://en.wikipedia.org/wiki/Neuroscience.

- (5) Neurosciences — Wikipédia. https://fr.wikipedia.org/wiki/Neurosciences.

- (6) Neuroscience | Psychology Today. https://www.psychologytoday.com/us/basics/neuroscience.


