# Quantum Computing - Notes

- [Overview](#overview)
- [Applications](#applications)
- [Tools and Frameworks](#tools-and-frameworks)
- [References](#references)


## Overview

![](https://azurecomcdn.azureedge.net/cvt-ceb7cdb75e3dccdf445c6b048ac025a2cf77a3352f654addaf1b880313cec3f3/images/page/resources/cloud-computing-dictionary/what-is-quantum-computing/supersition.jpg)

**Quantum Computing** : It's the use of quantum mechanics to run calculations on specialized hardware.


## Applications

- Artificial Intelligence
- optimization and simulation, and data management and searching.
- Cloud Computing
- Healthcare
- cybersecurity
- data analytics
- ...


## Tools and Frameworks

1. ProjectQ
2. Cirq
3. Q-CTRL Python Open Controls
4. Quantify
5. Intel Quantum Simulator
6. Perceval
7. Mitaq Tool
8. Berkeley Quantum Synthesis Toolkit
9. QCircuits
10. Yao
11. Silq
12. Paddle Quantum
13. Tequila
14. Qulacs
15. staq
16. Bayesforge
17. Bluqat
18. Quantum Programming Studio
19. Quirk
20. QuEST 
21. XACC
22. Quantum++
23. Quantum Inspire
24.QuCAT
1.  QuTiP
2.  OpenFermion
3.  TensorFlow Quantum
4.  Quipper
5.  QX Quantum Computing Simulator
6.  Quantum Algorithm Zoo
7.  ScaffCC
8.  TriQ
9.  Qbsolv from D-Wave
10. Quantum Computing Playground
11. Microsoft LIQUi|>
Other Quantum Computing Developer Tools

Src: [The Quantum insider](https://thequantuminsider.com/2022/05/27/quantum-computing-tools/)

## Quantum Computing Algorithms
@TODO

## Hello World!

```python

import cirq

# Pick a qubit.
qubit = cirq.GridQubit(0, 0)

# Create a circuit
circuit = cirq.Circuit(
    cirq.X(qubit)**0.5,  # Square root of NOT.
    cirq.measure(qubit, key='m')  # Measurement.
)
print("Circuit:")
print(circuit)

# Simulate the circuit several times.
simulator = cirq.Simulator()
result = simulator.run(circuit, repetitions=20)
print("Results:")
print(result)

```

Expected Output:

```
Circuit:
(0, 0): ───X^0.5───M('m')───
Results:
m=11000111111011001000
```
More examples on [Cirq Github repo](https://github.com/quantumlib/Cirq/tree/master)

## References

Wikipedia: 

- [Quantum Computing](https://en.wikipedia.org/wiki/Quantum_computing)
- [Quantum Programming](https://en.wikipedia.org/wiki/Quantum_programming)


Google:

- [Google Quantum AI Lab](https://quantumai.google/)

Microsoft:

- [Azure - Cloud Computing Dictionary](https://azure.microsoft.com/en-us/resources/cloud-computing-dictionary/what-is-quantum-computing/#introduction)

Amazon AWS:
- [Quantum Technologies - What Is Quantum Computing](https://aws.amazon.com/what-is/quantum-computing/?nc1=h_ls)
- [Quantum Technologies - Amazon Braket](https://aws.amazon.com/braket/)
  
> ### "If you think you understand quantum mechanics, you don't understand quantum mechanics"
> ### ~ Richard Feynman

