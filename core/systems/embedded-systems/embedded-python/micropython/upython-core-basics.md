# MicroPython Technical Notes
<!-- A rectangular diagram illustrating the MicroPython process, showing a microcontroller (e.g., ESP32) running a MicroPython interpreter, executing a simple script (e.g., blinking an LED), interacting with hardware (e.g., GPIO pins), and producing an output (e.g., LED on/off), with arrows indicating the flow from script to interpreter to hardware to output. -->

## Quick Reference
- **Definition**: MicroPython is a lightweight implementation of Python 3 designed to run on microcontrollers and resource-constrained devices, enabling easy programming of hardware for tasks like controlling sensors or LEDs.
- **Key Use Cases**: Prototyping IoT devices, controlling hardware in embedded systems, and educational projects with microcontrollers.
- **Prerequisites**: Basic understanding of Python programming and familiarity with microcontrollers.

## Table of Contents
- [Introduction](#introduction)
- [Core Concepts](#core-concepts)
  - [Fundamental Understanding](#fundamental-understanding)
  - [Visual Architecture](#visual-architecture)
- [Implementation Details](#implementation-details)
  - [Basic Implementation](#basic-implementation)
- [Real-World Applications](#real-world-applications)
  - [Industry Examples](#industry-examples)
  - [Hands-On Project](#hands-on-project)
- [Tools & Resources](#tools--resources)
  - [Essential Tools](#essential-tools)
  - [Learning Resources](#learning-resources)
- [References](#references)
- [Appendix](#appendix)
  - [Glossary](#glossary)
  - [Setup Guides](#setup-guides)
  - [Code Templates](#code-templates)

## Introduction
- **What**: MicroPython is a version of Python optimized for microcontrollers, allowing users to write Python scripts to control hardware like LEDs, sensors, or motors.
- **Why**: It simplifies embedded programming with Python’s easy syntax, enabling rapid prototyping and reducing the complexity of traditional C-based development.
- **Where**: Used in IoT devices, robotics, educational platforms, and hobbyist projects for tasks like home automation or environmental monitoring.

## Core Concepts
### Fundamental Understanding
- **Basic Principles**:
  - MicroPython runs a Python interpreter on microcontrollers, executing scripts to interact with hardware components like GPIO pins, sensors, or displays.
  - It provides a subset of Python 3 features, optimized for limited memory and processing power (e.g., 16-512 KB RAM, 8-32 MHz CPUs).
  - Scripts control hardware via libraries that interface with pins, timers, or communication protocols (e.g., I2C, SPI).
- **Key Components**:
  - **Microcontroller**: A small computer (e.g., ESP32, Raspberry Pi Pico) with CPU, memory, and GPIO pins for hardware control.
  - **MicroPython Interpreter**: A compact runtime that executes Python code on the microcontroller, including a REPL (Read-Eval-Print Loop) for interactive coding.
  - **GPIO (General Purpose Input/Output)**: Pins on the microcontroller used to send or receive signals (e.g., turn an LED on/off).
  - **Libraries**: Modules like `machine` and `utime` for hardware control and timing.
- **Common Misconceptions**:
  - Misconception: MicroPython is the same as desktop Python.
    - Reality: It’s a lightweight version with fewer features, tailored for microcontrollers.
  - Misconception: MicroPython requires internet connectivity.
    - Reality: It runs offline, though some boards support Wi-Fi for IoT tasks.

### Visual Architecture
```mermaid
graph TD
    A[Python Script <br> (e.g., Blink LED)] --> B[MicroPython Interpreter <br> (Running on Microcontroller)]
    B --> C[Hardware Interaction <br> (e.g., GPIO Pins)]
    C --> D[Output <br> (e.g., LED On/Off)]
```
- **System Overview**: The diagram shows a Python script executed by the MicroPython interpreter on a microcontroller, interacting with hardware to produce an output.
- **Component Relationships**: The script drives the interpreter, which controls hardware via GPIO, resulting in observable actions.

## Implementation Details
### Basic Implementation
```python
# MicroPython script to blink an LED on an ESP32 or Raspberry Pi Pico
from machine import Pin
import utime

# Configure GPIO pin for LED
led = Pin(2, Pin.OUT)  # Pin 2 (adjust for your board, e.g., GP25 for Pico)

# Blink LED in a loop
while True:
    led.value(1)  # Turn LED on
    utime.sleep(0.5)  # Wait 0.5 seconds
    led.value(0)  # Turn LED off
    utime.sleep(0.5)  # Wait 0.5 seconds
```
- **Step-by-Step Setup**:
  1. **Hardware**: Get a MicroPython-compatible microcontroller (e.g., ESP32, Raspberry Pi Pico) and connect an LED (with a 220Ω resistor) to GPIO pin 2 (or GP25 for Pico) and GND.
  2. **Install MicroPython**:
     - Download the MicroPython firmware (.uf2 or .bin) for your board from micropython.org.
     - Flash the firmware (e.g., for Pico, hold BOOTSEL, drag .uf2 to the drive; for ESP32, use `esptool.py`).
  3. **Install Tools**: Install a serial terminal (e.g., PuTTY, minicom) or IDE like Thonny (thonny.org).
  4. **Save Code**: Save the code as `main.py` on your computer.
  5. **Upload and Run**:
     - Connect the board via USB.
     - Use Thonny to upload `main.py` to the board’s filesystem.
     - Reset the board to run the script (LED should blink).
- **Code Walkthrough**:
  - Imports `machine.Pin` for GPIO control and `utime` for delays.
  - Configures pin 2 as an output for the LED.
  - Loops to toggle the LED on/off every 0.5 seconds using `led.value` and `utime.sleep`.
- **Common Pitfalls**:
  - Incorrect pin number (check board pinout diagram).
  - Missing resistor causing LED damage or unreliable behavior.
  - Firmware not flashed correctly, preventing MicroPython from running.

## Real-World Applications
### Industry Examples
- **Use Case**: IoT temperature monitor.
  - Reads a sensor and sends data to a server using an ESP32.
- **Implementation Patterns**: Use MicroPython to read sensor data via GPIO and send it over Wi-Fi.
- **Success Metrics**: Reliable data collection, low power consumption (<100mW).

### Hands-On Project
- **Project Goals**: Build a blinking LED circuit with MicroPython.
- **Implementation Steps**:
  1. Set up the circuit with an LED on pin 2 (or GP25 for Pico).
  2. Flash MicroPython firmware to your board.
  3. Upload the above code using Thonny.
  4. Observe the LED blinking every 0.5 seconds.
- **Validation Methods**: Confirm LED blinks consistently; verify code runs on reset.

## Tools & Resources
### Essential Tools
- **Development Environment**: Thonny IDE, Mu Editor, or serial terminals (PuTTY, minicom).
- **Key Hardware**: ESP32, Raspberry Pi Pico, or other MicroPython-compatible boards.
- **Key Software**: MicroPython firmware, `esptool.py` for flashing (if needed).

### Learning Resources
- **Documentation**: MicroPython docs (https://docs.micropython.org/en/latest/), ESP32 guide (https://docs.micropython.org/en/latest/esp32/quickref.html).
- **Tutorials**: MicroPython basics (https://learn.adafruit.com/micropython-basics).
- **Community Resources**: MicroPython Forum (forum.micropython.org), r/micropython on Reddit.

## References
- MicroPython documentation: https://docs.micropython.org/en/latest/
- ESP32 MicroPython guide: https://docs.micropython.org/en/latest/esp32/quickref.html
- Raspberry Pi Pico MicroPython: https://www.raspberrypi.com/documentation/microcontrollers/micropython.html
- Embedded Python overview: https://en.wikipedia.org/wiki/MicroPython

## Appendix
- **Glossary**:
  - **GPIO**: General Purpose Input/Output pins for hardware control.
  - **REPL**: Interactive shell for real-time coding.
  - **Firmware**: Low-level software enabling MicroPython on a microcontroller.
- **Setup Guides**:
  - Flash MicroPython: Follow https://micropython.org/download/.
  - Install Thonny: `pip install thonny` or download from thonny.org.
- **Code Templates**:
  - Read sensor: Use `machine.ADC` for analog inputs.
  - Wi-Fi connection: Use `network.WLAN` for IoT tasks.