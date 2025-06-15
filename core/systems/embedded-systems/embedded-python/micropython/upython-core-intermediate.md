# MicroPython Technical Notes
<!-- A rectangular diagram depicting an intermediate MicroPython pipeline, illustrating a microcontroller (e.g., ESP32) running a MicroPython interpreter, executing a script that manages multiple hardware components (e.g., sensor input, Wi-Fi communication), performing data processing (e.g., filtering sensor data), and producing outputs (e.g., sending data to a server, controlling actuators), with arrows indicating the flow from script execution to hardware interaction to networked output. -->

## Quick Reference
- **Definition**: MicroPython is a lightweight Python 3 implementation for microcontrollers, enabling complex hardware control, data processing, and network communication for tasks like IoT applications, sensor monitoring, and automation.
- **Key Use Cases**: Building IoT systems, real-time sensor data processing, and networked embedded applications with resource-constrained devices.
- **Prerequisites**: Familiarity with Python programming, basic microcontroller concepts (e.g., GPIO, I2C), and introductory knowledge of networking (e.g., HTTP, MQTT).

## Table of Contents
- [Introduction](#introduction)
- [Core Concepts](#core-concepts)
  - [Fundamental Understanding](#fundamental-understanding)
  - [Visual Architecture](#visual-architecture)
- [Implementation Details](#implementation-details)
  - [Intermediate Patterns](#intermediate-patterns)
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
- **What**: MicroPython allows users to write Python scripts on microcontrollers to control hardware (e.g., sensors, actuators), process data, and communicate over networks, leveraging Python’s simplicity for embedded systems.
- **Why**: It enables rapid development of complex embedded applications with low power consumption, supporting IoT and real-time tasks without the complexity of C/C++.
- **Where**: Applied in smart home devices, environmental monitoring systems, and industrial IoT for tasks like remote sensing or device control.

## Core Concepts
### Fundamental Understanding
- **Basic Principles**:
  - MicroPython runs an optimized Python interpreter on microcontrollers, supporting advanced features like multitasking, file systems, and network stacks within memory constraints (e.g., 32-512 KB RAM).
  - Scripts interact with hardware via protocols like I2C, SPI, or UART, and process data using lightweight algorithms suited for low compute power.
  - Networking capabilities (e.g., Wi-Fi, MQTT) enable IoT applications, allowing devices to send/receive data to/from servers or other devices.
- **Key Components**:
  - **Microcontroller**: Devices like ESP32 or Raspberry Pi Pico with GPIO, ADC, and communication interfaces.
  - **MicroPython Interpreter**: Executes scripts with a REPL, file system, and libraries for hardware (e.g., `machine`) and networking (e.g., `network`, `urequests`).
  - **Hardware Interfaces**:
    - **GPIO/ADC**: For digital/analog input/output.
    - **I2C/SPI**: For sensor/actuator communication.
  - **Networking**: Wi-Fi or Bluetooth for data exchange, often using protocols like HTTP or MQTT.
  - **Data Processing**: Lightweight filtering or aggregation for real-time applications.
- **Common Misconceptions**:
  - Misconception: MicroPython is too slow for real-time tasks.
    - Reality: Optimized code and hardware acceleration (e.g., ESP32’s dual-core) enable fast execution for many tasks.
  - Misconception: MicroPython supports all Python libraries.
    - Reality: Only a subset of standard libraries and specific MicroPython modules are available due to resource limits.

### Visual Architecture
```mermaid
graph TD
    A[Python Script <br> (e.g., Sensor + Wi-Fi)] --> B[MicroPython Interpreter <br> (ESP32/Pico)]
    B --> C[Hardware Interaction <br> (GPIO, I2C, ADC)]
    B --> D[Data Processing <br> (Filtering/Aggregation)]
    C --> E[Output <br> (e.g., Actuator Control)]
    D --> F[Network Output <br> (e.g., MQTT/HTTP)]
```
- **System Overview**: The diagram shows a MicroPython script executed on a microcontroller, interacting with hardware, processing data, and sending results over a network.
- **Component Relationships**: The interpreter orchestrates hardware control, data processing, and network communication for integrated outputs.

## Implementation Details
### Intermediate Patterns
```python
# MicroPython script for ESP32: Read temperature sensor (DHT11) and send data via MQTT
from machine import Pin
import dht
import utime
import network
import ujson
from umqtt.simple import MQTTClient

# Wi-Fi credentials
SSID = "your_ssid"
PASSWORD = "your_password"

# MQTT configuration
MQTT_BROKER = "broker.hivemq.com"
CLIENT_ID = "esp32_sensor"
TOPIC = b"sensor/temperature"

# Sensor configuration
dht_pin = Pin(4, Pin.IN)  # DHT11 on GPIO 4
sensor = dht.DHT11(dht_pin)

# Connect to Wi-Fi
def connect_wifi():
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    if not wlan.isconnected():
        print("Connecting to Wi-Fi...")
        wlan.connect(SSID, PASSWORD)
        while not wlan.isconnected():
            utime.sleep(1)
    print("Wi-Fi connected:", wlan.ifconfig())

# Connect to MQTT broker
def connect_mqtt():
    client = MQTTClient(CLIENT_ID, MQTT_BROKER)
    client.connect()
    print("Connected to MQTT broker")
    return client

# Read and filter sensor data (simple moving average)
def read_temperature(samples=5):
    temps = []
    for _ in range(samples):
        try:
            sensor.measure()
            temp = sensor.temperature()
            temps.append(temp)
            utime.sleep(0.5)
        except OSError as e:
            print("Sensor error:", e)
    if temps:
        return sum(temps) / len(temps)  # Moving average
    return None

# Main loop
connect_wifi()
mqtt_client = connect_mqtt()

while True:
    temp = read_temperature()
    if temp is not None:
        payload = ujson.dumps({"temperature": temp})
        mqtt_client.publish(TOPIC, payload)
        print(f"Published: {payload}")
    else:
        print("Failed to read temperature")
    utime.sleep(10)  # Publish every 10 seconds
```
- **Step-by-Step Setup**:
  1. **Hardware**: Connect an ESP32 board with a DHT11 temperature sensor to GPIO 4 (with a pull-up resistor if needed).
  2. **Install MicroPython**:
     - Download ESP32 MicroPython firmware (.bin) from micropython.org.
     - Flash using `esptool.py`: `esptool.py --port /dev/ttyUSB0 write_flash -z 0x1000 firmware.bin`.
  3. **Install Libraries**: Copy `umqtt/simple.py` from MicroPython’s `micropython-lib` to the board using Thonny or `ampy`.
  4. **Install Tools**: Use Thonny IDE (thonny.org) or `ampy` for file transfer (`pip install adafruit-ampy`).
  5. **Configure Wi-Fi**: Update `SSID` and `PASSWORD` in the script.
  6. **Save and Run**:
     - Save code as `main.py`.
     - Upload to ESP32 using Thonny or `ampy --port /dev/ttyUSB0 put main.py`.
     - Reset the board to run (monitor output via Thonny or serial terminal at 115200 baud).
- **Code Walkthrough**:
  - Uses `dht` library to read temperature from a DHT11 sensor.
  - Connects to Wi-Fi using `network.WLAN` and to an MQTT broker with `umqtt.simple`.
  - Filters sensor data with a moving average to reduce noise.
  - Publishes temperature data as JSON to an MQTT topic every 10 seconds.
- **Common Pitfalls**:
  - Incorrect DHT11 wiring or missing pull-up resistor causing read errors.
  - Wi-Fi connection failures due to incorrect credentials or weak signal.
  - Missing `umqtt` library or incompatible MQTT broker configuration.

## Real-World Applications
### Industry Examples
- **Use Case**: Smart agriculture monitoring.
  - Collects soil moisture and temperature data, sending it to a cloud server.
- **Implementation Patterns**: Use MicroPython to read sensors via I2C and publish data over MQTT.
- **Success Metrics**: >99% data delivery rate, <50mW average power.

### Hands-On Project
- **Project Goals**: Build a temperature monitoring system with MQTT.
- **Implementation Steps**:
  1. Set up the ESP32 with a DHT11 sensor.
  2. Flash MicroPython and upload the `umqtt` library.
  3. Upload the above code with your Wi-Fi credentials.
  4. Monitor MQTT messages using a client (e.g., MQTT Explorer) on `broker.hivemq.com`.
- **Validation Methods**: Verify temperature data appears in MQTT client; confirm stable Wi-Fi and sensor readings.

## Tools & Resources
### Essential Tools
- **Development Environment**: Thonny IDE, PlatformIO for advanced workflows.
- **Key Hardware**: ESP32, Raspberry Pi Pico, DHT11 or similar sensors.
- **Key Software**: MicroPython firmware, `esptool.py`, `ampy` for file management.
- **Libraries**: `umqtt`, `uasyncio` for networking and multitasking.

### Learning Resources
- **Documentation**: MicroPython docs (https://docs.micropython.org/en/latest/), MQTT guide (https://docs.micropython.org/en/latest/library/umqtt.simple.html).
- **Tutorials**: ESP32 IoT projects (https://randomnerdtutorials.com/micropython-esp32-esp8266/).
- **Community Resources**: MicroPython Forum (forum.micropython.org), r/micropython.

## References
- MicroPython documentation: https://docs.micropython.org/en/latest/
- ESP32 MicroPython: https://docs.micropython.org/en/latest/esp32/quickref.html
- MQTT library: https://github.com/micropython/micropython-lib/tree/master/micropython/umqtt.simple
- IoT with MicroPython: https://www.iotworldtoday.com/embedded-iot/micropython-for-iot

## Appendix
- **Glossary**:
  - **I2C/SPI**: Protocols for sensor communication.
  - **MQTT**: Lightweight messaging protocol for IoT.
  - **Moving Average**: Simple filter for smoothing sensor data.
- **Setup Guides**:
  - Flash ESP32: `esptool.py --port /dev/ttyUSB0 write_flash -z 0x1000 esp32.bin`.
  - Install `ampy`: `pip install adafruit-ampy`.
- **Code Templates**:
  - I2C sensor: Use `machine.I2C` for devices like BME280.
  - Async tasks: Use `uasyncio` for concurrent operations.