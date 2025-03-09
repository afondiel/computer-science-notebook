# **Zephyr RTOS - Intermediate Core Concepts**  

## **Introduction to Zephyr RTOS**  
**Zephyr RTOS** is a scalable, real-time operating system designed for **embedded, IoT, industrial, and automotive applications**. It offers:  
✅ **Real-time scheduling** for deterministic performance.  
✅ **Multi-threading with fine-grained synchronization**.  
✅ **Extensive driver support for peripherals**.  
✅ **Built-in networking and security features**.

---

## **Core Concepts in Zephyr RTOS (Intermediate Level)**  

### **1. Advanced Threading & Scheduling**  
Zephyr uses a **preemptive, priority-based scheduler** with:  
✅ **Time slicing** – Shares CPU time among threads.  
✅ **Cooperative scheduling** – Threads must yield manually.  
✅ **Priority-based preemption** – Higher-priority threads can interrupt lower-priority ones.  

🔹 **Example: Time Slicing in Zephyr**  
```c
#include <zephyr.h>
#include <sys/printk.h>

void thread_function(void *p1, void *p2, void *p3) {
    while (1) {
        printk("Thread running: %s\n", (char *)p1);
        k_yield();  // Yield to allow other threads to run
    }
}

K_THREAD_DEFINE(thread1, 1024, thread_function, "T1", NULL, NULL, 2, 0, 0);
K_THREAD_DEFINE(thread2, 1024, thread_function, "T2", NULL, NULL, 2, 0, 0);
```
✔ **Demonstrates time slicing by alternating between two threads**.

---

### **2. Advanced Synchronization and IPC**  
Zephyr supports multiple **inter-thread communication mechanisms**:  
✅ **Semaphores** – Synchronize between threads.  
✅ **Mutexes** – Prevent race conditions in shared resources.  
✅ **Message queues, FIFOs, and mailboxes** – Facilitate inter-thread data exchange.  

🔹 **Example: Using a FIFO for Inter-thread Communication**  
```c
#include <zephyr.h>
#include <sys/printk.h>

struct k_fifo my_fifo;
struct data_item {
    void *fifo_reserved;  // Reserved for FIFO
    int value;
};

void producer_thread(void) {
    struct data_item item;
    while (1) {
        item.value = k_uptime_get_32();
        k_fifo_put(&my_fifo, &item);
        k_sleep(K_MSEC(500));
    }
}

void consumer_thread(void) {
    while (1) {
        struct data_item *received = k_fifo_get(&my_fifo, K_FOREVER);
        printk("Received: %d\n", received->value);
    }
}

K_THREAD_DEFINE(producer, 1024, producer_thread, NULL, NULL, NULL, 5, 0, 0);
K_THREAD_DEFINE(consumer, 1024, consumer_thread, NULL, NULL, NULL, 5, 0, 0);
```
✔ **Demonstrates real-time data sharing between threads using a FIFO**.

---

### **3. Memory Management & Dynamic Allocation**  
Zephyr provides:  
✅ **Heap memory allocation** using `k_malloc()`.  
✅ **Memory pools** for efficient allocation of fixed-size blocks.  
✅ **Thread stack management** with runtime stack monitoring.  

🔹 **Example: Using a Memory Pool**  
```c
K_MEM_POOL_DEFINE(my_pool, 64, 256, 4, 4);

void my_thread(void) {
    void *ptr = k_mem_pool_malloc(&my_pool, 128);
    if (ptr) {
        printk("Memory allocated!\n");
        k_free(ptr);
    }
}

K_THREAD_DEFINE(thread_id, 1024, my_thread, NULL, NULL, NULL, 5, 0, 0);
```
✔ **Efficient memory handling prevents heap fragmentation**.

---

### **4. Device Driver and Peripheral Management**  
Zephyr provides a **hardware abstraction layer (HAL)** for:  
✅ GPIO, I2C, SPI, UART, PWM, ADC, CAN  
✅ Wireless connectivity (Wi-Fi, Bluetooth, LoRa)  
✅ Sensor frameworks (I2C/SPI-based sensors)  

🔹 **Example: UART Communication**  
```c
#include <zephyr.h>
#include <drivers/uart.h>

#define UART_DEVICE DT_LABEL(DT_NODELABEL(uart0))

void main() {
    const struct device *uart_dev = device_get_binding(UART_DEVICE);
    if (!uart_dev) {
        printk("UART device not found\n");
        return;
    }
    uart_poll_out(uart_dev, 'H');
    uart_poll_out(uart_dev, 'i');
}
```
✔ **Interfaces with the UART driver to send characters**.

---

### **5. Power Management**  
Zephyr includes **fine-grained power control**:  
✅ **Tickless idle** – Reduces CPU wake-ups.  
✅ **System power states** – Light sleep, deep sleep.  
✅ **Device power management** – Manages peripheral power states.  

🔹 **Example: Configuring Power Management**  
```c
#include <zephyr.h>

void main() {
    while (1) {
        printk("Entering low power mode\n");
        k_sleep(K_SECONDS(1));  // Triggers sleep mode
    }
}
```
✔ **Minimizes power usage in IoT and battery-operated devices**.

---

## **Real-World Applications**
🔹 **Zephyr in Industrial IoT** – Used in **sensor fusion, motor control, and industrial automation**.  
🔹 **Zephyr in Automotive** – Supports **CAN communication, ECU control, and ADAS systems**.  
🔹 **Zephyr in Wearables** – Optimized for **low-power applications with Bluetooth connectivity**.  

---

## **Getting Started with Zephyr Development**  
### **1. Installing Zephyr SDK**  
```sh
west init -m https://github.com/zephyrproject-rtos/zephyr.git zephyrproject
cd zephyrproject
west update
```
### **2. Building & Flashing a Sample Application**  
```sh
west build -b <board_name> samples/hello_world
west flash
```
✔ **Deploys Zephyr RTOS on supported boards like STM32, nRF, and ESP32**.

---

## **Conclusion**  
This guide covers Zephyr RTOS **core concepts at an intermediate level**, including:  
✅ **Thread scheduling & synchronization**  
✅ **Memory management**  
✅ **Peripheral driver interfaces**  
✅ **Power management**  
