# **Embedded Rust - Beginner Core Concepts**  

## **Overview**  
**Embedded Rust** is a **safe, efficient, and modern systems programming language** designed for **resource-constrained embedded systems**. It leverages Rust’s **memory safety guarantees**, **zero-cost abstractions**, and **strong type system** to prevent common errors like null dereferencing and buffer overflows.  

This guide covers **beginner** Embedded Rust concepts, including:  
✅ **Why Rust for Embedded Systems?**  
✅ **Setting up an Embedded Rust Environment**  
✅ **Basic Program Structure**  
✅ **Memory Safety & Ownership in Embedded Rust**  
✅ **GPIO & Hardware Abstraction Layer (HAL)**  
✅ **Basic Concurrency with RTIC (Real-Time Interrupt-driven Concurrency)**  

---

## **Table of Contents**  
1. **Introduction to Embedded Rust**  
2. **Why Use Rust for Embedded Development?**  
3. **Setting Up the Development Environment**  
4. **Basic Program Structure in Embedded Rust**  
5. **Understanding Memory Safety & Ownership**  
6. **Working with GPIO & Peripherals**  
7. **Basic Concurrency in Embedded Rust**  
8. **Tools & Learning Resources**  

---

## **1. Introduction to Embedded Rust**  

### **What is Embedded Rust?**  
🔹 Embedded Rust is a subset of the Rust language that runs on **bare-metal microcontrollers (MCUs)** and other embedded devices.  
🔹 It enables **memory safety without garbage collection**, making it ideal for **real-time and safety-critical applications**.  
🔹 Rust's **strong type system** helps **catch bugs at compile time**, reducing runtime errors.  

### **Common Use Cases**  
✅ **IoT Devices**  
✅ **Industrial Automation**  
✅ **Aerospace & Automotive Systems**  
✅ **Robotics & Drones**  

---

## **2. Why Use Rust for Embedded Development?**  

| Feature         | Benefit for Embedded Systems |
|----------------|-----------------------------|
| **Memory Safety** | No null pointer dereferencing, no buffer overflows |
| **Zero-Cost Abstractions** | No runtime overhead from high-level constructs |
| **Concurrency without Data Races** | Safe multi-threading and interrupts |
| **No Standard Library (`#![no_std]`)** | Runs on bare-metal devices without OS dependencies |
| **Performance Comparable to C** | Direct control over hardware like C, but safer |

---

## **3. Setting Up the Development Environment**  

### **Required Tools**  
🔹 **Rust Toolchain (rustup, cargo, rustc)** – Install Rust:  
```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```
🔹 **cargo-generate** – For setting up embedded projects:  
```sh
cargo install cargo-generate
```
🔹 **probe-rs** – For flashing and debugging:  
```sh
cargo install probe-rs
```
🔹 **Target Support for Embedded MCUs** – Example for ARM Cortex-M:  
```sh
rustup target add thumbv7em-none-eabihf
```

---

## **4. Basic Program Structure in Embedded Rust**  

A minimal **"Blinky LED"** program using the **`embedded-hal`** crate:

```rust
#![no_std]
#![no_main]

use cortex_m_rt::entry;
use embedded_hal::digital::v2::OutputPin;
use panic_halt as _; // Panic handler

#[entry]
fn main() -> ! {
    let mut led = ... // Configure GPIO as output
    loop {
        led.set_high().unwrap(); // Turn LED ON
        cortex_m::asm::delay(10_000_000);
        led.set_low().unwrap(); // Turn LED OFF
        cortex_m::asm::delay(10_000_000);
    }
}
```

✔ **`#![no_std]`** – No standard library, required for bare-metal applications.  
✔ **`#![no_main]`** – Disables default `main` function (MCUs have custom boot logic).  
✔ **`cortex_m_rt::entry`** – Marks entry point for embedded Rust programs.  

---

## **5. Understanding Memory Safety & Ownership**  

Rust prevents **common memory bugs** found in C and C++:  
✅ **No Null Pointers** – Rust enforces explicit handling of `Option<T>`.  
✅ **No Buffer Overflows** – Rust enforces safe array indexing.  
✅ **No Data Races** – Rust’s ownership system prevents concurrent memory corruption.  

### **Ownership Example in Embedded Context**  
```rust
fn configure_led(mut led: gpio::Pin<Output>) {
    led.set_high().unwrap(); // LED ON
} // `led` is dropped here, preventing accidental reuse
```
✔ Prevents **use-after-free** and **double free** errors.  

---

## **6. Working with GPIO & Peripherals**  

### **Blink an LED using `embedded-hal`**  
```rust
use embedded_hal::digital::v2::OutputPin;

fn blink_led(mut led: impl OutputPin) {
    led.set_high().unwrap();
    cortex_m::asm::delay(10_000_000);
    led.set_low().unwrap();
}
```
✔ Uses **HAL traits** for portability across different microcontrollers.  

### **Reading a Button Input**  
```rust
use embedded_hal::digital::v2::InputPin;

fn read_button(button: impl InputPin) -> bool {
    button.is_high().unwrap()
}
```
✔ Abstracts hardware details for **code reusability**.  

---

## **7. Basic Concurrency in Embedded Rust**  

### **Using RTIC (Real-Time Interrupt-driven Concurrency)**  
RTIC helps manage **tasks, interrupts, and resource sharing** safely.  

#### **Example: LED toggling with an interrupt-driven timer**  
```rust
#[rtic::app(device = stm32f4)]
mod app {
    use rtic::cyccnt::U32Ext;

    #[resources]
    struct Resources {
        led: gpio::Pin<Output>,
    }

    #[task(binds = TIM2, resources = [led])]
    fn timer_interrupt(ctx: timer_interrupt::Context) {
        ctx.resources.led.toggle().unwrap();
    }
}
```
✔ **RTIC prevents race conditions** through compile-time checks.  

---

## **8. Tools & Learning Resources**  

### **Essential Tools for Embedded Rust**  
🔹 **`probe-rs`** – Flash and debug firmware  
🔹 **`cargo-embed`** – Easy embedded development  
🔹 **`defmt`** – Efficient logging for microcontrollers  

### **Learning Resources**  
📘 [The Embedded Rust Book](https://docs.rust-embedded.org/book/)  
📘 [Rust Embedded HAL Documentation](https://docs.rs/embedded-hal/)  
📘 [RTIC (Real-Time Interrupt-driven Concurrency)](https://rtic.rs/)  
📘 [Writing Embedded Rust for ARM Cortex-M](https://docs.rust-embedded.org/discovery/)  

---

## **Conclusion**  

🎯 **Key Takeaways**  
✅ Rust offers **memory safety, concurrency, and high performance** for embedded systems.  
✅ **`#![no_std]`** enables Rust to run on **bare-metal** microcontrollers.  
✅ **Embedded HAL** abstracts hardware, enabling **portability across MCUs**.  
✅ RTIC provides **safe task scheduling and interrupt management**.  

🚀 Next Steps
🔹 Try blinking an LED on an STM32 or ESP32 board.
🔹 Explore embedded Rust projects like drone firmware or sensor integration.
🔹 Learn real-time operating systems (RTOS) in Rust.