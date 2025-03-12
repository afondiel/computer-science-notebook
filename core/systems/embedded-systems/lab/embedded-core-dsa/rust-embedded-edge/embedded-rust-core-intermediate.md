# **Embedded Rust - Intermediate Core Concepts**  

## **Overview**  
**Embedded Rust** is a **modern, memory-safe systems programming language** designed for **bare-metal microcontrollers (MCUs)** and **real-time embedded applications**. This guide explores **intermediate** concepts such as:  
✅ **Advanced Memory Management** (heapless design, `alloc`, static mutability)  
✅ **Interrupts & Concurrency with RTIC**  
✅ **Peripheral Access & Hardware Abstraction Layer (HAL)**  
✅ **Embedded Communication Protocols (I2C, SPI, UART)**  
✅ **Efficient Debugging & Logging with `defmt`**  

---

## **Table of Contents**  
1. **Recap: Why Rust for Embedded Systems?**  
2. **Advanced Memory Management & Ownership**  
3. **Interrupt Handling & RTIC for Safe Concurrency**  
4. **Peripheral Access & Hardware Abstraction Layers (HALs)**  
5. **Communication Protocols (I2C, SPI, UART)**  
6. **Efficient Debugging & Logging**  
7. **Performance Optimization Strategies**  
8. **Recommended Tools & Learning Resources**  

---

## **1. Recap: Why Rust for Embedded Systems?**  

| Rust Feature       | Benefit for Embedded Systems |
|--------------------|-----------------------------|
| **Memory Safety** | Prevents buffer overflows & null pointer issues |
| **Concurrency without Data Races** | Safe multi-threading & interrupt handling |
| **Zero-Cost Abstractions** | No runtime performance penalty |
| **`#![no_std]` & `core` Library** | Works on bare-metal without OS dependencies |
| **Efficient Error Handling (`Result`, `Option`)** | Prevents crashes & undefined behavior |

---

## **2. Advanced Memory Management & Ownership**  

Rust **eliminates manual memory management issues** found in C/C++ through **ownership, borrowing, and lifetimes**.  

### **Static Mutability in Embedded Systems**  
Since embedded applications often require **global variables (e.g., hardware registers, shared resources)**, Rust provides:  
- **`static` variables** for **persistent state across function calls**  
- **`unsafe` mutable access** (only if absolutely necessary)  

#### **Example: Static GPIO Pin Management**
```rust
use cortex_m::interrupt::{free, Mutex};
use core::cell::RefCell;

static GPIO_LED: Mutex<RefCell<Option<gpio::Pin<Output>>>> = Mutex::new(RefCell::new(None));

fn init_peripherals() {
    let led = gpio::Pin::new();
    free(|cs| GPIO_LED.borrow(cs).replace(Some(led)));
}

fn toggle_led() {
    free(|cs| {
        if let Some(ref mut led) = *GPIO_LED.borrow(cs).borrow_mut() {
            led.toggle().unwrap();
        }
    });
}
```
✔ **`Mutex<RefCell<T>>`** ensures **safe mutable access** across interrupts.  
✔ **`free(|cs| ...)`** executes in a **critical section**, preventing data races.  

---

## **3. Interrupt Handling & RTIC for Safe Concurrency**  

Rust provides **safe, preemptive concurrency** via the **Real-Time Interrupt-driven Concurrency (RTIC) framework**.  

### **Basic RTIC Example (Blink LED on Timer Interrupt)**
```rust
#[rtic::app(device = stm32f4)]
mod app {
    use rtic::cyccnt::U32Ext;
    use embedded_hal::digital::v2::OutputPin;

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
✔ **RTIC automatically manages shared resource access**, preventing race conditions.  
✔ **Interrupts are prioritized and handled safely**, ensuring real-time execution.  

---

## **4. Peripheral Access & Hardware Abstraction Layers (HALs)**  

Rust provides **abstraction layers** for **register-level hardware control** via:  
- **Peripheral Access Crates (PACs)** – Directly map MCU registers (unsafe, low-level).  
- **Hardware Abstraction Layer (HAL) crates** – Safe and ergonomic hardware control.  

#### **Example: Controlling GPIO Using HAL**
```rust
use stm32f4xx_hal::gpio::{Output, PushPull, gpioa::PA5};
use stm32f4xx_hal::prelude::*;

fn init_led() -> PA5<Output<PushPull>> {
    let dp = stm32f4xx_hal::pac::Peripherals::take().unwrap();
    let gpioa = dp.GPIOA.split();
    gpioa.pa5.into_push_pull_output()
}
```
✔ **HAL crates simplify register interactions**, making embedded Rust more portable.  

---

## **5. Communication Protocols (I2C, SPI, UART)**  

Rust supports **standard embedded communication protocols** through HAL implementations.  

### **Interfacing with an I2C Sensor**  
```rust
use embedded_hal::blocking::i2c::WriteRead;

fn read_sensor<T: WriteRead>(i2c: &mut T, address: u8, register: u8) -> u8 {
    let mut buf = [0u8];
    i2c.write_read(address, &[register], &mut buf).unwrap();
    buf[0]
}
```
✔ **`embedded-hal` provides a common interface** across different hardware platforms.  
✔ **Ensures reusability across microcontrollers**.  

---

## **6. Efficient Debugging & Logging**  

### **Using `defmt` for Lightweight Logging**  
**`defmt`** is an efficient **binary logging framework** designed for embedded systems.  

#### **Example: Logging Events in an Embedded Application**
```rust
use defmt::{info, warn};
use panic_probe as _;

fn main() {
    info!("System initialized");
    warn!("Low battery detected!");
}
```
✔ **Reduces memory & CPU overhead**, compared to traditional logging.  

---

## **7. Performance Optimization Strategies**  

### **Minimizing Flash & RAM Usage**
- **Use `#[inline(always)]`** for performance-critical functions.  
- **Prefer `heapless::Vec` over `Vec`** to avoid heap allocation.  
- **Use `panic-halt` instead of `std::panic!`** to eliminate unnecessary dependencies.  

### **Reducing Power Consumption**
- **Use `WFI` (Wait-For-Interrupt) instructions** in the main loop.  
- **Configure MCU sleep modes (`STOP`, `STANDBY`)** to reduce power draw.  

---

## **8. Recommended Tools & Learning Resources**  

### **Essential Tools**  
🔹 **`probe-rs`** – Flash & debug embedded Rust programs  
🔹 **`cargo-embed`** – Easy firmware deployment  
🔹 **`cargo-binutils`** – Inspect compiled binaries  

### **Learning Resources**  
📘 [The Embedded Rust Book](https://docs.rust-embedded.org/book/)  
📘 [Real-Time Interrupt-driven Concurrency (RTIC)](https://rtic.rs/)  
📘 [Defmt: Lightweight Logging Framework](https://github.com/knurling-rs/defmt)  
📘 [PACs & HALs on `crates.io`](https://crates.io/keywords/embedded)  

---

## **Conclusion**  

🎯 **Key Takeaways**  
✅ Rust ensures **memory safety, efficient concurrency, and high performance** in embedded systems.  
✅ **RTIC** provides **real-time task scheduling & interrupt handling** without race conditions.  
✅ **Hardware Abstraction Layers (HALs)** simplify **portability across MCUs**.  
✅ **Defmt logging & debugging tools** make development easier & more efficient.  

🚀 **Next Steps**  
🔹 Implement **I2C, SPI, or UART communication with an external sensor**.  
🔹 Explore **real-time applications using RTIC**.  
🔹 Optimize **power consumption & memory usage** for battery-powered devices.  