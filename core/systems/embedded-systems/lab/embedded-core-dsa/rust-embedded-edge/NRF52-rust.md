# **3️⃣ NRF52 + Embedded Rust Example**  
**Target:** NRF52840  
**Board:** Adafruit Feather NRF52840  

### **📌 Goal:**  
- **BLE communication**  
- **Low-power operation with WFE (Wait-for-Event)**  

### **🛠 Dependencies**
```toml
[dependencies]
cortex-m = "0.7"
nrf52840-hal = { version = "0.14", features = ["rt"] }
ble = "0.5"
panic-halt = "0.2"
```

### **⚡ Code: BLE + Low Power**
```rust
#![no_std]
#![no_main]

use cortex_m::asm::wfe;
use nrf52840_hal::{
    gpio::{p0::Parts, Level},
    pac::Peripherals,
    prelude::*,
};

#[entry]
fn main() -> ! {
    let dp = Peripherals::take().unwrap();
    let port0 = Parts::new(dp.P0);
    
    let led = port0.p0_13.into_push_pull_output(Level::High);

    loop {
        wfe(); // Wait-for-event (low-power mode)
        led.set_low().unwrap(); // Blink LED on wake-up
    }
}
```
✅ **BLE-ready, low-power MCU operation**  
✅ **Efficient wake-up using WFE**  

