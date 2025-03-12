# **2️⃣ ESP32 + Embedded Rust Example**  
**Target:** ESP32  
**Board:** ESP32-WROOM  

### **📌 Goal:**  
- **WiFi Communication** using `esp-wifi`  
- **Energy-efficient idle mode**  

### **🛠 Dependencies**
```toml
[dependencies]
esp-idf-sys = "0.30"
esp-idf-hal = "0.38"
embedded-svc = "0.24"
wifi = "0.1"
```

### **⚡ Code: WiFi + Low Power**
```rust
#![no_std]
#![no_main]

use embedded_svc::wifi::{AccessPointConfiguration, Configuration};
use esp_idf_hal::peripherals::Peripherals;
use esp_idf_sys as _;

#[entry]
fn main() {
    let peripherals = Peripherals::take().unwrap();
    let mut wifi = peripherals.wifi;
    
    wifi.set_configuration(&Configuration::AccessPoint(
        AccessPointConfiguration {
            ssid: "ESP_Rust".into(),
            password: "password123".into(),
            ..Default::default()
        },
    )).unwrap();

    loop {
        esp_idf_sys::esp_light_sleep_start(); // Low-power sleep mode
    }
}
```
✅ **ESP32 runs a WiFi Access Point**  
✅ **Energy-efficient with light sleep mode**  
