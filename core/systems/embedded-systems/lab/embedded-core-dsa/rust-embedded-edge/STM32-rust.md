# **1️⃣ STM32 + Embedded Rust Example**  
**Target MCU:** STM32F4xx  
**Board:** STM32F411 Black Pill / Nucleo F401RE  

### **📌 Goal:**  
- Read **sensor data using DMA**  
- Schedule tasks using **RTIC**  
- Implement **low-power mode** with WFI  

### **🛠 Required Dependencies**
Add these to `Cargo.toml`:  
```toml
[dependencies]
cortex-m = "0.7"
cortex-m-rt = "0.7"
rtic = "2.0.0"
stm32f4xx-hal = { version = "0.14", features = ["rt", "stm32f411"] }
panic-halt = "0.2"
```

### **⚡ Code: RTIC + DMA + Low Power**
```rust
#![no_std]
#![no_main]

use cortex_m::asm::wfi;
use cortex_m_rt::entry;
use rtic::app;
use stm32f4xx_hal::{
    adc::Adc,
    dma::{config::DmaConfig, Transfer, StreamsTuple},
    pac, prelude::*, 
    rcc::RccExt,
};

#[app(device = stm32f4xx_hal::pac, peripherals = true)]
mod app {
    use super::*;

    #[shared]
    struct Shared {
        sensor_data: heapless::Vec<u16, 512>,
    }

    #[local]
    struct Local {
        adc_dma: Transfer<dma::Stream0, adc::Adc, u16>,
    }

    #[init]
    fn init(ctx: init::Context) -> (Shared, Local) {
        let dp = ctx.device;
        let rcc = dp.RCC.constrain();
        let clocks = rcc.cfgr.freeze();
        let dma = StreamsTuple::new(dp.DMA2);
        let adc = Adc::adc1(dp.ADC1, true, &clocks);

        let adc_dma = Transfer::init_peripheral_to_memory(
            dma.0, 
            adc, 
            [0u16; 512], 
            DmaConfig::default()
        );

        (Shared { sensor_data: heapless::Vec::new() }, Local { adc_dma })
    }

    #[idle]
    fn idle(_: idle::Context) -> ! {
        loop {
            wfi(); // Low-power sleep mode
        }
    }

    #[task(binds = DMA2_STREAM0, priority = 2, local = [adc_dma])]
    fn dma_complete(ctx: dma_complete::Context) {
        let adc_dma = ctx.local.adc_dma;
        adc_dma.stop();
        // Process data
    }
}
```
✅ **DMA streams sensor data, freeing CPU**  
✅ **RTIC schedules tasks efficiently**  
✅ **Device enters sleep mode when idle**  

