use core::sync::atomic::{AtomicBool, Ordering};
use embedded_hal::watchdog::{Watchdog, WatchdogEnable};

pub struct SafeWatchdog<W: Watchdog + WatchdogEnable> {
    watchdog: W,
    enabled: AtomicBool,
}

impl<W: Watchdog + WatchdogEnable> SafeWatchdog<W> {
    pub fn new(watchdog: W) -> Self {
        SafeWatchdog {
            watchdog,
            enabled: AtomicBool::new(false),
        }
    }

    pub fn feed(&mut self) {
        if self.enabled.load(Ordering::Relaxed) {
            self.watchdog.feed();
        }
    }

    pub fn enable(&mut self, period: W::Time) {
        self.watchdog.start(period);
        self.enabled.store(true, Ordering::Release);
    }

    pub fn disable(&mut self) {
        self.enabled.store(false, Ordering::Release);
    }
}

// Usage (with stm32f4xx_hal as an example)
use stm32f4xx_hal::{pac, watchdog::IndependentWatchdog};

let dp = pac::Peripherals::take().unwrap();
let mut watchdog = IndependentWatchdog::new(dp.IWDG);
let mut safe_watchdog = SafeWatchdog::new(watchdog);

safe_watchdog.enable(100.millis());
// In main loop:
safe_watchdog.feed();
