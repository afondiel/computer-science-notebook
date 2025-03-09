import threading
import time

class WatchdogTimer:
    def __init__(self, timeout_seconds):
        self.timeout_seconds = timeout_seconds
        self.last_reset_time = time.time()
        self.timer_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self.is_running = False

    def start(self):
        self.is_running = True
        self.timer_thread.start()

    def reset(self):
        self.last_reset_time = time.time()

    def stop(self):
        self.is_running = False

    def _watchdog_loop(self):
        while self.is_running:
            if time.time() - self.last_reset_time > self.timeout_seconds:
                print("Watchdog timer expired! System reset required.")
                break
            time.sleep(0.1)

# Example usage
watchdog = WatchdogTimer(timeout_seconds=5)
watchdog.start()
time.sleep(3)
watchdog.reset()  # Reset before timeout
time.sleep(6)     # Let it expire after this point
