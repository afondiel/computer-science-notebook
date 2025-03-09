#include <chrono>
#include <thread>
#include <cassert>

class Watchdog {
    std::chrono::milliseconds timeout;
    std::chrono::steady_clock::time_point last_ping;
    bool armed = false;

public:
    Watchdog(uint32_t ms) : timeout(ms) {}

    void arm() {
        armed = true;
        last_ping = std::chrono::steady_clock::now();
    }

    void ping() {
        last_ping = std::chrono::steady_clock::now();
    }

    bool check_expired() const {
        if (!armed) return false;
        auto elapsed = std::chrono::steady_clock::now() - last_ping;
        return elapsed > timeout;
    }
};

// Test Case
void test_watchdog() {
    Watchdog wd(100);
    wd.arm();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    wd.ping();
    assert(!wd.check_expired());
    std::this_thread::sleep_for(std::chrono::milliseconds(60));
    assert(wd.check_expired());
}
