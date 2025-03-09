#include <queue>
#include <cstdint>
#include <functional>
#include <cassert>

struct Task {
    uint8_t priority; // Lower value = higher priority
    void (*handler)();
};

class TaskScheduler {
    std::priority_queue<Task, std::vector<Task>, 
        std::function<bool(const Task&, const Task&)>> queue{
        [](const Task& a, const Task& b) { return a.priority > b.priority; }
    };
    
public:
    void add_task(Task t) {
        queue.push(t);
    }

    void run_next() {
        if (!queue.empty()) {
            Task t = queue.top();
            queue.pop();
            t.handler();
        }
    }
};

// Test Task
void emergency_brake() { /* Safety-critical action */ }

void test_scheduler() {
    TaskScheduler ts;
    ts.add_task({1, emergency_brake});
    ts.add_task({3, []{ /* Low-priority task */ }});
    ts.run_next(); // Should execute emergency_brake first
}
