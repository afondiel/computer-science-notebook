#include <array>
#include <stdexcept>
#include <cassert>

template <typename T, size_t N>
class CircularBuffer {
    std::array<T, N> buffer;
    size_t head = 0;
    size_t tail = 0;
    bool full = false;

public:
    void push(const T& item) {
        buffer[head] = item;
        head = (head + 1) % N;
        if (full) tail = (tail + 1) % N;
        full = (head == tail);
    }

    T pop() {
        if (empty()) throw std::runtime_error("Buffer empty");
        T val = buffer[tail];
        full = false;
        tail = (tail + 1) % N;
        return val;
    }

    bool empty() const { return !full && (head == tail); }
    bool is_full() const { return full; }
};

// Test Case
void test_circular_buffer() {
    CircularBuffer<int, 3> cb;
    cb.push(10); cb.push(20); cb.push(30);
    assert(cb.is_full());
    assert(cb.pop() == 10);
    cb.push(40);
    assert(cb.pop() == 20);
}
