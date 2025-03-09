from collections import deque

class CircularBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def push(self, data):
        self.buffer.append(data)

    def pop(self):
        if self.buffer:
            return self.buffer.popleft()
        else:
            raise IndexError("Buffer is empty")

    def is_full(self):
        return len(self.buffer) == self.buffer.maxlen

    def is_empty(self):
        return len(self.buffer) == 0

# Test
def test_circular_buffer():
    cb = CircularBuffer(3)
    cb.push(1)
    cb.push(2)
    cb.push(3)
    assert cb.is_full()
    assert cb.pop() == 1
    cb.push(4)
    assert cb.pop() == 2
    assert not cb.is_empty()

test_circular_buffer()
