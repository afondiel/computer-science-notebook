#include <memory>
#include <stdexcept>

using namespace std;

template <typename T>
class FIFO {
    struct Node {
        T value;
        unique_ptr<Node> next = nullptr;

        Node(T _value): value(_value) {}
    };

    unique_ptr<Node> front = nullptr;
    unique_ptr<Node>* back = &front;

public:
    void enqueue(T _value) {
        *back = make_unique<Node>(_value);
        back = &(**back).next;
    }

    T dequeue() {
        if (front == nullptr)
            throw underflow_error("Nothing to dequeue");

        T value = front->value;
        front = move(front->next);
        
        return value;
    }
};