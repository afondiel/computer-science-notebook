template <typename T, size_t PoolSize>
class MemoryPool {
    union Node {
        T data;
        Node* next;
    };

    Node pool[PoolSize];
    Node* free_list;

public:
    MemoryPool() {
        for (size_t i = 0; i < PoolSize - 1; ++i) {
            pool[i].next = &pool[i + 1];
        }
        pool[PoolSize - 1].next = nullptr;
        free_list = pool;
    }

    T* allocate() {
        if (!free_list) throw std::bad_alloc();
        Node* node = free_list;
        free_list = free_list->next;
        return &node->data;
    }

    void deallocate(T* ptr) {
        Node* node = reinterpret_cast<Node*>(ptr);
        node->next = free_list;
        free_list = node;
    }
};

// Test Case
void test_memory_pool() {
    MemoryPool<float, 10> pool;
    float* ptr = pool.allocate();
    assert(ptr != nullptr);
    pool.deallocate(ptr);
}
