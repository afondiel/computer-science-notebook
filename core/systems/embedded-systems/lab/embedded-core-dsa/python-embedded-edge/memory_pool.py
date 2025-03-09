class MemoryPool:
    def __init__(self, block_size, num_blocks):
        self.pool = [None] * num_blocks
        self.free_list = list(range(num_blocks))

    def allocate(self):
        if not self.free_list:
            raise MemoryError("No free memory blocks available")
        return self.free_list.pop()

    def deallocate(self, block_id):
        if block_id == len(self.pool):
            raise ValueError("Invalid block ID")
        self.free_list.append(block_id)

# Test
def test_memory_pool():
    pool = MemoryPool(block_size=32, num_blocks=4)
    block1 = pool.allocate()
    block2 = pool.allocate()
    pool.deallocate(block1)
    block3 = pool.allocate()
    assert block1 == block3

test_memory_pool()
