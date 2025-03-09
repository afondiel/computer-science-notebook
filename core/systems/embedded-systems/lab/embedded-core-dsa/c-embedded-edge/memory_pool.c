#include <stdint.h>

#define POOL_SIZE 64

typedef union {
    uint32_t data;
    union MemoryBlock* next;
} MemoryBlock;

typedef struct {
    MemoryBlock blocks[POOL_SIZE];
    MemoryBlock* free_list;
} MemoryPool;

void pool_init(MemoryPool* pool) {
    for(int i=0; i<POOL_SIZE-1; i++) {
        pool->blocks[i].next = &pool->blocks[i+1];
    }
    pool->blocks[POOL_SIZE-1].next = NULL;
    pool->free_list = &pool->blocks[0];
}

void* pool_alloc(MemoryPool* pool) {
    if(!pool->free_list) return NULL;
    
    MemoryBlock* block = pool->free_list;
    pool->free_list = block->next;
    return &block->data;
}

void pool_free(MemoryPool* pool, void* ptr) {
    MemoryBlock* block = (MemoryBlock*)ptr;
    block->next = pool->free_list;
    pool->free_list = block;
}

int main(void)
{
    MemoryPool pool;
    pool_init(&pool);
    
    uint32_t* data1 = pool_alloc(&pool);
    *data1 = 42;
    
    uint32_t* data2 = pool_alloc(&pool);
    *data2 = 123;
    
    pool_free(&pool, data1);
    pool_free(&pool, data2);
    
    return 0;
}