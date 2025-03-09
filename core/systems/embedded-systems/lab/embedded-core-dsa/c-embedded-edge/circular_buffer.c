#include <stdint.h>
#include <stdbool.h>

#define BUFFER_SIZE 32

typedef struct {
    uint8_t buffer[BUFFER_SIZE];
    uint8_t head;
    uint8_t tail;
    bool full;
} CircularBuffer;

void cb_init(CircularBuffer* cb) {
    cb->head = 0;
    cb->tail = 0;
    cb->full = false;
}

void cb_push(CircularBuffer* cb, uint8_t data) {
    cb->buffer[cb->head] = data;
    cb->head = (cb->head + 1) % BUFFER_SIZE;
    if(cb->full) cb->tail = (cb->tail + 1) % BUFFER_SIZE;
    cb->full = (cb->head == cb->tail);
}

uint8_t cb_pop(CircularBuffer* cb) {
    if(!cb->full && (cb->head == cb->tail)) return 0;
    
    uint8_t data = cb->buffer[cb->tail];
    cb->tail = (cb->tail + 1) % BUFFER_SIZE;
    cb->full = false;
    return data;
}

// Test
#include <stdio.h>
void test_circular_buffer() {
    CircularBuffer cb;
    cb_init(&cb);
    for(int i=0; i<BUFFER_SIZE; i++) {
        cb_push(&cb, i);
    }
    for(int i=0; i<BUFFER_SIZE; i++) {
        printf("%d ", cb_pop(&cb));
    }
    printf("\n");
}


int main(void)
{
    test_circular_buffer();
    return 0;
}
