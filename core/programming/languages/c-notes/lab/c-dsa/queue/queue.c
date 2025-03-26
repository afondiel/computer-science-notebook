/* 
 * Queue implementation using array
 * 
 * enqueue: O(1)
 * dequeue: O(1)
 * 
 * Limitation: Fixed size
 *
 * Queue (FIFO)
 *                    +------------+
 *    front(head)     | 1 | 2 | 3  |  <-- rear(tail) // <= enqueue()
 *   (dequeue:del =>) +------------+ (<= enqueue:insert)
 *                      ^    ^   
 *                    front rear
 */

#include <stdio.h>

#define SIZE 3

// int queue[SIZE]; 
// int front = 0; 
// int rear = -1; 
// int count = 0;

// void enqueue(int val) { 
//     // if (count < SIZE) 
//     //     queue[++rear] = val, count++; 
//     // Increment rear and check bounds
//     if (rear + 1 >= SIZE) {
//         // Handle overflow error
//         return;
//     }
//     // rear++;
//     // queue[rear] = val;
//     queue[++rear] = val;

//     // Increment count
//     count++;
// }

// int dequeue() { 
//     if (count > 0) {
//         // Decrement count
//         count--;
//         // return queue[front++];
//         return queue[front++];
//     }
//     // Handle underflow error
//     return -1;
//     // return (count > 0) ? (count--, queue[front++]) : -1; 
// }

typedef struct {
    int data[SIZE];
    int front;
    int rear;
    int count;
} Queue;

void enqueue(Queue *q, int val) { 
    if (q->rear + 1 >= SIZE) {
        // Handle overflow error
        return;
    }
    q->data[++q->rear] = val;
    q->count++;
}

int dequeue(Queue *q) { 
    if (q->count > 0) {
        q->count--;
        return q->data[q->front++];
        // q->front = (q->front + 1) % SIZE; // Circular buffer
        // return q->data[q->front];

    }
    // Handle underflow error
    return -1;
}

int main() {
    // enqueue(1);
    // enqueue(2);
    // enqueue(3);
    // printf("%d %d %d", dequeue(), dequeue(), dequeue());
    // printf("\n");

    Queue q = {{0}, 0, -1, 0};

    enqueue(&q, 1); 
    enqueue(&q, 2); 
    enqueue(&q, 3);

    for(int i = 0; i < SIZE; i++) {
        printf("%d ", dequeue(&q));
    }

    printf("\n");   
    
    return 0;
}
