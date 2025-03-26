#include <stdio.h>
#define SIZE 3
int queue[SIZE], front = 0, rear = -1, count = 0;
void enqueue(int val) {
    if (count < SIZE) {
        rear = (rear + 1) % SIZE;
        queue[rear] = val;
        count++;
    }
}
int dequeue() {
    if (count > 0) {
        int val = queue[front];
        front = (front + 1) % SIZE;
        count--;
        return val;
    }
    return -1;
}
int main() {
    enqueue(1); enqueue(2); enqueue(3);
    printf("%d %d ", dequeue(), dequeue());
    return 0;
}
