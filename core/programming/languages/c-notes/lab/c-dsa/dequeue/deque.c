#include <stdio.h>
#include <stdlib.h>
struct Node {
    int data;
    struct Node *prev, *next;
};
struct Deque {
    struct Node *front, *rear;
};
void push_front(struct Deque* dq, int val) {
    struct Node* node = malloc(sizeof(struct Node));
    node->data = val; node->prev = NULL; node->next = dq->front;
    if (!dq->front) dq->rear = node; else dq->front->prev = node;
    dq->front = node;
}
int pop_front(struct Deque* dq) {
    if (!dq->front) return -1;
    struct Node* temp = dq->front;
    int val = temp->data;
    dq->front = temp->next;
    if (dq->front) dq->front->prev = NULL; else dq->rear = NULL;
    free(temp);
    return val;
}
int main() {
    struct Deque dq = {NULL, NULL};
    push_front(&dq, 1); push_front(&dq, 2);
    printf("%d %d", pop_front(&dq), pop_front(&dq));
    return 0;
}
