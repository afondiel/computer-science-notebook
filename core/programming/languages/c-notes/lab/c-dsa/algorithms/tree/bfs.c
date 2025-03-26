#include <stdio.h>
#include <stdlib.h>
#define SIZE 10
struct Queue {
    int items[SIZE], front, rear, count;
};
void enqueue(struct Queue* q, int val) {
    if (q->count < SIZE) q->items[++q->rear] = val, q->count++;
}
int dequeue(struct Queue* q) {
    if (q->count > 0) return q->count--, q->items[q->front++];
    return -1;
}
struct Node {
    int vertex;
    struct Node* next;
};
struct Graph {
    struct Node** adj;
    int vertices;
};
struct Graph* createGraph(int v) {
    struct Graph* g = malloc(sizeof(struct Graph));
    g->vertices = v;
    g->adj = malloc(v * sizeof(struct Node*));
    for (int i = 0; i < v; i++) g->adj[i] = NULL;
    return g;
}
void addEdge(struct Graph* g, int src, int dest) {
    struct Node* node = malloc(sizeof(struct Node));
    node->vertex = dest; node->next = g->adj[src];
    g->adj[src] = node;
}
void BFS(int start, struct Graph* g) {
    int visited[g->vertices];
    for (int i = 0; i < g->vertices; i++) visited[i] = 0;
    struct Queue q = {{0}, 0, -1, 0};
    visited[start] = 1;
    enqueue(&q, start);
    while (q.count > 0) {
        int v = dequeue(&q);
        printf("%d ", v);
        for (struct Node* curr = g->adj[v]; curr; curr = curr->next)
            if (!visited[curr->vertex]) {
                visited[curr->vertex] = 1;
                enqueue(&q, curr->vertex);
            }
    }
}
int main() {
    struct Graph* g = createGraph(3);
    addEdge(g, 0, 1); addEdge(g, 0, 2); addEdge(g, 1, 2);
    BFS(0, g);
    return 0; // Memory leak omitted
}
