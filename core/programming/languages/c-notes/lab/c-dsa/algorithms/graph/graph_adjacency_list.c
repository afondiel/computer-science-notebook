#include <stdio.h>
#include <stdlib.h>
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
int main() {
    struct Graph* g = createGraph(3);
    addEdge(g, 0, 1); addEdge(g, 0, 2); addEdge(g, 1, 2);
    for (int i = 0; i < 3; i++) {
        printf("%d: ", i);
        for (struct Node* curr = g->adj[i]; curr; curr = curr->next) printf("%d ", curr->vertex);
        printf("\n");
    }
    return 0; // Memory leak omitted
}
