#include <stdio.h>
#define V 3
int main() {
    int adj[V][V] = {{0, 1, 1}, {0, 0, 1}, {0, 0, 0}};
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) printf("%d ", adj[i][j]);
        printf("\n");
    }
    return 0;
}
