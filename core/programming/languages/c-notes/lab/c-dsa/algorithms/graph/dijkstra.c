#include <stdio.h>
#include <stdlib.h>
#define INF 99999
#define V 3
int minDistance(int dist[], int visited[], int n) {
    int min = INF, min_index;
    for (int v = 0; v < n; v++)
        if (!visited[v] && dist[v] <= min) min = dist[v], min_index = v;
    return min_index;
}
void dijkstra(int adj[V][V], int src) {
    int dist[V], visited[V];
    for (int i = 0; i < V; i++) dist[i] = INF, visited[i] = 0;
    dist[src] = 0;
    for (int count = 0; count < V-1; count++) {
        int u = minDistance(dist, visited, V);
        visited[u] = 1;
        for (int v = 0; v < V; v++)
            if (!visited[v] && adj[u][v] && dist[u] != INF && dist[u] + adj[u][v] < dist[v])
                dist[v] = dist[u] + adj[u][v];
    }
    for (int i = 0; i < V; i++) printf("%d ", dist[i]);
}
int main() {
    int adj[V][V] = {{0, 4, 8}, {4, 0, 2}, {8, 2, 0}};
    dijkstra(adj, 0);
    return 0;
}
