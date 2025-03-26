#include <stdio.h>
#define SIZE 10
void swap(int* a, int* b) { int t = *a; *a = *b; *b = t; }
void heapify(int arr[], int n, int i) {
    int largest = i, left = 2*i + 1, right = 2*i + 2;
    if (left < n && arr[left] > arr[largest]) largest = left;
    if (right < n && arr[right] > arr[largest]) largest = right;
    if (largest != i) {
        swap(&arr[i], &arr[largest]);
        heapify(arr, n, largest);
    }
}
int main() {
    int arr[] = {1, 3, 4};
    int n = 3;
    for (int i = n/2 - 1; i >= 0; i--) heapify(arr, n, i);
    for (int i = 0; i < n; i++) printf("%d ", arr[i]);
    return 0;
}
