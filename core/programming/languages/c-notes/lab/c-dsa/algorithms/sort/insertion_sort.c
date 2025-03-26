#include <stdio.h>
void insertionSort(int arr[], int n) {
    for (int i = 1; i < n; i++) {
        int key = arr[i], j = i - 1;
        while (j >= 0 && arr[j] > key) arr[j + 1] = arr[j--];
        arr[j + 1] = key;
    }
}
int main() {
    int arr[] = {4, 3, 2, 1};
    insertionSort(arr, 4);
    for (int i = 0; i < 4; i++) printf("%d ", arr[i]);
    return 0;
}
