#include <stdio.h>
void selectionSort(int arr[], int n) {
    for (int i = 0; i < n-1; i++) {
        int min = i;
        for (int j = i+1; j < n; j++) if (arr[j] < arr[min]) min = j;
        int temp = arr[i]; arr[i] = arr[min]; arr[min] = temp;
    }
}
int main() {
    int arr[] = {4, 3, 2, 1};
    selectionSort(arr, 4);
    for (int i = 0; i < 4; i++) printf("%d ", arr[i]);
    return 0;
}
