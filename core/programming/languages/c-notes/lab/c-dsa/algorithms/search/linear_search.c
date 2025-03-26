#include <stdio.h>
int linearSearch(int arr[], int n, int key) {
    for (int i = 0; i < n; i++) if (arr[i] == key) return i;
    return -1;
}
int main() {
    int arr[] = {1, 2, 3, 4};
    int result = linearSearch(arr, 4, 3);
    printf("%d", result);
    return 0;
}
