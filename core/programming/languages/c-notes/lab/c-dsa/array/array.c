#include <stdio.h>

int main() {
    
    int arr[] = {1, 2, 3, 4, 5}; // size of array is 5
    // int arr[5] = {1, 2, 3, 4, 5}; // size of array is 5
    // int arr[5] = {1, 2, 3}; // size of array is 5, rest of the elements will be initialized to 0
    // int arr[5] = {0}; // size of array is 5, all elements will be initialized to 0
    // int arr[] = {0}; // size of array is 1, element will be initialized to 0
    
    for (int i = 0; i < 5; i++){
        printf("%d ", arr[i]);
    }
    
    printf("\n");

    // Beginner: Calculate the sum of all elements in the array
    int sum = 0;
    for (int i = 0; i < 5; i++) {
        sum += arr[i];
    }
    printf("Sum of array elements: %d\n", sum);

    // Intermediate: Reverse the array
    printf("Reversed array: ");
    for (int i = 4; i >= 0; i--) {
        printf("%d ", arr[i]);
    }
    printf("\n");

    // Advanced: Find the maximum and minimum elements in the array
    // int max = arr[0], min = arr[0];
    // for (int i = 1; i < 5; i++) {
    //     if (arr[i] > max) {
    //         max = arr[i];
    //     }
    //     if (arr[i] < min) {
    //         min = arr[i];
    //     }
    // }
    // printf("Maximum element: %d\n", max);
    // printf("Minimum element: %d\n", min);

    // int max = arr[0], min = arr[0];
    // for (int i = 1; i < 10000; i++) {
    //     if (arr[i] > max) {
    //         max = arr[i];
    //     } else if (arr[i] < min) { // Use else if to reduce unnecessary comparisons
    //         min = arr[i];
    //     }
    // }
    // printf("Maximum element: %d\n", max);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    printf("Minimum element: %d\n", min);

    // Parallel computing approach for large scale data
    // int max, min;
    // int n = 100000000; // 100 million elements // int = 4 bytes, 100 million * 4 = 400 MB
    // int n = 1000000000; // 1 billion elements // int = 4 bytes, 1 billion * 4 = 4 GB
    // int n = 10000000000; // 10 billion elements // int = 4 bytes, 10 billion * 4 = 40 GB
    // int n = 100000000000; // 100 billion elements // int = 4 bytes, 100 billion * 4 = 400 GB
    // int n = 1000000000000; // 1 trillion elements // int = 4 bytes, 1 trillion * 4 = 4 TB
    // int n = 10000000000000; // 10 trillion elements // int = 4 bytes, 10 trillion * 4 = 40 TB
    // int n = 100000000000000; // 100 trillion elements // int = 4 bytes, 100 trillion * 4 = 400 TB
    // int n = 1000000000000000; // 1 quadrillion elements // int = 4 bytes, 1 quadrillion * 4 = 4 PB
    // int n = 10000000000000000; // 10 quadrillion elements // int = 4 bytes, 10 quadrillion * 4 = 40 PB
    // int n = 100000000000000000; // 100 quadrillion elements // int = 4 bytes, 100 quadrillion * 4 = 400 PB 

    // Initialize max/min with first pair (reduces comparisons by 1)
    int arr1[30000];
    int max, min;
    // int n = 30000; // (signed) int: 2^16 = [-2^16, 2^16] = [-32768, 32768] = 2^16 = 65536 = 64 KB (2^10 = 1024 bytes)
    int n = sizeof(arr1) / sizeof(arr1[0]);

    for (int i = 0; i < n; i++) {
        arr1[i] = i;
    }

    if (arr1[0] > arr1[1]) {
        max = arr1[0];
        min = arr1[1];
    } else {
        max = arr1[1];
        min = arr1[0];
    }

    // Process remaining elements in pairs
    for (int i = 2; i < n; i += 2) {
        int a = arr1[i];
        int b = arr1[i + 1];

        if (a > b) {
            if (a > max) max = a;
            if (b < min) min = b;
        } else {
            if (b > max) max = b;
            if (a < min) min = a;
        }
    }

    printf("Maximum: %d\nMinimum: %d\n", max, min);

    // Min and max algorithm using modulo operator
    int arr2[30000];
    int max2, min2;
    int n2 = sizeof(arr2) / sizeof(arr2[0]);
    // continue


    return 0;
}
