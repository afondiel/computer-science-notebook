#include <stdio.h>

int binarySearch(int arr[], int low, int high, int key) {
    while (low <= high) {
        int mid = low + (high - low) / 2;
        
        if (arr[mid] == key) 
            return mid;
        else if 
            (arr[mid] < key) low = mid + 1;
        else 
            high = mid - 1;
    }
    return -1;
}

int main() {

    int arr[] = {1, 2, 3, 4};
    
    int result = binarySearch(arr, 0, 3, 3);
    
    printf("%d", result);
    
    return 0;
}
