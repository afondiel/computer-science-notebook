/************************************************************************************
/*          Binary Search                       
/* **********************************************************************************
    Desc: A *divide and conquer* algorithm used to find the position of the searched 
          element by finding the middle element of the array
          - *Faster* than linear search (also known as sequential search) for large dataset
          - *Lower* time-space complexity O(logn) (linear: O(n)) in Worst cases, and space O(1)
          - Prerequisites : an array should be in *sorted order*
          
          The algorithm works as follows:
          - Case 1: a[mid] < data, then left=mid+1.
          - Case 2: a[mid] > data, then right=mid-1
          - Case 3: a[mid] == data, element is found
         
          Binary search can be implemented in two ways:
            - Iterative
            - Recursive
          
***********************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#define SIZE 5

/* Iterative Binary search function
* - It takes the arr, left/low and right/high and the key/target element x 
* - It returns the index of the element if found, otherwise -1
*/ 
int binary_search_i(int arr[], int l, int r, int x) {
      
    // Check base case
    while (l <= r) { 
        int m = (l + (r - l)) / 2; 

        // Check if x is present at mid 
        if (arr[m] == x) 
            return m; 

        // If x greater, ignore left half 
        if (arr[m] < x){
            l = m + 1; 
        } 
        // If x is smaller, ignore right half 
        else
            r = m - 1; 
    } 

    // if we reach here, then element was 
    // not present 
    return -1; 
} 


int main(void)
{

    // Tests
    int arr[SIZE] = {1,2,3,4,5};
    //Found
    int x = 3; // Expected: 2 => index of the 3rd element
    int result = binary_search_i(arr, 0, SIZE-1, x); 
    if (result == -1)
        printf("Element is not present in array\n");
    else
        printf("Element is present at index %d\n", result);

    return 0;
}


