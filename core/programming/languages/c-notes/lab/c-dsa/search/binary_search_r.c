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

# include <stdio.h>
# include <stdlib.h>
# include <math.h>
# include <stdint.h>

// A recursive binary search function. 
// It returns the location of x in given array arr[l..r] is present, otherwise -1

int binary_search_r(int arr[], int l, int r, int x) 
{ 
	if (r >= l) { 
		int mid = l + (r - l) / 2; 

		// If the element is present at the middle 
		// itself 
		if (arr[mid] == x) 
			return mid; 

		// If element is smaller than mid, then 
		// it can only be present in left subarray 
		if (arr[mid] > x) 
			return binary_search_r(arr, l, mid - 1, x); 

		// Else the element can only be present 
		// in right subarray 
		return binary_search_r(arr, mid + 1, r, x); 
	} 

	// We reach here when element is not 
	// present in array 
	return -1; 
} 


int main(void) 
{ 
	int arr[] = {1,2,3,4,5};
	int n = sizeof(arr) / sizeof(arr[0]); 
	int x = 3; // Expected: 2 => index of the 3rd element
	int result = binary_search_r(arr, 0, n - 1, x); 
	
	if (result == -1) 
		printf("Element is not present in array"); 
	else
		printf("Element is present at index %d\n", result);

	return 0; 
} 

// References: 
// - GeeksForGeeks (https://www.geeksforgeeks.org/binary-search/#iterative-binary-search-algorithm) 
// - Wikipedia (https://en.wikipedia.org/wiki/Binary_search_algorithm)