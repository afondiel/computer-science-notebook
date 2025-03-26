    //-----------------------------------------------//
    //          Binary Search                       //
    //-----------------------------------------------//


    /*
        DEF : It finds the position of the searched element by finding the middle element of the array
        - *Faster* than linear search (also known as sequential search) for large dataset
        - *Lower* time-space complexity O(log2n) and linear : O(n) in Worst cases, otherwise O(1)both

        prerequises : an array should be in *sorted order*

        There are three cases used in the binary search:

        - Case 1: data<a[mid] then left = mid+1.

        - Case 2: data>a[mid] then right=mid-1

        - Case 3: data = a[mid] // element is found

    */

    #include <stdio.h>
    #include <stdlib.h>
    #include <stdbool.h>

    #define SIZE 10

    /* === iterative binary search ====
    * - An iterative binary search function (size GIVEN).
    * - It gets arr, size, and target elements 
    * - It returns location of x in given array arr[l..r] if present, otherwise -1
    */ 

    // bool binary_search_i(int* arr, int size, int k)
    bool binary_search_i0(int arr[], int size, int k)
    {

        int i, j, m;
        bool found = false;
        i =0;
        j = size;
        //BS
        while(!found && (j - i)>1){
            m = (i + j)/2;
            found = (arr[m] == k);
            if(arr[m] > k)
                j = m;
            else
                i = m;
        }
        if(arr[i] == k)
            return true;
        else
            return false;

    }

    /* === iterative binary search ====
    * - An iterative binary search function (size NOT given).
    * - It gets the arr, Left(l) and Right(r) and target (x) elements 
    * - It returns location of x in given array arr[l..r] if present, otherwise -1
    */ 
    int binary_search_i1(int arr[], int l, int r, int x) 
    { 
        while (l <= r) { 
            int m = l + (r - l) / 2; 
    
            // Check if x is present at mid 
            if (arr[m] == x) 
                return m; 
    
            // If x greater, ignore left half 
            if (arr[m] < x) 
                l = m + 1; 
    
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
        int arr[SIZE] = {1,2,3,4,5,6,7,8,9,10};

        //if(binary_search_i0(arr, SIZE, 8))  //Found
        if(binary_search_i0(arr, SIZE, 11))   // NOT found
            printf("Element found\n");
        else
            printf("Element NOT found\n");

        return 0;
    }


