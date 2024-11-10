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


bool exists(int arr[], int size, int k);

int main(void)
{

    int arr[SIZE] ={1,2,3,4,5,6,7,8,9,10};

    //if(exists(arr, SIZE, 8))  //Found
    if(exists(arr, SIZE,11))   // NOT found
        printf("Element found\n");
    else
        printf("Element NOT found\n");

    return 0;
}


bool exists(int arr[], int size, int k)
{

    int i, j, m;
    bool foun_d = false;
    i =0;
    j = size;
    //BS
    while(!foun_d && (j - i)>1){
        m = (i + j)/2;
        foun_d = (arr[m] == k);
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
