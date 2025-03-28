# include <stdio.h>
# include <stdlib.h>
# include <stdbool.h>

int linear_search(int arr[], int size, int k)
{
    for (int i = 0; i < size; i++)
    {
        if(arr[i] == k )
        {
            return i; // Return the index of the found element
        }
    }
    return -1;
}

int main(void)
{
    int arr[] = {1,2,3,4,5};
    int size = sizeof(arr)/sizeof(arr[0]);
    int k = 3; // Expected: 2 => index of the 3rd element
    int result = linear_search(arr, size, k);
    if (result == -1)
        printf("Element is not present in array\n");
    else
        printf("Element is present at index %d\n", result);

    return 0;
}