#include <stdio.h>
#include <stdlib.h>

int main() {

    int *arr = malloc(3 * sizeof(int));
    
    arr[0] = 1;
    arr[1] = 2; 
    arr[2] = 3;
    
    arr = realloc(arr, 4 * sizeof(int));
    arr[3] = 4;
    
    for (int i = 0; i < 4; i++){
        printf("%d ", arr[i]);
    }

    printf("\n");
    
    free(arr);

    return 0;
}

