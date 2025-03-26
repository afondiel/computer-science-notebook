/* Two pointers technique
 * Find a pair of elements that sum to a target value
 * 
 * Time complexity: O(n)
 * Space complexity: O(1)
 *  
 * Limitation: Fixed size
 * 
 * 
 * Two pointers technique
 * +------------+
 * | 1 | 2 | 3  |  <-- right
 * +------------+
 * | 1 | 2 | 3  |  <-- left
 * +------------+
 *
 * 1. Sort the array
 * 2. Initialize two pointers, left and right
 * 3. Move left pointer to right if sum < target
 * 4. Move right pointer to left if sum > target    
 * 5. Return 1 if sum == target
 * 6. Return 0 if left >= right
 */
 
#include <stdio.h>

// Two pointers technique
// Find a pair of elements that sum to a target value
int find_pair(int arr[], int n, int target) {
    int left = 0;
    int right = n - 1;
    
    while (left < right) {
        int sum = arr[left] + arr[right];
        
        if (sum == target) {
            return 1;
        } else if (sum < target) {
            left++;
        } else {
            right--;
        }
    }
    
    return 0;
}

int main() {
    int arr[] = {1, 2, 3, 4, 5};
    int n = sizeof(arr) / sizeof(arr[0]);
    int target = 9;
    
    if (find_pair(arr, n, target)) {
        printf("Pair found\n");
    } else {
        printf("Pair not found\n");
    }
    
    return 0;
}



