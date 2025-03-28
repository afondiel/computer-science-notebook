/* ****************************************************************************** 
/*   Two pointers
// - The two pointers technique is a common algorithmic pattern that uses two pointers to solve problems.
// - It is often used to solve problems related to arrays or linked lists.
// - The two pointers can be used to traverse the data structure in different ways, such as:
//   - One pointer moving forward and the other moving backward.
//   - Both pointers moving in the same direction at different speeds.
// - The two pointers technique can be used to solve problems such as:
//   - Finding the middle of a linked list.
//   - Finding the intersection of two linked lists.
//   - Merging two sorted arrays.
*****************************************************************************/

# include <stdio.h>
# include <stdlib.h>
# include <string.h>
# include <stdbool.h>

// ========================== Use Cases =====================

// ### 1. **Finding a Pair with a Given Sum in a Sorted Array**
//    - Use two pointers: one starting at the beginning and the other at the end of the array.
//    - Move the pointers inward based on the sum comparison.
//    **Example Problem**: Find two numbers in a sorted array that add up to a target sum.

int findPairWithSum(int arr[], int n, int target) {
    int left = 0, right = n - 1;
    while (left < right) {
        int sum = arr[left] + arr[right];
        if (sum == target) {
            printf("Pair found: (%d, %d)\n", arr[left], arr[right]);
            return 1;
        } else if (sum < target) {
            left++;
        } else {
            right--;
        }
    }
    printf("No pair found.\n");
    return 0;
}

// ### 2. **Reversing an Array or String**
// - Use two pointers: one at the start and one at the end.
// - Swap the elements and move the pointers toward each other.
// **Example Problem**: Reverse a string.

void reverseString(char str[]) {
    int left = 0, right = strlen(str) - 1;
    while (left < right) {
        char temp = str[left];
        str[left] = str[right];
        str[right] = temp;
        left++;
        right--;
    }
}

// ### 3. **Checking for a Palindrome**
// - Use two pointers: one at the start and one at the end.
// - Compare characters and move the pointers inward.
// **Example Problem**: Check if a string is a palindrome.


int isPalindrome(char str[]) {
    int left = 0, right = strlen(str) - 1;
    while (left < right) {
        if (str[left] != str[right]) {
            return 0; // Not a palindrome
        }
        left++;
        right--;
    }
    return 1; // Palindrome
}

// ### 4. **Merging Two Sorted Arrays**
// - Use two pointers, one for each array.
// - Compare elements and merge them into a new array.
// **Example Problem**: Merge two sorted arrays into one sorted array.

void mergeSortedArrays(int arr1[], int n1, int arr2[], int n2, int result[]) {
    int i = 0, j = 0, k = 0;
    while (i < n1 && j < n2) {
        if (arr1[i] < arr2[j]) {
            result[k++] = arr1[i++];
        } else {
            result[k++] = arr2[j++];
        }
    }
    while (i < n1) {
        result[k++] = arr1[i++];
    }
    while (j < n2) {
        result[k++] = arr2[j++];
    }
}


// ### 5. **Finding the Middle of a Linked List**
// - Use two pointers: one moves one step at a time, and the other moves two steps at a time.
// - When the faster pointer reaches the end, the slower pointer will be at the middle.
// **Example Problem**: Find the middle node of a linked list.

struct Node {
    int data;
    struct Node* next;
};

struct Node* findMiddle(struct Node* head) {
    struct Node *slow = head, *fast = head;
    while (fast != NULL && fast->next != NULL) {
        slow = slow->next;
        fast = fast->next->next;
    }
    return slow; // Middle node
}


// ### 6. **Removing Duplicates from a Sorted Array**
// - Use two pointers: one to track the current unique element and the other to iterate through the array.
// **Example Problem**: Remove duplicates in-place from a sorted array.


int removeDuplicates(int arr[], int n) {
    if (n == 0) return 0;
    int j = 0;
    for (int i = 1; i < n; i++) {
        if (arr[i] != arr[j]) {
            j++;
            arr[j] = arr[i];
        }
    }
    return j + 1; // New length of the array
}


// ### 7. **Trapping Rainwater Problem**
// - Use two pointers to calculate the trapped water by comparing the heights of bars from both ends.

// **Example Problem**: Calculate the amount of water trapped between bars.

int trapRainwater(int height[], int n) {
    int left = 0, right = n - 1;
    int leftMax = 0, rightMax = 0, water = 0;
    while (left <= right) {
        if (height[left] < height[right]) {
            if (height[left] >= leftMax) {
                leftMax = height[left];
            } else {
                water += leftMax - height[left];
            }
            left++;
        } else {
            if (height[right] >= rightMax) {
                rightMax = height[right];
            } else {
                water += rightMax - height[right];
            }
            right--;
        }
    }
    return water;
}


// ### 8. **Partitioning an Array**
// - Use two pointers to partition an array around a pivot (e.g., for QuickSort).

// **Example Problem**: Partition an array into two parts based on a pivot.


int partition(int arr[], int low, int high) {
    int pivot = arr[high];
    int i = low - 1;
    for (int j = low; j < high; j++) {
        if (arr[j] < pivot) {
            i++;
            int temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }
    }
    int temp = arr[i + 1];
    arr[i + 1] = arr[high];
    arr[high] = temp;
    return i + 1;
}

int main(void) {
    // Example usage of the two pointers technique
    int arr[] = {1, 2, 3, 4, 5};
    int n = sizeof(arr) / sizeof(arr[0]);
    int target = 6;
    findPairWithSum(arr, n, target);

    char str[] = "hello";
    reverseString(str);
    printf("Reversed string: %s\n", str);

    char palindrome[] = "racecar";
    if (isPalindrome(palindrome)) {
        printf("%s is a palindrome.\n", palindrome);
    } else {
        printf("%s is not a palindrome.\n", palindrome);
    }

    int arr1[] = {1, 3, 5};
    int arr2[] = {2, 4, 6};
    int result[] = {0};
    int n1 = sizeof(arr1) / sizeof(arr1[0]);
    mergeSortedArrays(arr1, 3, arr2, 3, result);
    printf("Merged array: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", result[i]);
    }
    printf("\n");

    return 0;
}