/* Stack (LIFO) implementation using array
    * 
    * push: O(1)
    * pop: O(1)
    * 
    * Limitation: Fixed size
    * 
    * Stack (LIFO)
    * +------------+
    * | Last(head) | <--top (cursor)
    * +------------+
    * |    ...     |
    * +------------+
    * | First(base)|
    * +------------+
*/


#include <stdio.h>

#define SIZE 3

int stack[SIZE];
int top = -1;

void push(int val) {
    if (top < (SIZE - 1))
        stack[++top] = val; 
}

int pop() { 
    return (top >= 0) ? stack[top--] : -1; 
}

int main() {
    push(1); 
    push(2); 
    push(3);
    printf("%d %d %d", pop(), pop(), pop());
    printf("\n");
    
    return 0;
}
