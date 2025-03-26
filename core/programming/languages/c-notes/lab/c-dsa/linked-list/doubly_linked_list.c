#include <stdio.h>
#include <stdlib.h>
struct Node {
    int data;
    struct Node *prev, *next;
};
int main() {
    struct Node *first = malloc(sizeof(struct Node)), *second = malloc(sizeof(struct Node));
    first->data = 1; first->prev = NULL; first->next = second;
    second->data = 2; second->prev = first; second->next = NULL;
    for (struct Node* curr = first; curr; curr = curr->next) printf("%d ", curr->data);
    free(first); free(second);
    return 0;
}
