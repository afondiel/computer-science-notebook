#include <stdio.h>
#include <stdlib.h>

struct Node {
    int data;
    struct Node* next;
};

typedef struct Node_t{
    int data;
    struct Node* next;
} Node_t;

// Traversal: print the linked list, end by NULL
void printList(struct Node* head) {
    struct Node* temp = head;
    while (temp != NULL) {
        printf("%d -> ", temp->data);
        temp = temp->next;
    }
    printf("NULL\n");
}

// Create a new node
struct Node* createNode(int data) {
    struct Node* newNode = (struct Node*)malloc(sizeof(struct Node));
    newNode->data = data;
    newNode->next = NULL;
    return newNode;
}

// Insertion: insert a new node at the beginning
struct Node* insert_at_the_beginning(struct Node* head, int data) {
    struct Node* newNode = createNode(data); // head already points to the NULL node
    // printList(head);
    // printList(newNode);
    if(head != NULL) {
        newNode->next = head;
    }
    head = newNode;
    return head;
}

void insert_with_ref(struct Node** pointerToHead, int data) {
    struct Node* newNode = createNode(data); // head already points to the NULL node
    if(*pointerToHead != NULL) {
        newNode->next = *pointerToHead;
    }
    *pointerToHead = newNode;
}

// Insertion: insert a new node at the end
struct Node* insert_at_the_end(struct Node* curr_head, int data) {
    struct Node* newNode = createNode(data);
    if (curr_head == NULL) {
        // If the list is empty, the new node becomes the head
        return newNode;
    }
    struct Node* temp = curr_head;
    while (temp->next != NULL) {
        temp = temp->next;
    }
    temp->next = newNode;
    return curr_head;
}

// Deletion: delete the first node
struct Node* del_first_node(struct Node* head) {
    if(head == NULL) {
        return NULL;
    }
    struct Node* temp = head;
    head = head->next;
    free(temp);
    return head;
}

// Iterative Deletion: O(n) time, O(1) space
void del_list(struct Node** head) {
    struct Node* current = *head;
    struct Node* nextNode;

    while(current != NULL) {
        nextNode = current->next;
        free(current);
        current = nextNode;
    }
    *head = NULL; // Ensure head is NULL after deletion
}

// Deletion: using recursion
// O(n), No loop, does not set 'head=NULL' (dangling ptr or stack overflow prone)
void del_list_recursive(struct Node* head) {
    if(head == NULL) { return;}
    del_list_recursive(head->next);
    free(head);
}

int main() {
    // ===== Beginner: Manual creation of linked list === 
    // struct Node *head = malloc(sizeof(struct Node)); 
    // struct Node *second = malloc(sizeof(struct Node));
    // Traversal process
    // head->data = 1; 
    // head->next = second;
    // second->data = 2; 
    // second->next = NULL;
    
    // for (struct Node* curr = head; curr; curr = curr->next){
    //     printf("%d ", curr->data);
    // }

    // printf("\n");
    // free(head);  // Free allocated memory
    // free(second); // Free allocated memory
    // =================================================
    
    // Advanced: Dynamic creation of linked list
    // Traversal process // Propagate the head
    // struct Node* head = createNode(1);
    // head->next        = createNode(2);
    // head->next->next  = createNode(3);
    
    // printList(head);
    
    // // Free allocated memory
    // struct Node* current = head;
    // struct Node* next;
    // while (current != NULL) {
    //     next = current->next;
    //     free(current);
    //     current = next;
    // }


    // Insertion: insert a new node at the beginning
    // struct Node* head = NULL;
    // head = insert_at_the_beginning(head, 1);
    // head = insert_at_the_beginning(head, 2);
    // head = insert_at_the_beginning(head, 3);
    // printList(head);
    // head = del_first_node(head);
    // head= del_first_node(head);
    // head = del_first_node(head);
    // printList(head);

    struct Node* head = NULL;
    head = insert_at_the_end(head, 1); // List: 1 -> NULL
    head = insert_at_the_end(head, 2); // List: 1 -> 2 -> NULL
    head = insert_at_the_end(head, 3); // List: 1 -> 2 -> 3 -> NULL
    printList(head); // Output: 1 -> 2 -> 3 -> NULL

    return 0;
}

// Related resources: /home/muntu/Work/growth/job/interview-prep/technical/lab/coding/c/c-basics/20-Structs/20-Structs.c
