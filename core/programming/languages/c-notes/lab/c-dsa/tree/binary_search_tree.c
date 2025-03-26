#include <stdio.h>
#include <stdlib.h>
struct Node {
    int data;
    struct Node *left, *right;
};
struct Node* insert(struct Node* root, int val) {
    if (!root) {
        root = malloc(sizeof(struct Node));
        root->data = val; root->left = root->right = NULL;
    } else if (val < root->data) root->left = insert(root->left, val);
    else root->right = insert(root->right, val);
    return root;
}
void inorder(struct Node* root) {
    if (root) {
        inorder(root->left);
        printf("%d ", root->data);
        inorder(root->right);
    }
}
int main() {
    struct Node* root = NULL;
    root = insert(root, 3); insert(root, 1); insert(root, 4);
    inorder(root);
    return 0; // Memory leak omitted for brevity
}
