#include <stdio.h>
#include <stdlib.h>
struct Node {
    int data;
    struct Node *left, *right;
};
void inorder(struct Node* root) {
    if (root) {
        inorder(root->left);
        printf("%d ", root->data);
        inorder(root->right);
    }
}
int main() {
    struct Node *root = malloc(sizeof(struct Node));
    root->data = 1; root->left = malloc(sizeof(struct Node)); root->right = malloc(sizeof(struct Node));
    root->left->data = 2; root->left->left = root->left->right = NULL;
    root->right->data = 3; root->right->left = root->right->right = NULL;
    inorder(root);
    free(root->left); free(root->right); free(root);
    return 0;
}
