#include <stdio.h>
#include <stdlib.h>
#define SIZE 10
struct Entry {
    int key, value;
    struct Entry* next;
};
struct HashTable {
    struct Entry** table;
};
struct HashTable* createTable() {
    struct HashTable* ht = malloc(sizeof(struct HashTable));
    ht->table = malloc(SIZE * sizeof(struct Entry*));
    for (int i = 0; i < SIZE; i++) ht->table[i] = NULL;
    return ht;
}
void insert(struct HashTable* ht, int key, int value) {
    int index = key % SIZE;
    struct Entry* entry = malloc(sizeof(struct Entry));
    entry->key = key; entry->value = value; entry->next = ht->table[index];
    ht->table[index] = entry;
}
int main() {
    struct HashTable* ht = createTable();
    insert(ht, 1, 10); insert(ht, 11, 20);
    for (int i = 0; i < SIZE; i++)
        for (struct Entry* e = ht->table[i]; e; e = e->next)
            printf("%d: %d ", e->key, e->value);
    return 0; // Memory leak omitted
}
