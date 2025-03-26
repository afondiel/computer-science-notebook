#include <stdio.h>

int factorial(int n) {
    if (n == 0) return 1; // base case
    return n * factorial(n - 1);
}

int main() {
    printf("%d\n", factorial(5)); // 120
    return 0;
}
