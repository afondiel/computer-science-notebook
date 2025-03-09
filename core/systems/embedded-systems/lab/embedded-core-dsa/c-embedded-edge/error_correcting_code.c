#include <stdint.h>

uint8_t calculate_even_parity(uint32_t data) {
    uint8_t parity = 0;
    while(data) {
        parity ^= (data & 1);
        data >>= 1;
    }
    return parity;
}

uint32_t add_parity_bit(uint32_t data) {
    return (data << 1) | calculate_even_parity(data);
}

bool check_parity(uint32_t data) {
    uint8_t received_parity = data & 1;
    data >>= 1;
    return calculate_even_parity(data) == received_parity;
}


int main(void)
{
    uint32_t data = 0b10101010;
    uint32_t data_with_parity = add_parity_bit(data);
    printf("Data with parity: %x\n", data_with_parity);
    printf("Parity check: %d\n", check_parity(data_with_parity));
    return 0;
}