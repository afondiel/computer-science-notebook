constexpr uint8_t calculate_parity(uint16_t data) {
    uint8_t p1 = (data >> 0) & 0x1;
    uint8_t p2 = (data >> 1) & 0x1;
    uint8_t p4 = (data >> 3) & 0x1;
    return (p1 ^ p2 ^ p4);
}

constexpr uint16_t encode_hamming(uint8_t data) {
    uint16_t encoded = 0;
    encoded |= (data & 0x1) << 2;
    encoded |= (data & 0x2) << 3;
    encoded |= (data & 0x4) << 4;
    encoded |= (data & 0x8) << 5;
    encoded |= calculate_parity(encoded) << 0;
    encoded |= calculate_parity(encoded >> 1) << 1;
    encoded |= calculate_parity(encoded >> 3) << 3;
    return encoded;
}

uint8_t decode_hamming(uint16_t encoded) {
    uint8_t p1 = calculate_parity(encoded);
    uint8_t p2 = calculate_parity(encoded >> 1);
    uint8_t p4 = calculate_parity(encoded >> 3);
    uint8_t syndrome = (p4 << 2) | (p2 << 1) | p1;
    if (syndrome) {
        encoded ^= (1 << (syndrome - 1));
    }
    return (encoded >> 2) & 0xF;
}

// Test Case
void test_hamming_code() {
    uint8_t data = 0b1010;
    uint16_t encoded = encode_hamming(data);
    uint8_t decoded = decode_hamming(encoded);
    assert(data == decoded);
}
