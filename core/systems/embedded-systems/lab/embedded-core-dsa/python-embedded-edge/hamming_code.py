def hamming_encode(data: int) -> int:
    # Encode a 4-bit integer into a 7-bit Hamming code
    d1, d2, d3, d4 = [(data >> i) & 1 for i in range(4)]
    p1 = d1 ^ d2 ^ d4
    p2 = d1 ^ d3 ^ d4
    p3 = d2 ^ d3 ^ d4
    return (p1 << 6) | (p2 << 5) | (d1 << 4) | (p3 << 3) | (d2 << 2) | (d3 << 1) | d4

def hamming_decode(encoded: int) -> tuple[int, bool]:
    # Decode a 7-bit Hamming code into a 4-bit integer
    p1 = (encoded >> 6) & 1
    p2 = (encoded >> 5) & 1
    d1 = (encoded >> 4) & 1
    p3 = (encoded >> 3) & 1
    d2 = (encoded >> 2) & 1
    d3 = (encoded >> 1) & 1
    d4 = encoded & 1
    c1 = p1 ^ d1 ^ d2 ^ d4
    c2 = p2 ^ d1 ^ d3 ^ d4
    c3 = p3 ^ d2 ^ d3 ^ d4
    error_position = (c3 << 2) | (c2 << 1) | c1
    if error_position:
        encoded ^= 1 << (7 - error_position)
    data = (d1 << 3) | (d2 << 2) | (d3 << 1) | d4
    return data, error_position != 0

# Test
def test_hamming_code():
    data = 0b1011
    encoded = hamming_encode(data)
    decoded, error_corrected = hamming_decode(encoded)
    assert data == decoded
    assert not error_corrected

test_hamming_code()
