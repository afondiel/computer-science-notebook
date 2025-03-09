pub struct HammingCode;

impl HammingCode {
    pub fn encode(data: u8) -> u16 {
        let mut encoded = (data as u16) << 3;
        let p1 = (encoded & 0b0101_0101_0101_0101).count_ones() & 1;
        let p2 = (encoded & 0b0110_0110_0110_0110).count_ones() & 1;
        let p4 = (encoded & 0b0111_1000_0111_1000).count_ones() & 1;
        encoded |= p1 << 0 | p2 << 1 | p4 << 3;
        encoded
    }

    pub fn decode(encoded: u16) -> Result<u8, ()> {
        let p1 = (encoded & 0b0101_0101_0101_0101).count_ones() & 1;
        let p2 = (encoded & 0b0110_0110_0110_0110).count_ones() & 1;
        let p4 = (encoded & 0b0111_1000_0111_1000).count_ones() & 1;
        let error_pos = p1 | (p2 << 1) | (p4 << 3);

        if error_pos != 0 {
            if error_pos > 12 {
                return Err(());
            }
            let corrected = encoded ^ (1 << (error_pos - 1));
            Ok((corrected >> 3) as u8)
        } else {
            Ok((encoded >> 3) as u8)
        }
    }
}

// Usage
let encoded = HammingCode::encode(0b1010_1010);
let decoded = HammingCode::decode(encoded).unwrap();
assert_eq!(decoded, 0b1010_1010);
