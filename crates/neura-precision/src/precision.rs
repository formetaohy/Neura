use neura_abi::WORD_BYTES;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Precision {
    Single,
    Half,
}

impl Precision {
    pub const fn elements_per_word(self) -> u64 {
        match self {
            Self::Single => 1,
            Self::Half => 2,
        }
    }

    pub const fn code(self) -> u32 {
        match self {
            Self::Single => 0,
            Self::Half => 1,
        }
    }

    pub const fn half(self) -> bool {
        matches!(self, Self::Half)
    }

    pub const fn words(self, elements: u64) -> u64 {
        elements.div_ceil(self.elements_per_word())
    }

    pub fn pack(self, values: &[f32]) -> Vec<u8> {
        match self {
            Self::Single => bytemuck::cast_slice(values).to_vec(),
            Self::Half => {
                let words = values
                    .chunks(2)
                    .map(|pair| {
                        let low = half::f16::from_f32(pair[0]).to_bits() as u32;
                        let high = pair
                            .get(1)
                            .map_or(0, |value| half::f16::from_f32(*value).to_bits() as u32);
                        low | (high << 16)
                    })
                    .collect::<Vec<u32>>();
                bytemuck::cast_slice(&words).to_vec()
            }
        }
    }

    pub fn unpack(self, elements: usize, bytes: &[u8]) -> Vec<f32> {
        assert!(
            bytes.len() as u64 >= self.words(elements as u64) * WORD_BYTES,
            "unpacking {elements} {self:?} elements out of {} bytes",
            bytes.len(),
        );
        let mut values = match self {
            Self::Single => bytemuck::cast_slice::<u8, f32>(bytes).to_vec(),
            Self::Half => bytemuck::cast_slice::<u8, u32>(bytes)
                .iter()
                .flat_map(|word| {
                    [
                        half::f16::from_bits((word & 0xffff) as u16).to_f32(),
                        half::f16::from_bits((word >> 16) as u16).to_f32(),
                    ]
                })
                .collect(),
        };
        values.truncate(elements);
        values
    }
}
