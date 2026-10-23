#[repr(u32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Element {
    Single,
    Half,
    Bfloat16,
    Int8,
    Int4,
    Fp8E4M3,
    Fp8E5M2,
    Fp4E2M1,
    Q8_0,
    Q4K,
    Q5K,
    Q6K,
    Q3K,
}

pub const SINGLE: u32 = Element::Single as u32;
pub const HALF: u32 = Element::Half as u32;
pub const BFLOAT16: u32 = Element::Bfloat16 as u32;
pub const INT8: u32 = Element::Int8 as u32;
pub const INT4: u32 = Element::Int4 as u32;
pub const FP8_E4M3: u32 = Element::Fp8E4M3 as u32;
pub const FP8_E5M2: u32 = Element::Fp8E5M2 as u32;
pub const FP4_E2M1: u32 = Element::Fp4E2M1 as u32;
pub const Q8_0: u32 = Element::Q8_0 as u32;
pub const Q4_K: u32 = Element::Q4K as u32;
pub const Q5_K: u32 = Element::Q5K as u32;
pub const Q6_K: u32 = Element::Q6K as u32;
pub const Q3_K: u32 = Element::Q3K as u32;

pub const INT4_BLOCK: u32 = 128;
pub const FP4_BLOCK: u32 = 32;
pub const QUANT_BLOCK: u32 = 32;
pub const SUPER_BLOCK: u32 = 256;
pub const SUPER_SUB_BLOCK: u32 = 32;
pub const SUPER_SMALL_BLOCK: u32 = 16;
pub const DELTA_WIDTH: u32 = 128;
pub const SUPER_SCALE_BYTES: u32 = 12;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Block {
    pub elements: u32,
    pub bytes: u32,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct FloatFormat {
    pub bias: i32,
    pub mantissa_bits: u32,
    pub subnormal_shift: i32,
    pub smallest: f32,
    pub nan: u32,
    pub infinity: u32,
    pub max: u32,
    pub ceiling: u32,
}

impl Element {
    pub const ALL: &[Element] = &[
        Self::Single,
        Self::Half,
        Self::Bfloat16,
        Self::Int8,
        Self::Int4,
        Self::Fp8E4M3,
        Self::Fp8E5M2,
        Self::Fp4E2M1,
        Self::Q8_0,
        Self::Q4K,
        Self::Q5K,
        Self::Q6K,
        Self::Q3K,
    ];
    pub const COUNT: u32 = Self::ALL.len() as u32;

    pub const fn code(self) -> u32 {
        self as u32
    }

    pub fn of(code: u32) -> Self {
        *Self::ALL
            .get(code as usize)
            .unwrap_or_else(|| panic!("element {code} is not a declared tensor element"))
    }

    pub const fn symbol(self) -> &'static str {
        match self {
            Self::Single => "element::SINGLE",
            Self::Half => "element::HALF",
            Self::Bfloat16 => "element::BFLOAT16",
            Self::Int8 => "element::INT8",
            Self::Int4 => "element::INT4",
            Self::Fp8E4M3 => "element::FP8_E4M3",
            Self::Fp8E5M2 => "element::FP8_E5M2",
            Self::Fp4E2M1 => "element::FP4_E2M1",
            Self::Q8_0 => "element::Q8_0",
            Self::Q4K => "element::Q4_K",
            Self::Q5K => "element::Q5_K",
            Self::Q6K => "element::Q6_K",
            Self::Q3K => "element::Q3_K",
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Single => "single",
            Self::Half => "half",
            Self::Bfloat16 => "bfloat16",
            Self::Int8 => "int8",
            Self::Int4 => "int4",
            Self::Fp8E4M3 => "fp8e4m3",
            Self::Fp8E5M2 => "fp8e5m2",
            Self::Fp4E2M1 => "fp4e2m1",
            Self::Q8_0 => "q8_0",
            Self::Q4K => "q4_k",
            Self::Q5K => "q5_k",
            Self::Q6K => "q6_k",
            Self::Q3K => "q3_k",
        }
    }

    pub const fn integral(self) -> bool {
        matches!(
            self,
            Self::Int8 | Self::Int4 | Self::Q8_0 | Self::Q3K | Self::Q4K | Self::Q6K
        )
    }

    pub const fn elements_per_word(self) -> u64 {
        match self {
            Self::Single => 1,
            Self::Half | Self::Bfloat16 => 2,
            Self::Int8 | Self::Fp8E4M3 | Self::Fp8E5M2 => 4,
            Self::Int4 | Self::Fp4E2M1 => 8,
            _ => panic!("a block lays out its own bytes, and its numbers carry no per word count"),
        }
    }

    pub const fn bits_per_element(self) -> u32 {
        32 / self.elements_per_word() as u32
    }

    pub const fn block(self) -> Option<Block> {
        let block = match self {
            Self::Int4 => Block {
                elements: INT4_BLOCK,
                bytes: INT4_BLOCK / 2,
            },
            Self::Fp4E2M1 => Block {
                elements: FP4_BLOCK,
                bytes: FP4_BLOCK / 2,
            },
            Self::Q8_0 => Block {
                elements: QUANT_BLOCK,
                bytes: 2 + QUANT_BLOCK,
            },
            Self::Q4K => Block {
                elements: SUPER_BLOCK,
                bytes: 4 + SUPER_SCALE_BYTES + SUPER_BLOCK / 2,
            },
            Self::Q5K => Block {
                elements: SUPER_BLOCK,
                bytes: 4 + SUPER_SCALE_BYTES + SUPER_BLOCK / 2 + SUPER_BLOCK / 8,
            },
            Self::Q6K => Block {
                elements: SUPER_BLOCK,
                bytes: 2 + SUPER_BLOCK / 2 + SUPER_BLOCK / 4 + SUPER_BLOCK / 16,
            },
            Self::Q3K => Block {
                elements: SUPER_BLOCK,
                bytes: 2 + SUPER_BLOCK / 8 + SUPER_BLOCK / 4 + SUPER_SCALE_BYTES,
            },
            _ => return None,
        };
        Some(block)
    }

    pub const fn sidecar(self) -> bool {
        matches!(self, Self::Int4 | Self::Fp4E2M1)
    }

    pub const fn whole_blocks(self) -> bool {
        self.block().is_some() && !self.sidecar()
    }

    pub const fn per_tensor(self) -> bool {
        matches!(self, Self::Int8)
    }

    pub const fn quantized(self) -> bool {
        matches!(
            self,
            Self::Int8
                | Self::Int4
                | Self::Fp4E2M1
                | Self::Q8_0
                | Self::Q4K
                | Self::Q5K
                | Self::Q6K
                | Self::Q3K
        )
    }

    pub const fn packs_on_device(self) -> bool {
        matches!(self, Self::Q8_0)
    }

    pub const fn narrow(self) -> bool {
        self.block().is_some() || self.elements_per_word() > 1
    }

    pub const fn stages_a_run(self) -> bool {
        matches!(self, Self::Q8_0)
    }

    pub const fn packs_codes(self) -> bool {
        matches!(self, Self::Q4K | Self::Q5K | Self::Q6K | Self::Q3K)
    }

    pub const fn quanta(self, elements: u64) -> u64 {
        match self {
            Self::Int8 => 1,
            Self::Int4 => elements.div_ceil(INT4_BLOCK as u64),
            Self::Fp4E2M1 => elements.div_ceil(FP4_BLOCK as u64),
            _ => 0,
        }
    }

    pub const fn payload_bytes(self, elements: u64) -> u64 {
        if !self.whole_blocks() {
            return (elements * self.bits_per_element() as u64).div_ceil(8);
        }
        let block = match self.block() {
            Some(block) => block,
            None => return (elements * self.bits_per_element() as u64).div_ceil(8),
        };
        elements.div_ceil(block.elements as u64) * block.bytes as u64
    }

    pub const fn payload_words(self, elements: u64) -> u64 {
        self.payload_bytes(elements).div_ceil(crate::WORD_BYTES)
    }

    pub const fn storage_words(self, elements: u64) -> u64 {
        self.payload_words(elements) + self.quanta(elements)
    }

    pub const fn byte_span(self, first: u64, count: u64) -> (u64, u64) {
        match (self.whole_blocks(), self.block()) {
            (true, Some(block)) => {
                let per = block.elements as u64;
                let bytes = block.bytes as u64;
                (first / per * bytes, (first + count).div_ceil(per) * bytes)
            }
            _ => {
                let per = self.elements_per_word();
                let bytes = crate::WORD_BYTES;
                (first / per * bytes, (first + count).div_ceil(per) * bytes)
            }
        }
    }

    pub const fn word_span(self, first: u64, count: u64) -> (u64, u64) {
        let (first, end) = self.byte_span(first, count);
        (first / crate::WORD_BYTES, end.div_ceil(crate::WORD_BYTES))
    }

    pub const fn format(self) -> Option<FloatFormat> {
        match self {
            Self::Fp8E4M3 => Some(FloatFormat {
                bias: 7,
                mantissa_bits: 3,
                subnormal_shift: 14,
                smallest: 1.0 / 512.0,
                nan: 0x7f,
                infinity: 0x7f,
                max: 0x7e,
                ceiling: 0x43e0_0000,
            }),
            Self::Fp8E5M2 => Some(FloatFormat {
                bias: 15,
                mantissa_bits: 2,
                subnormal_shift: 7,
                smallest: 1.0 / 65536.0,
                nan: 0x7f,
                infinity: 0x7c,
                max: 0x7b,
                ceiling: 0x4760_0000,
            }),
            Self::Fp4E2M1 => Some(FloatFormat {
                bias: 1,
                mantissa_bits: 1,
                subnormal_shift: 22,
                smallest: 0.5,
                nan: u32::MAX,
                infinity: u32::MAX,
                max: 0b0111,
                ceiling: 0x40c0_0000,
            }),
            _ => None,
        }
    }

    pub const fn promote(self, other: Self) -> Self {
        match (self, other) {
            (Self::Fp8E4M3 | Self::Fp8E5M2, _)
            | (_, Self::Fp8E4M3 | Self::Fp8E5M2)
            | (Self::Fp4E2M1, _)
            | (_, Self::Fp4E2M1)
            | (Self::Single, _)
            | (_, Self::Single) => Self::Single,
            (Self::Half, Self::Half) => Self::Half,
            (Self::Bfloat16, Self::Bfloat16) => Self::Bfloat16,
            (Self::Half, Self::Bfloat16) | (Self::Bfloat16, Self::Half) => Self::Single,
            (first, second) => match (first.integral(), second.integral()) {
                (true, true) => Self::Single,
                (true, false) => second,
                (false, _) => first,
            },
        }
    }
}
