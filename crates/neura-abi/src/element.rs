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
}

pub const SINGLE: u32 = Element::Single as u32;
pub const HALF: u32 = Element::Half as u32;
pub const BFLOAT16: u32 = Element::Bfloat16 as u32;
pub const INT8: u32 = Element::Int8 as u32;
pub const INT4: u32 = Element::Int4 as u32;
pub const FP8_E4M3: u32 = Element::Fp8E4M3 as u32;
pub const FP8_E5M2: u32 = Element::Fp8E5M2 as u32;
pub const FP4_E2M1: u32 = Element::Fp4E2M1 as u32;

pub const INT4_BLOCK: u32 = 128;
pub const FP4_BLOCK: u32 = 32;

impl Element {
    pub const ALL: &'static [Element] = &[
        Self::Single,
        Self::Half,
        Self::Bfloat16,
        Self::Int8,
        Self::Int4,
        Self::Fp8E4M3,
        Self::Fp8E5M2,
        Self::Fp4E2M1,
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
        }
    }

    pub const fn elements_per_word(self) -> u64 {
        match self {
            Self::Single => 1,
            Self::Half | Self::Bfloat16 => 2,
            Self::Int8 | Self::Fp8E4M3 | Self::Fp8E5M2 => 4,
            Self::Int4 | Self::Fp4E2M1 => 8,
        }
    }

    pub const fn payload_words(self, elements: u64) -> u64 {
        elements.div_ceil(self.elements_per_word())
    }

    pub const fn quanta(self, elements: u64) -> u64 {
        match self.block() {
            0 if self.quantized() => 1,
            0 => 0,
            block => elements.div_ceil(block as u64),
        }
    }

    pub const fn storage_words(self, elements: u64) -> u64 {
        self.payload_words(elements) + self.quanta(elements)
    }

    pub const fn narrow(self) -> bool {
        self.elements_per_word() > 1
    }

    pub const fn quantized(self) -> bool {
        matches!(self, Self::Int8 | Self::Int4 | Self::Fp4E2M1)
    }

    pub const fn block(self) -> u32 {
        match self {
            Self::Int4 => INT4_BLOCK,
            Self::Fp4E2M1 => FP4_BLOCK,
            _ => 0,
        }
    }

    pub const fn per_tensor(self) -> bool {
        self.quantized() && self.block() == 0
    }

    pub const fn per_block(self) -> bool {
        self.block() != 0
    }

    pub const fn promote(self, other: Self) -> Self {
        match (self, other) {
            (Self::Fp8E4M3 | Self::Fp8E5M2, _)
            | (_, Self::Fp8E4M3 | Self::Fp8E5M2)
            | (Self::Fp4E2M1, _)
            | (_, Self::Fp4E2M1) => Self::Single,
            (Self::Single, _) | (_, Self::Single) => Self::Single,
            (Self::Int8 | Self::Int4, Self::Int8 | Self::Int4) => Self::Single,
            (Self::Int8 | Self::Int4, other) | (other, Self::Int8 | Self::Int4) => other,
            (Self::Half, Self::Half) => Self::Half,
            (Self::Bfloat16, Self::Bfloat16) => Self::Bfloat16,
            (Self::Half, Self::Bfloat16) | (Self::Bfloat16, Self::Half) => Self::Single,
        }
    }
}
