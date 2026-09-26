#[repr(u32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Element {
    Single,
    Half,
    Bfloat16,
}

pub const SINGLE: u32 = Element::Single as u32;
pub const HALF: u32 = Element::Half as u32;
pub const BFLOAT16: u32 = Element::Bfloat16 as u32;

impl Element {
    pub const ALL: &'static [Element] = &[Self::Single, Self::Half, Self::Bfloat16];
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
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Single => "single",
            Self::Half => "half",
            Self::Bfloat16 => "bfloat16",
        }
    }

    pub const fn elements_per_word(self) -> u64 {
        match self {
            Self::Single => 1,
            Self::Half | Self::Bfloat16 => 2,
        }
    }

    pub const fn words(self, elements: u64) -> u64 {
        elements.div_ceil(self.elements_per_word())
    }

    pub const fn narrow(self) -> bool {
        self.elements_per_word() > 1
    }

    pub const fn promote(self, other: Self) -> Self {
        match (self, other) {
            (Self::Single, _) | (_, Self::Single) => Self::Single,
            (Self::Half, Self::Half) => Self::Half,
            (Self::Bfloat16, Self::Bfloat16) => Self::Bfloat16,
            (Self::Half, Self::Bfloat16) | (Self::Bfloat16, Self::Half) => Self::Single,
        }
    }
}
