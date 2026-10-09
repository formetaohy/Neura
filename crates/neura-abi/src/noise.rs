#[repr(u32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Distribution {
    Uniform,
    Normal,
}

pub const UNIFORM: u32 = Distribution::Uniform as u32;
pub const NORMAL: u32 = Distribution::Normal as u32;

impl Distribution {
    pub const ALL: &'static [Self] = &[Self::Uniform, Self::Normal];
    pub const COUNT: u32 = Self::ALL.len() as u32;

    pub const fn code(self) -> u32 {
        self as u32
    }

    pub fn of(code: u32) -> Self {
        *Self::ALL
            .get(code as usize)
            .filter(|distribution| distribution.code() == code)
            .unwrap_or_else(|| panic!("distribution {code} is not a declared distribution"))
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Uniform => "uniform",
            Self::Normal => "normal",
        }
    }

    pub const fn symbol(self) -> &'static str {
        match self {
            Self::Uniform => "noise::UNIFORM",
            Self::Normal => "noise::NORMAL",
        }
    }
}
