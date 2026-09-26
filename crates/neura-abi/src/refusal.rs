use crate::Kind;

pub const KIND_BITS: u32 = 16;
pub const CATEGORY_BITS: u32 = 3;
pub const CODE_BITS: u32 = 32 - KIND_BITS - CATEGORY_BITS;
pub const CODE_LIMIT: u32 = 1 << CODE_BITS;

#[repr(u32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Refusal {
    Op,
    Partial,
    Index,
    Geometry,
    Origin,
    Task,
    Element,
}

impl Refusal {
    pub const ALL: &'static [Refusal] = &[
        Self::Op,
        Self::Partial,
        Self::Index,
        Self::Geometry,
        Self::Origin,
        Self::Task,
        Self::Element,
    ];

    pub const COUNT: u32 = Self::ALL.len() as u32;

    pub const fn code(self) -> u32 {
        self as u32
    }

    pub fn of(code: u32) -> Self {
        *Self::ALL
            .get(code as usize)
            .unwrap_or_else(|| panic!("refusal {code} is not a declared reason a task refuses"))
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Op => "op",
            Self::Partial => "partial",
            Self::Index => "index",
            Self::Geometry => "geometry",
            Self::Origin => "origin",
            Self::Task => "task",
            Self::Element => "element",
        }
    }

    pub fn word(self, subject: u32, code: u32) -> u32 {
        assert!(
            code + 1 < CODE_LIMIT,
            "a refusal of code {code} leaves no room beside {CODE_BITS} bits of subject and category",
        );
        assert!(
            subject < 1 << KIND_BITS,
            "a refusal subject of {subject} outruns {KIND_BITS} bits",
        );
        (subject << KIND_BITS) | (self.code() << CODE_BITS) | (code + 1)
    }

    pub fn read(word: u32) -> (u32, Self, u32) {
        let subject = word >> KIND_BITS;
        let code = word & (CODE_LIMIT - 1);
        assert!(code > 0, "a zeroed word holds no refusal");
        (
            subject,
            Self::of((word >> CODE_BITS) & ((1 << CATEGORY_BITS) - 1)),
            code - 1,
        )
    }
}

pub const TENSOR: u32 = Kind::COUNT;

const _: () = assert!(
    TENSOR < 1 << KIND_BITS,
    "a tensor refusal names no kind, and the subject it carries outruns the bits a kind fills",
);

pub const OP: u32 = Refusal::Op as u32;
pub const PARTIAL: u32 = Refusal::Partial as u32;
pub const INDEX: u32 = Refusal::Index as u32;
pub const GEOMETRY: u32 = Refusal::Geometry as u32;
pub const ORIGIN: u32 = Refusal::Origin as u32;
pub const TASK: u32 = Refusal::Task as u32;
pub const ELEMENT: u32 = Refusal::Element as u32;
