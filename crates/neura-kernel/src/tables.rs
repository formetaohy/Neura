use neura_abi::{FieldType, MEASURE, PATCH, PLACEMENT, RecordLayout, WORD_BYTES};

pub const MEASURE_WORDS: u32 = MEASURE.size / WORD_BYTES as u32;
pub const PATCH_WORDS: u32 = PATCH.size / WORD_BYTES as u32;

const _: () = assert!(
    MEASURE.size.is_multiple_of(WORD_BYTES as u32),
    "a measure record spans bytes beyond the device word grid",
);
const _: () = assert!(
    PATCH.size.is_multiple_of(WORD_BYTES as u32),
    "a patch record spans bytes beyond the device word grid",
);

pub fn field_word(layout: &RecordLayout, field: &str) -> u32 {
    let held = layout
        .fields
        .iter()
        .find(|held| held.name == field)
        .unwrap_or_else(|| {
            panic!("record {} declares no field {field}", layout.name);
        });
    assert_eq!(
        held.ty,
        FieldType::U32,
        "field {field} of record {} holds numbers another word table read cannot name",
        layout.name,
    );
    assert!(
        held.offset.is_multiple_of(WORD_BYTES as u32),
        "field {field} of record {} starts at byte {}, beyond the device word grid",
        layout.name,
        held.offset,
    );
    held.offset / WORD_BYTES as u32
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Tables {
    slots: u32,
    measures: u32,
    patches: u32,
    list: u32,
}

impl Tables {
    pub const fn of(slots: u32, measures: u32, patches: u32, list: u32) -> Self {
        Self {
            slots,
            measures,
            patches,
            list,
        }
    }

    pub const fn placement(self) -> u32 {
        0
    }

    pub const fn extents(self) -> u32 {
        PLACEMENT.size / WORD_BYTES as u32
    }

    pub const fn measures_first(self) -> u32 {
        self.extents() + self.slots
    }

    pub const fn patches_first(self) -> u32 {
        self.measures_first() + self.measures * MEASURE_WORDS
    }

    pub const fn list_first(self) -> u32 {
        self.patches_first() + self.patches * PATCH_WORDS
    }

    pub const fn words(self) -> u32 {
        self.list_first() + self.list
    }

    pub const fn bytes(self) -> u64 {
        self.words() as u64 * WORD_BYTES
    }
}
