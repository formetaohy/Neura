use crate::graph::{Residency, ValueInfo};
use neura_abi::{Placement, Precision, Store, WORD_BYTES};

#[derive(Clone, Debug)]
pub struct Region {
    words: u64,
    addresses: Vec<Option<u64>>,
    entries: Vec<(u64, u64)>,
    precision: Precision,
}

impl Region {
    fn of(
        values: &[ValueInfo],
        wants: impl Fn(&ValueInfo) -> bool,
        precision: Precision,
        alignment: u64,
    ) -> Self {
        let stride = (alignment / WORD_BYTES).max(1);
        let mut words = 0u64;
        let mut addresses = vec![None; values.len()];
        let mut entries = Vec::new();
        for (id, info) in values.iter().enumerate() {
            if info.storage as usize != id || !wants(info) {
                continue;
            }
            let elements = u64::from(info.shape.elements());
            words = words.next_multiple_of(stride);
            let address = words * precision.elements_per_word();
            addresses[id] = Some(address);
            entries.push((address, elements));
            words += precision.words(elements);
        }
        Self {
            words,
            addresses,
            entries,
            precision,
        }
    }

    pub fn words(&self) -> u64 {
        self.words
    }

    pub fn bytes(&self) -> u64 {
        self.words * WORD_BYTES
    }

    pub fn tensors(&self) -> usize {
        self.entries.len()
    }

    pub fn precision(&self) -> Precision {
        self.precision
    }

    pub(crate) fn address(&self, storage: u32) -> u64 {
        self.addresses
            .get(storage as usize)
            .copied()
            .flatten()
            .unwrap_or_else(|| panic!("value {storage} holds no tensor of its own"))
    }

    pub(crate) fn word_of(&self, address: u64) -> u64 {
        address / self.precision.elements_per_word()
    }
}

impl PartialEq for Region {
    fn eq(&self, other: &Self) -> bool {
        self.words == other.words && self.entries == other.entries
    }
}

impl Eq for Region {}

pub struct Layout {
    weights: Region,
    tensors: Region,
    uploads: Vec<(u64, Vec<f32>)>,
}

impl Layout {
    pub(crate) fn of(values: &[ValueInfo], precision: Precision, alignment: u64) -> Self {
        let weights = Region::of(
            values,
            |info| info.residency == Residency::Parameter,
            precision,
            alignment,
        );
        let tensors = Region::of(
            values,
            |info| info.residency == Residency::Resident,
            Precision::Single,
            alignment,
        );
        let uploads = values
            .iter()
            .enumerate()
            .filter(|(id, info)| {
                info.storage as usize == *id && info.residency == Residency::Parameter
            })
            .filter_map(|(id, info)| {
                let data = info.initial.as_ref()?;
                weights
                    .addresses
                    .get(id)
                    .copied()
                    .flatten()
                    .map(|address| (address, data.clone()))
            })
            .collect();
        Self {
            weights,
            tensors,
            uploads,
        }
    }

    pub fn weights(&self) -> &Region {
        &self.weights
    }

    pub fn tensors(&self) -> &Region {
        &self.tensors
    }

    pub fn uploads(&self) -> &[(u64, Vec<f32>)] {
        &self.uploads
    }

    pub(crate) fn store(&self, values: &[ValueInfo], value: u32) -> Store {
        store_of(values[values[value as usize].storage as usize].residency)
    }

    pub(crate) fn address(&self, values: &[ValueInfo], arena: &[u64], value: u32) -> u64 {
        let storage = values[value as usize].storage;
        match self.store(values, value) {
            Store::Weights => self.weights.address(storage),
            Store::Tensors => match values[storage as usize].residency {
                Residency::Resident => self.tensors.address(storage),
                _ => arena[storage as usize] / WORD_BYTES,
            },
        }
    }

    pub fn weight_bytes(&self, placement: Placement, address: u64) -> u64 {
        (placement.weights() + self.weights.word_of(address)) * WORD_BYTES
    }
}

pub(crate) fn store_of(residency: Residency) -> Store {
    match residency {
        Residency::Parameter => Store::Weights,
        Residency::Input | Residency::Resident | Residency::Derived | Residency::View => {
            Store::Tensors
        }
    }
}
