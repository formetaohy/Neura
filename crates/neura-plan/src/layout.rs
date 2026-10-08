use neura_abi::{Element, Placement, Store, WORD_BYTES};
use neura_graph::{Graph, Init, Residency, ValueInfo};
use std::sync::Arc;

#[derive(Clone, PartialEq, Debug)]
pub struct Entry {
    pub name: Option<Arc<str>>,
    pub shape: [u32; 4],
    pub word: u64,
    pub elements: u64,
    pub element: Element,
    pub scale: f32,
}

#[derive(Clone, Debug)]
pub struct Region {
    words: u64,
    entries: Vec<Entry>,
    placed: Vec<Option<(u64, Element, f32)>>,
}

impl Region {
    fn of(
        values: &[ValueInfo],
        wants: impl Fn(&ValueInfo) -> bool,
        alignment: u64,
        base: u64,
    ) -> Self {
        let stride = (alignment / WORD_BYTES).max(1);
        let mut words = base;
        let mut placed = vec![None; values.len()];
        let mut entries = Vec::new();
        for (id, info) in values.iter().enumerate() {
            if info.storage as usize != id || !wants(info) {
                continue;
            }
            let elements = u64::from(info.shape.elements());
            words = words.next_multiple_of(stride);
            placed[id] = Some((words, info.element, info.scale));
            entries.push(Entry {
                name: info.name.clone(),
                shape: info.shape.dims(),
                word: words,
                elements,
                element: info.element,
                scale: info.scale,
            });
            words += info.element.storage_words(elements);
        }
        Self {
            words,
            entries,
            placed,
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

    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    pub(crate) fn address(&self, storage: u32) -> u64 {
        self.placement(storage).0
    }

    pub(crate) fn holds(&self, storage: u32) -> bool {
        self.placed
            .get(storage as usize)
            .is_some_and(|placed| placed.is_some())
    }

    fn placement(&self, storage: u32) -> (u64, Element, f32) {
        self.placed
            .get(storage as usize)
            .copied()
            .flatten()
            .unwrap_or_else(|| panic!("value {storage} holds no tensor of its own"))
    }
}

impl PartialEq for Region {
    fn eq(&self, other: &Self) -> bool {
        self.words == other.words
            && self.entries.len() == other.entries.len()
            && self
                .entries
                .iter()
                .zip(&other.entries)
                .all(|(left, right)| left == right)
    }
}

pub struct Seed {
    word: u64,
    elements: u32,
    element: Element,
    scale: f32,
    init: Init,
}

impl Seed {
    pub fn address(&self) -> u64 {
        self.word
    }

    pub fn elements(&self) -> u32 {
        self.elements
    }

    pub fn element(&self) -> Element {
        self.element
    }

    pub fn scale(&self) -> f32 {
        self.scale
    }

    pub fn init(&self) -> Init {
        self.init
    }
}

pub struct Layout {
    weights: Region,
    state: Region,
    tensors: Region,
    seeds: Vec<Seed>,
}

impl Layout {
    pub fn of(graph: &Graph<'_>, alignment: u64) -> Self {
        Self::of_values(graph.snapshot().values(), alignment)
    }

    pub(crate) fn of_values(values: &[ValueInfo], alignment: u64) -> Self {
        let weights = Region::of(
            values,
            |info| info.residency == Residency::Parameter,
            alignment,
            0,
        );
        let state = Region::of(
            values,
            |info| info.residency == Residency::State,
            alignment,
            weights.words(),
        );
        let tensors = Region::of(
            values,
            |info| info.residency == Residency::Resident,
            alignment,
            0,
        );
        let seeds = values
            .iter()
            .enumerate()
            .filter(|(id, info)| {
                info.storage as usize == *id
                    && matches!(info.residency, Residency::Parameter | Residency::State)
            })
            .map(|(id, info)| {
                let (word, element, scale) = match info.residency {
                    Residency::State => state.placement(id as u32),
                    _ => weights.placement(id as u32),
                };
                Seed {
                    word,
                    element,
                    scale,
                    elements: info.shape.elements(),
                    init: info
                        .seed
                        .expect("a stored tensor carries the sampler it was declared with"),
                }
            })
            .collect();
        Self {
            weights,
            state,
            tensors,
            seeds,
        }
    }

    pub fn weights(&self) -> &Region {
        &self.weights
    }

    pub fn state(&self) -> &Region {
        &self.state
    }

    pub fn tensors(&self) -> &Region {
        &self.tensors
    }

    pub fn words(&self) -> u64 {
        self.state.words().max(self.weights.words())
    }

    pub fn seeds(&self) -> &[Seed] {
        &self.seeds
    }

    pub(crate) fn store(&self, values: &[ValueInfo], value: u32) -> Store {
        store_of(values[values[value as usize].storage as usize].residency)
    }

    pub(crate) fn element(&self, values: &[ValueInfo], value: u32) -> Element {
        let info = &values[value as usize];
        let owner = &values[info.storage as usize];
        assert!(
            info.element == owner.element,
            "value {value} holds {} numbers of another storage",
            info.element.name(),
        );
        info.element
    }

    pub(crate) fn scale(&self, values: &[ValueInfo], value: u32) -> f32 {
        let info = &values[value as usize];
        let owner = &values[info.storage as usize];
        assert!(
            info.scale == owner.scale,
            "value {value} reconstructs numbers of another storage",
        );
        info.scale
    }

    pub(crate) fn address(&self, values: &[ValueInfo], arena: &[u64], value: u32) -> u64 {
        let storage = values[value as usize].storage;
        match self.store(values, value) {
            Store::Weights => match values[storage as usize].residency {
                Residency::State => self.state.address(storage),
                _ => self.weights.address(storage),
            },
            Store::Tensors => match values[storage as usize].residency {
                Residency::Resident => self.tensors.address(storage),
                _ => arena[storage as usize] / WORD_BYTES,
            },
        }
    }

    pub fn weight_bytes(&self, placement: Placement, address: u64) -> u64 {
        (placement.weights() + address) * WORD_BYTES
    }
}

pub(crate) fn store_of(residency: Residency) -> Store {
    match residency {
        Residency::Parameter | Residency::State => Store::Weights,
        Residency::Input | Residency::Resident | Residency::Derived | Residency::View => {
            Store::Tensors
        }
    }
}
