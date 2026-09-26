use neura_abi::{Element, Placement, Store, WORD_BYTES};
use neura_graph::{Graph, Init, Residency, ValueInfo};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Entry {
    pub word: u64,
    pub elements: u64,
    pub element: Element,
}

#[derive(Clone, Debug)]
pub struct Region {
    words: u64,
    entries: Vec<Entry>,
    placed: Vec<Option<(u64, Element)>>,
}

impl Region {
    fn of(values: &[ValueInfo], wants: impl Fn(&ValueInfo) -> bool, alignment: u64) -> Self {
        let stride = (alignment / WORD_BYTES).max(1);
        let mut words = 0u64;
        let mut placed = vec![None; values.len()];
        let mut entries = Vec::new();
        for (id, info) in values.iter().enumerate() {
            if info.storage as usize != id || !wants(info) {
                continue;
            }
            let elements = u64::from(info.shape.elements());
            words = words.next_multiple_of(stride);
            placed[id] = Some((words, info.element));
            entries.push(Entry {
                word: words,
                elements,
                element: info.element,
            });
            words += info.element.words(elements);
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

    pub(crate) fn element(&self, storage: u32) -> Element {
        self.placement(storage).1
    }

    fn placement(&self, storage: u32) -> (u64, Element) {
        self.placed
            .get(storage as usize)
            .copied()
            .flatten()
            .unwrap_or_else(|| panic!("value {storage} holds no tensor of its own"))
    }
}

impl PartialEq for Region {
    fn eq(&self, other: &Self) -> bool {
        self.words == other.words && self.entries == other.entries
    }
}

impl Eq for Region {}

pub struct Seed {
    word: u64,
    elements: u32,
    element: Element,
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

    pub fn init(&self) -> Init {
        self.init
    }
}

pub struct Layout {
    weights: Region,
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
        );
        let tensors = Region::of(
            values,
            |info| info.residency == Residency::Resident,
            alignment,
        );
        let seeds = values
            .iter()
            .enumerate()
            .filter(|(id, info)| {
                info.storage as usize == *id && info.residency == Residency::Parameter
            })
            .map(|(id, info)| Seed {
                word: weights.address(id as u32),
                element: weights.element(id as u32),
                elements: info.shape.elements(),
                init: info
                    .seed
                    .expect("a parameter carries the sampler it was declared with"),
            })
            .collect();
        Self {
            weights,
            tensors,
            seeds,
        }
    }

    pub fn weights(&self) -> &Region {
        &self.weights
    }

    pub fn tensors(&self) -> &Region {
        &self.tensors
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
        (placement.weights() + address) * WORD_BYTES
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
