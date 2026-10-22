use crate::region::{Region, Touches, Values};
use neura_abi::PAGE_WORDS;
use std::collections::BTreeMap;

const QUANTUM: u64 = PAGE_WORDS;
const QUANTUM_MASK: u64 = QUANTUM - 1;
const EXACT_BAND_RUNS: u64 = 256;

#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub(crate) struct Hazard {
    pub(crate) deepest: Option<u32>,
}

impl Hazard {
    pub(crate) fn deep(deepest: u32) -> Self {
        Self {
            deepest: Some(deepest),
        }
    }

    pub(crate) fn join(&mut self, other: &Self) {
        self.deepest = match (self.deepest, other.deepest) {
            (Some(kept), Some(carried)) => Some(kept.max(carried)),
            (kept, carried) => kept.or(carried),
        };
    }
}

#[derive(Default)]
pub(crate) struct Accesses {
    cells: BTreeMap<u64, Hazard>,
    whole: Hazard,
    every: Hazard,
}

impl Accesses {
    fn split(&mut self, at: u64) {
        if self.cells.contains_key(&at) {
            return;
        }
        let inherited = self
            .cells
            .range(..at)
            .next_back()
            .map_or_else(Hazard::default, |(_, hazard)| *hazard);
        self.cells.insert(at, inherited);
    }

    pub(crate) fn record(&mut self, region: Region, hazard: &Hazard, quantized: bool) {
        if matches!(region, Region::Empty) {
            return;
        }
        self.every.join(hazard);
        if matches!(region, Region::Whole) {
            self.whole.join(hazard);
            return;
        }
        for (at, end) in quanta(region, quantized) {
            self.record_span(at, end, hazard);
        }
    }

    fn record_span(&mut self, first: u64, end: Option<u64>, hazard: &Hazard) {
        self.split(first);
        if let Some(end) = end {
            self.split(end);
        }
        let end = end.unwrap_or(u64::MAX);
        for (_, cell) in self.cells.range_mut(first..end) {
            cell.join(hazard);
        }
    }

    pub(crate) fn query(&self, region: Region, quantized: bool) -> Hazard {
        if matches!(region, Region::Empty) {
            return Hazard::default();
        }
        if matches!(region, Region::Whole) {
            return self.every;
        }
        let mut hazard = Hazard::default();
        hazard.join(&self.whole);
        for (first, end) in quanta(region, quantized) {
            let from = self
                .cells
                .range(..=first)
                .next_back()
                .map_or(0, |(start, _)| *start);
            let end = end.unwrap_or(u64::MAX);
            for (_, cell) in self.cells.range(from..end) {
                hazard.join(cell);
            }
        }
        hazard
    }

    pub(crate) fn clear(&mut self) {
        self.cells.clear();
        self.whole = Hazard::default();
        self.every = Hazard::default();
    }
}

pub(crate) struct Quanta {
    state: QuantaState,
}

enum QuantaState {
    Empty,
    Once(Option<(u64, Option<u64>)>),
    Runs {
        at: u64,
        span: u64,
        stride: u64,
        index: u64,
        count: u64,
    },
    Merged {
        at: u64,
        span: u64,
        stride: u64,
        index: u64,
        count: u64,
        pending: Option<(u64, u64)>,
    },
}

impl Iterator for Quanta {
    type Item = (u64, Option<u64>);

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.state {
            QuantaState::Empty => None,
            QuantaState::Once(span) => span.take(),
            QuantaState::Runs {
                at,
                span,
                stride,
                index,
                count,
            } => {
                if *index >= *count {
                    return None;
                }
                let here = at.checked_add(index.checked_mul(*stride)?)?;
                *index += 1;
                Some((here, here.checked_add(*span)))
            }
            QuantaState::Merged {
                at,
                span,
                stride,
                index,
                count,
                pending,
            } => {
                while *index < *count {
                    let here = at.checked_add(index.checked_mul(*stride)?)?;
                    *index += 1;
                    let from = here & !QUANTUM_MASK;
                    let to = (here + *span).next_multiple_of(QUANTUM);
                    match pending {
                        Some((_, last)) if from <= *last => *last = (*last).max(to),
                        Some(_) => {
                            let done = *pending;
                            *pending = Some((from, to));
                            let (first, last) = done.expect("a merged quantum holds its span");
                            return Some((first, Some(last)));
                        }
                        None => *pending = Some((from, to)),
                    }
                }
                pending.take().map(|(first, last)| (first, Some(last)))
            }
        }
    }
}

pub(crate) fn quanta(region: Region, quantized: bool) -> Quanta {
    let state = match region {
        Region::Empty | Region::Whole => QuantaState::Empty,
        Region::Run { first, count } => QuantaState::Once(Some((first, first.checked_add(count)))),
        Region::Band {
            first,
            span,
            stride,
            count,
        } if quantized && (stride >= QUANTUM || count > EXACT_BAND_RUNS) => QuantaState::Merged {
            at: first,
            span,
            stride,
            index: 0,
            count,
            pending: None,
        },
        Region::Band {
            first,
            span,
            stride,
            count,
        } => QuantaState::Runs {
            at: first,
            span,
            stride,
            index: 0,
            count,
        },
    };
    Quanta { state }
}

pub(crate) struct Hazards {
    readers: Vec<Accesses>,
    writers: Vec<Accesses>,
    partition_readers: Vec<std::collections::BTreeMap<u32, Accesses>>,
    partition_writers: Vec<std::collections::BTreeMap<u32, Accesses>>,
    partitioned_readers: Vec<Hazard>,
    partitioned_writers: Vec<Hazard>,
}

impl Hazards {
    pub(crate) fn of(values: usize) -> Self {
        Self {
            readers: (0..values).map(|_| Accesses::default()).collect(),
            writers: (0..values).map(|_| Accesses::default()).collect(),
            partition_readers: (0..values)
                .map(|_| std::collections::BTreeMap::new())
                .collect(),
            partition_writers: (0..values)
                .map(|_| std::collections::BTreeMap::new())
                .collect(),
            partitioned_readers: vec![Hazard::default(); values],
            partitioned_writers: vec![Hazard::default(); values],
        }
    }

    pub(crate) fn inspect<V: Values>(
        &self,
        values: &V,
        touches: &Touches,
        in_place: bool,
        partition: Option<u32>,
    ) -> Hazard {
        let mut hazard = Hazard::default();
        for (storage, region) in &touches.reads {
            let quantized = quantized(values, *storage);
            hazard.join(&self.writers[*storage as usize].query(*region, quantized));
            match partition {
                Some(partition) => {
                    if let Some(accesses) =
                        self.partition_writers[*storage as usize].get(&partition)
                    {
                        hazard.join(&accesses.query(*region, quantized));
                    }
                }
                None => hazard.join(&self.partitioned_writers[*storage as usize]),
            }
        }
        for (storage, region) in &touches.writes {
            let quantized = quantized(values, *storage);
            hazard.join(&self.readers[*storage as usize].query(*region, quantized));
            match partition {
                Some(partition) => {
                    if let Some(accesses) =
                        self.partition_readers[*storage as usize].get(&partition)
                    {
                        hazard.join(&accesses.query(*region, quantized));
                    }
                }
                None => {
                    hazard.join(&self.partitioned_readers[*storage as usize]);
                    if in_place {
                        hazard.join(&self.partitioned_writers[*storage as usize]);
                    }
                }
            }
        }
        if in_place {
            for (storage, region) in &touches.writes {
                let quantized = quantized(values, *storage);
                hazard.join(&self.writers[*storage as usize].query(*region, quantized));
                if let Some(partition) = partition
                    && let Some(accesses) =
                        self.partition_writers[*storage as usize].get(&partition)
                {
                    hazard.join(&accesses.query(*region, quantized));
                }
            }
        }
        hazard
    }

    pub(crate) fn record<V: Values>(
        &mut self,
        values: &V,
        touches: &Touches,
        hazard: &Hazard,
        partition: Option<u32>,
    ) {
        let Some(partition) = partition else {
            for (storage, region) in &touches.reads {
                let quantized = quantized(values, *storage);
                self.readers[*storage as usize].record(*region, hazard, quantized);
            }
            for (storage, region) in &touches.writes {
                let quantized = quantized(values, *storage);
                self.writers[*storage as usize].record(*region, hazard, quantized);
                if covers(values, *storage, *region) {
                    self.readers[*storage as usize].clear();
                }
            }
            return;
        };
        for (storage, region) in &touches.reads {
            let quantized = quantized(values, *storage);
            self.partition_readers[*storage as usize]
                .entry(partition)
                .or_default()
                .record(*region, hazard, quantized);
            self.partitioned_readers[*storage as usize].join(hazard);
        }
        for (storage, region) in &touches.writes {
            let quantized = quantized(values, *storage);
            self.partition_writers[*storage as usize]
                .entry(partition)
                .or_default()
                .record(*region, hazard, quantized);
            if covers(values, *storage, *region)
                && let Some(readers) = self.partition_readers[*storage as usize].get_mut(&partition)
            {
                readers.clear();
            }
            self.partitioned_writers[*storage as usize].join(hazard);
        }
    }
}

fn quantized<V: Values>(values: &V, storage: u32) -> bool {
    values.elements(storage) > QUANTUM
}

fn covers<V: Values>(values: &V, storage: u32, region: Region) -> bool {
    match region {
        Region::Empty => false,
        Region::Whole => true,
        Region::Run { first, count } => first == 0 && count >= values.elements(storage),
        Region::Band { .. } => false,
    }
}
