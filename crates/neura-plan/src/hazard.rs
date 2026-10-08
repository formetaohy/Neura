use crate::region::{Region, Touches, Values};
use neura_abi::PAGE_WORDS;

const QUANTUM: u64 = PAGE_WORDS;
const QUANTUM_MASK: u64 = QUANTUM - 1;
const HAZARD_SEGMENTS: usize = 4;

#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub(crate) struct Hazard {
    pub(crate) wave: Option<u32>,
    pub(crate) segments: Segments,
    pub(crate) deepest: Option<u32>,
}

impl Hazard {
    pub(crate) fn deep(deepest: u32) -> Self {
        Self {
            deepest: Some(deepest),
            ..Self::default()
        }
    }

    pub(crate) fn at(wave: u32, segment: u32) -> Self {
        Self {
            wave: Some(wave),
            segments: Segments::of(segment),
            deepest: None,
        }
    }

    pub(crate) fn join(&mut self, other: &Self) {
        self.combine(other, true);
    }

    pub(crate) fn gather(&mut self, other: &Self) {
        self.combine(other, false);
    }

    fn combine(&mut self, other: &Self, ordered: bool) {
        match (self.wave, other.wave) {
            (_, None) => {}
            (None, Some(wave)) => {
                self.wave = Some(wave);
                self.segments.copy_from(&other.segments);
            }
            (Some(kept), Some(wave)) if wave > kept => {
                self.wave = Some(wave);
                self.segments.copy_from(&other.segments);
            }
            (Some(kept), Some(wave)) if wave == kept => {
                if ordered {
                    self.segments.merge(&other.segments);
                } else {
                    self.segments.extend_from(other.segments.slice());
                }
            }
            _ => {}
        }
        self.deepest = match (self.deepest, other.deepest) {
            (Some(kept), Some(carried)) => Some(kept.max(carried)),
            (kept, carried) => kept.or(carried),
        };
    }

    fn settle(&mut self) {
        self.segments.settle();
    }
}

impl PartialEq<Vec<u32>> for Segments {
    fn eq(&self, other: &Vec<u32>) -> bool {
        self.slice() == other.as_slice()
    }
}

#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub(crate) struct Segments {
    inline: [u32; HAZARD_SEGMENTS],
    held: u8,
    overflow: bool,
}

impl Segments {
    pub(crate) fn of(segment: u32) -> Self {
        let mut segments = Self::default();
        segments.inline[0] = segment;
        segments.held = 1;
        segments
    }

    pub(crate) fn slice(&self) -> &[u32] {
        &self.inline[..self.held as usize]
    }

    pub(crate) fn len(&self) -> usize {
        self.held as usize
    }

    pub(crate) fn overflowed(&self) -> bool {
        self.overflow
    }

    pub(crate) fn get(&self, at: usize) -> u32 {
        self.inline[at]
    }

    pub(crate) fn iter(&self) -> std::slice::Iter<'_, u32> {
        self.slice().iter()
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [u32] {
        &mut self.inline[..self.held as usize]
    }

    pub(crate) fn clear(&mut self) {
        self.held = 0;
        self.overflow = false;
    }

    pub(crate) fn copy_from(&mut self, other: &Self) {
        *self = *other;
    }

    pub(crate) fn merge(&mut self, other: &Self) {
        for segment in other.slice() {
            self.insert(*segment);
        }
        self.overflow |= other.overflow;
    }

    fn insert(&mut self, segment: u32) {
        let at = match self.slice().binary_search(&segment) {
            Ok(_) => return,
            Err(at) => at,
        };
        if self.held as usize == HAZARD_SEGMENTS {
            self.overflow = true;
            return;
        }
        self.inline.copy_within(at..self.held as usize, at + 1);
        self.inline[at] = segment;
        self.held += 1;
    }

    pub(crate) fn extend_from(&mut self, other: &[u32]) {
        for segment in other {
            self.insert(*segment);
        }
    }

    pub(crate) fn settle(&mut self) {
        self.inline[..self.held as usize].sort_unstable();
        let mut kept = 0usize;
        for at in 0..self.held as usize {
            if kept == 0 || self.inline[kept - 1] != self.inline[at] {
                self.inline[kept] = self.inline[at];
                kept += 1;
            }
        }
        self.held = kept as u8;
        self.overflow |= kept == HAZARD_SEGMENTS;
    }
}

#[derive(Default)]
pub(crate) struct Accesses {
    cells: Vec<(u64, Hazard)>,
    whole: Hazard,
    every: Hazard,
}

impl Accesses {
    fn split(&mut self, at: u64) {
        let position = match self.cells.binary_search_by_key(&at, |(start, _)| *start) {
            Ok(_) => return,
            Err(position) => position,
        };
        let inherited = match position {
            0 => Hazard::default(),
            _ => self.cells[position - 1].1,
        };
        self.cells.insert(position, (at, inherited));
    }

    fn first_cell(&self, at: u64) -> usize {
        match self.cells.binary_search_by_key(&at, |(start, _)| *start) {
            Ok(position) => position,
            Err(0) => 0,
            Err(position) => position - 1,
        }
    }

    pub(crate) fn record(&mut self, region: Region, hazard: &Hazard, quantized: bool) {
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
        let from = self.first_cell(first);
        for (start, cell) in &mut self.cells[from..] {
            if end.is_some_and(|end| *start >= end) {
                break;
            }
            cell.join(hazard);
        }
    }

    pub(crate) fn query(&self, region: Region, quantized: bool) -> Hazard {
        if matches!(region, Region::Whole) {
            return self.every;
        }
        let mut hazard = Hazard::default();
        hazard.gather(&self.whole);
        for (first, end) in quanta(region, quantized) {
            let from = self.first_cell(first);
            for (start, cell) in &self.cells[from..] {
                if end.is_some_and(|end| *start >= end) {
                    break;
                }
                hazard.gather(cell);
            }
        }
        hazard.settle();
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
        Region::Whole => QuantaState::Empty,
        Region::Run { first, count } => QuantaState::Once(Some((first, first.checked_add(count)))),
        Region::Band {
            first,
            span,
            stride,
            count,
        } if quantized => QuantaState::Merged {
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
}

impl Hazards {
    pub(crate) fn of(values: usize) -> Self {
        Self {
            readers: (0..values).map(|_| Accesses::default()).collect(),
            writers: (0..values).map(|_| Accesses::default()).collect(),
        }
    }

    pub(crate) fn inspect<V: Values>(
        &self,
        values: &V,
        touches: &Touches,
        in_place: bool,
    ) -> Hazard {
        let mut hazard = Hazard::default();
        for (storage, region) in &touches.reads {
            let quantized = quantized(values, *storage);
            hazard.gather(&self.writers[*storage as usize].query(*region, quantized));
        }
        for (storage, region) in &touches.writes {
            let quantized = quantized(values, *storage);
            hazard.gather(&self.readers[*storage as usize].query(*region, quantized));
        }
        if in_place {
            for (storage, region) in &touches.writes {
                let quantized = quantized(values, *storage);
                hazard.gather(&self.writers[*storage as usize].query(*region, quantized));
            }
        }
        hazard.settle();
        hazard
    }

    pub(crate) fn record<V: Values>(&mut self, values: &V, touches: &Touches, hazard: &Hazard) {
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
    }
}

fn quantized<V: Values>(values: &V, storage: u32) -> bool {
    values.elements(storage) > QUANTUM
}

fn covers<V: Values>(values: &V, storage: u32, region: Region) -> bool {
    match region {
        Region::Whole => true,
        Region::Run { first, count } => first == 0 && count >= values.elements(storage),
        Region::Band { .. } => false,
    }
}
