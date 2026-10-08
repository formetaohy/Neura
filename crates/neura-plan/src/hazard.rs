use crate::region::{Region, Touches, Values};
use neura_abi::PAGE_WORDS;

const QUANTUM: u64 = PAGE_WORDS;
const QUANTUM_MASK: u64 = QUANTUM - 1;

#[derive(Clone, Default, PartialEq, Eq, Debug)]
pub(crate) struct Hazard {
    pub(crate) wave: Option<u32>,
    pub(crate) segments: Vec<u32>,
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
            segments: vec![segment],
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
                self.segments.clear();
                self.segments.extend_from_slice(&other.segments);
            }
            (Some(kept), Some(wave)) if wave > kept => {
                self.wave = Some(wave);
                self.segments.clear();
                self.segments.extend_from_slice(&other.segments);
            }
            (Some(kept), Some(wave)) if wave == kept => {
                if ordered {
                    for segment in &other.segments {
                        if let Err(position) = self.segments.binary_search(segment) {
                            self.segments.insert(position, *segment);
                        }
                    }
                } else {
                    self.segments.extend_from_slice(&other.segments);
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
        self.segments.sort_unstable();
        self.segments.dedup();
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
            _ => self.cells[position - 1].1.clone(),
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
            return self.every.clone();
        }
        let mut hazard = self.whole.clone();
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

pub(crate) fn quanta(region: Region, quantized: bool) -> Vec<(u64, Option<u64>)> {
    match region {
        Region::Whole => Vec::new(),
        Region::Run { first, count } => vec![(first, first.checked_add(count))],
        Region::Band {
            first,
            span,
            stride,
            count,
        } if quantized => {
            let mut spans: Vec<(u64, u64)> = Vec::new();
            for index in 0..count {
                let at = first + index * stride;
                let from = at & !QUANTUM_MASK;
                let to = (at + span).next_multiple_of(QUANTUM);
                match spans.last_mut() {
                    Some((_, last)) if from <= *last => *last = (*last).max(to),
                    _ => spans.push((from, to)),
                }
            }
            spans
                .into_iter()
                .map(|(from, to)| (from, Some(to)))
                .collect()
        }
        Region::Band {
            first,
            span,
            stride,
            count,
        } => (0..count)
            .map(|index| {
                let at = first + index * stride;
                (at, at.checked_add(span))
            })
            .collect(),
    }
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

    pub(crate) fn record<V: Values>(&mut self, values: &V, touches: Touches, hazard: &Hazard) {
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
