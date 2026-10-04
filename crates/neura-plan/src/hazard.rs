use crate::region::{Region, Touches};
use neura_graph::ValueInfo;

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

    pub(crate) fn record(&mut self, region: Region, hazard: &Hazard) {
        self.every.join(hazard);
        let (first, end) = bounds(region);
        if end.is_none() && first == 0 {
            self.whole.join(hazard);
            return;
        }
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

    pub(crate) fn query(&self, region: Region) -> Hazard {
        let (first, end) = bounds(region);
        if end.is_none() && first == 0 {
            return self.every.clone();
        }
        let mut hazard = self.whole.clone();
        let from = self.first_cell(first);
        for (start, cell) in &self.cells[from..] {
            if end.is_some_and(|end| *start >= end) {
                break;
            }
            hazard.gather(cell);
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

fn bounds(region: Region) -> (u64, Option<u64>) {
    match region {
        Region::Whole => (0, None),
        Region::Run { first, count } => (first, first.checked_add(count)),
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

    pub(crate) fn inspect(&self, touches: &Touches, in_place: bool) -> Hazard {
        let mut hazard = Hazard::default();
        for (storage, region) in &touches.reads {
            hazard.gather(&self.writers[*storage as usize].query(*region));
        }
        for (storage, region) in &touches.writes {
            hazard.gather(&self.readers[*storage as usize].query(*region));
        }
        if in_place {
            for (storage, region) in &touches.writes {
                hazard.gather(&self.writers[*storage as usize].query(*region));
            }
        }
        hazard.settle();
        hazard
    }

    pub(crate) fn record(&mut self, values: &[ValueInfo], touches: Touches, hazard: &Hazard) {
        for (storage, region) in &touches.reads {
            self.readers[*storage as usize].record(*region, hazard);
        }
        for (storage, region) in &touches.writes {
            self.writers[*storage as usize].record(*region, hazard);
            if covers(values, *storage, *region) {
                self.readers[*storage as usize].clear();
            }
        }
    }
}

fn covers(values: &[ValueInfo], storage: u32, region: Region) -> bool {
    match region {
        Region::Whole => true,
        Region::Run { first, count } => {
            first == 0 && count >= u64::from(values[storage as usize].shape.elements())
        }
    }
}
