use crate::hazard::{Quanta, quanta};
use crate::region::{Region, Touches, Values};
use neura_abi::PAGE_WORDS;

const QUANTUM: u64 = PAGE_WORDS;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Query {
    Writers,
    Readers,
    Both,
}

#[derive(Clone, Default)]
struct Cell {
    writer: Option<(u32, u32)>,
    readers: Vec<(u32, u32)>,
}

impl Cell {
    fn hold(&mut self, wave: u32, segment: u32) {
        if self
            .readers
            .last()
            .is_some_and(|held| *held == (wave, segment))
        {
            return;
        }
        if !self.readers.contains(&(wave, segment)) {
            self.readers.push((wave, segment));
        }
    }
}

#[derive(Default)]
struct Store {
    cells: Vec<(u64, Cell)>,
}

impl Store {
    fn split(&mut self, at: u64) {
        let position = match self.cells.binary_search_by_key(&at, |(start, _)| *start) {
            Ok(_) => return,
            Err(position) => position,
        };
        let inherited = match position {
            0 => Cell::default(),
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

    fn for_each_cell(
        &mut self,
        region: Region,
        elements: u64,
        quantized: bool,
        mut hold: impl FnMut(&mut Cell),
    ) {
        for (start, end) in spans(region, elements, quantized) {
            self.split(start);
            if let Some(end) = end {
                self.split(end);
            }
            let from = self.first_cell(start);
            for (cell_start, cell) in &mut self.cells[from..] {
                if end.is_some_and(|end| *cell_start >= end) {
                    break;
                }
                hold(cell);
            }
        }
    }

    fn query(
        &self,
        region: Region,
        elements: u64,
        quantized: bool,
        query: Query,
        out: &mut Vec<u32>,
    ) {
        for (start, end) in spans(region, elements, quantized) {
            let from = self.first_cell(start);
            for (cell_start, cell) in &self.cells[from..] {
                if end.is_some_and(|end| *cell_start >= end) {
                    break;
                }
                collect(cell, query, out);
            }
        }
    }
}

fn collect(cell: &Cell, query: Query, out: &mut Vec<u32>) {
    if matches!(query, Query::Writers | Query::Both)
        && let Some((_, segment)) = cell.writer
    {
        out.push(segment);
    }
    if matches!(query, Query::Readers | Query::Both) {
        for (_, segment) in &cell.readers {
            out.push(*segment);
        }
    }
}

pub(crate) struct Flow {
    stores: Vec<Store>,
    written: Vec<bool>,
}

impl Flow {
    pub(crate) fn of<V: Values, W>(values: &V, writes: W) -> Self
    where
        W: Iterator<Item = u32>,
    {
        let mut written = vec![false; values.len()];
        for value in writes {
            written[values.storage(value) as usize] = true;
        }
        Self {
            stores: (0..values.len()).map(|_| Store::default()).collect(),
            written,
        }
    }

    pub(crate) fn inspect<V: Values>(
        &self,
        values: &V,
        touches: &Touches,
        in_place: bool,
        out: &mut Vec<u32>,
    ) {
        out.clear();
        for (storage, region) in &touches.reads {
            if !self.written[*storage as usize] {
                continue;
            }
            let store = &self.stores[*storage as usize];
            store.query(
                *region,
                values.elements(*storage),
                quantized(values, *storage),
                Query::Writers,
                out,
            );
        }
        let query = if in_place {
            Query::Both
        } else {
            Query::Readers
        };
        for (storage, region) in &touches.writes {
            let store = &self.stores[*storage as usize];
            store.query(
                *region,
                values.elements(*storage),
                quantized(values, *storage),
                query,
                out,
            );
        }
        out.sort_unstable();
        out.dedup();
    }

    pub(crate) fn record<V: Values>(
        &mut self,
        values: &V,
        touches: &Touches,
        wave: u32,
        segment: u32,
    ) {
        for (storage, region) in &touches.reads {
            if !self.written[*storage as usize] {
                continue;
            }
            let store = &mut self.stores[*storage as usize];
            store.for_each_cell(
                *region,
                values.elements(*storage),
                quantized(values, *storage),
                |cell| {
                    cell.hold(wave, segment);
                },
            );
        }
        for (storage, region) in &touches.writes {
            let store = &mut self.stores[*storage as usize];
            store.for_each_cell(
                *region,
                values.elements(*storage),
                quantized(values, *storage),
                |cell| {
                    cell.writer = Some((wave, segment));
                    cell.readers.clear();
                },
            );
        }
    }
}

fn spans(region: Region, elements: u64, quantized: bool) -> Quanta {
    match region {
        Region::Whole => quanta(Region::run(0, elements), quantized),
        _ => quanta(region, quantized),
    }
}

fn quantized<V: Values>(values: &V, storage: u32) -> bool {
    values.elements(storage) > QUANTUM
}

#[derive(Clone, Default)]
struct WordCell {
    writer: Option<u32>,
    readers: Vec<u32>,
}

#[derive(Default)]
pub(crate) struct WordFlow {
    cells: std::collections::BTreeMap<u64, WordCell>,
}

impl WordFlow {
    fn split(&mut self, at: u64) {
        if self.cells.contains_key(&at) {
            return;
        }
        let inherited = self
            .cells
            .range(..at)
            .next_back()
            .map(|(_, cell)| cell.clone())
            .unwrap_or_default();
        self.cells.insert(at, inherited);
    }

    pub(crate) fn read(&mut self, spans: &[(u64, u64)], segment: u32, out: &mut Vec<u32>) {
        for (start, end) in spans {
            if end <= start {
                continue;
            }
            self.split(*start);
            self.split(*end);
            for (_, cell) in self.cells.range(*start..*end) {
                if let Some(writer) = cell.writer {
                    out.push(writer);
                }
            }
            for (_, cell) in self.cells.range_mut(*start..*end) {
                if !cell.readers.contains(&segment) {
                    cell.readers.push(segment);
                }
            }
        }
    }

    pub(crate) fn write(
        &mut self,
        spans: &[(u64, u64)],
        segment: u32,
        in_place: bool,
        out: &mut Vec<u32>,
    ) {
        for (start, end) in spans {
            if end <= start {
                continue;
            }
            self.split(*start);
            self.split(*end);
            for (_, cell) in self.cells.range(*start..*end) {
                if in_place && let Some(writer) = cell.writer {
                    out.push(writer);
                }
                out.extend_from_slice(&cell.readers);
            }
            for (_, cell) in self.cells.range_mut(*start..*end) {
                cell.writer = Some(segment);
                cell.readers.clear();
            }
        }
    }
}

pub(crate) fn arena_spans(
    base: u64,
    element: neura_abi::Element,
    elements: u64,
    region: Region,
    out: &mut Vec<(u64, u64)>,
) {
    out.clear();
    match region {
        Region::Empty => return,
        Region::Whole => {
            let words = element.storage_words(elements);
            out.push((base, base + words));
        }
        Region::Run { first, count } => {
            let (first, end) = element.word_span(first, count);
            out.push((base + first, base + end));
        }
        Region::Band {
            first,
            span,
            stride,
            count,
        } => {
            for index in 0..count {
                let (first, end) = element.word_span(first + index * stride, span);
                out.push((base + first, base + end));
            }
        }
    }
    if element.quantized() && !matches!(region, Region::Whole) {
        out.push((
            base + element.payload_words(elements),
            base + element.storage_words(elements),
        ));
    }
    out.sort_unstable();
    let mut kept = 0usize;
    for at in 0..out.len() {
        if kept == 0 || out[at].0 > out[kept - 1].1 {
            out[kept] = out[at];
            kept += 1;
        } else {
            out[kept - 1].1 = out[kept - 1].1.max(out[at].1);
        }
    }
    out.truncate(kept);
}
