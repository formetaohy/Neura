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
    writers: Vec<u32>,
    readers: Vec<u32>,
}

impl Cell {
    fn hold(&mut self, segment: u32) {
        if self.readers.last() != Some(&segment) {
            self.readers.push(segment);
        }
    }

    fn write(&mut self, segment: u32) {
        if self.writers.last() != Some(&segment) {
            self.writers.push(segment);
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
    if matches!(query, Query::Writers | Query::Both) {
        out.extend_from_slice(&cell.writers);
    }
    if matches!(query, Query::Readers | Query::Both) {
        out.extend_from_slice(&cell.readers);
    }
}

pub(crate) struct Flow {
    stores: Vec<Store>,
    partition_stores: Vec<std::collections::BTreeMap<u32, Store>>,
    partitioned_readers: Vec<Vec<u32>>,
    partitioned_writers: Vec<Vec<u32>>,
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
            partition_stores: (0..values.len())
                .map(|_| std::collections::BTreeMap::new())
                .collect(),
            partitioned_readers: (0..values.len()).map(|_| Vec::new()).collect(),
            partitioned_writers: (0..values.len()).map(|_| Vec::new()).collect(),
            written,
        }
    }

    pub(crate) fn inspect<V: Values>(
        &self,
        values: &V,
        touches: &Touches,
        in_place: bool,
        partition: Option<u32>,
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
            match partition {
                Some(partition) => {
                    if let Some(store) = self.partition_stores[*storage as usize].get(&partition) {
                        store.query(
                            *region,
                            values.elements(*storage),
                            quantized(values, *storage),
                            Query::Writers,
                            out,
                        );
                    }
                }
                None => out.extend_from_slice(&self.partitioned_writers[*storage as usize]),
            }
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
            match partition {
                Some(partition) => {
                    if let Some(store) = self.partition_stores[*storage as usize].get(&partition) {
                        store.query(
                            *region,
                            values.elements(*storage),
                            quantized(values, *storage),
                            query,
                            out,
                        );
                    }
                }
                None => {
                    out.extend_from_slice(&self.partitioned_readers[*storage as usize]);
                    if in_place {
                        out.extend_from_slice(&self.partitioned_writers[*storage as usize]);
                    }
                }
            }
        }
        out.sort_unstable();
        out.dedup();
    }

    pub(crate) fn record<V: Values>(
        &mut self,
        values: &V,
        touches: &Touches,
        segment: u32,
        partition: Option<u32>,
    ) {
        let Some(partition) = partition else {
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
                        cell.hold(segment);
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
                        cell.write(segment);
                        cell.readers.clear();
                    },
                );
            }
            return;
        };
        for (storage, region) in &touches.reads {
            if !self.written[*storage as usize] {
                continue;
            }
            let store = self.partition_stores[*storage as usize]
                .entry(partition)
                .or_default();
            store.for_each_cell(
                *region,
                values.elements(*storage),
                quantized(values, *storage),
                |cell| {
                    cell.hold(segment);
                },
            );
            let collected = &mut self.partitioned_readers[*storage as usize];
            if collected.last() != Some(&segment) {
                collected.push(segment);
            }
        }
        for (storage, region) in &touches.writes {
            let store = self.partition_stores[*storage as usize]
                .entry(partition)
                .or_default();
            store.for_each_cell(
                *region,
                values.elements(*storage),
                quantized(values, *storage),
                |cell| {
                    cell.write(segment);
                    cell.readers.clear();
                },
            );
            let collected = &mut self.partitioned_writers[*storage as usize];
            if collected.last() != Some(&segment) {
                collected.push(segment);
            }
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

#[derive(Clone, Copy)]
pub(crate) struct WordWrite {
    pub(crate) segment: u32,
    pub(crate) in_place: bool,
    pub(crate) orders_writers: bool,
    pub(crate) partition: Option<u32>,
}

type Cells = std::collections::BTreeMap<u64, WordCell>;

#[derive(Default)]
pub(crate) struct WordFlow {
    cells: Cells,
    partition_cells: std::collections::BTreeMap<u32, Cells>,
}

impl WordFlow {
    fn split(cells: &mut Cells, at: u64) {
        if cells.contains_key(&at) {
            return;
        }
        let inherited = cells
            .range(..at)
            .next_back()
            .map(|(_, cell)| cell.clone())
            .unwrap_or_default();
        cells.insert(at, inherited);
    }

    fn read_cells(cells: &mut Cells, spans: &[(u64, u64)], segment: u32, out: &mut Vec<u32>) {
        for (start, end) in spans {
            if end <= start {
                continue;
            }
            Self::split(cells, *start);
            Self::split(cells, *end);
            for (_, cell) in cells.range(*start..*end) {
                if let Some(writer) = cell.writer {
                    out.push(writer);
                }
            }
            for (_, cell) in cells.range_mut(*start..*end) {
                if !cell.readers.contains(&segment) {
                    cell.readers.push(segment);
                }
            }
        }
    }

    fn inspect_write_cells(
        cells: &mut Cells,
        spans: &[(u64, u64)],
        write: WordWrite,
        out: &mut Vec<u32>,
    ) {
        let WordWrite {
            segment,
            in_place,
            orders_writers,
            ..
        } = write;
        for (start, end) in spans {
            if end <= start {
                continue;
            }
            Self::split(cells, *start);
            Self::split(cells, *end);
            for (_, cell) in cells.range(*start..*end) {
                if let Some(writer) = cell.writer
                    && (in_place || orders_writers)
                    && writer != segment
                {
                    out.push(writer);
                }
                out.extend_from_slice(&cell.readers);
            }
        }
    }

    fn record_write_cells(cells: &mut Cells, spans: &[(u64, u64)], segment: u32) {
        for (start, end) in spans {
            if end <= start {
                continue;
            }
            Self::split(cells, *start);
            Self::split(cells, *end);
            for (_, cell) in cells.range_mut(*start..*end) {
                cell.writer = Some(segment);
                cell.readers.clear();
            }
        }
    }

    fn write_cells(cells: &mut Cells, spans: &[(u64, u64)], write: WordWrite, out: &mut Vec<u32>) {
        Self::inspect_write_cells(cells, spans, write, out);
        Self::record_write_cells(cells, spans, write.segment);
    }

    pub(crate) fn read(
        &mut self,
        spans: &[(u64, u64)],
        partition: Option<u32>,
        segment: u32,
        out: &mut Vec<u32>,
    ) {
        Self::read_cells(&mut self.cells, spans, segment, out);
        match partition {
            Some(partition) => {
                if let Some(cells) = self.partition_cells.get_mut(&partition) {
                    Self::read_cells(cells, spans, segment, out);
                }
            }
            None => {
                for cells in self.partition_cells.values_mut() {
                    Self::read_cells(cells, spans, segment, out);
                }
            }
        }
    }

    pub(crate) fn write(&mut self, spans: &[(u64, u64)], write: WordWrite, out: &mut Vec<u32>) {
        match write.partition {
            Some(partition) => {
                let cells = self.partition_cells.entry(partition).or_default();
                Self::write_cells(cells, spans, write, out);
            }
            None => {
                Self::write_cells(&mut self.cells, spans, write, out);
                for cells in self.partition_cells.values_mut() {
                    Self::inspect_write_cells(cells, spans, write, out);
                }
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
    let base = base / neura_abi::WORD_BYTES;
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

#[cfg(test)]
mod tests {
    use super::{Query, Store, arena_spans};
    use crate::region::Region;
    use neura_abi::Element;

    #[test]
    fn arena_spans_address_words() {
        let mut spans = Vec::new();
        arena_spans(256, Element::Single, 64, Region::run(4, 8), &mut spans);
        assert_eq!(spans, vec![(68, 76)]);
        arena_spans(256, Element::Single, 64, Region::Whole, &mut spans);
        assert_eq!(spans, vec![(64, 128)]);
    }

    #[test]
    fn a_reader_holds_every_writer_of_a_cell() {
        let mut store = Store::default();
        let region = Region::run(0, 8);
        store.for_each_cell(region, 8, false, |cell| cell.write(0));
        store.for_each_cell(region, 8, false, |cell| cell.write(1));
        let mut held = Vec::new();
        store.query(region, 8, false, Query::Writers, &mut held);
        assert_eq!(held, vec![0, 1]);
    }
}
