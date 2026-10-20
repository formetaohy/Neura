use super::{Encoding, Placed, Plan, Quantum};
use crate::authored::{Authored, Counted, Patches};
use crate::span::{Extents, Measure, Split};
use neura_abi::{Element, Kind, PatchRecord, SegmentRecord, Store, ValueRecord};
use neura_graph::{Graph, Shape, ValueInfo};
use neura_profile::{
    AttentionTile, ConvolutionInputTile, ConvolutionTile, ConvolutionWeightTile, MatmulStrategy,
    MatmulTile, Profile,
};
use std::mem::size_of;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex};

const MAGIC: u32 = 0x4e_50_4c_41;
const FORMAT: u32 = 2;

impl Plan {
    pub fn image(&self) -> Vec<u8> {
        let mut out = Writer::default();
        out.word(MAGIC);
        out.word(FORMAT);
        out.long(crate::PLAN_BUILD);
        out.long(self.fingerprint);
        out.long(self.alignment);
        out.word(self.paged_word());
        out.word(self.profile.workgroup());
        out.word(self.profile.registers());
        out.long(self.profile.shared_bytes());
        out.word(self.tiles().len() as u32);
        for tile in self.tiles() {
            for word in tile_words(*tile) {
                out.word(word);
            }
        }
        out.word(self.shapes.len() as u32);
        for info in self.shapes.iter() {
            for word in shape_words(info.shape) {
                out.word(word);
            }
        }
        out.words(&self.slot_bounds);
        out.words(
            &self
                .kinds
                .iter()
                .map(|kind| kind.code())
                .collect::<Vec<_>>(),
        );
        out.words(
            &self
                .elements
                .iter()
                .map(|element| element.code())
                .collect::<Vec<_>>(),
        );
        out.words(
            &self
                .matmul_elements
                .iter()
                .map(|element| element.code())
                .collect::<Vec<_>>(),
        );
        out.word(self.strategies);
        out.flag(self.updates_weights);
        out.long(self.work);
        out.bytes(&self.tasks);
        out.bytes(&self.steps);
        out.words(&self.geometries);
        out.word(self.products.len() as u32);
        for product in &self.products {
            out.word(product.planes());
            out.word(product.rows());
            out.word(product.columns());
            out.word(product.depth());
            out.word(product.weight().element().code());
            out.word(contiguous_word(product.weight()));
        }
        write_tiles(&mut out, self.attention.iter().map(|tile| tile.words()));
        write_tiles(&mut out, self.convolution.iter().map(|tile| tile.words()));
        write_tiles(
            &mut out,
            self.convolution_weight.iter().map(|tile| tile.words()),
        );
        write_tiles(
            &mut out,
            self.convolution_input.iter().map(|tile| tile.words()),
        );
        write_measures(&mut out, self.extents.measures());
        out.word(self.splits.len() as u32);
        for split in &self.splits {
            write_split(&mut out, *split);
        }
        out.longs(&self.works);
        out.words(&self.depends);
        out.words(&self.depends_at);
        let (slots, value_slots, measure_slots, measures) = self.authored.image_parts();
        out.words(slots);
        write_nested(&mut out, value_slots);
        write_nested(&mut out, measure_slots);
        out.pods(measures);
        out.words(&self.patches.owners);
        out.word(self.patches.counted.len() as u32);
        for counted in &self.patches.counted {
            let (count, writer, slots, segment, values, tasks) = counted.image_parts();
            out.word(count);
            out.word(writer);
            out.words(slots);
            out.word(segment);
            out.words(values);
            out.words(tasks);
        }
        out.words(&self.gather_tables);
        write_encoding(&mut out, &self.bound);
        out.bytes
    }

    pub fn restore(
        graph: &Graph,
        alignment: u64,
        profile: Profile,
        encoding_bytes: u64,
        weight_slots: u32,
        image: &[u8],
    ) -> Option<Self> {
        let mut held = Reader::of(image);
        if held.word()? != MAGIC || held.word()? != FORMAT {
            return None;
        }
        if held.long()? != crate::PLAN_BUILD || held.long()? != graph.fingerprint() {
            return None;
        }
        let header_alignment = held.long()?;
        let header_paged = held.word()?;
        let header_workgroup = held.word()?;
        let header_registers = held.word()?;
        let header_shared = held.long()?;
        if header_alignment != alignment
            || header_paged != u32::from(weight_slots > 0)
            || header_workgroup != profile.workgroup()
            || header_registers != profile.registers()
            || header_shared != profile.shared_bytes()
        {
            return None;
        }
        let tile_count = held.word()? as usize;
        if tile_count != profile.tiles().len() {
            return None;
        }
        for tile in profile.tiles() {
            if held.take::<8>()? != tile_words(*tile) {
                return None;
            }
        }
        let plan_values = held.word()? as usize;
        let mut shapes = graph.snapshot().values().to_vec();
        if plan_values < shapes.len() {
            return None;
        }
        for id in 0..plan_values {
            let words = held.take::<8>()?;
            if id < shapes.len() {
                if shape_words(shapes[id].shape) != words {
                    return None;
                }
            } else {
                let shape = shape_of_words(words)?;
                shapes.push(ValueInfo::derived(shape, id as u32));
            }
        }
        let shapes = Arc::new(shapes);
        let values = &shapes;
        let slot_bounds = held.words()?;
        if slot_bounds != slot_bounds_of(values) {
            return None;
        }
        let kinds = held
            .words()?
            .into_iter()
            .map(|code| Kind::ALL.get(code as usize).copied())
            .collect::<Option<Vec<_>>>()?;
        let elements = held
            .words()?
            .into_iter()
            .map(|code| Element::ALL.get(code as usize).copied())
            .collect::<Option<Vec<_>>>()?;
        let matmul_elements = held
            .words()?
            .into_iter()
            .map(|code| Element::ALL.get(code as usize).copied())
            .collect::<Option<Vec<_>>>()?;
        let strategies = held.word()?;
        let updates_weights = held.flag()?;
        let work = held.long()?;
        let tasks = held.bytes()?.to_vec();
        let steps = held.bytes()?.to_vec();
        let geometries = held.words()?;
        if geometries.len() != profile.tiles().len() {
            return None;
        }
        let mut products = Vec::new();
        for _ in 0..held.word()? {
            let planes = held.word()?;
            let rows = held.word()?;
            let columns = held.word()?;
            let depth = held.word()?;
            let element = Element::ALL.get(held.word()? as usize).copied()?;
            let contiguous = contiguous_of(held.word()?)?;
            if planes == 0 || rows == 0 || columns == 0 || depth == 0 {
                return None;
            }
            products.push(crate::product::Product::of(
                planes,
                rows,
                columns,
                depth,
                crate::product::Weight::of(element, contiguous),
            ));
        }
        let attention = read_tiles(&mut held, AttentionTile::of_words)?;
        let convolution = read_tiles(&mut held, ConvolutionTile::of_words)?;
        let convolution_weight = read_tiles(&mut held, ConvolutionWeightTile::of_words)?;
        let convolution_input = read_tiles(&mut held, ConvolutionInputTile::of_words)?;
        let measures = read_measures(&mut held)?;
        let splits = read_splits(&mut held)?;
        let works = held.longs()?;
        let depends = held.words()?;
        let depends_at = held.words()?;
        let authored = Authored::of_image_parts(
            held.words()?,
            read_nested(&mut held)?,
            read_nested(&mut held)?,
            held.pods()?,
        );
        let owners = held.words()?;
        let mut counted = Vec::new();
        for _ in 0..held.word()? {
            counted.push(Counted::of_image_parts(
                held.word()?,
                held.word()?,
                held.words()?,
                held.word()?,
                held.words()?,
                held.words()?,
            ));
        }
        let patches = Patches::of_image_parts(owners, counted);
        let gather_tables = held.words()?;
        let bound = read_encoding(&mut held)?;
        if !held.done() {
            return None;
        }
        let extents = Extents::of(values, profile.tiles(), &convolution, &measures);
        let plan = Self {
            profile,
            fingerprint: graph.fingerprint(),
            kinds,
            elements,
            matmul_elements,
            strategies,
            tasks,
            steps,
            geometries,
            products,
            attention,
            convolution,
            convolution_weight,
            convolution_input,
            layout: crate::layout::Layout::of_values(values, alignment),
            updates_weights,
            work,
            extents,
            splits,
            works,
            slot_bounds,
            depends,
            depends_at,
            authored,
            patches,
            declared_waves: Vec::new(),
            gather_tables,
            shapes,
            alignment,
            bound: Arc::new(Encoding::empty()),
            remembered: Mutex::new(crate::remembered::Remembered::of(encoding_bytes)),
            derived: AtomicU64::new(0),
            paged: weight_slots > 0,
        };
        plan.validate(&bound)?;
        let mut plan = plan;
        plan.declared_waves = super::declared_waves(&bound);
        plan.bound = Arc::new(bound);
        Some(plan)
    }

    fn validate(&self, bound: &Encoding) -> Option<()> {
        let values = self.shapes.len();
        let (_, value_slots, measure_slots, _) = self.authored.image_parts();
        if self.splits.len() != self.works.len()
            || self.splits.len() != self.patches.owners.len()
            || self.depends_at.len() != self.splits.len() + 1
            || self.depends_at.last().copied()? as usize != self.depends.len()
            || value_slots.len() != values
            || measure_slots.len() != self.extents.measures().len()
            || bound.lengths() != self.slot_bounds
            || bound.values().len() != values * size_of::<ValueRecord>()
            || !bound
                .tasks()
                .len()
                .is_multiple_of(size_of::<neura_abi::TaskRecord>())
            || bound.task_count() as usize
                != bound.tasks().len() / size_of::<neura_abi::TaskRecord>()
            || bound.readable.len() != values
            || bound.held.len() != values
            || bound.spans.len() != values
            || bound.dependencies_at.len() != bound.segments().len() + 1
            || bound.dependencies_at.last().copied()? as usize != bound.dependencies().len()
            || bound
                .order()
                .iter()
                .any(|at| *at as usize >= bound.task_count() as usize)
            || bound
                .waves()
                .iter()
                .any(|wave| *wave as usize >= bound.wave_count() as usize)
            || bound
                .dependencies()
                .iter()
                .any(|at| *at as usize >= bound.segments().len())
            || bound
                .successors()
                .iter()
                .any(|at| *at as usize >= bound.segments().len())
            || bound
                .roots()
                .iter()
                .any(|at| *at as usize >= bound.segments().len())
            || bound.wave_table().len() != 1 + bound.wave_count() as usize * 4
            || bound.wave_table().first().copied() != Some(bound.wave_count())
        {
            return None;
        }
        for segment in bound.segments() {
            if segment.first as usize + segment.count as usize > bound.order().len()
                || segment.successors as usize + segment.successors_count as usize
                    > bound.successors().len()
            {
                return None;
            }
        }
        Some(())
    }

    fn paged_word(&self) -> u32 {
        u32::from(self.paged)
    }
}

fn shape_words(shape: Shape) -> [u32; 8] {
    let dims = shape.dims();
    [
        dims[0],
        dims[1],
        dims[2],
        dims[3],
        shape.free(0).unwrap_or(u32::MAX),
        shape.free(1).unwrap_or(u32::MAX),
        shape.free(2).unwrap_or(u32::MAX),
        shape.free(3).unwrap_or(u32::MAX),
    ]
}

fn shape_of_words(words: [u32; 8]) -> Option<Shape> {
    let dims = [words[0], words[1], words[2], words[3]];
    if dims.contains(&0) {
        return None;
    }
    let mut frees = [None; 4];
    for axis in 0..4 {
        frees[axis] = match words[4 + axis] {
            u32::MAX => None,
            slot => Some(slot),
        };
    }
    if frees
        .iter()
        .any(|slot| slot.is_some_and(|slot| slot > u32::from(u8::MAX)))
    {
        return None;
    }
    Some(Shape::with_frees(dims, frees))
}

fn slot_bounds_of(values: &[ValueInfo]) -> Vec<u32> {
    let mut bounds = Vec::new();
    for info in values {
        for axis in 0..neura_abi::MAX_RANK {
            if let Some(slot) = info.shape.free(axis) {
                let bound = info.shape.dims()[axis as usize];
                if bounds.len() <= slot as usize {
                    bounds.resize(slot as usize + 1, 0);
                }
                bounds[slot as usize] = bound;
            }
        }
    }
    bounds
}

fn tile_words(tile: MatmulTile) -> [u32; 8] {
    [
        strategy_word(tile.strategy()),
        tile.rows(),
        tile.columns(),
        tile.depth(),
        tile.threads(),
        tile.thread_rows(),
        tile.thread_columns(),
        u32::from(tile.stages_a_run()),
    ]
}

fn strategy_word(strategy: MatmulStrategy) -> u32 {
    match strategy {
        MatmulStrategy::Staged => 0,
        MatmulStrategy::Streamed => 1,
        MatmulStrategy::Cooperative => 2,
        MatmulStrategy::Folded => 3,
    }
}

fn contiguous_word(weight: crate::product::Weight) -> u32 {
    match weight.contiguous() {
        crate::product::Contiguous::Reduction => 0,
        crate::product::Contiguous::Output => 1,
    }
}

fn contiguous_of(word: u32) -> Option<crate::product::Contiguous> {
    match word {
        0 => Some(crate::product::Contiguous::Reduction),
        1 => Some(crate::product::Contiguous::Output),
        _ => None,
    }
}

fn write_tiles<const N: usize>(out: &mut Writer, tiles: impl Iterator<Item = [u32; N]>) {
    let tiles = tiles.collect::<Vec<_>>();
    out.word(tiles.len() as u32);
    for tile in tiles {
        for word in tile {
            out.word(word);
        }
    }
}

fn read_tiles<const N: usize, T>(
    held: &mut Reader<'_>,
    of_words: impl Fn([u32; N]) -> Option<T>,
) -> Option<Vec<T>> {
    let mut tiles = Vec::new();
    for _ in 0..held.word()? {
        tiles.push(of_words(held.take::<N>()?)?);
    }
    Some(tiles)
}

fn write_measures(out: &mut Writer, measures: &[Measure]) {
    out.word(measures.len() as u32);
    for measure in measures {
        let (tag, value, input, filter, geometry) = match *measure {
            Measure::Elements(value) => (0, value, 0, 0, 0),
            Measure::Rows(value) => (1, value, 0, 0, 0),
            Measure::Words(value) => (2, value, 0, 0, 0),
            Measure::Tiles { value, geometry } => (3, value, 0, 0, geometry),
            Measure::Tokens(value) => (4, value, 0, 0, 0),
            Measure::Windows {
                value,
                input,
                filter,
                geometry,
            } => (5, value, input, filter, geometry),
        };
        out.word(tag);
        out.word(value);
        out.word(input);
        out.word(filter);
        out.word(geometry);
    }
}

fn read_measures(held: &mut Reader<'_>) -> Option<Vec<Measure>> {
    let mut measures = Vec::new();
    for _ in 0..held.word()? {
        let tag = held.word()?;
        let value = held.word()?;
        let input = held.word()?;
        let filter = held.word()?;
        let geometry = held.word()?;
        measures.push(match tag {
            0 => Measure::Elements(value),
            1 => Measure::Rows(value),
            2 => Measure::Words(value),
            3 => Measure::Tiles { value, geometry },
            4 => Measure::Tokens(value),
            5 => Measure::Windows {
                value,
                input,
                filter,
                geometry,
            },
            _ => return None,
        });
    }
    Some(measures)
}

fn write_split(out: &mut Writer, split: Split) {
    let (tag, words) = match split {
        Split::Range { first, count } => (0, [first, count, 0, 0, 0, 0]),
        Split::Uniform {
            measure,
            index,
            group,
        } => (1, [measure, index, group, 0, 0, 0]),
        Split::Plane {
            measure,
            planes,
            plane,
            index,
            group,
        } => (2, [measure, planes, plane, index, group, 0]),
        Split::Segment {
            measure,
            plane,
            index,
            group,
        } => (3, [measure, plane, index, group, 0, 0]),
        Split::Ragged {
            planes,
            plane,
            index,
            group,
        } => (4, [planes, plane, index, group, 0, 0]),
    };
    out.word(tag);
    for word in words {
        out.word(word);
    }
}

fn read_splits(held: &mut Reader<'_>) -> Option<Vec<Split>> {
    let mut splits = Vec::new();
    for _ in 0..held.word()? {
        let tag = held.word()?;
        let words = held.take::<6>()?;
        splits.push(match tag {
            0 => Split::Range {
                first: words[0],
                count: words[1],
            },
            1 => Split::Uniform {
                measure: words[0],
                index: words[1],
                group: words[2],
            },
            2 => Split::Plane {
                measure: words[0],
                planes: words[1],
                plane: words[2],
                index: words[3],
                group: words[4],
            },
            3 => Split::Segment {
                measure: words[0],
                plane: words[1],
                index: words[2],
                group: words[3],
            },
            4 => Split::Ragged {
                planes: words[0],
                plane: words[1],
                index: words[2],
                group: words[3],
            },
            _ => return None,
        });
    }
    Some(splits)
}

fn write_nested(out: &mut Writer, values: &[Vec<u32>]) {
    out.word(values.len() as u32);
    for value in values {
        out.words(value);
    }
}

fn read_nested(held: &mut Reader<'_>) -> Option<Vec<Vec<u32>>> {
    let mut values = Vec::new();
    for _ in 0..held.word()? {
        values.push(held.words()?);
    }
    Some(values)
}

fn write_encoding(out: &mut Writer, encoding: &Encoding) {
    out.bytes(&encoding.values);
    out.bytes(&encoding.tasks);
    out.word(encoding.spans.len() as u32);
    for span in &encoding.spans {
        out.flag(span.is_some());
        if let Some(placed) = span {
            out.word(placed.store.code());
            out.long(placed.address);
            out.word(placed.element.code());
            out.single(placed.scale);
            out.long(placed.table);
        }
    }
    out.flags(&encoding.readable);
    out.word(encoding.quanta.len() as u32);
    for quantum in &encoding.quanta {
        out.long(quantum.offset);
        out.single(quantum.scale);
    }
    out.words(&encoding.lengths);
    out.long(encoding.arena_bytes);
    out.long(encoding.tensor_bytes);
    out.words(&encoding.order);
    out.words(&encoding.waves);
    out.pods(&encoding.segments);
    out.words(&encoding.wave_segments);
    out.words(&encoding.dependencies);
    out.words(&encoding.dependencies_at);
    out.words(&encoding.successors);
    out.words(&encoding.wave_table);
    out.words(&encoding.roots);
    out.long(encoding.work);
    out.pods(&encoding.patches);
    out.words(&encoding.patch_list);
    out.flags(&encoding.held);
}

fn read_encoding(held: &mut Reader<'_>) -> Option<Encoding> {
    let values = held.bytes()?.to_vec();
    let tasks = held.bytes()?.to_vec();
    let mut spans = Vec::new();
    for _ in 0..held.word()? {
        if !held.flag()? {
            spans.push(None);
            continue;
        }
        let store = *Store::ALL.get(held.word()? as usize)?;
        let address = held.long()?;
        let element = *Element::ALL.get(held.word()? as usize)?;
        let scale = held.single()?;
        let table = held.long()?;
        spans.push(Some(Placed {
            store,
            address,
            element,
            scale,
            table,
        }));
    }
    let readable = held.flags()?;
    let mut quanta = Vec::new();
    for _ in 0..held.word()? {
        quanta.push(Quantum {
            offset: held.long()?,
            scale: held.single()?,
        });
    }
    let lengths = held.words()?;
    let arena_bytes = held.long()?;
    let tensor_bytes = held.long()?;
    let order = held.words()?;
    let waves = held.words()?;
    let segments: Vec<SegmentRecord> = held.pods()?;
    let wave_segments = held.words()?;
    let dependencies = held.words()?;
    let dependencies_at = held.words()?;
    let successors = held.words()?;
    let wave_table = held.words()?;
    let roots = held.words()?;
    let work = held.long()?;
    let patches: Vec<PatchRecord> = held.pods()?;
    let patch_list = held.words()?;
    let held_flags = held.flags()?;
    Some(Encoding {
        values,
        tasks,
        spans,
        readable,
        quanta,
        lengths,
        arena_bytes,
        tensor_bytes,
        order,
        waves,
        segments,
        wave_segments,
        dependencies,
        dependencies_at,
        successors,
        wave_table,
        roots,
        work,
        patches,
        patch_list,
        held: held_flags,
    })
}

#[derive(Default)]
struct Writer {
    bytes: Vec<u8>,
}

impl Writer {
    fn word(&mut self, word: u32) {
        self.bytes.extend_from_slice(&word.to_le_bytes());
    }

    fn long(&mut self, long: u64) {
        self.bytes.extend_from_slice(&long.to_le_bytes());
    }

    fn single(&mut self, single: f32) {
        self.word(single.to_bits());
    }

    fn flag(&mut self, flag: bool) {
        self.word(u32::from(flag));
    }

    fn words(&mut self, words: &[u32]) {
        self.word(words.len() as u32);
        for word in words {
            self.word(*word);
        }
    }

    fn longs(&mut self, longs: &[u64]) {
        self.word(longs.len() as u32);
        for long in longs {
            self.long(*long);
        }
    }

    fn bytes(&mut self, bytes: &[u8]) {
        self.word(bytes.len() as u32);
        self.bytes.extend_from_slice(bytes);
    }

    fn flags(&mut self, flags: &[bool]) {
        self.word(flags.len() as u32);
        self.bytes.extend(flags.iter().map(|flag| u8::from(*flag)));
    }

    fn pods<T: bytemuck::Pod>(&mut self, pods: &[T]) {
        self.word(pods.len() as u32);
        self.bytes.extend_from_slice(bytemuck::cast_slice(pods));
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn of(bytes: &'a [u8]) -> Self {
        Self { bytes, at: 0 }
    }

    fn word(&mut self) -> Option<u32> {
        let word = self.take_bytes(4)?;
        Some(u32::from_le_bytes(word.try_into().ok()?))
    }

    fn long(&mut self) -> Option<u64> {
        let long = self.take_bytes(8)?;
        Some(u64::from_le_bytes(long.try_into().ok()?))
    }

    fn single(&mut self) -> Option<f32> {
        Some(f32::from_bits(self.word()?))
    }

    fn flag(&mut self) -> Option<bool> {
        match self.word()? {
            0 => Some(false),
            1 => Some(true),
            _ => None,
        }
    }

    fn take<const N: usize>(&mut self) -> Option<[u32; N]> {
        let mut words = [0u32; N];
        for word in &mut words {
            *word = self.word()?;
        }
        Some(words)
    }

    fn take_bytes(&mut self, count: usize) -> Option<&'a [u8]> {
        let end = self.at.checked_add(count)?;
        let taken = self.bytes.get(self.at..end)?;
        self.at = end;
        Some(taken)
    }

    fn words(&mut self) -> Option<Vec<u32>> {
        let count = self.word()? as usize;
        let bytes = self.take_bytes(count.checked_mul(4)?)?;
        let (words, rest) = bytes.as_chunks::<4>();
        if !rest.is_empty() {
            return None;
        }
        Some(words.iter().map(|word| u32::from_le_bytes(*word)).collect())
    }

    fn longs(&mut self) -> Option<Vec<u64>> {
        let count = self.word()? as usize;
        let bytes = self.take_bytes(count.checked_mul(8)?)?;
        let (longs, rest) = bytes.as_chunks::<8>();
        if !rest.is_empty() {
            return None;
        }
        Some(longs.iter().map(|long| u64::from_le_bytes(*long)).collect())
    }

    fn bytes(&mut self) -> Option<&'a [u8]> {
        let count = self.word()? as usize;
        self.take_bytes(count)
    }

    fn flags(&mut self) -> Option<Vec<bool>> {
        let count = self.word()? as usize;
        let bytes = self.take_bytes(count)?;
        bytes
            .iter()
            .map(|byte| match byte {
                0 => Some(false),
                1 => Some(true),
                _ => None,
            })
            .collect()
    }

    fn pods<T: bytemuck::Pod>(&mut self) -> Option<Vec<T>> {
        let count = self.word()? as usize;
        let bytes = self.take_bytes(count.checked_mul(size_of::<T>())?)?;
        let mut pods = Vec::with_capacity(count);
        let mut at = 0;
        while at < bytes.len() {
            pods.push(bytemuck::pod_read_unaligned(
                &bytes[at..at + size_of::<T>()],
            ));
            at += size_of::<T>();
        }
        Some(pods)
    }

    fn done(&self) -> bool {
        self.at == self.bytes.len()
    }
}
