use neura_abi::Element;
use neura_graph::{AttentionOptions, Free, Graph, Shape, Value};
use neura_runtime::{Program, Runtime};

#[path = "support/mod.rs"]
mod support;

use support::{assert_close, open};

const PLANES: u32 = 3;
const CAPACITY: u32 = 6;
const BOUND: u32 = 16;
const WIDTH: u32 = 4;
const SCALE: f32 = 0.5;

fn refuses(action: impl FnOnce()) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(action)).is_err()
}

fn data(count: u32, seed: u32) -> Vec<f32> {
    let mut entropy = seed | 1;
    (0..count)
        .map(|_| {
            entropy ^= entropy << 13;
            entropy ^= entropy >> 17;
            entropy ^= entropy << 5;
            (entropy >> 8) as f32 / 16_777_216.0 - 0.5
        })
        .collect()
}

fn offsets(lengths: &[f32]) -> Vec<u32> {
    let mut offsets = Vec::with_capacity(lengths.len() + 1);
    let mut running = 0u32;
    offsets.push(0);
    for length in lengths {
        running += *length as u32;
        offsets.push(running);
    }
    offsets
}

fn flags(lengths: &[f32], bound: u32) -> Vec<f32> {
    let mut flags = Vec::new();
    for plane in lengths {
        for row in 0..bound {
            flags.push(f32::from((row as f32) < *plane));
        }
    }
    flags
}

fn cursors(lengths: &[f32], planes: u32) -> Vec<f32> {
    lengths
        .iter()
        .map(|length| (length - 1.0).max(0.0))
        .chain(std::iter::repeat_n(
            0.0,
            (planes - lengths.len() as u32) as usize,
        ))
        .collect()
}

struct Batch {
    runtime: Runtime,
    graph: Graph<'static>,
    written: Value<'static>,
    masked: bool,
    binding: bool,
    plane: Value<'static>,
    position: Value<'static>,
    source: Value<'static>,
    dest: Value<'static>,
    batched: Value<'static>,
    packed: Value<'static>,
    unpacked: Value<'static>,
    query: Value<'static>,
    cursor: Value<'static>,
    out: Value<'static>,
    offsets: Value<'static>,
}

impl Batch {
    fn of(planes: u32, bound: u32, capacity: u32, width: u32, masked: bool, binding: bool) -> Self {
        let graph: Graph<'static> = Graph::new();
        let live: Option<Free> = binding.then(|| graph.free(planes));
        let planed = |dims: &[u32], axis: u32| match live {
            Some(free) => Shape::of(dims).freed(&[(axis, free)]),
            None => Shape::of(dims),
        };
        let mask = masked.then(|| graph.input(planed(&[planes, 1, bound, 1], 0), Element::Single));
        let lengths = match mask {
            Some(mask) => graph.sum_axis(mask, 2),
            None => graph.input(planed(&[planes], 3), Element::Single),
        };
        let written = mask.unwrap_or(lengths);
        let ragged = graph.ragged(bound, lengths);
        let batched = graph.input(planed(&[planes, 1, capacity, width], 0), Element::Single);
        let unpacked = graph.resident(Shape::matrix(planes * capacity, width), Element::Single);
        let packed = graph.resident(
            Shape::of([1, 1, bound, width]).freed(&[(2, ragged.extent)]),
            Element::Single,
        );
        let rows = graph.rows(ragged);
        let source = graph.add(
            graph.mul(rows.plane, graph.fill(Shape::scalar(), capacity as f32)),
            rows.position,
        );
        let dest = graph.add(graph.gather(ragged.offsets, rows.plane), rows.position);
        graph.write_into(packed, dest, graph.gather(batched, source));
        let doubled = graph.mul(packed, graph.fill(Shape::scalar(), 2.0));
        graph.write_into(unpacked, source, graph.gather(doubled, dest));
        let query = graph.input(planed(&[1, planes, 1, width], 1), Element::Single);
        let cursor = graph.input(Shape::of([1, planes, 1, 1]), Element::Single);
        let out = graph.attention(
            query,
            packed,
            packed,
            AttentionOptions {
                scale: SCALE,
                causal: true,
                origin: Some(cursor),
                segments: Some(ragged.offsets),
                reach: None,
            },
        );
        for value in [
            rows.plane,
            rows.position,
            source,
            dest,
            packed,
            unpacked,
            out,
            ragged.offsets,
        ] {
            graph.retain(value);
        }
        Self {
            runtime: open(),
            graph,
            written,
            masked,
            binding,
            plane: rows.plane,
            position: rows.position,
            source,
            dest,
            batched,
            packed,
            unpacked,
            query,
            cursor,
            out,
            offsets: ragged.offsets,
        }
    }

    fn compile(&self) -> Program<'_> {
        let weights = self.runtime.weights(&self.graph);
        self.runtime.compile(&self.graph, &weights)
    }

    fn step(&self, program: &Program<'_>, lengths: &[f32]) -> Produced {
        let planes = self.graph.shape(self.query).dims()[1];
        let bound = self.graph.shape(self.packed).dims()[2];
        let capacity = self.graph.shape(self.batched).dims()[2];
        let width = self.graph.shape(self.batched).dims()[3];
        let live = lengths.len() as u32;
        if self.binding {
            self.runtime.bind(program, &[live]);
        }
        let batched = data(live * capacity * width, 37);
        let query = data(live * width, 29);
        let flags = flags(lengths, bound);
        let written: &[f32] = if self.masked { &flags } else { lengths };
        self.runtime.write(program, self.written, written);
        self.runtime.write(program, self.batched, &batched);
        self.runtime.write(program, self.query, &query);
        self.runtime
            .write(program, self.cursor, &cursors(lengths, planes));
        self.runtime.run(program);
        Produced {
            plane: self.runtime.read(program, self.plane),
            position: self.runtime.read(program, self.position),
            source: self.runtime.read(program, self.source),
            dest: self.runtime.read(program, self.dest),
            offsets: self.runtime.read(program, self.offsets),
            packed: self.runtime.read(program, self.packed),
            unpacked: self.runtime.read(program, self.unpacked),
            out: self.runtime.read(program, self.out),
            batched,
            query,
        }
    }
}

struct Produced {
    plane: Vec<f32>,
    position: Vec<f32>,
    source: Vec<f32>,
    dest: Vec<f32>,
    offsets: Vec<f32>,
    packed: Vec<f32>,
    unpacked: Vec<f32>,
    out: Vec<f32>,
    batched: Vec<f32>,
    query: Vec<f32>,
}

struct Reference {
    plane: Vec<f32>,
    position: Vec<f32>,
    source: Vec<f32>,
    dest: Vec<f32>,
    offsets: Vec<f32>,
    packed: Vec<f32>,
    unpacked: Vec<f32>,
}

impl Reference {
    fn of(lengths: &[f32], batched: &[f32], capacity: u32, width: u32) -> Self {
        let offsets = offsets(lengths);
        let rows = *offsets.last().unwrap_or(&0);
        let mut reference = Self {
            plane: vec![0.0; rows as usize],
            position: vec![0.0; rows as usize],
            source: vec![0.0; rows as usize],
            dest: vec![0.0; rows as usize],
            offsets: offsets.iter().map(|offset| *offset as f32).collect(),
            packed: vec![0.0; rows as usize * width as usize],
            unpacked: vec![0.0; lengths.len() * capacity as usize * width as usize],
        };
        for plane in 0..lengths.len() {
            let start = offsets[plane] as usize;
            for row in 0..lengths[plane] as usize {
                let packed_row = start + row;
                let source_row = plane * capacity as usize + row;
                reference.plane[packed_row] = plane as f32;
                reference.position[packed_row] = row as f32;
                reference.source[packed_row] = source_row as f32;
                reference.dest[packed_row] = packed_row as f32;
                for depth in 0..width as usize {
                    reference.packed[packed_row * width as usize + depth] =
                        batched[source_row * width as usize + depth];
                    reference.unpacked[source_row * width as usize + depth] =
                        2.0 * batched[source_row * width as usize + depth];
                }
            }
        }
        reference
    }

    fn check(&self, produced: &Produced) {
        assert_close(&produced.plane, &self.plane, 0.0);
        assert_close(&produced.position, &self.position, 0.0);
        assert_close(&produced.source, &self.source, 0.0);
        assert_close(&produced.dest, &self.dest, 0.0);
        assert_close(&produced.offsets[..self.offsets.len()], &self.offsets, 0.0);
        assert_close(&produced.packed, &self.packed, 1e-6);
        assert_close(
            &produced.unpacked[..self.unpacked.len()],
            &self.unpacked,
            1e-6,
        );
    }
}

fn attended(
    lengths: &[f32],
    cached: &[f32],
    query: &[f32],
    cursors: &[f32],
    width: u32,
) -> Vec<f32> {
    let offsets = offsets(lengths);
    let mut out = vec![0.0f32; lengths.len() * width as usize];
    for plane in 0..lengths.len() {
        let start = offsets[plane] as usize;
        let row = &query[plane * width as usize..][..width as usize];
        let mut scores = Vec::new();
        let mut at = Vec::new();
        for slot in 0..lengths[plane] as usize {
            if slot as f32 > cursors[plane] {
                break;
            }
            let key = &cached[(start + slot) * width as usize..][..width as usize];
            scores.push(
                row.iter()
                    .zip(key)
                    .map(|(left, right)| left * right)
                    .sum::<f32>()
                    * SCALE,
            );
            at.push(start + slot);
        }
        if at.is_empty() {
            continue;
        }
        let peak = scores.iter().copied().fold(f32::MIN, f32::max);
        let weights = scores
            .iter()
            .map(|score| (score - peak).exp())
            .collect::<Vec<f32>>();
        let total = weights.iter().sum::<f32>();
        for depth in 0..width as usize {
            let mut value = 0.0f32;
            for (slot, weight) in at.iter().zip(&weights) {
                value += weight * cached[slot * width as usize + depth];
            }
            out[plane * width as usize + depth] = value / total;
        }
    }
    out
}

#[test]
fn a_row_map_names_the_plane_and_position_of_every_packed_row() {
    let batch = Batch::of(PLANES, BOUND, CAPACITY, WIDTH, false, false);
    let program = batch.compile();
    let lengths = [3.0f32, 0.0, 5.0];
    let produced = batch.step(&program, &lengths);
    Reference::of(&lengths, &produced.batched, CAPACITY, WIDTH).check(&produced);
    assert_eq!(produced.plane, [0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0, 2.0]);
    assert_eq!(produced.position, [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 3.0, 4.0]);
    assert_eq!(
        produced.source,
        [0.0, 1.0, 2.0, 12.0, 13.0, 14.0, 15.0, 16.0]
    );
}

#[test]
fn a_packed_batch_attends_the_keys_every_plane_holds() {
    let batch = Batch::of(PLANES, BOUND, CAPACITY, WIDTH, false, false);
    let program = batch.compile();
    let lengths = [3.0f32, 1.0, 4.0];
    let produced = batch.step(&program, &lengths);
    let reference = Reference::of(&lengths, &produced.batched, CAPACITY, WIDTH);
    reference.check(&produced);
    assert_close(
        &produced.out,
        &attended(
            &lengths,
            &reference.packed,
            &produced.query,
            &cursors(&lengths, PLANES),
            WIDTH,
        ),
        1e-5,
    );
}

#[test]
fn a_row_map_follows_the_lengths_the_device_authors() {
    let batch = Batch::of(PLANES, BOUND, CAPACITY, WIDTH, true, false);
    let program = batch.compile();
    let lengths = [2.0f32, 4.0, 0.0];
    let produced = batch.step(&program, &lengths);
    Reference::of(&lengths, &produced.batched, CAPACITY, WIDTH).check(&produced);
    assert_eq!(produced.plane, [0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
    let empty = batch.step(&program, &[0.0, 0.0, 0.0]);
    assert!(
        empty.plane.is_empty()
            && empty.position.is_empty()
            && empty.source.is_empty()
            && empty.dest.is_empty()
            && empty.packed.is_empty(),
        "a ragged axis of no rows maps no row",
    );
    assert!(empty.out.iter().all(|value| *value == 0.0));
}

#[test]
fn a_row_map_walks_only_the_planes_a_binding_holds() {
    let batch = Batch::of(PLANES, BOUND, CAPACITY, WIDTH, true, true);
    let program = batch.compile();
    let wide = batch.step(&program, &[3.0f32, 2.0, 4.0]);
    let narrow = batch.step(&program, &[3.0, 2.0]);
    let reference = Reference::of(&[3.0, 2.0], &narrow.batched, CAPACITY, WIDTH);
    reference.check(&narrow);
    assert_eq!(narrow.plane, wide.plane[..5].to_vec());
    assert_eq!(
        narrow.offsets[..3],
        [0.0, 3.0, 5.0],
        "the prefix closes the offsets at the planes a binding holds",
    );
}

#[test]
fn a_row_map_of_a_table_no_ragged_axis_closed_is_refused() {
    assert!(refuses(|| {
        let graph: Graph<'static> = Graph::new();
        let offsets = graph.input(Shape::matrix(PLANES + 1, 1), Element::Single);
        let extent = graph.free(BOUND);
        graph.rows(neura_graph::Ragged { extent, offsets });
    }));
}

#[test]
fn a_row_map_names_the_extent_its_axis_closes() {
    assert!(refuses(|| {
        let graph: Graph<'static> = Graph::new();
        let lengths = graph.input(Shape::vector(PLANES), Element::Single);
        let ragged = graph.ragged(BOUND, lengths);
        let other = graph.free(BOUND);
        graph.rows(neura_graph::Ragged {
            extent: other,
            offsets: ragged.offsets,
        });
    }));
}
