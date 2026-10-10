use crate::hazard::{Accesses, Hazard, quanta};
use crate::lower;
use crate::plan;
use crate::region::{self, Region};
use crate::remembered::Remembered;
use crate::span;
use neura_abi::{Element, Kind, NO_VALUE, StepFields, StepRecord};
use neura_graph::{Graph, Shape, ValueInfo, Window};
use neura_profile::{Budget, MatmulStrategy, MatmulTile, Profile};

fn spans(region: Region) -> Vec<(u64, Option<u64>)> {
    match region {
        Region::Empty => Vec::new(),
        Region::Whole => vec![(0, None)],
        Region::Run { first, count } => vec![(first, first.checked_add(count))],
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

fn overlaps_span(left: (u64, Option<u64>), right: (u64, Option<u64>)) -> bool {
    let left_end = left.1.unwrap_or(u64::MAX);
    let right_end = right.1.unwrap_or(u64::MAX);
    left.0 < right_end && right.0 < left_end
}

fn overlaps(left: Region, right: Region) -> bool {
    match (left, right) {
        (Region::Empty, _) | (_, Region::Empty) => false,
        (Region::Whole, _) | (_, Region::Whole) => true,
        _ => spans(left)
            .iter()
            .any(|l| spans(right).iter().any(|r| overlaps_span(*l, *r))),
    }
}

fn naive(entries: &[(Region, Hazard)], region: Region) -> Hazard {
    let mut hazard = Hazard::default();
    let query = quanta(region, true).collect::<Vec<_>>();
    for (kept, carried) in entries {
        if matches!((kept, region), (Region::Empty, _) | (_, Region::Empty)) {
            continue;
        }
        let whole = matches!((kept, region), (Region::Whole, _) | (_, Region::Whole));
        let quantized = quanta(*kept, true).collect::<Vec<_>>();
        if whole
            || quantized
                .iter()
                .any(|l| query.iter().any(|r| overlaps_span(*l, *r)))
        {
            hazard.join(carried);
        }
    }
    hazard
}

fn region(first: u64, count: u64) -> Region {
    if count == 0 {
        Region::Whole
    } else {
        Region::Run { first, count }
    }
}

struct Chaos(u64);

impl Chaos {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }

    fn below(&mut self, bound: u64) -> u64 {
        self.next() % bound
    }

    fn region(&mut self) -> Region {
        match self.below(4) {
            0 => Region::Whole,
            1 => Region::run(self.below(12000), self.below(5000).max(1)),
            _ => {
                let span = self.below(2000).max(1);
                let stride = span + self.below(3000);
                let count = self.below(4).max(1);
                Region::band(self.below(12000), span, stride, count)
            }
        }
    }
}

#[test]
fn a_product_narrows_every_operand_to_the_tile_it_walks() {
    let left = ValueInfo::derived(Shape::of([1, 1, 3, 4]), 0);
    let right = ValueInfo::derived(Shape::of([1, 1, 4, 5]), 1);
    let bias = ValueInfo::derived(Shape::of([1, 1, 1, 5]), 2);
    let out = ValueInfo::derived(Shape::of([1, 1, 1, 30]), 3);
    let values = vec![left, right, bias, out];
    let tiles = vec![MatmulTile::new(MatmulStrategy::Staged, 2, 2, 2, 1, 1)];
    let task = lower::Task {
        kind: Kind::Matmul,
        op: neura_pointwise::NONE,
        geometry: 0,
        first: 1,
        count: 1,
        slot: 1,
        out: 3,
        extra: NO_VALUE,
        inputs: [0, 1, NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE],
        origin: NO_VALUE,
        literal: 0.0,
        knob: neura_abi::NO_VALUE,
        window: Window::sliding([1, 1]),
        splits: 2,
        work: 0,
        in_place: false,
        axis: 0,
        offset: 0,
        prelude: Vec::new(),
        chain: vec![StepRecord::of(StepFields {
            op: neura_pointwise::MUL,
            operand: 2,
            swapped: 0,
        })],
        unit: 0,
        split: span::Split::Range { first: 1, count: 1 },
        depends: Vec::new(),
        patch: NO_VALUE,
        segments: NO_VALUE,
        reach: 0,
        keys: 0,
        plane: 0,
        queries: NO_VALUE,
        tokens: 0,
        grid: NO_VALUE,
        keep: 0,
    };
    let mut touched = region::Touches::default();
    region::touches(
        &values.as_slice(),
        &tiles,
        &task,
        (task.first, task.count),
        &mut touched,
    );
    assert_eq!(
        touched.writes,
        vec![(
            3,
            Region::Band {
                first: 17,
                span: 2,
                stride: 5,
                count: 2,
            },
        )],
        "a product writes the tile it carries",
    );
    assert_eq!(
        touched.reads,
        vec![
            (
                0,
                Region::Band {
                    first: 2,
                    span: 2,
                    stride: 4,
                    count: 2,
                },
            ),
            (
                1,
                Region::Band {
                    first: 12,
                    span: 2,
                    stride: 5,
                    count: 2,
                },
            ),
            (2, Region::Run { first: 2, count: 2 },),
        ],
        "a product reads the rows, the columns and the depth slice its tile walks",
    );
}

#[test]
fn a_quantum_covers_every_run_a_band_names() {
    let mut chaos = Chaos(0x1c3);
    assert!(quanta(Region::Whole, true).next().is_none());
    for _ in 0..2000 {
        let left = chaos.region();
        let right = chaos.region();
        if left == Region::Whole {
            continue;
        }
        for (first, end) in spans(left) {
            let end = end.unwrap_or(u64::MAX);
            assert!(
                quanta(left, true)
                    .any(|(from, to)| { from <= first && to.unwrap_or(u64::MAX) >= end }),
                "the quanta of {left:?} cover the run [{first}, {end})",
            );
        }
        if overlaps(left, right) && left != Region::Whole && right != Region::Whole {
            let quantized_left = quanta(left, true).collect::<Vec<_>>();
            let quantized_right = quanta(right, true).collect::<Vec<_>>();
            assert!(
                quantized_left
                    .iter()
                    .any(|a| quantized_right.iter().any(|b| overlaps_span(*a, *b))),
                "the quanta of {left:?} and {right:?} meet where the runs they widen meet",
            );
        }
    }
}

#[test]
fn a_store_that_fits_a_quantum_widens_no_band() {
    let band = Region::band(3, 2, 5, 4);
    assert_eq!(quanta(band, false).collect::<Vec<_>>(), spans(band));
    assert_ne!(quanta(band, true).collect::<Vec<_>>(), spans(band));
}

#[test]
fn an_interval_map_answers_what_a_scan_of_every_access_answers() {
    let mut chaos = Chaos(0x5eed);
    let mut accesses = Accesses::default();
    let mut entries = Vec::new();
    for _ in 0..4000 {
        let choice = chaos.below(4);
        let region = chaos.region();
        let hazard = if chaos.below(6) == 0 {
            Hazard::deep(chaos.below(7) as u32)
        } else {
            Hazard::at(chaos.below(7) as u32, chaos.below(5) as u32)
        };
        if choice == 3 {
            accesses.clear();
            entries.clear();
            continue;
        }
        let expected = naive(&entries, region);
        if choice == 0 {
            accesses.record(region, &hazard, true);
            entries.push((region, hazard));
        } else {
            let found = accesses.query(region, true);
            assert_eq!(found.wave, expected.wave, "wave of {region:?}");
            assert_eq!(found.segments, expected.segments, "segments of {region:?}");
            assert_eq!(found.deepest, expected.deepest, "deepest of {region:?}");
        }
    }
}

#[test]
fn an_interval_map_keeps_every_run_exact() {
    let mut chaos = Chaos(0x9a71);
    let mut accesses = Accesses::default();
    let mut entries = Vec::new();
    for _ in 0..4000 {
        let choice = chaos.below(4);
        let region = if chaos.below(8) == 0 {
            Region::Whole
        } else {
            region(chaos.below(9), chaos.below(9).max(1))
        };
        let hazard = if chaos.below(6) == 0 {
            Hazard::deep(chaos.below(7) as u32)
        } else {
            Hazard::at(chaos.below(7) as u32, chaos.below(5) as u32)
        };
        if choice == 3 {
            accesses.clear();
            entries.clear();
            continue;
        }
        let expected = naive(&entries, region);
        if choice == 0 {
            accesses.record(region, &hazard, true);
            entries.push((region, hazard));
        } else {
            let found = accesses.query(region, true);
            assert_eq!(found.wave, expected.wave, "wave of {region:?}");
            assert_eq!(found.segments, expected.segments, "segments of {region:?}");
            assert_eq!(found.deepest, expected.deepest, "deepest of {region:?}");
        }
    }
}

#[test]
fn a_covering_write_forgets_every_reader_it_overwrites() {
    let mut accesses = Accesses::default();
    accesses.record(region(0, 4), &Hazard::at(3, 1), true);
    accesses.record(region(8, 4), &Hazard::at(5, 2), true);
    let found = accesses.query(Region::Whole, true);
    assert_eq!(found.wave, Some(5));
    assert_eq!(found.segments, vec![2]);
    accesses.clear();
    assert_eq!(accesses.query(Region::Whole, true), Hazard::default());
    accesses.record(region(2, 2), &Hazard::at(1, 3), true);
    let found = accesses.query(region(1, 2), true);
    assert_eq!(found.wave, Some(1));
    assert_eq!(found.segments, vec![3]);
    assert_eq!(accesses.query(region(0, 1), true), Hazard::default());
}

#[test]
fn a_wave_keeps_only_the_segments_that_reach_it() {
    let mut accesses = Accesses::default();
    accesses.record(region(0, 8), &Hazard::at(1, 4), true);
    accesses.record(region(2, 2), &Hazard::at(1, 2), true);
    accesses.record(region(6, 2), &Hazard::at(7, 9), true);
    let found = accesses.query(region(0, 8), true);
    assert_eq!(found.wave, Some(7));
    assert_eq!(found.segments, vec![9]);
    let found = accesses.query(region(1, 3), true);
    assert_eq!(found.wave, Some(1));
    assert_eq!(found.segments, vec![2, 4]);
    assert_eq!(found.deepest, None);
}

#[test]
fn a_split_walks_the_numbers_of_its_tensor_once() {
    for total in [1u32, 5, 10, 4096, 21_000] {
        for per_task in [1u32, 7, 256, 2048, 65536] {
            let mut walked = 0;
            for (first, count, _) in span::chunks(total, per_task, Some(0)) {
                assert_eq!(
                    first, walked,
                    "a split of {total} numbers in pieces of {per_task} leaves {first} where the walk reached {walked}",
                );
                walked += count;
            }
            assert_eq!(
                walked, total,
                "a split of {total} numbers in pieces of {per_task} walks {walked} of them",
            );
        }
    }
}

#[test]
fn a_boundary_stays_inside_the_numbers_of_its_tensor() {
    for (total, group) in [
        (1u32 << 22, 1u32 << 11),
        (1u32 << 24, 1u32 << 13),
        (i32::MAX as u32, 1u32 << 16),
        (i32::MAX as u32, i32::MAX as u32),
    ] {
        let mut previous = 0;
        for piece in [0u32, 1, group / 3, group / 2, group - 1, group] {
            let boundary = span::boundary(total, piece, group);
            assert!(
                boundary <= total,
                "boundary {piece} of {group} walks {boundary} past the {total} numbers it holds",
            );
            assert!(
                boundary >= previous,
                "boundary {piece} of {group} walks back to {boundary} from {previous}",
            );
            previous = boundary;
        }
        assert_eq!(span::boundary(total, group, group), total);
    }
}

#[test]
fn a_row_walk_counts_the_rows_of_a_tensor_that_holds_no_numbers() {
    let graph = Graph::new();
    let width = graph.free(8);
    let shape = Shape::of([1, 1, 4, 8]).freed(&[(3, width)]);
    let extents = span::Extents::of(
        &[ValueInfo::derived(shape, 0)],
        &[],
        &[span::Measure::Rows(0), span::Measure::Elements(0)],
    );
    assert_eq!(
        extents.count(1, &[0]),
        0,
        "a tensor of no numbers holds none",
    );
    assert_eq!(
        extents.count(0, &[0]),
        4,
        "a row of no numbers still walks the rows the shape of its tensor holds",
    );
    assert_eq!(extents.count(0, &[8]), 4);
    assert_eq!(
        extents.span(
            span::Split::Uniform {
                measure: 0,
                index: 0,
                group: 1,
            },
            &[0],
        ),
        (0, 4),
        "a row walk names the rows a body must leave alone when they hold no numbers",
    );
}

fn lowered(dynamic: bool) -> lower::Plan {
    let graph = Graph::new();
    let shape = if dynamic {
        let tokens = graph.free(4096);
        Shape::of([1, 1, 4096, 2]).freed(&[(2, tokens)])
    } else {
        Shape::of([1, 1, 4096, 2])
    };
    let input = graph.input(shape, Element::Single);
    let doubled = graph.mul(input, input);
    graph.retain(doubled);
    let snapshot = graph.snapshot();
    lower::lower(
        snapshot.values(),
        snapshot.tasks(),
        *Profile::derive(Budget::BASELINE, None)
            .last()
            .expect("a profile"),
        &[],
        0,
    )
}

#[test]
fn a_walk_a_binding_rules_touches_whole_storages() {
    let dynamic = lowered(true);
    let mut walking = 0;
    for task in &dynamic.tasks {
        let mut touches = region::Touches::default();
        region::touches(
            &dynamic.values.as_slice(),
            &dynamic.tiles,
            task,
            (task.first, task.count),
            &mut touches,
        );
        for (storage, region) in touches.reads.iter().chain(&touches.writes) {
            assert_eq!(
                *region,
                Region::Whole,
                "a {} task walks storage {storage} through a length a binding rules, and its range moves with the binding",
                task.kind.name(),
            );
        }
        walking += 1;
    }
    assert!(
        walking > 1,
        "a walk of 8192 numbers a binding rules schedules more than one task",
    );
    let frozen = lowered(false);
    let mut narrowed = 0;
    for task in &frozen.tasks {
        if task.kind != Kind::Binary {
            continue;
        }
        let mut touches = region::Touches::default();
        region::touches(
            &frozen.values.as_slice(),
            &frozen.tiles,
            task,
            (task.first, task.count),
            &mut touches,
        );
        assert_eq!(
            touches.writes,
            vec![(
                frozen.values[task.out as usize].storage,
                Region::run(u64::from(task.first), u64::from(task.count)),
            )],
            "a {} task of a shape the plan freezes writes the range it names",
            task.kind.name(),
        );
        narrowed += 1;
    }
    assert!(
        narrowed > 1,
        "a walk of 8192 numbers a binding does not rule schedules more than one task",
    );
}

fn refuses(action: impl FnOnce()) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(action)).is_err()
}

#[test]
fn a_task_that_addresses_the_length_the_plan_froze_walks_only_exact_lengths() {
    let graph = Graph::new();
    let free = graph.free(4);
    let values = vec![
        ValueInfo::derived(Shape::of([1, 1, 1, 4]).freed(&[(3, free)]), 0),
        ValueInfo::derived(Shape::of([1, 1, 1, 4]), 1),
    ];
    let exact = lower::Task {
        kind: Kind::Concat,
        op: neura_pointwise::NONE,
        geometry: 0,
        first: 0,
        count: 4,
        slot: 0,
        out: 1,
        extra: NO_VALUE,
        inputs: [1, NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE],
        origin: NO_VALUE,
        literal: 0.0,
        knob: neura_abi::NO_VALUE,
        window: Window::sliding([1, 1]),
        splits: 1,
        work: 0,
        in_place: false,
        axis: 3,
        offset: 0,
        prelude: Vec::new(),
        chain: Vec::new(),
        unit: 0,
        split: span::Split::Range { first: 0, count: 4 },
        depends: Vec::new(),
        patch: NO_VALUE,
        segments: NO_VALUE,
        reach: 0,
        keys: 0,
        plane: 0,
        queries: NO_VALUE,
        tokens: 0,
        grid: NO_VALUE,
        keep: 0,
    };
    assert!(
        !refuses(|| plan::assert_a_task_needs_exact_lengths_the_plan_froze(
            &values,
            std::slice::from_ref(&exact),
        )),
        "a concatenation of tensors that hold the length it shifts by",
    );
    let mut walked = exact.clone();
    walked.inputs = [0, NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE];
    assert!(
        refuses(|| plan::assert_a_task_needs_exact_lengths_the_plan_froze(
            &values,
            std::slice::from_ref(&walked),
        )),
        "a concatenation shifts every tensor beside it by the length the plan froze",
    );
    let mut taps = exact;
    taps.kind = Kind::Conv2d;
    taps.inputs = [1, 0, NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE];
    assert!(
        refuses(|| plan::assert_a_task_needs_exact_lengths_the_plan_froze(
            &values,
            std::slice::from_ref(&taps),
        )),
        "a window walks the taps of its filter row by row over the reach the plan froze",
    );
}

fn rotation(out: u32, segments: u32) -> lower::Task {
    lower::Task {
        kind: Kind::Rope,
        op: neura_pointwise::NONE,
        geometry: 0,
        first: 0,
        count: 4,
        slot: 0,
        out,
        extra: NO_VALUE,
        inputs: [out, NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE],
        origin: NO_VALUE,
        literal: 0.0,
        knob: neura_abi::NO_VALUE,
        window: Window::sliding([1, 1]),
        splits: 1,
        work: 0,
        in_place: false,
        axis: 0,
        offset: 0,
        prelude: Vec::new(),
        chain: Vec::new(),
        unit: 0,
        split: span::Split::Range { first: 0, count: 4 },
        depends: Vec::new(),
        patch: NO_VALUE,
        segments,
        reach: 0,
        keys: 0,
        plane: 0,
        queries: NO_VALUE,
        tokens: 0,
        grid: NO_VALUE,
        keep: 0,
    }
}

#[test]
fn a_rotation_of_a_packed_tensor_walks_the_segments_of_its_axis() {
    let graph = Graph::new();
    let extent = graph.free(4);
    let ragged = [(extent.slot(), 7u32)];
    let values = vec![
        ValueInfo::derived(Shape::of([4, 1, 1, 2]).freed(&[(0, extent)]), 0),
        ValueInfo::derived(Shape::of([1, 1, 4, 2]).freed(&[(2, extent)]), 1),
    ];
    let loose = rotation(0, NO_VALUE);
    assert!(
        refuses(
            || plan::assert_a_packed_rope_turns_the_rows_of_one_plane_at_a_time(
                &values,
                std::slice::from_ref(&loose),
                &ragged,
            )
        ),
        "a rotation of the rows a ragged axis packs named no segments",
    );
    let mut sideways = loose;
    sideways.segments = 7;
    sideways.grid = 7;
    sideways.split = span::Split::Ragged {
        planes: 1,
        plane: 0,
        index: 0,
        group: 1,
    };
    assert!(
        refuses(
            || plan::assert_a_packed_rope_turns_the_rows_of_one_plane_at_a_time(
                &values,
                std::slice::from_ref(&sideways),
                &ragged,
            )
        ),
        "a packed rotation packed the rows of every plane into axis 0",
    );
    let mut packed = rotation(1, 7);
    packed.grid = 7;
    packed.split = span::Split::Ragged {
        planes: 1,
        plane: 0,
        index: 0,
        group: 1,
    };
    assert!(
        !refuses(
            || plan::assert_a_packed_rope_turns_the_rows_of_one_plane_at_a_time(
                &values,
                std::slice::from_ref(&packed),
                &ragged,
            )
        ),
        "a rotation of the rows a ragged axis packs into axis 2",
    );
}

#[test]
fn an_encoding_store_evicts_the_shape_it_touched_longest_ago() {
    let graph = Graph::new();
    let batch = graph.free(64);
    let mut value = graph.input(Shape::matrix(64, 64).freed(&[(2, batch)]), Element::Single);
    for _ in 0..3 {
        let weight = graph.parameter(
            Shape::matrix(64, 64),
            neura_graph::Init::Uniform {
                low: -0.2,
                high: 0.2,
            },
            Element::Single,
        );
        value = graph.relu(graph.matmul(value, weight));
    }
    graph.retain(value);
    let plan = plan::Plan::of(
        &graph,
        256,
        Profile::derive(Budget::BASELINE, None)[0],
        crate::remembered::DEFAULT_ENCODING_BYTES,
    );
    let mut shapes = [8u32, 16, 24]
        .into_iter()
        .map(|rows| (rows, std::sync::Arc::new(plan.deliver(&[rows]))))
        .collect::<Vec<_>>();
    shapes.sort_by_key(|(_, encoding)| encoding.bytes());
    let (smallest_shape, smallest) = shapes[0].clone();
    let (middle_shape, middle) = shapes[1].clone();
    let (largest_shape, largest) = shapes[2].clone();
    let room = largest.bytes() + middle.bytes() + 8;
    let mut store = Remembered::of(room);
    store.insert(&[largest_shape], largest.clone());
    store.insert(&[middle_shape], middle.clone());
    assert_eq!(
        store.len(),
        2,
        "the store holds the two shapes its budget has room for",
    );
    assert!(
        store.bytes() <= room,
        "the store holds {} bytes where its budget is {room}",
        store.bytes(),
    );
    store.find(&[largest_shape]);
    store.insert(&[smallest_shape], smallest.clone());
    assert!(
        store.find(&[middle_shape]).is_none(),
        "the store kept a shape it touched before the one it had touched last",
    );
    assert!(
        store.find(&[largest_shape]).is_some(),
        "the store evicted the shape it touched last",
    );
    assert!(store.find(&[smallest_shape]).is_some());
    assert!(
        store.bytes() <= room,
        "the store holds {} bytes where its budget is {room}",
        store.bytes(),
    );
    let held = store.insert(&[largest_shape], largest.clone());
    assert!(
        std::sync::Arc::ptr_eq(&held, &largest),
        "the store answers a shape it holds with the encoding it holds",
    );
}

fn windowed(kind: Kind, out: u32, inputs: [u32; 6], first: u32, count: u32) -> lower::Task {
    lower::Task {
        kind,
        op: neura_pointwise::NONE,
        geometry: 0,
        first,
        count,
        slot: 0,
        out,
        extra: NO_VALUE,
        inputs,
        origin: NO_VALUE,
        literal: 0.0,
        knob: neura_abi::NO_VALUE,
        window: Window::sliding([3, 3]),
        splits: 1,
        work: 0,
        in_place: false,
        axis: 0,
        offset: 0,
        prelude: Vec::new(),
        chain: Vec::new(),
        unit: 0,
        split: span::Split::Range { first, count },
        depends: Vec::new(),
        patch: NO_VALUE,
        segments: NO_VALUE,
        reach: 0,
        keys: 0,
        plane: 0,
        queries: NO_VALUE,
        tokens: 0,
        grid: NO_VALUE,
        keep: 0,
    }
}

fn holds(region: Region, at: u64) -> bool {
    match region {
        Region::Empty => false,
        Region::Whole => true,
        Region::Run { first, count } => first <= at && at < first + count,
        Region::Band {
            first,
            span,
            stride,
            count,
        } => (0..count).any(|index| {
            let start = first + index * stride;
            start <= at && at < start + span
        }),
    }
}

fn declared(reads: &[(u32, Region)], storage: u32) -> Vec<Region> {
    reads
        .iter()
        .filter(|(kept, _)| *kept == storage)
        .map(|(_, region)| *region)
        .collect()
}

fn touches_of(values: &[ValueInfo], task: &lower::Task) -> region::Touches {
    let mut touched = region::Touches::default();
    region::touches(&values, &[], task, (task.first, task.count), &mut touched);
    touched
}

fn plane_of(index: u32, dims: [u32; 4]) -> (u32, u32, u32, u32) {
    let plane = dims[2] * dims[3];
    (
        index / (plane * dims[1]),
        (index / plane) % dims[1],
        (index % plane) / dims[3],
        index % dims[3],
    )
}

fn convolution_values() -> ([ValueInfo; 3], u32, u32) {
    (
        [
            ValueInfo::derived(Shape::of([1, 2, 6, 6]), 0),
            ValueInfo::derived(Shape::of([4, 2, 3, 3]), 1),
            ValueInfo::derived(Shape::of([1, 4, 4, 4]), 2),
        ],
        18,
        4,
    )
}

#[test]
fn a_convolution_reads_the_filter_channels_its_walk_names() {
    let (values, taps, elements) = convolution_values();
    let values = values.to_vec();
    for first in 0..elements {
        for count in [1u32, 3, 16, elements - first] {
            if count == 0 || first + count > elements {
                continue;
            }
            let task = windowed(
                Kind::Conv2d,
                2,
                [0, 1, NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE],
                first,
                count,
            );
            let touched = touches_of(&values, &task);
            let filter_reads = declared(&touched.reads, 1);
            assert!(!filter_reads.is_empty(), "a convolution reads its filter");
            for index in first..first + count {
                let (_, channel, _, _) = plane_of(index, [1, 4, 4, 4]);
                for tap in 0..taps {
                    assert!(
                        filter_reads
                            .iter()
                            .any(|region| holds(*region, u64::from(channel * taps + tap))),
                        "a filter walk of {count} elements from {first} declared {filter_reads:?} and the device reads element {} of channel {channel}",
                        channel * taps + tap,
                    );
                }
            }
        }
    }
    let one_channel = windowed(
        Kind::Conv2d,
        2,
        [0, 1, NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE],
        32,
        16,
    );
    assert_eq!(
        touches_of(&values, &one_channel).reads,
        vec![
            (0, Region::Whole),
            (1, Region::run(2 * u64::from(taps), u64::from(taps)))
        ],
        "a convolution that walks one output channel weighs that channel of the filter and no other",
    );
}

#[test]
fn a_convolution_gradient_reads_the_filter_blocks_its_walk_names() {
    let filter = ValueInfo::derived(Shape::of([4, 2, 3, 3]), 0);
    let gradient = ValueInfo::derived(Shape::of([1, 4, 6, 6]), 1);
    let out = ValueInfo::derived(Shape::of([1, 2, 6, 6]), 2);
    let values = vec![filter, gradient, out];
    let per_channel = 18u64;
    let elements = 2 * 6 * 6;
    for first in 0..elements {
        for count in [1u32, 6, 36, elements - first] {
            if count == 0 || first + count > elements {
                continue;
            }
            let task = windowed(
                Kind::Conv2dInputGrad,
                2,
                [0, 1, NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE],
                first,
                count,
            );
            let touched = touches_of(&values, &task);
            let filter_reads = declared(&touched.reads, 0);
            assert!(
                !filter_reads.is_empty(),
                "a convolution gradient reads its filter"
            );
            for index in first..first + count {
                let (_, channel, _, _) = plane_of(index, [1, 2, 6, 6]);
                for block in 0..4u64 {
                    for tap in 0..9u64 {
                        let at = block * per_channel + u64::from(channel) * 9 + tap;
                        assert!(
                            filter_reads.iter().any(|region| holds(*region, at)),
                            "a gradient walk of {count} elements from {first} declared {filter_reads:?} and the device reads element {at} of output channel {block}",
                        );
                    }
                }
            }
        }
    }
    let one_channel = windowed(
        Kind::Conv2dInputGrad,
        2,
        [0, 1, NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE],
        6,
        6,
    );
    assert_eq!(
        touches_of(&values, &one_channel).reads,
        vec![
            (
                0,
                Region::Band {
                    first: 0,
                    span: 9,
                    stride: per_channel,
                    count: 4,
                },
            ),
            (1, Region::Whole),
        ],
        "a gradient that walks one input channel weighs that channel of every filter row",
    );
}

#[test]
fn a_convolution_gradient_over_groups_reads_the_blocks_its_channels_close() {
    let filter = ValueInfo::derived(Shape::of([8, 2, 3, 3]), 0);
    let gradient = ValueInfo::derived(Shape::of([1, 8, 6, 6]), 1);
    let out = ValueInfo::derived(Shape::of([1, 4, 6, 6]), 2);
    let values = vec![filter, gradient, out];
    let per_channel = 18u64;
    let elements = 4 * 6 * 6;
    for first in (0..elements).step_by(7) {
        let count = 12.min(elements - first);
        let task = windowed(
            Kind::Conv2dInputGrad,
            2,
            [0, 1, NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE],
            first,
            count,
        );
        let touched = touches_of(&values, &task);
        let filter_reads = declared(&touched.reads, 0);
        assert!(
            !filter_reads.is_empty(),
            "a grouped gradient reads its filter"
        );
        for index in first..first + count {
            let (_, channel, _, _) = plane_of(index, [1, 4, 6, 6]);
            let group = u64::from(channel / 2);
            let local = u64::from(channel % 2);
            for block in 0..4u64 {
                for tap in 0..9u64 {
                    let at = (group * 4 + block) * per_channel + local * 9 + tap;
                    assert!(
                        filter_reads.iter().any(|region| holds(*region, at)),
                        "a gradient walk of channel {channel} declared {filter_reads:?} and the device reads element {at} of block {block} of group {group}",
                    );
                }
            }
        }
    }
}

#[test]
fn a_weight_gradient_carries_no_element_of_its_filter() {
    let input = ValueInfo::derived(Shape::of([1, 2, 6, 6]), 0);
    let gradient = ValueInfo::derived(Shape::of([1, 4, 4, 4]), 1);
    let filter = ValueInfo::derived(Shape::of([4, 2, 3, 3]), 2);
    let partials = ValueInfo::derived(Shape::of([1, 1, 4, 72]), 3);
    let values = vec![input, gradient, filter, partials];
    let task = windowed(
        Kind::Conv2dWeightGrad,
        3,
        [0, 1, 2, NO_VALUE, NO_VALUE, NO_VALUE],
        0,
        18,
    );
    let touched = touches_of(&values, &task);
    assert_eq!(
        touched
            .reads
            .iter()
            .find(|(storage, _)| *storage == 2)
            .map(|(_, region)| *region),
        Some(Region::Empty),
        "a weight gradient weighs its filter through the dims it reads and no element of it",
    );
}

#[test]
fn a_walk_that_touches_no_element_carries_no_hazard() {
    let mut accesses = Accesses::default();
    accesses.record(Region::Empty, &Hazard::deep(3), true);
    assert_eq!(
        accesses.query(Region::Whole, true),
        Hazard::default(),
        "a walk that reads no element of a tensor stands in no wave a whole walk must wait for",
    );
    accesses.record(Region::run(0, 8), &Hazard::deep(1), true);
    assert_eq!(
        accesses.query(Region::Empty, true),
        Hazard::default(),
        "a walk that reads no element carries no hazard a later walk can reach",
    );
}

#[test]
fn a_weight_gradient_writes_the_rows_it_folds_and_reads_the_row_it_updates() {
    let input = ValueInfo::derived(Shape::of([1, 2, 6, 6]), 0);
    let gradient = ValueInfo::derived(Shape::of([1, 4, 4, 4]), 1);
    let filter = ValueInfo::derived(Shape::of([4, 2, 3, 3]), 2);
    let partials = ValueInfo::derived(Shape::of([1, 1, 4, 72]), 3);
    let values = vec![input, gradient, filter, partials];
    let mut chunk = windowed(
        Kind::Conv2dWeightGrad,
        3,
        [0, 1, 2, NO_VALUE, NO_VALUE, NO_VALUE],
        18,
        18,
    );
    chunk.slot = 2;
    assert_eq!(
        touches_of(&values, &chunk).writes,
        vec![(3, Region::run(2 * 72 + 18, 18))],
        "a weight gradient chunk writes the span of its own seat in the partials",
    );
    let mut fold = windowed(
        Kind::Conv2dWeightGrad,
        2,
        [3, NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE],
        18,
        18,
    );
    fold.in_place = true;
    fold.chain = vec![StepRecord::of(StepFields {
        op: neura_pointwise::ADD,
        operand: 2,
        swapped: 1,
    })];
    let touched = touches_of(&values, &fold);
    assert_eq!(
        touched.writes,
        vec![(2, Region::run(18, 18))],
        "a weight gradient fold updates the rows of the filter it folds",
    );
    assert_eq!(
        touched.reads,
        vec![(3, Region::Whole), (2, Region::run(18, 18))],
        "a weight gradient fold reads the partials it folds and the rows of the filter it updates",
    );
}

#[test]
fn a_convolution_gradient_chunk_declares_the_partials_it_writes_and_the_band_it_weighs() {
    let filter = ValueInfo::derived(Shape::of([4, 2, 3, 3]), 0);
    let gradient = ValueInfo::derived(Shape::of([1, 4, 4, 4]), 1);
    let out = ValueInfo::derived(Shape::of([1, 2, 6, 6]), 2);
    let partials = ValueInfo::derived(Shape::of([1, 1, 4, 72]), 3);
    let values = vec![filter, gradient, out, partials];
    let mut chunk = windowed(
        Kind::Conv2dInputGrad,
        2,
        [0, 1, NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE],
        6,
        6,
    );
    chunk.geometry = neura_abi::strategy::INPUT_CHUNK;
    chunk.extra = 3;
    chunk.slot = 1;
    chunk.splits = 4;
    let touched = touches_of(&values, &chunk);
    assert_eq!(
        touched.writes,
        vec![(2, Region::empty()), (3, Region::run(72 + 6, 6))],
        "a gradient chunk leaves the input gradient alone and writes the seat of its own chunk",
    );
    assert_eq!(
        declared(&touched.reads, 0),
        vec![Region::run(18, 9)],
        "a gradient chunk weighs the block of the filter its own channels name",
    );
    assert_eq!(
        declared(&touched.reads, 1),
        vec![Region::Whole],
        "a gradient chunk reads the incoming gradient whole",
    );
}

#[test]
fn a_convolution_gradient_fold_reads_every_chunk_of_the_span_it_folds() {
    let gradient = ValueInfo::derived(Shape::of([1, 4, 4, 4]), 0);
    let out = ValueInfo::derived(Shape::of([1, 2, 6, 6]), 1);
    let partials = ValueInfo::derived(Shape::of([1, 1, 4, 72]), 2);
    let values = vec![gradient, out, partials];
    let mut fold = windowed(
        Kind::Conv2dInputGrad,
        1,
        [2, NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE],
        6,
        12,
    );
    fold.geometry = neura_abi::strategy::INPUT_FOLD;
    fold.splits = 4;
    let touched = touches_of(&values, &fold);
    assert_eq!(
        touched.writes,
        vec![(1, Region::run(6, 12))],
        "a gradient fold lands the sum of its chunks in the input gradient",
    );
    let expected = Region::band(6, 12, 72, 4);
    assert_eq!(
        declared(&touched.reads, 2),
        vec![expected],
        "a gradient fold reads every chunk of the span it folds",
    );
}

#[test]
fn a_convolution_gradient_fold_rides_the_chain_that_updates_its_rows() {
    let out = ValueInfo::derived(Shape::of([1, 2, 6, 6]), 0);
    let partials = ValueInfo::derived(Shape::of([1, 1, 4, 72]), 1);
    let carried = ValueInfo::derived(Shape::of([1, 2, 6, 6]), 2);
    let values = vec![out, partials, carried];
    let mut fold = windowed(
        Kind::Conv2dInputGrad,
        0,
        [1, NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE],
        6,
        12,
    );
    fold.geometry = neura_abi::strategy::INPUT_FOLD;
    fold.splits = 4;
    fold.in_place = true;
    fold.chain = vec![StepRecord::of(StepFields {
        op: neura_pointwise::ADD,
        operand: 2,
        swapped: 0,
    })];
    let touched = touches_of(&values, &fold);
    assert_eq!(
        touched.writes,
        vec![(0, Region::run(6, 12))],
        "a gradient fold lands the sum of its chunks in the input gradient",
    );
    assert_eq!(
        declared(&touched.reads, 1),
        vec![Region::band(6, 12, 72, 4)],
        "a gradient fold reads every chunk of the span it folds",
    );
    assert_eq!(
        declared(&touched.reads, 2),
        vec![Region::run(6, 12)],
        "a gradient fold updates the rows of the input gradient it rides into",
    );
}
