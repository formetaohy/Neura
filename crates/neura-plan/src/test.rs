use crate::encode;
use crate::hazard::{Accesses, Hazard};
use crate::lower;
use crate::region::{self, Region};
use crate::span;
use neura_abi::{Element, Kind, NO_VALUE};
use neura_graph::{Graph, Shape, ValueInfo, Window};
use neura_profile::{Budget, Profile};

fn overlaps(left: Region, right: Region) -> bool {
    match (left, right) {
        (Region::Whole, _) | (_, Region::Whole) => true,
        (
            Region::Run {
                first: left,
                count: l,
            },
            Region::Run {
                first: right,
                count: r,
            },
        ) => left < right + r && right < left + l,
    }
}

fn naive(entries: &[(Region, Hazard)], region: Region) -> Hazard {
    let mut hazard = Hazard::default();
    for (kept, carried) in entries {
        if overlaps(*kept, region) {
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
}

#[test]
fn an_interval_map_answers_what_a_scan_of_every_access_answers() {
    let mut chaos = Chaos(0x5eed);
    let mut accesses = Accesses::default();
    let mut entries = Vec::new();
    for _ in 0..4000 {
        let choice = chaos.below(4);
        let width = chaos.below(9);
        let first = chaos.below(9);
        let region = if chaos.below(8) == 0 {
            Region::Whole
        } else {
            region(first, width.max(1))
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
            accesses.record(region, &hazard);
            entries.push((region, hazard.clone()));
        } else {
            let found = accesses.query(region);
            assert_eq!(found.wave, expected.wave, "wave of {region:?}");
            assert_eq!(found.segments, expected.segments, "segments of {region:?}");
            assert_eq!(found.deepest, expected.deepest, "deepest of {region:?}");
        }
    }
}

#[test]
fn a_covering_write_forgets_every_reader_it_overwrites() {
    let mut accesses = Accesses::default();
    accesses.record(region(0, 4), &Hazard::at(3, 1));
    accesses.record(region(8, 4), &Hazard::at(5, 2));
    let found = accesses.query(Region::Whole);
    assert_eq!(found.wave, Some(5));
    assert_eq!(found.segments, vec![2]);
    accesses.clear();
    assert_eq!(accesses.query(Region::Whole), Hazard::default());
    accesses.record(region(2, 2), &Hazard::at(1, 3));
    let found = accesses.query(region(1, 2));
    assert_eq!(found.wave, Some(1));
    assert_eq!(found.segments, vec![3]);
    assert_eq!(accesses.query(region(0, 1)), Hazard::default());
}

#[test]
fn a_wave_keeps_only_the_segments_that_reach_it() {
    let mut accesses = Accesses::default();
    accesses.record(region(0, 8), &Hazard::at(1, 4));
    accesses.record(region(2, 2), &Hazard::at(1, 2));
    accesses.record(region(6, 2), &Hazard::at(7, 9));
    let found = accesses.query(region(0, 8));
    assert_eq!(found.wave, Some(7));
    assert_eq!(found.segments, vec![9]);
    let found = accesses.query(region(1, 3));
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
    )
}

#[test]
fn a_walk_a_binding_rules_touches_whole_storages() {
    let dynamic = lowered(true);
    let mut walking = 0;
    for task in &dynamic.tasks {
        let touches = region::touches(&dynamic.values, &dynamic.tiles, task);
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
        let touches = region::touches(&frozen.values, &frozen.tiles, task);
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
        param: 0.0,
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
    };
    assert!(
        !refuses(|| encode::assert_a_task_needs_exact_lengths_the_plan_froze(
            &values,
            std::slice::from_ref(&exact),
        )),
        "a concatenation of tensors that hold the length it shifts by",
    );
    let mut walked = exact.clone();
    walked.inputs = [0, NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE];
    assert!(
        refuses(|| encode::assert_a_task_needs_exact_lengths_the_plan_froze(
            &values,
            std::slice::from_ref(&walked),
        )),
        "a concatenation shifts every tensor beside it by the length the plan froze",
    );
    let mut taps = exact;
    taps.kind = Kind::Conv2d;
    taps.inputs = [1, 0, NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE];
    assert!(
        refuses(|| encode::assert_a_task_needs_exact_lengths_the_plan_froze(
            &values,
            std::slice::from_ref(&taps),
        )),
        "a window walks the taps of its filter row by row over the reach the plan froze",
    );
}
