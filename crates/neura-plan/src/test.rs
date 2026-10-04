use crate::hazard::{Accesses, Hazard};
use crate::region::Region;

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
