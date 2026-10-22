use crate::hazard::quanta;
use crate::region::{GeometryTiles, Region, Touches, Values, touches};
use crate::schedule::Scheduled;
use neura_abi::PAGE_WORDS;

pub(crate) struct AccessDomain {
    starts: Vec<u64>,
}

impl AccessDomain {
    pub(crate) fn of(mut starts: Vec<u64>) -> Self {
        starts.sort_unstable();
        starts.dedup();
        Self { starts }
    }

    pub(crate) fn cells(&self) -> usize {
        self.starts.len().saturating_sub(1)
    }

    pub(crate) fn range(&self, first: u64, end: Option<u64>) -> std::ops::Range<usize> {
        let last = self.starts.len().saturating_sub(1);
        let start = self.starts.partition_point(|held| *held < first).min(last);
        let end = match end {
            Some(end) => self.starts.partition_point(|held| *held < end).min(last),
            None => last,
        };
        start..end.max(start)
    }
}

pub(crate) struct AccessDomains {
    domains: Vec<AccessDomain>,
}

impl AccessDomains {
    pub(crate) fn of<V: Values, W: Scheduled>(
        values: &V,
        tiles: GeometryTiles<'_>,
        tasks: &[W],
    ) -> Self {
        let mut starts = vec![Vec::new(); values.len()];
        let mut touched = Touches::default();
        for task in tasks {
            touches(values, tiles, task, task.span(), &mut touched);
            for (storage, region) in touched.reads.iter().chain(&touched.writes) {
                collect(
                    &mut starts[*storage as usize],
                    *region,
                    values.elements(*storage),
                );
            }
        }
        Self {
            domains: starts.into_iter().map(AccessDomain::of).collect(),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.domains.len()
    }

    pub(crate) fn domain(&self, storage: u32) -> &AccessDomain {
        &self.domains[storage as usize]
    }
}

fn collect(starts: &mut Vec<u64>, region: Region, elements: u64) {
    match region {
        Region::Empty => {}
        Region::Whole => {
            if elements > 0 {
                starts.push(0);
                starts.push(elements);
            }
        }
        region => {
            for (first, end) in quanta(region, elements > PAGE_WORDS) {
                starts.push(first);
                starts.push(end.unwrap_or(u64::MAX));
            }
        }
    }
}
