use neura_plan::{Encoding, Plan, WeightPages};

pub(crate) struct WeightSchedule {
    pins: Vec<u32>,
    groups: Vec<WeightGroup>,
}

impl WeightSchedule {
    pub(crate) fn empty() -> Self {
        Self {
            pins: Vec::new(),
            groups: Vec::new(),
        }
    }

    pub(crate) fn pins(&self) -> &[u32] {
        &self.pins
    }

    pub(crate) fn groups(&self) -> &[WeightGroup] {
        &self.groups
    }

    pub(crate) fn group_of(&self, segment: u32) -> &WeightGroup {
        let at = self
            .groups
            .partition_point(|group| group.end_segment() <= segment);
        let group = self
            .groups
            .get(at)
            .unwrap_or_else(|| panic!("segment {segment} of a plan walks no weight window"));
        assert!(
            segment >= group.first_segment() && segment < group.end_segment(),
            "segment {segment} lies outside the weight window a schedule names for it",
        );
        group
    }
}

pub(crate) struct WeightGroup {
    first_segment: u32,
    end_segment: u32,
    pages: Vec<u32>,
    writes: Vec<u32>,
    roots: Vec<u32>,
    counts: Vec<u32>,
}

impl WeightGroup {
    fn of(
        encoding: &Encoding,
        first: usize,
        end: usize,
        pages: Vec<u32>,
        writes: Vec<u32>,
    ) -> Self {
        let mut roots = Vec::new();
        let mut counts = Vec::with_capacity(end - first);
        for segment in first..end {
            let count = encoding.predecessors(segment, first);
            if count == 0 {
                roots.push(segment as u32);
            }
            counts.push(count);
        }
        Self {
            first_segment: first as u32,
            end_segment: end as u32,
            pages,
            writes,
            roots,
            counts,
        }
    }

    pub(crate) fn first_segment(&self) -> u32 {
        self.first_segment
    }

    pub(crate) fn end_segment(&self) -> u32 {
        self.end_segment
    }

    pub(crate) fn pages(&self) -> &[u32] {
        &self.pages
    }

    pub(crate) fn writes(&self) -> &[u32] {
        &self.writes
    }

    pub(crate) fn roots(&self) -> &[u32] {
        &self.roots
    }

    pub(crate) fn counts(&self) -> &[u32] {
        &self.counts
    }
}

fn raise_or_declare(plan: &Plan) -> &'static str {
    match plan.gather_tables().is_empty() {
        true => "raise the resident weight budget",
        false => {
            "raise the resident weight budget, or name the rows a table walk reads with Runtime::declare_rows"
        }
    }
}

pub(crate) fn weight_schedule(
    plan: &Plan,
    encoding: &Encoding,
    slots: u32,
    pages: u32,
    tasks: &[WeightPages],
) -> WeightSchedule {
    let segments = encoding.segments();
    if segments.is_empty() {
        return WeightSchedule::empty();
    }
    let mut walked = Vec::with_capacity(segments.len());
    let mut written = Vec::with_capacity(segments.len());
    let mut counts = vec![0u32; pages as usize];
    for record in segments {
        let first = record.first as usize;
        let end = first + record.count as usize;
        let mut segment_pages = Vec::new();
        let mut segment_writes = Vec::new();
        for task in &tasks[first..end] {
            segment_pages.extend_from_slice(task.pages());
            segment_writes.extend_from_slice(task.writes());
        }
        segment_pages.sort_unstable();
        segment_pages.dedup();
        segment_writes.sort_unstable();
        segment_writes.dedup();
        for page in &segment_pages {
            counts[*page as usize] += 1;
        }
        walked.push(segment_pages);
        written.push(segment_writes);
    }
    let touched = counts.iter().filter(|count| **count > 0).count() as u32;
    if touched <= slots {
        let pins = (0..pages)
            .filter(|page| counts[*page as usize] > 0)
            .collect::<Vec<_>>();
        let mut group_pages = pins.clone();
        let mut writes = Vec::new();
        for held in &written {
            writes.extend_from_slice(held);
        }
        writes.sort_unstable();
        writes.dedup();
        return WeightSchedule {
            pins,
            groups: vec![WeightGroup::of(
                encoding,
                0,
                segments.len(),
                std::mem::take(&mut group_pages),
                writes,
            )],
        };
    }
    let reserve = (slots / 8).clamp(1, 512).min(slots - 1);
    let Some(pins) = pinned_pages(slots, reserve, pages, &walked, &counts) else {
        return WeightSchedule {
            pins: Vec::new(),
            groups: streamed_groups(plan, encoding, slots, tasks),
        };
    };
    let pinned = pins
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    let mut groups = Vec::new();
    let mut group_pages = std::collections::BTreeSet::new();
    let mut group_unpinned = 0usize;
    let mut group_writes = std::collections::BTreeSet::new();
    let mut start = 0usize;
    for (segment, segment_pages) in walked.iter().enumerate() {
        let mut added = 0usize;
        for page in segment_pages {
            if !pinned.contains(page) && !group_pages.contains(page) {
                added += 1;
            }
        }
        if group_unpinned + added > reserve as usize {
            groups.push(WeightGroup::of(
                encoding,
                start,
                segment,
                group_pages.iter().copied().collect(),
                group_writes.iter().copied().collect(),
            ));
            group_pages.clear();
            group_writes.clear();
            group_unpinned = 0;
            start = segment;
        }
        for page in segment_pages {
            if group_pages.insert(*page) && !pinned.contains(page) {
                group_unpinned += 1;
            }
        }
        group_writes.extend(written[segment].iter().copied());
    }
    groups.push(WeightGroup::of(
        encoding,
        start,
        segments.len(),
        group_pages.into_iter().collect(),
        group_writes.into_iter().collect(),
    ));
    WeightSchedule { pins, groups }
}

fn pinned_pages(
    slots: u32,
    reserve: u32,
    pages: u32,
    walked: &[Vec<u32>],
    counts: &[u32],
) -> Option<Vec<u32>> {
    let budget = (slots - reserve) as usize;
    let mut marked = vec![false; pages as usize];
    let mut pinned = Vec::new();
    for segment in walked {
        if segment.len() > reserve as usize {
            for page in segment {
                if !marked[*page as usize] {
                    marked[*page as usize] = true;
                    pinned.push(*page);
                }
            }
        }
    }
    if pinned.len() > budget {
        return None;
    }
    let mut ranked = (0..pages)
        .filter(|page| counts[*page as usize] > 0)
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        counts[*right as usize]
            .cmp(&counts[*left as usize])
            .then(left.cmp(right))
    });
    for page in ranked {
        if pinned.len() == budget {
            break;
        }
        if !marked[page as usize] {
            marked[page as usize] = true;
            pinned.push(page);
        }
    }
    if pinned.len() < budget {
        return None;
    }
    pinned.sort_unstable();
    Some(pinned)
}

fn streamed_groups(
    plan: &Plan,
    encoding: &Encoding,
    slots: u32,
    tasks: &[WeightPages],
) -> Vec<WeightGroup> {
    let mut groups = Vec::new();
    let mut pages = Vec::<u32>::new();
    let mut writes = Vec::<u32>::new();
    let mut start = 0usize;
    for (segment, record) in encoding.segments().iter().enumerate() {
        let first = record.first as usize;
        let end = first + record.count as usize;
        let mut walking = Vec::<u32>::new();
        let mut writing = Vec::<u32>::new();
        for task in &tasks[first..end] {
            walking.extend_from_slice(task.pages());
            writing.extend_from_slice(task.writes());
        }
        walking.sort_unstable();
        walking.dedup();
        writing.sort_unstable();
        writing.dedup();
        let added = walking
            .iter()
            .filter(|page| pages.binary_search(page).is_err())
            .count();
        if pages.len() + added > slots as usize {
            if pages.is_empty() {
                panic!(
                    "segment {segment} of this plan walks {} weight pages, and the resident weights of {slots} pages cannot hold them; {}, or bind a graph whose segments walk no more pages than it holds",
                    walking.len(),
                    raise_or_declare(plan),
                );
            }
            groups.push(WeightGroup::of(
                encoding,
                start,
                segment,
                std::mem::take(&mut pages),
                std::mem::take(&mut writes),
            ));
            start = segment;
        }
        for page in walking {
            match pages.binary_search(&page) {
                Ok(_) => {}
                Err(at) => pages.insert(at, page),
            }
        }
        assert!(
            pages.len() <= slots as usize,
            "segment {segment} of this plan walks {} weight pages, and the resident weights of {slots} pages cannot hold them; {}, or bind a graph whose tasks walk no more pages than it holds",
            pages.len(),
            raise_or_declare(plan),
        );
        for write in writing {
            match writes.binary_search(&write) {
                Ok(_) => {}
                Err(at) => writes.insert(at, write),
            }
        }
    }
    groups.push(WeightGroup::of(
        encoding,
        start,
        encoding.segments().len(),
        pages,
        writes,
    ));
    groups
}
