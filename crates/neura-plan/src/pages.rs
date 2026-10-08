use crate::layout::Layout;
use crate::lower::Task;
use crate::region::{Region, touches};
use neura_abi::{PAGE_SHIFT, pages_of};
use neura_graph::{Residency, ValueInfo};
use neura_profile::MatmulTile;
use std::collections::BTreeSet;

#[derive(Default)]
pub struct WeightPages {
    pages: Vec<u32>,
    writes: Vec<u32>,
}

impl WeightPages {
    pub fn pages(&self) -> &[u32] {
        &self.pages
    }

    pub fn writes(&self) -> &[u32] {
        &self.writes
    }
}

pub(crate) fn weight_pages(
    values: &[ValueInfo],
    tiles: &[MatmulTile],
    tasks: &[Task],
    order: &[u32],
    layout: &Layout,
) -> Vec<WeightPages> {
    order
        .iter()
        .map(|index| {
            let touched = touches(values, tiles, &tasks[*index as usize]);
            let mut pages = BTreeSet::new();
            let mut writes = BTreeSet::new();
            for (storage, region) in &touched.reads {
                pages.extend(pages_of_storage(values, *storage, *region, layout));
            }
            for (storage, region) in &touched.writes {
                let written = pages_of_storage(values, *storage, *region, layout);
                pages.extend(written.iter().copied());
                writes.extend(written);
            }
            WeightPages {
                pages: pages.into_iter().collect(),
                writes: writes.into_iter().collect(),
            }
        })
        .collect()
}

fn pages_of_storage(
    values: &[ValueInfo],
    storage: u32,
    region: Region,
    layout: &Layout,
) -> Vec<u32> {
    let info = &values[storage as usize];
    let address = match info.residency {
        Residency::Parameter => layout.weights().address(storage),
        Residency::State => layout.state().address(storage),
        _ => return Vec::new(),
    };
    let elements = u64::from(info.shape.elements());
    let words = info.element.storage_words(elements);
    let mut runs = vec![match region {
        Region::Whole => (0, words),
        Region::Run { first, count } => info.element.word_span(first, count),
    }];
    if info.element.quantized() {
        runs.push((info.element.payload_words(elements), words));
    }
    let mut pages = Vec::new();
    for (first, end) in runs {
        if end <= first {
            continue;
        }
        let from = (address + first) >> PAGE_SHIFT;
        let to = pages_of(address + end);
        for page in from..to {
            pages.push(u32::try_from(page).expect("a weight page fits the page table"));
        }
    }
    pages
}
