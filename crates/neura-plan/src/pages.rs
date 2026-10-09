use crate::layout::Layout;
use crate::region::{Region, Touches, Values};
use neura_abi::{PAGE_SHIFT, pages_of};

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

pub(crate) fn weight_pages<V: Values>(
    touched: &Touches,
    values: &V,
    layout: &Layout,
) -> WeightPages {
    let mut pages = Vec::new();
    let mut writes = Vec::new();
    for (storage, region) in &touched.reads {
        collect(values, *storage, *region, layout, &mut pages);
    }
    for (storage, region) in &touched.writes {
        let mark = writes.len();
        collect(values, *storage, *region, layout, &mut writes);
        pages.extend_from_slice(&writes[mark..]);
    }
    pages.sort_unstable();
    pages.dedup();
    writes.sort_unstable();
    writes.dedup();
    WeightPages { pages, writes }
}

fn collect<V: Values>(
    values: &V,
    storage: u32,
    region: Region,
    layout: &Layout,
    pages: &mut Vec<u32>,
) {
    let placed = match layout.weights().holds(storage) {
        true => layout.weights(),
        false if layout.state().holds(storage) => layout.state(),
        false => return,
    };
    let element = values.element(storage);
    let base = placed.address(storage);
    let elements = values
        .bounds(storage)
        .iter()
        .map(|dim| u64::from(*dim))
        .product::<u64>();
    let words = element.storage_words(elements);
    match region {
        Region::Empty => return,
        Region::Whole => span(base, 0, words, pages),
        Region::Run { first, count } => {
            let (first, end) = element.word_span(first, count);
            span(base, first, end, pages);
        }
        Region::Band {
            first,
            span: width,
            stride,
            count,
        } => {
            for index in 0..count {
                let (first, end) = element.word_span(first + index * stride, width);
                span(base, first, end, pages);
            }
        }
    }
    if element.quantized() {
        span(base, element.payload_words(elements), words, pages);
    }
}

fn span(base: u64, first: u64, end: u64, pages: &mut Vec<u32>) {
    if end <= first {
        return;
    }
    let from = (base + first) >> PAGE_SHIFT;
    let to = pages_of(base + end);
    for page in from..to {
        pages.push(u32::try_from(page).expect("a weight page fits the page table"));
    }
}
