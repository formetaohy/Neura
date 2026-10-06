use crate::source::{self, Split};
use std::path::Path;
use std::slice::Chunks;

pub(crate) struct Dataset {
    pixels: Vec<u8>,
    labels: Vec<u8>,
    rows: u32,
    columns: u32,
}

impl Dataset {
    pub(crate) fn load(directory: &Path, split: Split) -> Self {
        let (pixels, rows, columns) = source::images(directory, split);
        let labels = source::labels(directory, split);
        let dataset = Self {
            pixels,
            labels,
            rows,
            columns,
        };
        assert_eq!(
            dataset.pixels.len(),
            dataset.labels.len() * dataset.width(),
            "an MNIST split holds one {rows} by {columns} image per label",
        );
        dataset
    }

    pub(crate) fn len(&self) -> u32 {
        self.labels.len() as u32
    }

    pub(crate) fn rows(&self) -> u32 {
        self.rows
    }

    pub(crate) fn columns(&self) -> u32 {
        self.columns
    }

    pub(crate) fn width(&self) -> usize {
        (self.rows * self.columns) as usize
    }

    pub(crate) fn image(&self, at: u32) -> &[u8] {
        let start = at as usize * self.width();
        &self.pixels[start..start + self.width()]
    }

    pub(crate) fn label(&self, at: u32) -> u8 {
        self.labels[at as usize]
    }
}

pub(crate) struct Order {
    indices: Vec<u32>,
}

impl Order {
    pub(crate) fn ordered(len: u32) -> Self {
        Self {
            indices: (0..len).collect(),
        }
    }

    pub(crate) fn shuffled(len: u32, seed: u64) -> Self {
        let mut indices = (0..len).collect::<Vec<_>>();
        let mut state = seed | 1;
        for at in (1..indices.len()).rev() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            indices.swap(at, (state % (at as u64 + 1)) as usize);
        }
        Self { indices }
    }

    pub(crate) fn batches(&self, size: u32) -> Chunks<'_, u32> {
        self.indices.chunks(size as usize)
    }
}

pub(crate) fn materialize(
    dataset: &Dataset,
    batch: &[u32],
    images: &mut Vec<f32>,
    labels: &mut Vec<f32>,
) {
    images.clear();
    labels.clear();
    for at in batch {
        images.extend(dataset.image(*at).iter().map(|ink| f32::from(*ink) / 255.0));
        labels.push(f32::from(dataset.label(*at)));
    }
}
