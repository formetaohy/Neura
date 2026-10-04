use crate::lower;
use neura_abi::Element;
use neura_graph::{Shape, ValueInfo};
use neura_profile::MatmulTile;

#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) enum Measure {
    Elements(u32),
    Rows(u32),
    Words(u32),
    Tiles { value: u32, geometry: u32 },
    Tokens(u32),
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) enum Split {
    Range {
        first: u32,
        count: u32,
    },
    Uniform {
        measure: u32,
        index: u32,
        group: u32,
    },
    Plane {
        measure: u32,
        planes: u32,
        plane: u32,
        index: u32,
        group: u32,
    },
    Segment {
        measure: u32,
        plane: u32,
        index: u32,
        group: u32,
    },
}

struct Extent {
    shape: Shape,
    strides_source: Option<[u8; 4]>,
    storage: u32,
    element: Element,
}

pub(crate) struct Extents {
    values: Vec<Extent>,
    measures: Vec<Measure>,
    tiles: Vec<MatmulTile>,
}

impl Extents {
    pub(crate) fn of(values: &[ValueInfo], tiles: &[MatmulTile], measures: &[Measure]) -> Self {
        Self {
            values: values
                .iter()
                .map(|info| Extent {
                    shape: info.shape,
                    strides_source: info.strides_source,
                    storage: info.storage,
                    element: info.element,
                })
                .collect(),
            measures: measures.to_vec(),
            tiles: tiles.to_vec(),
        }
    }

    pub(crate) fn dims(&self, value: u32, extents: &[u32]) -> [u32; 4] {
        self.values[value as usize].shape.actual_dims(extents)
    }

    pub(crate) fn strides(&self, value: u32, extents: &[u32]) -> [u32; 4] {
        match self.values[value as usize].strides_source {
            None => Shape::dense_strides(self.dims(value, extents)),
            Some(source) => {
                let owner = self.values[value as usize].storage;
                let owner_strides = Shape::dense_strides(self.dims(owner, extents));
                let mut strides = [0u32; 4];
                for axis in 0..4 {
                    strides[axis] = owner_strides[source[axis] as usize];
                }
                strides
            }
        }
    }

    pub(crate) fn count(&self, measure: u32, extents: &[u32]) -> u32 {
        match self.measures[measure as usize] {
            Measure::Elements(value) => self.elements(value, extents),
            Measure::Rows(value) => self.rows(value, extents),
            Measure::Words(value) => self.words(value, extents),
            Measure::Tokens(value) => self.dims(value, extents)[2],
            Measure::Tiles { value, geometry } => self.tiles(value, geometry, extents),
        }
    }

    pub(crate) fn span(&self, split: Split, extents: &[u32]) -> (u32, u32) {
        match split {
            Split::Range { first, count } => (first, count),
            Split::Segment { index, group, .. } => {
                assert!(
                    index < group,
                    "a segment walks tile {index} where a segment of its shape holds {group}",
                );
                (index, 1)
            }
            Split::Uniform {
                measure,
                index,
                group,
            } => {
                let total = self.count(measure, extents);
                uniform(total, index, group)
            }
            Split::Plane {
                measure,
                planes,
                plane,
                index,
                group,
            } => {
                let Measure::Tokens(value) = self.measures[measure as usize] else {
                    panic!("a plane walks the tokens of one value");
                };
                let actual = self.dims(value, extents);
                if plane >= actual[0] * actual[1] || plane >= planes {
                    return (0, 0);
                }
                let total = actual[2];
                let (within, count) = uniform(total, index, group);
                (plane * total + within, count)
            }
        }
    }

    fn elements(&self, value: u32, extents: &[u32]) -> u32 {
        self.dims(value, extents).iter().product()
    }

    fn rows(&self, value: u32, extents: &[u32]) -> u32 {
        let dims = self.dims(value, extents);
        dims.iter().product::<u32>() / dims[3]
    }

    fn words(&self, value: u32, extents: &[u32]) -> u32 {
        u32::try_from(
            self.values[value as usize]
                .element
                .payload_words(u64::from(self.elements(value, extents))),
        )
        .expect("a tensor of words fits the device word space")
    }

    fn tiles(&self, value: u32, geometry: u32, extents: &[u32]) -> u32 {
        let dims = self.dims(value, extents);
        let tile = self.tiles[geometry as usize];
        dims[0] * dims[1] * dims[2].div_ceil(tile.rows()) * dims[3].div_ceil(tile.columns())
    }
}

pub(crate) fn boundary(total: u32, piece: u32, group: u32) -> u32 {
    assert!(
        group > 0 && piece <= group,
        "a split of {total} numbers in {group} pieces names no boundary {piece}",
    );
    let shared = total / group;
    let rest = total % group;
    piece * shared + piece.min(rest)
}

fn uniform(total: u32, index: u32, group: u32) -> (u32, u32) {
    let first = boundary(total, index, group);
    (first, boundary(total, index + 1, group) - first)
}

pub(crate) fn chunks(total: u32, per_task: u32, measure: Option<u32>) -> Vec<(u32, u32, Split)> {
    match measure {
        None => lower::spans(total, per_task)
            .map(|(first, count)| (first, count, Split::Range { first, count }))
            .collect(),
        Some(measure) => {
            let group = total.div_ceil(per_task).max(1);
            (0..group)
                .map(|index| {
                    let (first, count) = uniform(total, index, group);
                    (
                        first,
                        count,
                        Split::Uniform {
                            measure,
                            index,
                            group,
                        },
                    )
                })
                .collect()
        }
    }
}

pub(crate) fn plane_chunks(
    tokens: u32,
    per_task: u32,
    planes: u32,
    measure: u32,
) -> Vec<(u32, u32, Split)> {
    let group = tokens.div_ceil(per_task).max(1);
    let mut spans = Vec::with_capacity((planes * group) as usize);
    for plane in 0..planes {
        for index in 0..group {
            let (first, count) = uniform(tokens, index, group);
            spans.push((
                plane * tokens + first,
                count,
                Split::Plane {
                    measure,
                    planes,
                    plane,
                    index,
                    group,
                },
            ));
        }
    }
    spans
}
