use crate::window::Window;
use neura_abi::MAX_RANK;

const ELEMENT_LIMIT: u64 = i32::MAX as u64;
const FIXED: u8 = u8::MAX;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct Free {
    slot: u32,
    bound: u32,
}

impl Free {
    pub(crate) fn of(slot: u32, bound: u32) -> Self {
        assert!(
            bound > 0,
            "a free extent bounded at {bound} holds no length a tensor can take",
        );
        Self { slot, bound }
    }

    pub const fn slot(self) -> u32 {
        self.slot
    }

    pub const fn bound(self) -> u32 {
        self.bound
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum PlaneLayout {
    Grid { heads: u32, batch: u32 },
    Flat { planes: u32 },
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) struct Domain {
    statics: u32,
    frees: Vec<(u32, u32)>,
}

impl Domain {
    fn over(shape: Shape, axes: impl Iterator<Item = u32>) -> Self {
        let mut statics = 1u32;
        let mut frees = Vec::new();
        for axis in axes {
            let dim = shape.dims()[axis as usize];
            match shape.free(axis) {
                Some(slot) => frees.push((slot, dim)),
                None => statics *= dim,
            }
        }
        frees.sort_unstable();
        Self { statics, frees }
    }

    pub(crate) fn meets(&self, other: &Self) -> bool {
        self.statics == other.statics && self.frees == other.frees
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct Shape {
    dims: [u32; 4],
    frees: [u8; 4],
    elements: u32,
}

impl Shape {
    pub const fn scalar() -> Self {
        Self {
            dims: [1; 4],
            frees: [FIXED; 4],
            elements: 1,
        }
    }

    pub fn vector(elements: u32) -> Self {
        Self::of([elements])
    }

    pub fn matrix(rows: u32, columns: u32) -> Self {
        Self::of([rows, columns])
    }

    pub fn of(dims: impl AsRef<[u32]>) -> Self {
        let dims = dims.as_ref();
        assert!(
            !dims.is_empty() && dims.len() <= MAX_RANK as usize,
            "a shape declares {} dimensions where {MAX_RANK} is the limit",
            dims.len(),
        );
        let mut padded = [1u32; 4];
        for (slot, dim) in padded[4 - dims.len()..].iter_mut().zip(dims) {
            assert!(*dim > 0, "a tensor dimension must be positive");
            *slot = *dim;
        }
        Self::from_axes(padded, [None; 4])
    }

    pub fn freed(self, frees: &[(u32, Free)]) -> Self {
        let mut shape = self;
        for (axis, free) in frees {
            assert!(
                *axis < MAX_RANK,
                "a free extent names one of the {MAX_RANK} axes of {:?}",
                shape.dims,
            );
            assert_eq!(
                shape.dims[*axis as usize], free.bound,
                "axis {axis} of {:?} declares {} numbers where the free extent bound at {} holds them",
                shape.dims, shape.dims[*axis as usize], free.bound,
            );
            assert!(
                shape.free(*axis).is_none(),
                "axis {axis} of {:?} already carries a free extent",
                shape.dims,
            );
            shape.frees[*axis as usize] = u8::try_from(free.slot)
                .expect("a graph holds fewer free extents than one byte names");
        }
        shape
    }

    pub(crate) fn from_axes(dims: [u32; 4], frees: [Option<u32>; 4]) -> Self {
        let mut elements = 1u64;
        for dim in dims {
            assert!(dim > 0, "a tensor dimension must be positive");
            elements *= u64::from(dim);
        }
        assert!(
            elements <= ELEMENT_LIMIT,
            "a tensor of {elements} elements outruns the index space the device addresses",
        );
        let mut slots = [FIXED; 4];
        for axis in 0..4 {
            slots[axis] = match frees[axis] {
                Some(slot) => u8::try_from(slot)
                    .expect("a graph holds fewer free extents than one byte names"),
                None => FIXED,
            };
        }
        Self {
            dims,
            frees: slots,
            elements: elements as u32,
        }
    }

    pub const fn dims(self) -> [u32; 4] {
        self.dims
    }

    pub const fn elements(self) -> u32 {
        self.elements
    }

    pub(crate) fn domain(self) -> Domain {
        Domain::over(self, 0..MAX_RANK)
    }

    pub(crate) fn plane_domain(self) -> Domain {
        Domain::over(self, 0..2)
    }

    pub(crate) fn plane_layout(self) -> Option<PlaneLayout> {
        if self.dims[2] * self.dims[3] == 1 {
            return Some(PlaneLayout::Grid {
                heads: self.dims[0],
                batch: self.dims[1],
            });
        }
        (self.dims[0] * self.dims[1] == 1).then_some(PlaneLayout::Flat {
            planes: self.elements,
        })
    }

    pub fn free(self, axis: u32) -> Option<u32> {
        assert!(
            axis < MAX_RANK,
            "a shape holds {MAX_RANK} axes, and a free extent names axis {axis}",
        );
        match self.frees[axis as usize] {
            FIXED => None,
            slot => Some(u32::from(slot)),
        }
    }

    pub const fn dynamic(self) -> bool {
        self.frees[0] != FIXED
            || self.frees[1] != FIXED
            || self.frees[2] != FIXED
            || self.frees[3] != FIXED
    }

    pub fn slots(self) -> u32 {
        self.frees
            .iter()
            .filter(|slot| **slot != FIXED)
            .map(|slot| u32::from(*slot) + 1)
            .max()
            .unwrap_or(0)
    }

    pub fn meets(self, other: Self, left: u32, right: u32) -> bool {
        let (left_free, right_free) = (self.free(left), other.free(right));
        match (left_free, right_free) {
            (Some(slot), Some(other_slot)) => slot == other_slot,
            (None, None) => self.dims[left as usize] == other.dims[right as usize],
            _ => false,
        }
    }

    pub fn meeting(self, other: Self, left: u32, right: u32) -> (u32, Option<u32>) {
        assert!(
            self.meets(other, left, right),
            "axis {left} of {:?} meets axis {right} of {:?}, and the two walk extents that never agree",
            self.dims,
            other.dims,
        );
        let free = self.free(left).or_else(|| other.free(right));
        (self.dims[left as usize], free)
    }

    pub fn paired_axis(self, other: Self, left: u32, right: u32) -> (u32, Option<u32>) {
        let (left_free, right_free) = (self.free(left), other.free(right));
        let (left_dim, right_dim) = (self.dims[left as usize], other.dims[right as usize]);
        match (left_free, right_free) {
            (Some(slot), Some(other_slot)) => {
                assert_eq!(
                    slot, other_slot,
                    "axis {left} of {:?} and axis {right} of {:?} walk two free extents that never agree",
                    self.dims, other.dims,
                );
                (left_dim.max(right_dim), Some(slot))
            }
            (Some(slot), None) => {
                assert_eq!(
                    right_dim, 1,
                    "axis {left} of {:?} walks a free extent where axis {right} of {:?} holds {right_dim} numbers: a product walks the planes of both operands by one count, and the planes of a static operand stand beside a free extent only as one number",
                    self.dims, other.dims,
                );
                (left_dim, Some(slot))
            }
            (None, Some(slot)) => {
                assert_eq!(
                    left_dim, 1,
                    "axis {right} of {:?} walks a free extent where axis {left} of {:?} holds {left_dim} numbers: a product walks the planes of both operands by one count, and the planes of a static operand stand beside a free extent only as one number",
                    other.dims, self.dims,
                );
                (right_dim, Some(slot))
            }
            (None, None) => {
                assert!(
                    left_dim == right_dim || left_dim == 1 || right_dim == 1,
                    "axis {left} of {:?} meets axis {right} of {:?}",
                    self.dims,
                    other.dims,
                );
                (left_dim.max(right_dim), None)
            }
        }
    }

    pub(crate) fn joined(shapes: &[Self], axis: u32) -> Self {
        assert!(
            axis < MAX_RANK,
            "a concatenation names one of the {MAX_RANK} axes",
        );
        let first = *shapes
            .first()
            .expect("a concatenation joins at least one tensor");
        for shape in shapes {
            assert!(
                shape.free(axis).is_none(),
                "a concatenation along axis {axis} shifts every tensor beside it by the lengths it holds, and the shift a plan carries is one number; axis {axis} of {:?} walks a free extent whose length a binding rules",
                shape.dims,
            );
            for walked in 0..MAX_RANK {
                if walked == axis {
                    continue;
                }
                assert!(
                    first.meets(*shape, walked, walked),
                    "a concatenation along axis {axis} meets {:?} and {:?}",
                    first.dims,
                    shape.dims,
                );
            }
        }
        let mut dims = first.dims;
        dims[axis as usize] = shapes.iter().map(|shape| shape.dims[axis as usize]).sum();
        Self::from_axes(dims, first.frees())
    }

    pub(crate) fn tapped(self, window: Window) -> Self {
        assert!(
            self.free(2).is_none() && self.free(3).is_none(),
            "a window walks the taps of {:?} row by row, and a free extent of either tap axis hands the taps a length a binding rules",
            self.dims,
        );
        assert_eq!(
            [self.dims[2], self.dims[3]],
            [window.reach_rows(), window.reach_columns()],
            "a window of {} by {} taps walks a filter of {} by {} taps",
            window.reach_rows(),
            window.reach_columns(),
            self.dims[2],
            self.dims[3],
        );
        self
    }

    pub(crate) fn windowed(self, window: Window) -> [u32; 2] {
        assert!(
            self.free(2).is_none() && self.free(3).is_none(),
            "a window walks the rows and columns of {:?}, and a free extent of either hands every length the taps reach",
            self.dims,
        );
        let padded_rows = self.dims[2] + 2 * window.pad_rows();
        let padded_columns = self.dims[3] + 2 * window.pad_columns();
        assert!(
            padded_rows >= window.reach_rows() && padded_columns >= window.reach_columns(),
            "a window of {} by {} taps over {:?} padded by {} by {} reaches no position",
            window.reach_rows(),
            window.reach_columns(),
            self.dims,
            window.pad_rows(),
            window.pad_columns(),
        );
        [
            (padded_rows - window.reach_rows()) / window.stride_rows() + 1,
            (padded_columns - window.reach_columns()) / window.stride_columns() + 1,
        ]
    }

    pub fn fixed_axis(self, axis: u32, dim: u32) -> Self {
        assert!(
            axis < MAX_RANK,
            "a shape holds {MAX_RANK} axes, and {axis} is not one of them",
        );
        assert!(
            self.free(axis).is_none(),
            "axis {axis} of {:?} walks a free extent, and a fixed extent replaces every length it takes",
            self.dims,
        );
        assert!(dim > 0, "a tensor dimension must be positive");
        let mut dims = self.dims;
        dims[axis as usize] = dim;
        Self::from_axes(dims, self.frees())
    }

    pub(crate) fn frees(self) -> [Option<u32>; 4] {
        let mut frees = [None; 4];
        for (axis, free) in frees.iter_mut().enumerate() {
            *free = self.free(axis as u32);
        }
        frees
    }

    pub fn actual(self, extents: &[u32]) -> Self {
        let dims = self.actual_dims(extents);
        Self::from_axes(dims, [None; 4])
    }

    pub fn actual_dims(self, extents: &[u32]) -> [u32; 4] {
        let mut dims = self.dims;
        for (axis, dim) in dims.iter_mut().enumerate() {
            if let Some(slot) = self.free(axis as u32) {
                *dim = extents.get(slot as usize).copied().unwrap_or_else(|| {
                    panic!(
                        "axis {axis} walks free extent {slot}, and the binding names {} extents",
                        extents.len(),
                    )
                });
            }
        }
        dims
    }

    pub fn strides(self) -> [u32; 4] {
        Self::dense_strides(self.dims)
    }

    pub fn dense_strides(dims: [u32; 4]) -> [u32; 4] {
        let mut strides = [0u32; 4];
        let mut stride = 1u32;
        for axis in (0..4).rev() {
            strides[axis] = if dims[axis] == 1 { 0 } else { stride };
            stride *= dims[axis];
        }
        strides
    }

    pub fn combines_with(self, other: Self) -> bool {
        (0..4).all(|axis| {
            let axis = axis as u32;
            match (self.free(axis), other.free(axis)) {
                (Some(slot), Some(other_slot)) => slot == other_slot,
                (Some(_), None) => {
                    covers_a_walk(other.dims[axis as usize], self.dims[axis as usize])
                }
                (None, Some(_)) => {
                    covers_a_walk(self.dims[axis as usize], other.dims[axis as usize])
                }
                (None, None) => {
                    self.dims[axis as usize] == other.dims[axis as usize]
                        || self.dims[axis as usize] == 1
                        || other.dims[axis as usize] == 1
                }
            }
        })
    }

    pub fn combined(self, other: Self) -> Self {
        assert!(
            self.combines_with(other),
            "shapes {:?} and {:?} cannot meet element by element: two axes meet when they walk the same free extent, hold the same length, or one holds one number, and a tensor that stands beside a free extent holds one number or as many numbers as the bound of that extent",
            self.dims,
            other.dims,
        );
        let mut dims = [1u32; 4];
        let mut frees = [None; 4];
        for axis in 0..4 {
            let axis = axis as u32;
            frees[axis as usize] = self.free(axis).or_else(|| other.free(axis));
            dims[axis as usize] = self.dims[axis as usize].max(other.dims[axis as usize]);
        }
        Self::from_axes(dims, frees)
    }

    pub fn fits_within(self, other: Self) -> bool {
        (0..4).all(|axis| {
            let axis = axis as u32;
            match (self.free(axis), other.free(axis)) {
                (Some(slot), Some(other_slot)) => slot == other_slot,
                (Some(_), None) => false,
                (None, Some(_)) => {
                    covers_a_walk(self.dims[axis as usize], other.dims[axis as usize])
                }
                (None, None) => {
                    self.dims[axis as usize] == other.dims[axis as usize]
                        || self.dims[axis as usize] == 1
                }
            }
        })
    }

    pub fn reduced(self, axis: u32) -> Self {
        assert!(
            axis < MAX_RANK,
            "a fold names one of the {MAX_RANK} axes of {:?}",
            self.dims,
        );
        let mut dims = self.dims;
        let mut frees = self.frees();
        frees[axis as usize] = None;
        dims[axis as usize] = 1;
        Self::from_axes(dims, frees)
    }

    pub fn rows(self) -> u32 {
        self.dims[0] * self.dims[1] * self.dims[2]
    }

    pub fn columns(self) -> u32 {
        self.dims[3]
    }

    pub fn is_scalar(self) -> bool {
        self.elements == 1
    }
}

fn covers_a_walk(held: u32, bound: u32) -> bool {
    held == 1 || held == bound
}
