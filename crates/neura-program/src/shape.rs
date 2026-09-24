use neura_abi::MAX_RANK;

const ELEMENT_LIMIT: u64 = i32::MAX as u64;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct Shape {
    dims: [u32; 4],
    elements: u32,
}

impl Shape {
    pub const fn scalar() -> Self {
        Self {
            dims: [1; 4],
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
        let mut elements = 1u64;
        for (slot, dim) in padded[4 - dims.len()..].iter_mut().zip(dims) {
            assert!(*dim > 0, "a tensor dimension must be positive");
            *slot = *dim;
            elements *= u64::from(*dim);
        }
        assert!(
            elements <= ELEMENT_LIMIT,
            "a tensor of {elements} elements outruns the index space the device addresses",
        );
        Self {
            dims: padded,
            elements: elements as u32,
        }
    }

    pub const fn dims(self) -> [u32; 4] {
        self.dims
    }

    pub const fn elements(self) -> u32 {
        self.elements
    }

    pub fn strides(self) -> [u32; 4] {
        let dims = self.dims;
        let mut strides = [0u32; 4];
        let mut stride = 1u32;
        for axis in (0..4).rev() {
            strides[axis] = if dims[axis] == 1 { 0 } else { stride };
            stride *= dims[axis];
        }
        strides
    }

    pub fn combines_with(self, other: Self) -> bool {
        self.dims
            .iter()
            .zip(other.dims)
            .all(|(left, right)| left == &right || *left == 1 || right == 1)
    }

    pub fn combined(self, other: Self) -> Self {
        assert!(
            self.combines_with(other),
            "shapes {:?} and {:?} cannot meet element by element",
            self.dims,
            other.dims,
        );
        let mut dims = [1u32; 4];
        for (axis, dim) in dims.iter_mut().enumerate() {
            *dim = self.dims[axis].max(other.dims[axis]);
        }
        Self::of(dims)
    }

    pub fn fits_within(self, other: Self) -> bool {
        (0..4).all(|axis| self.dims[axis] == other.dims[axis] || self.dims[axis] == 1)
    }

    pub fn batch(self) -> [u32; 2] {
        [self.dims[0], self.dims[1]]
    }

    pub fn reduced(self, axis: u32) -> Self {
        assert!(
            axis < MAX_RANK,
            "a fold names one of the {MAX_RANK} axes of {:?}",
            self.dims,
        );
        let mut dims = self.dims;
        dims[axis as usize] = 1;
        Self::of(dims)
    }

    pub fn rows(self) -> u32 {
        self.elements / self.dims[3]
    }

    pub fn columns(self) -> u32 {
        self.dims[3]
    }

    pub fn is_scalar(self) -> bool {
        self.elements == 1
    }
}
