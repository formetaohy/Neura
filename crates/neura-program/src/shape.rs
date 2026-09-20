use neura_abi::MAX_RANK;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct Shape {
    dims: [u32; 4],
}

impl Shape {
    pub const fn scalar() -> Self {
        Self { dims: [1; 4] }
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
        Self { dims: padded }
    }

    pub const fn dims(self) -> [u32; 4] {
        self.dims
    }

    pub fn elements(self) -> u32 {
        let elements = self.dims.iter().product::<u32>();
        assert!(
            elements < i32::MAX as u32,
            "a tensor of {elements} elements outruns the device index space",
        );
        elements
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
        Self { dims }
    }

    pub fn fits_within(self, other: Self) -> bool {
        (0..4).all(|axis| self.dims[axis] == other.dims[axis] || self.dims[axis] == 1)
    }

    pub fn as_matrix(self) -> Option<(u32, u32)> {
        if self.dims[0] == 1 && self.dims[1] == 1 {
            Some((self.dims[2], self.dims[3]))
        } else {
            None
        }
    }

    pub fn rows(self) -> u32 {
        self.elements() / self.dims[3]
    }

    pub fn columns(self) -> u32 {
        self.dims[3]
    }

    pub fn is_scalar(self) -> bool {
        self.elements() == 1
    }
}
