#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct Window {
    reach_rows: u32,
    reach_columns: u32,
    stride_rows: u32,
    stride_columns: u32,
    pad_rows: u32,
    pad_columns: u32,
}

impl Window {
    pub fn new(reach: [u32; 2], stride: [u32; 2], padding: [u32; 2]) -> Self {
        assert!(
            reach[0] > 0 && reach[1] > 0,
            "a window of {reach:?} taps covers no element",
        );
        assert!(
            stride[0] > 0 && stride[1] > 0,
            "a window of stride {stride:?} never reaches its next position",
        );
        Self {
            reach_rows: reach[0],
            reach_columns: reach[1],
            stride_rows: stride[0],
            stride_columns: stride[1],
            pad_rows: padding[0],
            pad_columns: padding[1],
        }
    }

    pub fn sliding(reach: [u32; 2]) -> Self {
        Self::new(reach, [1, 1], [0, 0])
    }

    pub const fn reach_rows(self) -> u32 {
        self.reach_rows
    }

    pub const fn reach_columns(self) -> u32 {
        self.reach_columns
    }

    pub const fn stride_rows(self) -> u32 {
        self.stride_rows
    }

    pub const fn stride_columns(self) -> u32 {
        self.stride_columns
    }

    pub const fn pad_rows(self) -> u32 {
        self.pad_rows
    }

    pub const fn pad_columns(self) -> u32 {
        self.pad_columns
    }
}
