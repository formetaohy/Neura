use crate::WORD_BYTES;
use std::fmt::Write as _;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct MatmulTile {
    rows: u32,
    columns: u32,
    depth: u32,
    thread_rows: u32,
    thread_columns: u32,
}

impl MatmulTile {
    pub const fn new(
        rows: u32,
        columns: u32,
        depth: u32,
        thread_rows: u32,
        thread_columns: u32,
    ) -> Self {
        assert!(
            rows > 0 && columns > 0 && depth > 0,
            "a matmul tile spans no element"
        );
        assert!(
            thread_rows > 0 && thread_columns > 0,
            "a matmul tile is carried by no thread",
        );
        assert!(
            rows.is_multiple_of(thread_rows),
            "a matmul tile does not divide its rows by the threads along rows",
        );
        assert!(
            columns.is_multiple_of(thread_columns),
            "a matmul tile does not divide its columns by the threads along columns",
        );
        Self {
            rows,
            columns,
            depth,
            thread_rows,
            thread_columns,
        }
    }

    pub const fn rows(self) -> u32 {
        self.rows
    }

    pub const fn columns(self) -> u32 {
        self.columns
    }

    pub const fn depth(self) -> u32 {
        self.depth
    }

    pub const fn thread_rows(self) -> u32 {
        self.thread_rows
    }

    pub const fn thread_columns(self) -> u32 {
        self.thread_columns
    }

    pub const fn threads(self) -> u32 {
        self.thread_rows * self.thread_columns
    }

    pub const fn register_rows(self) -> u32 {
        self.rows / self.thread_rows
    }

    pub const fn register_columns(self) -> u32 {
        self.columns / self.thread_columns
    }

    pub const fn registers(self) -> u32 {
        self.register_rows() * self.register_columns()
    }

    pub const fn tile_work(self) -> u64 {
        self.rows as u64 * self.columns as u64 * self.depth as u64
    }

    pub const fn shared_bytes(self) -> u64 {
        (self.rows as u64 * self.depth as u64 + self.depth as u64 * self.columns as u64)
            * WORD_BYTES
            * 2
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct Schedule {
    matmul: MatmulTile,
    elements_per_task: u32,
    elements_per_reduction: u32,
    rows_per_task: u32,
}

impl Schedule {
    pub const fn new(
        matmul: MatmulTile,
        elements_per_task: u32,
        elements_per_reduction: u32,
        rows_per_task: u32,
    ) -> Self {
        assert!(
            matmul.threads().is_power_of_two(),
            "a workgroup reduction folds a power of two of threads",
        );
        assert!(
            elements_per_task >= matmul.threads(),
            "a task hands some thread no element",
        );
        assert!(
            elements_per_reduction >= matmul.threads(),
            "a reduction chunk hands some thread no element",
        );
        assert!(rows_per_task > 0, "a task folds no softmax row");
        Self {
            matmul,
            elements_per_task,
            elements_per_reduction,
            rows_per_task,
        }
    }

    pub const fn matmul(self) -> MatmulTile {
        self.matmul
    }

    pub const fn workgroup(self) -> u32 {
        self.matmul.threads()
    }

    pub const fn elements_per_task(self) -> u32 {
        self.elements_per_task
    }

    pub const fn elements_per_reduction(self) -> u32 {
        self.elements_per_reduction
    }

    pub const fn rows_per_task(self) -> u32 {
        self.rows_per_task
    }

    pub const fn shared_bytes(self) -> u64 {
        self.matmul.shared_bytes() + self.workgroup() as u64 * WORD_BYTES
    }

    pub const fn fits(self, threads: u32, shared_bytes: u64) -> bool {
        self.workgroup() <= threads && self.shared_bytes() <= shared_bytes
    }

    pub fn declarations(self) -> String {
        let matmul = self.matmul;
        let mut out = String::new();
        writeln!(out, "const WORKGROUP_SIZE: u32 = {}u;", self.workgroup()).unwrap();
        writeln!(out, "const MATMUL_ROW_TILE: u32 = {}u;", matmul.rows()).unwrap();
        writeln!(out, "const MATMUL_COL_TILE: u32 = {}u;", matmul.columns()).unwrap();
        writeln!(out, "const MATMUL_DEPTH_TILE: u32 = {}u;", matmul.depth()).unwrap();
        writeln!(
            out,
            "const MATMUL_REGISTER_ROWS: u32 = {}u;",
            matmul.register_rows()
        )
        .unwrap();
        writeln!(
            out,
            "const MATMUL_REGISTER_COLUMNS: u32 = {}u;",
            matmul.register_columns()
        )
        .unwrap();
        writeln!(
            out,
            "const MATMUL_THREAD_ROWS: u32 = {}u;",
            matmul.thread_rows()
        )
        .unwrap();
        writeln!(
            out,
            "const MATMUL_THREAD_COLUMNS: u32 = {}u;",
            matmul.thread_columns()
        )
        .unwrap();
        out
    }
}

pub const NARROW: Schedule = Schedule::new(MatmulTile::new(16, 16, 16, 8, 8), 1024, 4096, 8);
pub const MEDIUM: Schedule = Schedule::new(MatmulTile::new(32, 32, 16, 16, 8), 2048, 8192, 8);
pub const WIDE: Schedule = Schedule::new(MatmulTile::new(64, 64, 16, 16, 16), 2048, 8192, 8);

pub const SCHEDULES: &[Schedule] = &[NARROW, MEDIUM, WIDE];
