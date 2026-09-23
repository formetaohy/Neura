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

    pub const fn left_stage(self) -> u64 {
        self.rows as u64 * self.depth as u64
    }

    pub const fn right_stage(self) -> u64 {
        self.depth as u64 * self.columns as u64
    }

    pub const fn shared_bytes(self) -> u64 {
        2 * (self.left_stage() + self.right_stage()) * WORD_BYTES
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct Profile {
    workgroup: u32,
    tiles: &'static [MatmulTile],
    shared_bytes: u64,
}

const REDUCTION_SCRATCH: u64 = 2 * WORD_BYTES;

impl Profile {
    pub const fn of(tiles: &'static [MatmulTile]) -> Self {
        let count = tiles.len();
        assert!(
            count > 0,
            "a profile offers no matmul tile for the device to run",
        );
        let workgroup = tiles[0].threads();
        let mut left_stage = 0u64;
        let mut right_stage = 0u64;
        let mut index = 0;
        while index < count {
            let tile = tiles[index];
            assert!(
                tile.threads() == workgroup,
                "a profile hands one device program two workgroup sizes",
            );
            if tile.left_stage() > left_stage {
                left_stage = tile.left_stage();
            }
            if tile.right_stage() > right_stage {
                right_stage = tile.right_stage();
            }
            index += 1;
        }
        Self {
            workgroup,
            tiles,
            shared_bytes: 2 * (left_stage + right_stage) * WORD_BYTES
                + REDUCTION_SCRATCH * workgroup as u64,
        }
    }

    pub const fn workgroup(self) -> u32 {
        self.workgroup
    }

    pub const fn tiles(self) -> &'static [MatmulTile] {
        self.tiles
    }

    pub const fn shared_bytes(self) -> u64 {
        self.shared_bytes
    }

    pub const fn fits(self, threads: u32, shared_bytes: u64) -> bool {
        self.workgroup <= threads && self.shared_bytes <= shared_bytes
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Geometry {
    workgroup: u32,
    tiles: Vec<MatmulTile>,
    left_stage: u64,
    right_stage: u64,
}

impl Geometry {
    pub fn of(profile: Profile) -> Self {
        let tiles = profile.tiles();
        let mut left_stage = 0u64;
        let mut right_stage = 0u64;
        for tile in tiles {
            left_stage = left_stage.max(tile.left_stage());
            right_stage = right_stage.max(tile.right_stage());
        }
        Self {
            workgroup: profile.workgroup(),
            tiles: tiles.to_vec(),
            left_stage,
            right_stage,
        }
    }

    pub const fn workgroup(&self) -> u32 {
        self.workgroup
    }

    pub fn tiles(&self) -> &[MatmulTile] {
        &self.tiles
    }

    pub fn geometry(&self, tile: MatmulTile) -> u32 {
        self.tiles
            .iter()
            .position(|candidate| *candidate == tile)
            .unwrap_or_else(|| panic!("a device program carries no {tile:?}"))
            .try_into()
            .expect("a device program carries fewer tiles than a word holds")
    }

    pub fn tile(&self, geometry: u32) -> MatmulTile {
        self.tiles
            .get(geometry as usize)
            .copied()
            .unwrap_or_else(|| {
                panic!(
                    "geometry {geometry} lies outside the {} a device program carries",
                    self.tiles.len(),
                )
            })
    }

    pub fn declarations(&self) -> String {
        let mut out = String::new();
        writeln!(out, "const WORKGROUP_SIZE: u32 = {}u;", self.workgroup).unwrap();
        writeln!(
            out,
            "const MATMUL_LEFT_STAGE: u32 = {}u;",
            2 * self.left_stage
        )
        .unwrap();
        writeln!(
            out,
            "const MATMUL_RIGHT_STAGE: u32 = {}u;",
            2 * self.right_stage
        )
        .unwrap();
        for (geometry, tile) in self.tiles.iter().enumerate() {
            writeln!(out, "const MATMUL_ROWS_{geometry}: u32 = {}u;", tile.rows()).unwrap();
            writeln!(
                out,
                "const MATMUL_COLUMNS_{geometry}: u32 = {}u;",
                tile.columns()
            )
            .unwrap();
            writeln!(
                out,
                "const MATMUL_DEPTH_{geometry}: u32 = {}u;",
                tile.depth()
            )
            .unwrap();
            writeln!(
                out,
                "const MATMUL_THREAD_ROWS_{geometry}: u32 = {}u;",
                tile.thread_rows()
            )
            .unwrap();
            writeln!(
                out,
                "const MATMUL_THREAD_COLUMNS_{geometry}: u32 = {}u;",
                tile.thread_columns()
            )
            .unwrap();
            writeln!(
                out,
                "const MATMUL_REGISTER_ROWS_{geometry}: u32 = {}u;",
                tile.register_rows()
            )
            .unwrap();
            writeln!(
                out,
                "const MATMUL_REGISTER_COLUMNS_{geometry}: u32 = {}u;",
                tile.register_columns()
            )
            .unwrap();
        }
        out
    }
}

pub const NARROW: Profile = Profile::of(&[
    MatmulTile::new(8, 8, 16, 8, 8),
    MatmulTile::new(8, 16, 16, 8, 8),
    MatmulTile::new(16, 8, 16, 8, 8),
    MatmulTile::new(8, 32, 16, 8, 8),
    MatmulTile::new(16, 16, 16, 8, 8),
    MatmulTile::new(32, 8, 16, 8, 8),
    MatmulTile::new(16, 32, 16, 8, 8),
    MatmulTile::new(32, 16, 16, 8, 8),
    MatmulTile::new(32, 32, 16, 8, 8),
    MatmulTile::new(64, 16, 16, 8, 8),
]);
pub const MEDIUM: Profile = Profile::of(&[
    MatmulTile::new(8, 16, 16, 8, 16),
    MatmulTile::new(16, 8, 16, 16, 8),
    MatmulTile::new(8, 32, 16, 8, 16),
    MatmulTile::new(32, 8, 16, 16, 8),
    MatmulTile::new(16, 16, 16, 8, 16),
    MatmulTile::new(16, 32, 16, 8, 16),
    MatmulTile::new(32, 16, 16, 8, 16),
    MatmulTile::new(64, 16, 16, 16, 8),
    MatmulTile::new(32, 32, 16, 8, 16),
    MatmulTile::new(64, 32, 16, 8, 16),
]);
pub const WIDE: Profile = Profile::of(&[
    MatmulTile::new(8, 32, 16, 8, 32),
    MatmulTile::new(32, 8, 16, 32, 8),
    MatmulTile::new(16, 16, 16, 16, 16),
    MatmulTile::new(8, 64, 16, 8, 32),
    MatmulTile::new(64, 8, 16, 32, 8),
    MatmulTile::new(16, 32, 16, 16, 16),
    MatmulTile::new(32, 16, 16, 16, 16),
    MatmulTile::new(16, 64, 16, 16, 16),
    MatmulTile::new(64, 16, 16, 16, 16),
    MatmulTile::new(32, 32, 16, 16, 16),
    MatmulTile::new(32, 64, 16, 16, 16),
    MatmulTile::new(64, 32, 16, 16, 16),
    MatmulTile::new(64, 64, 16, 16, 16),
]);

pub const PROFILES: &[Profile] = &[NARROW, MEDIUM, WIDE];
