use neura_abi::WORD_BYTES;

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
pub struct Budget {
    threads: u32,
    shared_bytes: u64,
}

impl Budget {
    pub const BASELINE: Self = Self {
        threads: 256,
        shared_bytes: 16 << 10,
    };

    pub const BALANCED_THREADS: u32 = 256;

    pub const NOMINAL_RESIDENT_THREADS: u32 = 65_536;

    pub const fn of(threads: u32, shared_bytes: u64) -> Self {
        assert!(
            threads >= 32,
            "a device schedules workgroups of at least 32 threads",
        );
        assert!(
            shared_bytes >= 16 << 10,
            "a device offers its workgroups at least the baseline pool of shared memory",
        );
        Self {
            threads: 1 << (31 - threads.leading_zeros()),
            shared_bytes,
        }
    }

    pub const fn threads(self) -> u32 {
        self.threads
    }

    pub const fn shared_bytes(self) -> u64 {
        self.shared_bytes
    }
}

pub const MAX_TILES: usize = 32;
const PLAINEST_BLOCKING: (u32, u32) = (1, 1);
const WORKGROUP_SIZES: [u32; 5] = [64, 128, 256, 512, 1024];
const BLOCKINGS: [(u32, u32); 8] = [
    (4, 4),
    (2, 4),
    (4, 2),
    (2, 2),
    (1, 4),
    (4, 1),
    (1, 2),
    (2, 1),
];
const GRID_ASPECT: u32 = 4;
const DEPTH: u32 = 8;
const REDUCTION_SCRATCH: u64 = 2 * WORD_BYTES;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Profile {
    workgroup: u32,
    shared_bytes: u64,
    tiles: [MatmulTile; MAX_TILES],
    count: u8,
}

impl std::fmt::Debug for Profile {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        out.debug_struct("Profile")
            .field("workgroup", &self.workgroup)
            .field("shared_bytes", &self.shared_bytes)
            .field("tiles", &self.tiles())
            .finish()
    }
}

impl Profile {
    pub const fn of(tiles: &[MatmulTile]) -> Self {
        assert!(
            !tiles.is_empty() && tiles.len() <= MAX_TILES,
            "a profile offers between one and thirty-two matmul tiles",
        );
        let workgroup = tiles[0].threads();
        let mut entries = [tiles[0]; MAX_TILES];
        let mut left_stage = 0u64;
        let mut right_stage = 0u64;
        let mut index = 0;
        while index < tiles.len() {
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
            entries[index] = tile;
            index += 1;
        }
        Self {
            workgroup,
            shared_bytes: 2 * (left_stage + right_stage) * WORD_BYTES
                + REDUCTION_SCRATCH * workgroup as u64,
            tiles: entries,
            count: tiles.len() as u8,
        }
    }

    pub fn derive(budget: Budget) -> Vec<Self> {
        let mut profiles = Vec::new();
        for workgroup in WORKGROUP_SIZES {
            if workgroup > budget.threads() {
                continue;
            }
            let mut tiles: Vec<MatmulTile> = Vec::new();
            let mut left_stage = 0u64;
            let mut right_stage = 0u64;
            for (rows, columns) in grids(workgroup) {
                for (register_rows, register_columns) in blockings(
                    rows,
                    columns,
                    left_stage,
                    right_stage,
                    workgroup,
                    budget.shared_bytes(),
                ) {
                    let tile = MatmulTile::new(
                        rows * register_rows,
                        columns * register_columns,
                        DEPTH,
                        rows,
                        columns,
                    );
                    left_stage = left_stage.max(tile.left_stage());
                    right_stage = right_stage.max(tile.right_stage());
                    tiles.push(tile);
                }
            }
            if tiles.is_empty() {
                continue;
            }
            tiles.sort_by_key(|tile| (tile.rows() * tile.columns(), tile.depth()));
            tiles.dedup();
            profiles.push(Self::of(&tiles));
        }
        profiles
    }

    pub const fn workgroup(self) -> u32 {
        self.workgroup
    }

    pub fn tiles(&self) -> &[MatmulTile] {
        &self.tiles[..self.count as usize]
    }

    pub const fn shared_bytes(self) -> u64 {
        self.shared_bytes
    }

    pub const fn workgroups(self) -> u32 {
        let count = Budget::NOMINAL_RESIDENT_THREADS / self.workgroup;
        if count == 0 { 1 } else { count }
    }

    pub const fn fits(self, threads: u32, shared_bytes: u64) -> bool {
        self.workgroup <= threads && self.shared_bytes <= shared_bytes
    }
}

fn blockings(
    rows: u32,
    columns: u32,
    left_stage: u64,
    right_stage: u64,
    workgroup: u32,
    shared_bytes: u64,
) -> [(u32, u32); 2] {
    [
        PLAINEST_BLOCKING,
        widest(
            rows,
            columns,
            left_stage,
            right_stage,
            workgroup,
            shared_bytes,
        ),
    ]
}

fn widest(
    rows: u32,
    columns: u32,
    left_stage: u64,
    right_stage: u64,
    workgroup: u32,
    shared_bytes: u64,
) -> (u32, u32) {
    for (register_rows, register_columns) in BLOCKINGS {
        let tile = MatmulTile::new(
            rows * register_rows,
            columns * register_columns,
            DEPTH,
            rows,
            columns,
        );
        let staged = 2
            * (left_stage.max(tile.left_stage()) + right_stage.max(tile.right_stage()))
            * WORD_BYTES;
        if staged + REDUCTION_SCRATCH * u64::from(workgroup) <= shared_bytes {
            return (register_rows, register_columns);
        }
    }
    PLAINEST_BLOCKING
}

fn grids(workgroup: u32) -> Vec<(u32, u32)> {
    let mut grids = Vec::new();
    let mut rows = 1;
    while rows <= workgroup {
        if workgroup.is_multiple_of(rows) {
            let columns = workgroup / rows;
            if rows <= GRID_ASPECT * columns && columns <= GRID_ASPECT * rows {
                grids.push((rows, columns));
            }
        }
        rows *= 2;
    }
    grids
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct Geometry {
    workgroup: u32,
    tiles: Vec<MatmulTile>,
    left_stage: u64,
    right_stage: u64,
}

impl Geometry {
    pub fn of(workgroup: u32, tiles: &[MatmulTile]) -> Self {
        assert!(
            tiles.is_empty() || tiles.iter().all(|tile| tile.threads() == workgroup),
            "a device program carries a tile another workgroup stages",
        );
        let mut left_stage = 0u64;
        let mut right_stage = 0u64;
        for tile in tiles {
            left_stage = left_stage.max(tile.left_stage());
            right_stage = right_stage.max(tile.right_stage());
        }
        Self {
            workgroup,
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

    pub fn stage_lengths(&self) -> (u32, u32) {
        (
            (2 * self.left_stage)
                .try_into()
                .expect("a left tile fits in device memory"),
            (2 * self.right_stage)
                .try_into()
                .expect("a right tile fits in device memory"),
        )
    }
}
