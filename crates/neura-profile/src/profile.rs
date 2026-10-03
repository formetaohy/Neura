use neura_abi::{Kind, Module, WORD_BYTES};

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct AttentionTile {
    keys: u32,
    width: u32,
}

impl AttentionTile {
    pub const REGISTER_CEILING: u32 = 200;
    pub const KEYS_CEILING: u32 = 16;

    pub fn new(keys: u32, width: u32) -> Self {
        assert!(
            keys > 0 && keys <= Self::KEYS_CEILING && width > 0,
            "an attention tile walks no key of no width",
        );
        Self { keys, width }
    }

    pub fn fit(pool: u64, width: u32) -> Self {
        let row = 3 * width + 1;
        assert!(
            row <= Self::REGISTER_CEILING,
            "an attention of width {width} carries a query row of {} numbers and its gradient in one thread, beyond the {} a device thread holds",
            row - 1,
            Self::REGISTER_CEILING,
        );
        let staged = 2 * WORD_BYTES * u64::from(width);
        let room = pool / staged;
        let registers = u64::from(Self::REGISTER_CEILING - 3 * width);
        let keys = u32::try_from(room.min(registers))
            .unwrap_or(u32::MAX)
            .clamp(1, Self::KEYS_CEILING);
        assert!(
            u64::from(keys) <= room,
            "an attention of width {width} stages {staged} bytes of keys and values for one key, beyond the {pool} bytes of workgroup scratch its profile offers",
        );
        Self::new(keys, width)
    }

    pub const fn keys(self) -> u32 {
        self.keys
    }

    pub const fn width(self) -> u32 {
        self.width
    }

    pub const fn stage_words(self) -> u32 {
        self.keys * self.width
    }

    pub const fn shared_bytes(self) -> u64 {
        2 * self.stage_words() as u64 * WORD_BYTES
    }

    pub const fn registers(self) -> u32 {
        3 * self.width + self.keys
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum MatmulStrategy {
    Staged,
    Streamed,
    Cooperative,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct CooperativeMatrix {
    subgroup: u32,
    rows: u32,
    columns: u32,
    depth: u32,
}

impl CooperativeMatrix {
    pub fn new(subgroup: u32, rows: u32, columns: u32, depth: u32) -> Self {
        assert!(
            subgroup > 0 && rows > 0 && columns > 0 && depth > 0,
            "a cooperative matrix unit spans no element",
        );
        Self {
            subgroup,
            rows,
            columns,
            depth,
        }
    }

    pub const fn subgroup(self) -> u32 {
        self.subgroup
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
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct MatmulTile {
    strategy: MatmulStrategy,
    rows: u32,
    columns: u32,
    depth: u32,
    thread_rows: u32,
    thread_columns: u32,
}

impl MatmulTile {
    pub const fn new(
        strategy: MatmulStrategy,
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
        assert!(
            matches!(strategy, MatmulStrategy::Streamed) || depth.is_multiple_of(2),
            "a staged tile walks two of its rows at one stride into the same bank of the workgroup scratch",
        );
        Self {
            strategy,
            rows,
            columns,
            depth,
            thread_rows,
            thread_columns,
        }
    }

    pub const fn strategy(self) -> MatmulStrategy {
        self.strategy
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
        assert!(
            !matches!(self.strategy, MatmulStrategy::Cooperative),
            "a cooperative tile holds its fragment across a subgroup",
        );
        self.register_rows() * self.register_columns()
    }

    pub const fn tile_work(self) -> u64 {
        self.rows as u64 * self.columns as u64 * self.depth as u64
    }

    pub const fn left_stride(self) -> u32 {
        match self.strategy {
            MatmulStrategy::Staged => self.depth + 1,
            MatmulStrategy::Streamed | MatmulStrategy::Cooperative => 0,
        }
    }

    pub const fn left_stage(self) -> u64 {
        match self.strategy {
            MatmulStrategy::Staged => self.rows as u64 * self.left_stride() as u64,
            MatmulStrategy::Streamed => 0,
            MatmulStrategy::Cooperative => self.rows as u64 * self.depth as u64,
        }
    }

    pub const fn right_stage(self) -> u64 {
        match self.strategy {
            MatmulStrategy::Staged => self.depth as u64 * self.columns as u64,
            MatmulStrategy::Streamed => 0,
            MatmulStrategy::Cooperative => self.columns as u64 * self.depth as u64,
        }
    }

    pub const fn shared_bytes(self) -> u64 {
        2 * (self.left_stage() + self.right_stage()) * WORD_BYTES
    }

    pub const fn cooperative(
        rows: u32,
        columns: u32,
        depth: u32,
        subgroup_rows: u32,
        subgroup_columns: u32,
    ) -> Self {
        assert!(
            rows > 0 && columns > 0 && depth > 0,
            "a cooperative tile spans no element"
        );
        assert!(
            subgroup_rows > 0 && subgroup_columns > 0,
            "a cooperative tile is carried by no subgroup"
        );
        Self {
            strategy: MatmulStrategy::Cooperative,
            rows,
            columns,
            depth,
            thread_rows: subgroup_rows,
            thread_columns: subgroup_columns,
        }
    }

    pub const fn subgroup_rows(self) -> u32 {
        assert!(
            matches!(self.strategy, MatmulStrategy::Cooperative),
            "only a cooperative tile is carried by subgroups",
        );
        self.thread_rows
    }

    pub const fn subgroup_columns(self) -> u32 {
        assert!(
            matches!(self.strategy, MatmulStrategy::Cooperative),
            "only a cooperative tile is carried by subgroups",
        );
        self.thread_columns
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

    pub const NOMINAL_RESIDENT_THREADS: u32 = 32_768;

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
const WIDEST_BLOCKINGS: usize = 4;
const WORKGROUP_SIZES: [u32; 5] = [64, 128, 256, 512, 1024];
const BLOCKINGS: [(u32, u32); 11] = [
    (8, 8),
    (8, 4),
    (4, 8),
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
const STREAMED_ASPECT: u32 = 4;
const DEPTH: u32 = 8;
pub const CLAIM_WORDS: u32 = 2;
pub const CLAIM_BYTES: u64 = CLAIM_WORDS as u64 * WORD_BYTES;
const SCRATCH_WORDS_PER_THREAD: u64 = 2;

const fn scratch_budget(workgroup: u32, staging: u64, copy: u64) -> u64 {
    let floor = SCRATCH_WORDS_PER_THREAD * workgroup as u64 * WORD_BYTES;
    let widest = if staging > floor { staging } else { floor };
    if copy > widest {
        copy + CLAIM_BYTES
    } else {
        widest + CLAIM_BYTES
    }
}

const fn cooperative_panels(rows: u32, columns: u32, depth: u32) -> u64 {
    2 * (rows as u64 * depth as u64 + columns as u64 * depth as u64) * 2
}

const fn cooperative_copy(rows: u32, columns: u32) -> u64 {
    rows as u64 * columns as u64 * WORD_BYTES
}

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
        let mut plainest = tiles[0];
        let mut index = 0;
        while index < tiles.len() {
            if !matches!(tiles[index].strategy, MatmulStrategy::Cooperative) {
                plainest = tiles[index];
                index = tiles.len();
            } else {
                index += 1;
            }
        }
        let workgroup = plainest.threads();
        let mut entries = [plainest; MAX_TILES];
        let mut left_stage = 0u64;
        let mut right_stage = 0u64;
        let mut panels = 0u64;
        let mut copy = 0u64;
        let mut index = 0;
        while index < tiles.len() {
            let tile = tiles[index];
            if matches!(tile.strategy, MatmulStrategy::Cooperative) {
                assert!(
                    tile.rows().is_multiple_of(tile.subgroup_rows()),
                    "a cooperative tile does not divide its rows by its subgroups"
                );
                assert!(
                    tile.columns().is_multiple_of(tile.subgroup_columns()),
                    "a cooperative tile does not divide its columns by its subgroups"
                );
                let here = cooperative_panels(tile.rows(), tile.columns(), tile.depth());
                if here > panels {
                    panels = here;
                }
                let here = cooperative_copy(tile.rows(), tile.columns());
                if here > copy {
                    copy = here;
                }
            } else {
                assert!(
                    tile.threads() == workgroup,
                    "a profile hands one device program two workgroup sizes"
                );
                if tile.left_stage() > left_stage {
                    left_stage = tile.left_stage();
                }
                if tile.right_stage() > right_stage {
                    right_stage = tile.right_stage();
                }
            }
            entries[index] = tile;
            index += 1;
        }
        let staging = 2 * (left_stage + right_stage) * WORD_BYTES;
        Self {
            workgroup,
            shared_bytes: scratch_budget(workgroup, staging, copy) + panels,
            tiles: entries,
            count: tiles.len() as u8,
        }
    }

    pub fn derive(budget: Budget, cooperative: Option<CooperativeMatrix>) -> Vec<Self> {
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
                        MatmulStrategy::Staged,
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
            for (rows, columns) in streamed_grids(workgroup) {
                tiles.push(MatmulTile::new(
                    MatmulStrategy::Streamed,
                    rows,
                    columns,
                    DEPTH,
                    rows,
                    columns,
                ));
            }
            if let Some(shape) = cooperative
                && workgroup.is_multiple_of(shape.subgroup())
                && panels_fit(
                    workgroup,
                    budget.shared_bytes(),
                    left_stage,
                    right_stage,
                    shape,
                )
            {
                let subgroups = workgroup / shape.subgroup();
                let subgroup_columns = if subgroups >= 2 { subgroups / 2 } else { 1 };
                let subgroup_rows = subgroups / subgroup_columns;
                let tile = MatmulTile::cooperative(
                    subgroup_rows * shape.rows(),
                    subgroup_columns * shape.columns(),
                    shape.depth(),
                    subgroup_rows,
                    subgroup_columns,
                );
                assert!(
                    tile.rows() / subgroup_rows == shape.rows()
                        && tile.columns() / subgroup_columns == shape.columns(),
                    "a cooperative tile carries the fragment of its device",
                );
                tiles.push(tile);
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

    pub const fn scratch_bytes(self) -> u64 {
        self.shared_bytes - CLAIM_BYTES
    }

    pub const fn workgroups(self) -> u32 {
        let count = Budget::NOMINAL_RESIDENT_THREADS / self.workgroup;
        if count == 0 { 1 } else { count }
    }

    pub const fn fits(self, threads: u32, shared_bytes: u64) -> bool {
        self.workgroup <= threads && self.shared_bytes <= shared_bytes
    }

    pub const fn staging_bytes(self) -> u64 {
        let mut left = 0u64;
        let mut right = 0u64;
        let mut index = 0;
        while index < self.count as usize {
            if !matches!(self.tiles[index].strategy, MatmulStrategy::Cooperative) {
                if self.tiles[index].left_stage() > left {
                    left = self.tiles[index].left_stage();
                }
                if self.tiles[index].right_stage() > right {
                    right = self.tiles[index].right_stage();
                }
            }
            index += 1;
        }
        2 * (left + right) * WORD_BYTES
    }

    pub const fn cooperative_panels(self) -> u64 {
        let mut panels = 0u64;
        let mut index = 0;
        while index < self.count as usize {
            let tile = self.tiles[index];
            if matches!(tile.strategy, MatmulStrategy::Cooperative) {
                let here = cooperative_panels(tile.rows(), tile.columns(), tile.depth());
                if here > panels {
                    panels = here;
                }
            }
            index += 1;
        }
        panels
    }
}

fn blockings(
    rows: u32,
    columns: u32,
    left_stage: u64,
    right_stage: u64,
    workgroup: u32,
    shared_bytes: u64,
) -> Vec<(u32, u32)> {
    let mut chosen = vec![PLAINEST_BLOCKING];
    let mut left = left_stage;
    let mut right = right_stage;
    for blocking in BLOCKINGS {
        if chosen.len() > WIDEST_BLOCKINGS {
            break;
        }
        let (register_rows, register_columns) = blocking;
        let tile = MatmulTile::new(
            MatmulStrategy::Staged,
            rows * register_rows,
            columns * register_columns,
            DEPTH,
            rows,
            columns,
        );
        if !carried(
            left.max(tile.left_stage()),
            right.max(tile.right_stage()),
            workgroup,
            shared_bytes,
        ) {
            continue;
        }
        left = left.max(tile.left_stage());
        right = right.max(tile.right_stage());
        chosen.push(blocking);
    }
    chosen
}

fn carried(left_stage: u64, right_stage: u64, workgroup: u32, shared_bytes: u64) -> bool {
    scratch_budget(workgroup, 2 * (left_stage + right_stage) * WORD_BYTES, 0) <= shared_bytes
}

fn panels_fit(
    workgroup: u32,
    shared_bytes: u64,
    left_stage: u64,
    right_stage: u64,
    shape: CooperativeMatrix,
) -> bool {
    let rows = shape.rows() * 2;
    let columns = shape.columns() * 2;
    let staging = scratch_budget(workgroup, 2 * (left_stage + right_stage) * WORD_BYTES, 0);
    let cooperative =
        cooperative_copy(rows, columns) + cooperative_panels(rows, columns, shape.depth());
    staging + cooperative <= shared_bytes
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

fn streamed_grids(workgroup: u32) -> Vec<(u32, u32)> {
    let mut grids = Vec::new();
    let mut rows = 1;
    while rows <= STREAMED_ASPECT && rows < workgroup {
        let columns = workgroup / rows;
        assert!(
            columns * rows == workgroup,
            "a streamed tile hands its threads a grid that does not cover the workgroup",
        );
        grids.push((rows, columns));
        rows *= 2;
    }
    grids
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct Geometry {
    workgroup: u32,
    budget: u64,
    tiles: Vec<MatmulTile>,
    attention: Vec<AttentionTile>,
    left_stage: u64,
    right_stage: u64,
    half_stage: u64,
    copy: u64,
}

impl Geometry {
    pub fn of(
        workgroup: u32,
        budget: u64,
        tiles: &[MatmulTile],
        attention: &[AttentionTile],
    ) -> Self {
        assert!(
            tiles.iter().all(|tile| {
                matches!(tile.strategy, MatmulStrategy::Cooperative) || tile.threads() == workgroup
            }),
            "a device program carries a tile another workgroup stages",
        );
        let mut left_stage = 0u64;
        let mut right_stage = 0u64;
        let mut half_stage = 0u64;
        let mut copy = 0u64;
        for tile in tiles {
            if matches!(tile.strategy, MatmulStrategy::Cooperative) {
                half_stage = half_stage.max(2 * (tile.left_stage() + tile.right_stage()));
                copy = copy.max(tile.rows() as u64 * tile.columns() as u64);
            } else {
                left_stage = left_stage.max(tile.left_stage());
                right_stage = right_stage.max(tile.right_stage());
            }
        }
        Self {
            workgroup,
            budget,
            tiles: tiles.to_vec(),
            attention: attention.to_vec(),
            left_stage,
            right_stage,
            half_stage,
            copy,
        }
    }

    pub const fn workgroup(&self) -> u32 {
        self.workgroup
    }

    pub const fn budget(&self) -> u64 {
        self.budget
    }

    pub const fn staging_bytes(&self) -> u64 {
        2 * (self.left_stage + self.right_stage) * WORD_BYTES
    }

    pub fn scratch_bytes(&self, kinds: &[Kind]) -> u64 {
        u64::from(self.scratch_words(kinds)) * WORD_BYTES
    }

    pub fn scratch_words(&self, kinds: &[Kind]) -> u32 {
        Module::ALL
            .iter()
            .copied()
            .filter(|module| kinds.iter().any(|kind| kind.carries(*module)))
            .map(|module| self.module_scratch(module))
            .max()
            .unwrap_or(0)
    }

    fn module_scratch(&self, module: Module) -> u32 {
        match module {
            Module::MatmulTiles => (2 * (self.left_stage + self.right_stage)).max(self.copy) as u32,
            Module::Attention => 2 * self.attention_stage_words(),
            Module::Reduce => self.workgroup,
            Module::Choice => 2 * self.workgroup,
            _ => 0,
        }
    }

    pub fn subgroup(&self) -> u32 {
        let tile = self
            .tiles
            .iter()
            .find(|tile| matches!(tile.strategy, MatmulStrategy::Cooperative))
            .unwrap_or_else(|| panic!("a device program carries no cooperative tile"));
        self.workgroup / (tile.subgroup_rows() * tile.subgroup_columns())
    }

    pub const fn half_stage_words(&self) -> u32 {
        self.half_stage as u32
    }

    pub const fn matmul_right(&self) -> u32 {
        (2 * self.left_stage) as u32
    }

    pub const fn choice(&self) -> u32 {
        self.workgroup
    }

    pub fn attention(&self) -> &[AttentionTile] {
        &self.attention
    }

    pub fn attention_geometry(&self, tile: AttentionTile) -> u32 {
        self.attention
            .iter()
            .position(|candidate| *candidate == tile)
            .unwrap_or_else(|| panic!("a device program carries no {tile:?}"))
            .try_into()
            .expect("a device program carries fewer attention tiles than a word holds")
    }

    pub fn attention_tile(&self, geometry: u32) -> AttentionTile {
        self.attention
            .get(geometry as usize)
            .copied()
            .unwrap_or_else(|| {
                panic!(
                    "attention geometry {geometry} lies outside the {} a device program carries",
                    self.attention.len(),
                )
            })
    }

    pub fn attention_stage_words(&self) -> u32 {
        self.attention
            .iter()
            .map(|tile| tile.stage_words())
            .max()
            .unwrap_or(0)
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
}
