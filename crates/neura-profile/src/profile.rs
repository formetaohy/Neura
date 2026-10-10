use neura_abi::{DeviceModule, Kind, WORD_BYTES};
use std::collections::HashMap;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct AttentionTile {
    keys: u32,
    width: u32,
    slices: u32,
}

impl AttentionTile {
    pub const REGISTER_CEILING: u32 = 200;
    pub const KEYS_CEILING: u32 = 16;
    pub const WIDTH_PER_THREAD: u32 = 64;
    pub const REDUCTIONS: u32 = 1;

    pub fn new(keys: u32, width: u32) -> Self {
        assert!(
            keys > 0 && keys <= Self::KEYS_CEILING && width > 0,
            "an attention tile walks no key of no width",
        );
        let slices = Self::slices_of(width);
        let slice_width = width.div_ceil(slices);
        assert!(
            3 * slice_width < Self::REGISTER_CEILING,
            "an attention of width {width} spreads one query row over {slices} threads of {slice_width} numbers, and the {} numbers of that row and its gradients still outrun the {} a device thread holds",
            3 * slice_width,
            Self::REGISTER_CEILING,
        );
        Self {
            keys,
            width,
            slices,
        }
    }

    pub fn fit(pool: u64, workgroup: u32, width: u32) -> Self {
        let slices = Self::slices_of(width);
        assert!(
            slices <= workgroup,
            "an attention of width {width} spreads one query row over {slices} threads, and a workgroup of {workgroup} threads holds no whole row",
        );
        let slice_width = width.div_ceil(slices);
        let staged = 2 * WORD_BYTES * u64::from(width);
        let reduced = u64::from(Self::reduction_words_per_key(workgroup, slices)) * WORD_BYTES;
        let room = pool / (staged + reduced);
        let registers = u64::from(Self::REGISTER_CEILING - 3 * slice_width) / 2;
        let keys = u32::try_from(room.min(registers))
            .unwrap_or(u32::MAX)
            .clamp(1, Self::KEYS_CEILING);
        assert!(
            3 * slice_width < Self::REGISTER_CEILING,
            "an attention of width {width} spreads one query row over {slices} threads of {slice_width} numbers, and the {} numbers of that row and its gradients still outrun the {} a device thread holds",
            3 * slice_width,
            Self::REGISTER_CEILING,
        );
        assert!(
            u64::from(keys) <= room,
            "an attention of width {width} stages {staged} bytes of keys and values and {reduced} bytes of reductions for one key, beyond the {pool} bytes of workgroup scratch its profile offers",
        );
        Self::new(keys, width)
    }

    fn slices_of(width: u32) -> u32 {
        width.div_ceil(Self::WIDTH_PER_THREAD).next_power_of_two()
    }

    const fn reduction_words_per_key(workgroup: u32, slices: u32) -> u32 {
        if slices == 1 {
            return 0;
        }
        Self::REDUCTIONS * workgroup
    }

    pub const fn keys(self) -> u32 {
        self.keys
    }

    pub const fn width(self) -> u32 {
        self.width
    }

    pub const fn slices(self) -> u32 {
        self.slices
    }

    pub const fn slice_width(self) -> u32 {
        self.width.div_ceil(self.slices)
    }

    pub const fn stage_words(self) -> u32 {
        self.keys * self.width
    }

    pub const fn reduction_words(self, workgroup: u32) -> u32 {
        if self.slices == 1 {
            return 0;
        }
        (workgroup / self.slices) * self.keys * Self::REDUCTIONS * self.slices
    }

    pub const fn scratch_words(self, workgroup: u32) -> u32 {
        2 * self.stage_words() + self.reduction_words(workgroup)
    }

    pub const fn shared_bytes(self, workgroup: u32) -> u64 {
        self.scratch_words(workgroup) as u64 * WORD_BYTES
    }

    pub const fn registers(self) -> u32 {
        3 * self.slice_width() + 2 * self.keys
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
pub struct CooperativeTile {
    fragment: CooperativeMatrix,
    subgroups: (u32, u32),
    fragments: (u32, u32),
    stage: u32,
}

impl CooperativeTile {
    pub fn new(
        fragment: CooperativeMatrix,
        subgroups: (u32, u32),
        fragments: (u32, u32),
        stage: u32,
    ) -> Self {
        assert!(
            subgroups.0 > 0 && subgroups.1 > 0,
            "a cooperative tile hands its fragments to no subgroup",
        );
        assert!(
            fragments.0 > 0 && fragments.1 > 0,
            "a subgroup of a cooperative tile carries no fragment",
        );
        assert!(
            stage > 0 && stage.is_multiple_of(fragment.depth()),
            "a cooperative tile stages {stage} deep where a whole fragment of its device is {} deep",
            fragment.depth(),
        );
        Self {
            fragment,
            subgroups,
            fragments,
            stage,
        }
    }

    pub const fn fragment(self) -> CooperativeMatrix {
        self.fragment
    }

    pub const fn fragments(self) -> (u32, u32) {
        self.fragments
    }

    pub const fn stage(self) -> u32 {
        self.stage
    }

    pub const fn subgroup(self) -> u32 {
        self.fragment.subgroup()
    }

    pub const fn subgroup_rows(self) -> u32 {
        self.subgroups.0
    }

    pub const fn subgroup_columns(self) -> u32 {
        self.subgroups.1
    }

    pub const fn fragment_rows(self) -> u32 {
        self.fragments.0
    }

    pub const fn fragment_columns(self) -> u32 {
        self.fragments.1
    }

    pub const fn rows(self) -> u32 {
        self.subgroups.0 * self.fragments.0 * self.fragment.rows()
    }

    pub const fn columns(self) -> u32 {
        self.subgroups.1 * self.fragments.1 * self.fragment.columns()
    }

    pub const fn depth(self) -> u32 {
        self.stage
    }

    pub const fn threads(self) -> u32 {
        self.subgroups.0 * self.subgroups.1 * self.fragment.subgroup()
    }

    pub const fn accumulators(self) -> u32 {
        self.fragments.0 * self.fragments.1
    }

    pub const fn registers(self) -> u32 {
        let accumulator = self.accumulators() * self.fragment.rows() * self.fragment.columns();
        let operands = self.fragment.rows() * self.fragment.depth()
            + self.fragment.depth() * self.fragment.columns();
        (accumulator + operands) / self.fragment.subgroup()
    }

    pub const fn left_stride(self) -> u32 {
        self.stage
    }

    pub const fn left_stage(self) -> u64 {
        self.rows() as u64 * self.stage as u64
    }

    pub const fn right_stage(self) -> u64 {
        self.columns() as u64 * self.stage as u64
    }

    pub const fn half_panels(self) -> u64 {
        2 * (self.left_stage() + self.right_stage())
    }

    pub const fn half_bytes(self) -> u64 {
        self.half_panels() * 2
    }

    pub const fn copy(self) -> u64 {
        self.subgroups.0 as u64
            * self.subgroups.1 as u64
            * self.fragment.rows() as u64
            * self.fragment.columns() as u64
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct PlainTile {
    rows: u32,
    columns: u32,
    depth: u32,
    thread_rows: u32,
    thread_columns: u32,
}

impl PlainTile {
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

    pub const fn operands(self) -> u32 {
        self.register_rows() + self.register_columns()
    }

    pub const fn tile_work(self) -> u64 {
        self.rows as u64 * self.columns as u64 * self.depth as u64
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum MatmulTile {
    Staged(PlainTile),
    Streamed(PlainTile),
    Cooperative(CooperativeTile),
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
        let tile = PlainTile {
            rows,
            columns,
            depth,
            thread_rows,
            thread_columns,
        };
        match strategy {
            MatmulStrategy::Staged => Self::Staged(tile),
            MatmulStrategy::Streamed => Self::Streamed(tile),
            MatmulStrategy::Cooperative => panic!(
                "a cooperative tile divides the fragments of its device among subgroups instead of registers",
            ),
        }
    }

    pub const fn cooperative(tile: CooperativeTile) -> Self {
        Self::Cooperative(tile)
    }

    pub const fn strategy(self) -> MatmulStrategy {
        match self {
            Self::Staged(_) => MatmulStrategy::Staged,
            Self::Streamed(_) => MatmulStrategy::Streamed,
            Self::Cooperative(_) => MatmulStrategy::Cooperative,
        }
    }

    pub const fn rows(self) -> u32 {
        match self {
            Self::Staged(tile) | Self::Streamed(tile) => tile.rows(),
            Self::Cooperative(tile) => tile.rows(),
        }
    }

    pub const fn columns(self) -> u32 {
        match self {
            Self::Staged(tile) | Self::Streamed(tile) => tile.columns(),
            Self::Cooperative(tile) => tile.columns(),
        }
    }

    pub const fn depth(self) -> u32 {
        match self {
            Self::Staged(tile) | Self::Streamed(tile) => tile.depth(),
            Self::Cooperative(tile) => tile.depth(),
        }
    }

    pub const fn thread_rows(self) -> u32 {
        match self {
            Self::Staged(tile) | Self::Streamed(tile) => tile.thread_rows(),
            Self::Cooperative(tile) => tile.subgroup_rows(),
        }
    }

    pub const fn thread_columns(self) -> u32 {
        match self {
            Self::Staged(tile) | Self::Streamed(tile) => tile.thread_columns(),
            Self::Cooperative(tile) => tile.subgroup_columns(),
        }
    }

    pub const fn threads(self) -> u32 {
        match self {
            Self::Staged(tile) | Self::Streamed(tile) => tile.threads(),
            Self::Cooperative(tile) => tile.threads(),
        }
    }

    pub const fn registers(self) -> u32 {
        match self {
            Self::Staged(tile) | Self::Streamed(tile) => tile.registers(),
            Self::Cooperative(tile) => tile.registers(),
        }
    }

    pub const fn operands(self) -> u32 {
        match self {
            Self::Staged(tile) | Self::Streamed(tile) => tile.operands(),
            Self::Cooperative(_) => 0,
        }
    }

    pub const fn register_rows(self) -> u32 {
        match self {
            Self::Staged(tile) | Self::Streamed(tile) => tile.register_rows(),
            Self::Cooperative(_) => panic!("a cooperative tile accumulates across a subgroup"),
        }
    }

    pub const fn register_columns(self) -> u32 {
        match self {
            Self::Staged(tile) | Self::Streamed(tile) => tile.register_columns(),
            Self::Cooperative(_) => panic!("a cooperative tile accumulates across a subgroup"),
        }
    }

    pub const fn tile_work(self) -> u64 {
        self.rows() as u64 * self.columns() as u64 * self.depth() as u64
    }

    pub const fn left_stride(self) -> u32 {
        match self {
            Self::Staged(tile) => tile.depth() + 1,
            Self::Streamed(_) => 0,
            Self::Cooperative(tile) => tile.left_stride(),
        }
    }

    pub const fn left_stage(self) -> u64 {
        match self {
            Self::Staged(tile) => tile.rows() as u64 * (tile.depth() + 1) as u64,
            Self::Streamed(_) => 0,
            Self::Cooperative(tile) => tile.left_stage(),
        }
    }

    pub const fn right_stage(self) -> u64 {
        match self {
            Self::Staged(tile) => tile.depth() as u64 * tile.columns() as u64,
            Self::Streamed(_) => 0,
            Self::Cooperative(tile) => tile.right_stage(),
        }
    }

    pub const fn half_panels(self) -> u64 {
        match self {
            Self::Cooperative(tile) => tile.half_panels(),
            _ => panic!("only a cooperative tile stages its panels on the half grid"),
        }
    }

    pub const fn shared_bytes(self) -> u64 {
        match self {
            Self::Cooperative(tile) => tile.half_bytes(),
            _ => 2 * (self.left_stage() + self.right_stage()) * WORD_BYTES,
        }
    }

    pub const fn subgroup_rows(self) -> u32 {
        match self {
            Self::Cooperative(tile) => tile.subgroup_rows(),
            _ => panic!("only a cooperative tile is carried by subgroups"),
        }
    }

    pub const fn subgroup_columns(self) -> u32 {
        match self {
            Self::Cooperative(tile) => tile.subgroup_columns(),
            _ => panic!("only a cooperative tile is carried by subgroups"),
        }
    }

    pub const fn subgroup(self) -> u32 {
        match self {
            Self::Cooperative(tile) => tile.subgroup(),
            _ => panic!("only a cooperative tile is carried by subgroups"),
        }
    }

    pub const fn fragment_rows(self) -> u32 {
        match self {
            Self::Cooperative(tile) => tile.fragment().rows(),
            _ => panic!("only a cooperative tile holds the fragment of its device"),
        }
    }

    pub const fn fragment_columns(self) -> u32 {
        match self {
            Self::Cooperative(tile) => tile.fragment().columns(),
            _ => panic!("only a cooperative tile holds the fragment of its device"),
        }
    }

    pub const fn fragment_depth(self) -> u32 {
        match self {
            Self::Cooperative(tile) => tile.fragment().depth(),
            _ => panic!("only a cooperative tile holds the fragment of its device"),
        }
    }

    pub const fn fragment_grid(self) -> (u32, u32) {
        match self {
            Self::Cooperative(tile) => tile.fragments(),
            _ => panic!("only a cooperative tile holds the fragment of its device"),
        }
    }

    pub const fn accumulators(self) -> u32 {
        match self {
            Self::Cooperative(tile) => tile.accumulators(),
            _ => panic!("only a cooperative tile accumulates across a subgroup"),
        }
    }

    pub const fn copy(self) -> u64 {
        match self {
            Self::Cooperative(tile) => tile.copy(),
            _ => 0,
        }
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
const COOPERATIVE_ASPECT: u32 = 4;
const COOPERATIVE_FRAGMENTS: [(u32, u32); 4] = [(2, 2), (2, 1), (1, 2), (1, 1)];
const COOPERATIVE_STAGE_FRAGMENTS: u32 = 1;
const COOPERATIVE_MENU: usize = 6;
const POOL_BYTES: u64 = 21 << 10;
const BLOCKINGS_PER_GRID: usize = 2;
const DEPTH: u32 = 8;
pub const CLAIM_WORDS: u32 = 4;
pub const CLAIM_BYTES: u64 = CLAIM_WORDS as u64 * WORD_BYTES;
const SCRATCH_WORDS_PER_THREAD: u64 = 2;
const REGISTER_FILE: u32 = 65_536;
const REGISTER_SLACK: u32 = 24;
const THREAD_REGISTER_CEILING: u32 = 255;

fn carries_registers(workgroup: u32, registers: u32) -> bool {
    let per_thread = registers + REGISTER_SLACK;
    per_thread <= THREAD_REGISTER_CEILING && per_thread.saturating_mul(workgroup) <= REGISTER_FILE
}

const fn scratch_budget(workgroup: u32, staging: u64, copy: u64) -> u64 {
    let floor = SCRATCH_WORDS_PER_THREAD * workgroup as u64 * WORD_BYTES;
    let widest = if staging > floor { staging } else { floor };
    if copy > widest {
        copy + CLAIM_BYTES
    } else {
        widest + CLAIM_BYTES
    }
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
    pub fn of(tiles: &[MatmulTile]) -> Self {
        assert!(
            !tiles.is_empty() && tiles.len() <= MAX_TILES,
            "a profile offers between one and thirty-two matmul tiles",
        );
        let workgroup = tiles[0].threads();
        let mut entries = [tiles[0]; MAX_TILES];
        let mut fragment = None;
        let mut left_stage = 0u64;
        let mut right_stage = 0u64;
        let mut panels = 0u64;
        let mut copy = 0u64;
        let mut index = 0;
        while index < tiles.len() {
            let tile = tiles[index];
            assert!(
                tile.threads() == workgroup,
                "a profile hands one device program two workgroup sizes",
            );
            assert!(
                carries_registers(workgroup, tile.registers() + tile.operands()),
                "a {workgroup} thread workgroup of {tile:?} carries {} registers a thread beside the {} a device holds for it",
                tile.registers() + tile.operands(),
                REGISTER_FILE,
            );
            if let MatmulTile::Cooperative(cooperative) = tile {
                assert!(
                    *fragment.get_or_insert(cooperative.fragment()) == cooperative.fragment(),
                    "a device program holds one cooperative fragment shape, and {tile:?} holds another",
                );
                let here = cooperative.half_bytes();
                if here > panels {
                    panels = here;
                }
                let here = cooperative.copy() * WORD_BYTES;
                if here > copy {
                    copy = here;
                }
            } else {
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
        let pool = budget.shared_bytes().min(POOL_BYTES);
        for workgroup in WORKGROUP_SIZES {
            if workgroup > budget.threads() {
                continue;
            }
            let mut tiles = staged_tiles(workgroup, pool);
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
            let mut left_stage = 0u64;
            let mut right_stage = 0u64;
            for tile in &tiles {
                left_stage = left_stage.max(tile.left_stage());
                right_stage = right_stage.max(tile.right_stage());
            }
            if let Some(fragment) = cooperative {
                tiles.extend(cooperative_tiles(
                    workgroup,
                    budget.shared_bytes(),
                    fragment,
                    left_stage,
                    right_stage,
                ));
            }
            if tiles.is_empty() {
                continue;
            }
            tiles.sort_by_key(|tile| (tile.rows() * tile.columns(), tile.depth()));
            tiles.dedup();
            assert!(
                tiles.len() <= MAX_TILES,
                "a device derives {} tiles for one workgroup of {workgroup} threads where a profile carries {MAX_TILES}",
                tiles.len(),
            );
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
            if !matches!(self.tiles[index], MatmulTile::Cooperative(_)) {
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
            if let MatmulTile::Cooperative(cooperative) = tile {
                let here = cooperative.half_bytes();
                if here > panels {
                    panels = here;
                }
            }
            index += 1;
        }
        panels
    }
}

fn staged_tiles(workgroup: u32, budget: u64) -> Vec<MatmulTile> {
    let mut candidates = Vec::new();
    for (rows, columns) in grids(workgroup) {
        for (register_rows, register_columns) in BLOCKINGS.into_iter().chain([PLAINEST_BLOCKING]) {
            let tile = MatmulTile::new(
                MatmulStrategy::Staged,
                rows * register_rows,
                columns * register_columns,
                DEPTH,
                rows,
                columns,
            );
            if !carries_registers(workgroup, tile.registers() + tile.operands()) {
                continue;
            }
            if !carried(tile.left_stage(), tile.right_stage(), workgroup, budget) {
                continue;
            }
            candidates.push(tile);
        }
    }
    candidates.sort_by_key(|tile| {
        (
            std::cmp::Reverse(tile.rows() * tile.columns()),
            tile.left_stage() + tile.right_stage(),
        )
    });
    candidates.dedup();
    let mut chosen: Vec<MatmulTile> = Vec::new();
    let mut held = HashMap::<(u32, u32), usize>::new();
    for tile in &candidates {
        let grid = (tile.thread_rows(), tile.thread_columns());
        let count = held.entry(grid).or_insert(0);
        if *count >= BLOCKINGS_PER_GRID {
            continue;
        }
        if !carried(
            joint_stage(&chosen, *tile).0,
            joint_stage(&chosen, *tile).1,
            workgroup,
            budget,
        ) {
            continue;
        }
        *count += 1;
        chosen.push(*tile);
    }
    for tile in &candidates {
        if tile.register_rows() * tile.register_columns() != 1 {
            continue;
        }

        if !carried(
            joint_stage(&chosen, *tile).0,
            joint_stage(&chosen, *tile).1,
            workgroup,
            budget,
        ) {
            continue;
        }
        chosen.push(*tile);
    }
    chosen
}

fn joint_stage(held: &[MatmulTile], tile: MatmulTile) -> (u64, u64) {
    let mut left = tile.left_stage();
    let mut right = tile.right_stage();
    for held in held {
        left = left.max(held.left_stage());
        right = right.max(held.right_stage());
    }
    (left, right)
}

fn carried(left_stage: u64, right_stage: u64, workgroup: u32, shared_bytes: u64) -> bool {
    scratch_budget(workgroup, 2 * (left_stage + right_stage) * WORD_BYTES, 0) <= shared_bytes
}

fn cooperative_tiles(
    workgroup: u32,
    shared_bytes: u64,
    fragment: CooperativeMatrix,
    left_stage: u64,
    right_stage: u64,
) -> Vec<MatmulTile> {
    if !workgroup.is_multiple_of(fragment.subgroup()) {
        return Vec::new();
    }
    let mut tiles: Vec<CooperativeTile> = Vec::new();
    for (subgroup_rows, subgroup_columns) in cooperative_subgroups(workgroup / fragment.subgroup())
    {
        for (fragments_rows, fragments_columns) in COOPERATIVE_FRAGMENTS {
            let fragments = (fragments_rows, fragments_columns);
            let mut stage = fragment.depth();
            let mut staged = None;
            while stage <= COOPERATIVE_STAGE_FRAGMENTS * fragment.depth() {
                let tile = CooperativeTile::new(
                    fragment,
                    (subgroup_rows, subgroup_columns),
                    fragments,
                    stage,
                );
                if !carries_registers(workgroup, tile.registers())
                    || !cooperative_fits(workgroup, shared_bytes, left_stage, right_stage, tile)
                {
                    break;
                }
                staged = Some(tile);
                stage += fragment.depth();
            }
            if let Some(tile) = staged {
                tiles.push(tile);
            }
        }
    }
    tiles.sort_by_key(|tile| std::cmp::Reverse((tile.rows() * tile.columns(), tile.stage())));
    tiles.dedup();
    tiles.truncate(COOPERATIVE_MENU);
    tiles.into_iter().map(MatmulTile::cooperative).collect()
}

fn cooperative_subgroups(subgroups: u32) -> Vec<(u32, u32)> {
    let mut grids = Vec::new();
    let mut rows = 1;
    while rows <= subgroups {
        if subgroups.is_multiple_of(rows) {
            let columns = subgroups / rows;
            if rows <= COOPERATIVE_ASPECT * columns {
                grids.push((rows, columns));
            }
        }
        rows *= 2;
    }
    grids
}

fn cooperative_fits(
    workgroup: u32,
    shared_bytes: u64,
    left_stage: u64,
    right_stage: u64,
    tile: CooperativeTile,
) -> bool {
    let staging = scratch_budget(
        workgroup,
        2 * (left_stage + right_stage) * WORD_BYTES,
        tile.copy() * WORD_BYTES,
    );
    staging + tile.half_bytes() <= shared_bytes
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
    walked: Vec<(u32, MatmulTile)>,
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
        walked: &[(u32, MatmulTile)],
        attention: &[AttentionTile],
    ) -> Self {
        assert!(
            walked.iter().all(|(_, tile)| tile.threads() == workgroup),
            "a device program carries a tile another workgroup stages",
        );
        assert!(
            walked.windows(2).all(|pair| pair[0].0 < pair[1].0),
            "a device program walks its tiles in the order its profile carries them",
        );
        let mut left_stage = 0u64;
        let mut right_stage = 0u64;
        let mut half_stage = 0u64;
        let mut copy = 0u64;
        for (_, tile) in walked {
            if let MatmulTile::Cooperative(cooperative) = tile {
                half_stage = half_stage.max(cooperative.half_panels());
                copy = copy.max(cooperative.copy());
            } else {
                left_stage = left_stage.max(tile.left_stage());
                right_stage = right_stage.max(tile.right_stage());
            }
        }
        Self {
            workgroup,
            budget,
            walked: walked.to_vec(),
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

    pub const fn scratch_half_bytes(&self) -> u64 {
        self.half_stage * 2
    }

    pub fn workgroup_bytes(&self, kinds: &[Kind]) -> u64 {
        CLAIM_BYTES + self.scratch_bytes(kinds) + self.scratch_half_bytes()
    }

    fn scratch_words(&self, kinds: &[Kind]) -> u32 {
        DeviceModule::ALL
            .iter()
            .copied()
            .filter(|module| kinds.iter().any(|kind| kind.carries(*module)))
            .map(|module| self.module_scratch(module))
            .max()
            .unwrap_or(0)
    }

    fn module_scratch(&self, module: DeviceModule) -> u32 {
        match module {
            DeviceModule::MatmulTiles => {
                (2 * (self.left_stage + self.right_stage)).max(self.copy) as u32
            }
            DeviceModule::MatmulWeight => (self.left_stage + self.right_stage) as u32,
            DeviceModule::Attention => self.attention_scratch_words(),
            DeviceModule::Reduce => self.workgroup,
            DeviceModule::Choice => 2 * self.workgroup + 2 * neura_abi::CANDIDATES,
            DeviceModule::Scan => self.workgroup,
            _ => 0,
        }
    }

    pub fn cooperative(&self) -> Option<CooperativeTile> {
        self.walked.iter().find_map(|(_, tile)| match tile {
            MatmulTile::Cooperative(cooperative) => Some(*cooperative),
            _ => None,
        })
    }

    pub const fn half_panels(&self) -> u64 {
        self.half_stage
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

    pub fn attention_scratch_words(&self) -> u32 {
        self.attention
            .iter()
            .map(|tile| tile.scratch_words(self.workgroup))
            .max()
            .unwrap_or(0)
    }

    pub fn walked(&self) -> &[(u32, MatmulTile)] {
        &self.walked
    }
}
