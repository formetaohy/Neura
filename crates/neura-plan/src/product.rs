use neura_profile::{MatmulStrategy, MatmulTile, Profile};

const MATMUL_SPLITS_CEILING: u32 = 64;
const MATMUL_SPLIT_BLOCKS: u32 = 4;
const MATMUL_PARTIALS_CEILING: u32 = 1 << 20;
const NARROW_ROWS: u32 = 2;
const BARRIER_SLOTS: u128 = 16;
const STREAMED_LOAD_WEIGHT: u128 = 6;
const STREAMED_TRAFFIC_WEIGHT: u128 = 16;
const COOPERATIVE_STAGE_INSTRUCTIONS: u128 = 3;
const COOPERATIVE_TENSOR_INSTRUCTIONS: u128 = 1;
const COOPERATIVE_COPY_INSTRUCTIONS: u128 = 3;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Product {
    planes: u32,
    rows: u32,
    columns: u32,
    depth: u32,
}

impl Product {
    pub fn of(planes: u32, rows: u32, columns: u32, depth: u32) -> Self {
        assert!(
            planes > 0 && rows > 0 && columns > 0 && depth > 0,
            "a product of {planes} planes, {rows} rows, {columns} columns and {depth} of depth spans no element",
        );
        Self {
            planes,
            rows,
            columns,
            depth,
        }
    }

    pub const fn planes(self) -> u32 {
        self.planes
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

    pub fn planned(self, profile: Profile) -> MatmulTile {
        self.cheapest(profile, |tile| {
            tile.strategy() != MatmulStrategy::Cooperative
        })
        .or_else(|| self.cheapest(profile, |_| true))
        .expect("a profile offers no tile for a product")
    }

    pub fn gathered(self, profile: Profile) -> Option<MatmulTile> {
        self.cheapest(profile, |tile| {
            tile.strategy() == MatmulStrategy::Cooperative
        })
    }

    pub fn shortlist(self, profile: Profile) -> Vec<MatmulTile> {
        let planned = self.planned(profile);
        let mut tiles = vec![planned];
        if let Some(gathered) = self.gathered(profile)
            && gathered != planned
        {
            tiles.push(gathered);
        }
        tiles
    }

    pub(crate) fn splits(self, tile: MatmulTile, profile: Profile) -> u32 {
        let elements = u64::from(self.planes) * u64::from(self.rows) * u64::from(self.columns);
        let tiles = u64::from(self.planes)
            * u64::from(self.rows.div_ceil(tile.rows()))
            * u64::from(self.columns.div_ceil(tile.columns()));
        let splits = u64::from(profile.workgroups()).div_ceil(tiles);
        let splits = splits.min(u64::from(
            self.depth.div_ceil(tile.depth()) / MATMUL_SPLIT_BLOCKS,
        ));
        let splits = splits.min(u64::from(MATMUL_SPLITS_CEILING));
        let splits = splits.min(u64::from(MATMUL_PARTIALS_CEILING) / elements);
        let splits =
            splits.min(u64::from(self.depth) * u64::from(self.rows + self.columns) / elements);
        splits.max(1) as u32
    }

    fn cheapest(self, profile: Profile, keep: impl Fn(&MatmulTile) -> bool) -> Option<MatmulTile> {
        let narrow = self.rows <= NARROW_ROWS
            && profile
                .tiles()
                .iter()
                .any(|tile| tile.strategy() == MatmulStrategy::Streamed);
        let mut chosen = None;
        let mut cheapest = u128::MAX;
        for tile in profile.tiles() {
            if !keep(tile) || (narrow && tile.strategy() == MatmulStrategy::Staged) {
                continue;
            }
            let cost = self.cost(*tile, profile);
            if cost < cheapest {
                cheapest = cost;
                chosen = Some(*tile);
            }
        }
        chosen
    }

    fn cost(self, tile: MatmulTile, profile: Profile) -> u128 {
        let rows = u128::from(self.rows);
        let columns = u128::from(self.columns);
        let depth = u128::from(self.depth);
        let block_rows = u128::from(tile.rows());
        let block_columns = u128::from(tile.columns());
        let tiles =
            u128::from(self.planes) * rows.div_ceil(block_rows) * columns.div_ceil(block_columns);
        let blocks = depth.div_ceil(u128::from(tile.depth()));
        let threads = u128::from(tile.threads());
        let splits = u128::from(self.splits(tile, profile));
        let work = tiles * block_rows * block_columns * depth;
        let cost = match tile.strategy() {
            MatmulStrategy::Staged => {
                let registers = u128::from(tile.registers());
                let operands = u128::from(tile.register_rows() + tile.register_columns());
                let staged = registers + operands;
                let panels =
                    tiles * blocks * (block_rows + block_columns) * u128::from(tile.depth());
                work * staged / registers + panels + tiles * blocks * threads * BARRIER_SLOTS
            }
            MatmulStrategy::Streamed => {
                let registers = u128::from(tile.registers());
                let operands = u128::from(tile.register_rows() + tile.register_columns());
                let bands = rows.div_ceil(block_rows);
                let reads = u128::from(self.planes)
                    * (bands * depth * columns + columns.div_ceil(block_columns) * depth * rows);
                STREAMED_LOAD_WEIGHT * work * operands / registers + STREAMED_TRAFFIC_WEIGHT * reads
            }
            MatmulStrategy::Cooperative => {
                let fragment = u128::from(tile.fragment_rows())
                    * u128::from(tile.fragment_columns())
                    * u128::from(tile.fragment_depth());
                let staged = tiles * (block_rows + block_columns) * depth;
                let tensor = tiles * block_rows * block_columns * depth / fragment;
                let copy = tiles * block_rows * block_columns;
                COOPERATIVE_STAGE_INSTRUCTIONS * staged
                    + COOPERATIVE_TENSOR_INSTRUCTIONS * tensor
                    + COOPERATIVE_COPY_INSTRUCTIONS * copy
                    + tiles * blocks * threads * BARRIER_SLOTS
            }
        };
        cost.div_ceil((tiles * splits).min(u128::from(profile.workgroups())))
    }
}
