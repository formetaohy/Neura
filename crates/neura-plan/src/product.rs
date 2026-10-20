use neura_abi::Element;
use neura_profile::{
    FOLDED_LANES, FOLDED_STAGE, MatmulStrategy, MatmulTile, Profile, Run, folded_splits, run_words,
};

pub(crate) const SPLITS_CEILING: u32 = 64;
const MATMUL_SPLIT_BLOCKS: u32 = 4;
pub(crate) const PARTIALS_CEILING: u64 = 1 << 20;
const NARROW_ROWS: u32 = 2;
const WIDE_COLUMNS: u32 = 64;
const WIDE_FLOOR: u32 = 2048;
const SPLIT_COLUMNS_FLOOR: u32 = 1024;
const SPLIT_DEPTH_FLOOR: u32 = 1024;
const BARRIER_SLOTS: u128 = 16;
const STREAMED_LOAD_WEIGHT: u128 = 6;
const COALESCED_TRAFFIC_WEIGHT: u128 = 4;
const STAGED_RUN_TRAFFIC_WEIGHT: u128 = 1;
const STRIDED_TRAFFIC_WEIGHT: u128 = 16;
const RUN_DECODE_WEIGHT: u128 = 4;
const RUN_COPY_WEIGHT: u128 = 1;
const FOLDED_STAGE_WEIGHT: u128 = 8;
const COOPERATIVE_STAGE_INSTRUCTIONS: u128 = 3;
const COOPERATIVE_TENSOR_INSTRUCTIONS: u128 = 1;
const COOPERATIVE_COPY_INSTRUCTIONS: u128 = 3;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Contiguous {
    Reduction,
    Output,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct Weight {
    element: Element,
    contiguous: Contiguous,
}

impl Weight {
    pub const fn of(element: Element, contiguous: Contiguous) -> Self {
        Self {
            element,
            contiguous,
        }
    }

    pub const fn element(self) -> Element {
        self.element
    }

    pub const fn contiguous(self) -> Contiguous {
        self.contiguous
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Product {
    planes: u32,
    rows: u32,
    columns: u32,
    depth: u32,
    weight: Weight,
}

impl Product {
    pub fn of(planes: u32, rows: u32, columns: u32, depth: u32, weight: Weight) -> Self {
        assert!(
            planes > 0 && rows > 0 && columns > 0 && depth > 0,
            "a product of {planes} planes, {rows} rows, {columns} columns and {depth} of depth spans no element",
        );
        Self {
            planes,
            rows,
            columns,
            depth,
            weight,
        }
    }

    pub const fn weight(self) -> Weight {
        self.weight
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
        if let Some(run) = self.run_walk(profile) {
            return run;
        }
        self.narrow_walk(profile)
            .or_else(|| {
                self.cheapest(profile, |tile| {
                    tile.strategy() != MatmulStrategy::Cooperative
                })
            })
            .or_else(|| self.cheapest_any(profile, |_| true))
            .expect("a profile offers no tile for a product")
    }

    fn run_walk(self, profile: Profile) -> Option<MatmulTile> {
        if self.rows <= NARROW_ROWS
            || self.weight.contiguous != Contiguous::Reduction
            || !self.weight.element().stages_a_run() && !self.weight.element().packs_codes()
        {
            return None;
        }
        self.cheapest(profile, |tile| {
            tile.strategy() == MatmulStrategy::Staged
                && tile.stages_a_run()
                && tile.rows() >= self.rows
                && self.stages(*tile)
        })
    }

    fn stages(self, tile: MatmulTile) -> bool {
        match tile.run() {
            Some(Run::Blocks) => self.weight.element().stages_a_run(),
            Some(Run::Codes) => self.weight.element().packs_codes(),
            None => true,
        }
    }

    fn narrow_walk(self, profile: Profile) -> Option<MatmulTile> {
        if !self.narrow(profile) {
            return None;
        }
        let strategy = match self.weight.contiguous {
            Contiguous::Reduction => MatmulStrategy::Folded,
            Contiguous::Output => MatmulStrategy::Streamed,
        };
        let wide = self.columns >= WIDE_FLOOR
            || (self.columns >= SPLIT_COLUMNS_FLOOR && self.depth > SPLIT_DEPTH_FLOOR);
        if strategy == MatmulStrategy::Folded && wide && self.weight.element().stages_a_run() {
            let widest = profile
                .tiles()
                .iter()
                .copied()
                .filter(|tile| {
                    tile.strategy() == MatmulStrategy::Folded
                        && tile.stages_a_run()
                        && tile.columns() <= WIDE_COLUMNS
                })
                .max_by_key(|tile| tile.columns());
            if let Some(tile) = widest {
                return Some(tile);
            }
        }
        self.cheapest(profile, |tile| tile.strategy() == strategy)
    }

    pub fn gathered(self, profile: Profile) -> Option<MatmulTile> {
        self.cheapest(profile, |tile| {
            tile.strategy() == MatmulStrategy::Cooperative
        })
    }

    pub(crate) fn streamed(self, profile: Profile) -> MatmulTile {
        self.cheapest(profile, |tile| {
            tile.strategy() == MatmulStrategy::Streamed
        })
        .unwrap_or_else(|| {
            panic!("a profile offers no streamed tile for a product, and a streamed tile walks every shape")
        })
    }

    pub(crate) fn staged(self, profile: Profile) -> MatmulTile {
        self.cheapest(profile, |tile| {
            tile.strategy() == MatmulStrategy::Staged && !tile.stages_a_run()
        })
        .unwrap_or_else(|| {
            panic!(
                "a segmented product weighs the rows of every segment through the staged panels a tile stages, and {profile:?} stages no tile",
            )
        })
    }

    pub fn shortlist(self, profile: Profile) -> Vec<MatmulTile> {
        let planned = self.planned(profile);
        let mut tiles = vec![planned];
        if let Some(other) = self.other_narrow(profile, planned) {
            tiles.push(other);
        }
        if let Some(gathered) = self.gathered(profile)
            && !tiles.contains(&gathered)
        {
            tiles.push(gathered);
        }
        tiles
    }

    fn other_narrow(self, profile: Profile, planned: MatmulTile) -> Option<MatmulTile> {
        let wanted = match planned.strategy() {
            MatmulStrategy::Folded => MatmulStrategy::Streamed,
            MatmulStrategy::Streamed => MatmulStrategy::Folded,
            _ => return None,
        };
        self.cheapest(profile, |tile| tile.strategy() == wanted)
    }

    pub(crate) fn splits(self, tile: MatmulTile, profile: Profile) -> u32 {
        if let MatmulTile::Folded(_) = tile {
            let blocks = self.depth.div_ceil(FOLDED_LANES);
            let splits = folded_splits(self.depth);
            assert!(
                blocks <= splits * (FOLDED_STAGE / FOLDED_LANES),
                "a folded product walks {blocks} blocks of {FOLDED_LANES} numbers where {splits} tasks stage {} blocks each",
                FOLDED_STAGE / FOLDED_LANES,
            );
            return splits;
        }
        let elements = u64::from(self.planes) * u64::from(self.rows) * u64::from(self.columns);
        let tiles = u64::from(self.planes)
            * u64::from(self.rows.div_ceil(tile.rows()))
            * u64::from(self.columns.div_ceil(tile.columns()));
        let splits = u64::from(profile.workgroups()).div_ceil(tiles);
        let splits = splits.min(u64::from(
            self.depth.div_ceil(tile.depth()) / MATMUL_SPLIT_BLOCKS,
        ));
        let splits = splits.min(u64::from(SPLITS_CEILING));
        let splits = splits.min(PARTIALS_CEILING / elements);
        let splits =
            splits.min(u64::from(self.depth) * u64::from(self.rows + self.columns) / elements);
        splits.max(1) as u32
    }

    fn cheapest(self, profile: Profile, keep: impl Fn(&MatmulTile) -> bool) -> Option<MatmulTile> {
        self.cheapest_any(profile, |tile| keep(tile) && self.walks(*tile, profile))
    }

    fn cheapest_any(
        self,
        profile: Profile,
        keep: impl Fn(&MatmulTile) -> bool,
    ) -> Option<MatmulTile> {
        let mut chosen = None;
        let mut cheapest = u128::MAX;
        for tile in profile.tiles() {
            if !keep(tile) {
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

    fn walks(self, tile: MatmulTile, profile: Profile) -> bool {
        match tile.strategy() {
            MatmulStrategy::Staged => {
                !self.narrow(profile)
                    && (!tile.stages_a_run()
                        || (self.weight.contiguous == Contiguous::Reduction && self.stages(tile)))
            }
            MatmulStrategy::Streamed | MatmulStrategy::Cooperative => true,
            MatmulStrategy::Folded => {
                self.narrow(profile)
                    && self.weight.contiguous == Contiguous::Reduction
                    && (!tile.stages_a_run() || self.weight.element().stages_a_run())
            }
        }
    }

    fn narrow(self, profile: Profile) -> bool {
        self.rows <= NARROW_ROWS
            && profile.tiles().iter().any(|tile| {
                matches!(
                    tile.strategy(),
                    MatmulStrategy::Streamed | MatmulStrategy::Folded
                )
            })
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
            MatmulStrategy::Staged if tile.stages_a_run() => {
                let registers = u128::from(tile.registers());
                let operands = u128::from(tile.register_rows())
                    + RUN_DECODE_WEIGHT * u128::from(tile.register_columns());
                let staged = registers + operands;
                let run = tile
                    .run()
                    .expect("a staged run stages the codes of the element it weighs");
                let words = block_columns * u128::from(run_words(tile.depth(), run));
                let panels = (tiles * blocks * (block_rows * depth + words * RUN_COPY_WEIGHT)) / 4;
                work * staged / registers + panels + tiles * blocks * threads * BARRIER_SLOTS
            }
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
                STREAMED_LOAD_WEIGHT * work * operands / registers + self.traffic(reads)
            }
            MatmulStrategy::Folded => {
                let registers = u128::from(tile.registers());
                let operands = u128::from(tile.registers() + tile.operands());
                let bands = rows.div_ceil(block_rows);
                let reads = u128::from(self.planes)
                    * (bands * depth * columns + columns.div_ceil(block_columns) * depth * rows);
                let traffic = match tile.stages_a_run() {
                    true => STAGED_RUN_TRAFFIC_WEIGHT,
                    false => COALESCED_TRAFFIC_WEIGHT,
                };
                STREAMED_LOAD_WEIGHT * work * operands / registers
                    + traffic * reads
                    + FOLDED_STAGE_WEIGHT
                        * (tiles * depth.min(u128::from(FOLDED_STAGE))
                            + tiles * u128::from(FOLDED_LANES))
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

    fn traffic(self, reads: u128) -> u128 {
        let weight = match self.weight.contiguous {
            Contiguous::Output => COALESCED_TRAFFIC_WEIGHT,
            Contiguous::Reduction => STRIDED_TRAFFIC_WEIGHT,
        };
        weight * reads
    }
}
