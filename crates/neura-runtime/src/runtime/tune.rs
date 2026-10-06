use super::{Program, Runtime, Weights};
use neura_abi::{Element, WORD_BYTES};
use neura_graph::{Graph, Init, Shape};
use neura_plan::{Layout, Plan, Product};
use neura_profile::{MatmulTile, Profile};

const TILE_WARMUP: u32 = 1;
const TILE_ROUNDS: u32 = 4;
const PLAN_WARMUP: u32 = 1;
const PLAN_ROUNDS: u32 = 4;

impl Runtime {
    pub fn tune(&self, graph: &Graph, weights: &Weights) -> Program {
        let scratch = graph.updates_weights().then(|| self.scratch_weights(graph));
        let measuring = scratch.as_ref().unwrap_or(weights);
        let products = self.products(graph);
        let default = self.default_profile();
        let mut candidates = Vec::new();
        for profile in self.profiles() {
            let planned = planned_tiles(&products, profile);
            let measured = self.measured_tiles(&products, profile);
            candidates.push((profile, planned.clone()));
            if measured != planned {
                candidates.push((profile, measured));
            }
        }
        let index = candidates
            .iter()
            .position(|(profile, _)| *profile == default)
            .expect("the device offers the profile it balances on");
        candidates.swap(0, index);
        let winner = self.fastest_plan(graph, measuring, &candidates);
        let (profile, chosen) = if winner == 0 {
            candidates[0].clone()
        } else {
            self.verified(
                graph,
                measuring,
                candidates[0].clone(),
                candidates[winner].clone(),
            )
        };
        self.compile_chosen(graph, weights, profile, &chosen)
    }

    fn products(&self, graph: &Graph) -> Vec<Product> {
        Plan::of(graph, self.alignment, self.default_profile())
            .products()
            .to_vec()
    }

    fn fastest_plan(
        &self,
        graph: &Graph,
        weights: &Weights,
        candidates: &[(Profile, Vec<(Product, MatmulTile)>)],
    ) -> usize {
        let reference = self.compile_chosen(graph, weights, candidates[0].0, &candidates[0].1);
        self.run(&reference);
        let mut best = 0;
        let mut seconds = 1.0;
        let mut batch = Vec::new();
        let mut at = 1;
        while at < candidates.len() {
            let (profile, chosen) = &candidates[at];
            let words =
                Plan::chosen(graph, self.alignment, *profile, chosen).tensor_bytes() / WORD_BYTES;
            if !self.heap.holds(words) {
                self.score(&reference, &batch, &mut best, &mut seconds);
                batch.clear();
                if !self.heap.holds(words) {
                    at += 1;
                    continue;
                }
            }
            let program = self.compile_chosen(graph, weights, *profile, chosen);
            batch.push((at, program));
            at += 1;
        }
        self.score(&reference, &batch, &mut best, &mut seconds);
        best
    }

    fn score(
        &self,
        reference: &Program,
        batch: &[(usize, Program)],
        best: &mut usize,
        seconds: &mut f64,
    ) {
        if batch.is_empty() {
            return;
        }
        for (_, program) in batch {
            for _ in 0..PLAN_WARMUP {
                self.run(program);
            }
        }
        let mut ratios = vec![f64::MAX; batch.len()];
        for round in 0..PLAN_ROUNDS {
            if round.is_multiple_of(2) {
                let reference_seconds = self.run(reference).seconds();
                for (ratio, (_, program)) in ratios.iter_mut().zip(batch) {
                    *ratio = ratio.min(self.run(program).seconds() / reference_seconds);
                }
            } else {
                let measured = batch
                    .iter()
                    .map(|(_, program)| self.run(program).seconds())
                    .collect::<Vec<_>>();
                let reference_seconds = self.run(reference).seconds();
                for (ratio, measured) in ratios.iter_mut().zip(&measured) {
                    *ratio = ratio.min(measured / reference_seconds);
                }
            }
        }
        for (ratio, (index, _)) in ratios.into_iter().zip(batch) {
            if ratio < *seconds {
                *seconds = ratio;
                *best = *index;
            }
        }
    }

    fn verified(
        &self,
        graph: &Graph,
        weights: &Weights,
        incumbent: (Profile, Vec<(Product, MatmulTile)>),
        challenger: (Profile, Vec<(Product, MatmulTile)>),
    ) -> (Profile, Vec<(Product, MatmulTile)>) {
        let incumbent_words = Plan::chosen(graph, self.alignment, incumbent.0, &incumbent.1)
            .tensor_bytes()
            / WORD_BYTES;
        let challenger_words = Plan::chosen(graph, self.alignment, challenger.0, &challenger.1)
            .tensor_bytes()
            / WORD_BYTES;
        if !self.heap.holds(incumbent_words + challenger_words) {
            return challenger;
        }
        let incumbent_program = self.compile_chosen(graph, weights, incumbent.0, &incumbent.1);
        let challenger_program = self.compile_chosen(graph, weights, challenger.0, &challenger.1);
        for _ in 0..PLAN_WARMUP {
            self.run(&incumbent_program);
            self.run(&challenger_program);
        }
        let mut incumbent_seconds = f64::MAX;
        let mut challenger_seconds = f64::MAX;
        for round in 0..PLAN_ROUNDS {
            if round.is_multiple_of(2) {
                incumbent_seconds = incumbent_seconds.min(self.run(&incumbent_program).seconds());
                challenger_seconds =
                    challenger_seconds.min(self.run(&challenger_program).seconds());
            } else {
                challenger_seconds =
                    challenger_seconds.min(self.run(&challenger_program).seconds());
                incumbent_seconds = incumbent_seconds.min(self.run(&incumbent_program).seconds());
            }
        }
        if challenger_seconds < incumbent_seconds {
            challenger
        } else {
            incumbent
        }
    }

    fn measured_tiles(&self, products: &[Product], profile: Profile) -> Vec<(Product, MatmulTile)> {
        products
            .iter()
            .map(|product| {
                let tile = self
                    .measured_tile(*product, profile)
                    .unwrap_or_else(|| product.planned(profile));
                (*product, tile)
            })
            .collect()
    }

    fn measured_tile(&self, product: Product, profile: Profile) -> Option<MatmulTile> {
        let graph = product_graph(product);
        let layout = Layout::of(&graph, self.alignment);
        let tensors = Plan::of(&graph, self.alignment, profile).tensor_bytes() / WORD_BYTES;
        let tiles = product.shortlist(profile);
        if !self.heap.holds(layout.words() + tensors) {
            return None;
        }
        let weights = self.weights(&graph);
        let mut fastest = None;
        let mut seconds = f64::MAX;
        if self
            .heap
            .holds(layout.words() + tensors * tiles.len() as u64)
        {
            let programs = tiles
                .iter()
                .map(|tile| self.program(&graph, &weights, *tile))
                .collect::<Vec<_>>();
            for program in &programs {
                self.run(program);
            }
            for round in 0..TILE_ROUNDS {
                let order: Vec<usize> = if round.is_multiple_of(2) {
                    (0..programs.len()).collect()
                } else {
                    (0..programs.len()).rev().collect()
                };
                for index in order {
                    let elapsed = self.run(&programs[index]).seconds();
                    if elapsed < seconds {
                        seconds = elapsed;
                        fastest = Some(tiles[index]);
                    }
                }
            }
        } else {
            for tile in &tiles {
                let program = self.program(&graph, &weights, *tile);
                for _ in 0..TILE_WARMUP {
                    self.run(&program);
                }
                for _ in 0..TILE_ROUNDS {
                    let elapsed = self.run(&program).seconds();
                    if elapsed < seconds {
                        seconds = elapsed;
                        fastest = Some(*tile);
                    }
                }
            }
        }
        fastest
    }

    fn program(&self, graph: &Graph, weights: &Weights, tile: MatmulTile) -> Program {
        self.compile_chosen(graph, weights, Profile::of(&[tile]), &[])
    }
}

fn planned_tiles(products: &[Product], profile: Profile) -> Vec<(Product, MatmulTile)> {
    products
        .iter()
        .map(|product| (*product, product.planned(profile)))
        .collect()
}

fn product_graph(product: Product) -> Graph<'static> {
    let graph = Graph::new();
    let left = graph.parameter(
        Shape::of([product.planes(), 1, product.rows(), product.depth()]),
        Init::Zero,
        Element::Single,
    );
    let right = graph.parameter(
        Shape::of([product.planes(), 1, product.depth(), product.columns()]),
        Init::Zero,
        Element::Single,
    );
    graph.retain(graph.matmul(left, right));
    graph
}
