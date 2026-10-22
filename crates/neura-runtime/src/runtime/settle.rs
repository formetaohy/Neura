use super::choice::{Choice, ChoiceMode};
use super::tune::planned_tiles;
use super::{Program, Runtime, Weights};
use neura_graph::Graph;
use neura_plan::{Plan, Product};

const SETTLE_WORKGROUP: u32 = 512;
const SETTLE_WEIGHT_WORDS: u64 = 1 << 20;

impl Runtime {
    pub fn compile_measured(&self, graph: &Graph, weights: &Weights) -> Program {
        match self.remembered(graph, ChoiceMode::Settled) {
            Some(choice) => self.compile_chosen(graph, weights, choice.profile(), choice.chosen()),
            None => self.settle(graph, weights),
        }
    }

    fn settle(&self, graph: &Graph, weights: &Weights) -> Program {
        let planned = self.planned_profile(graph);
        let sketch = Plan::sketch(
            graph,
            self.alignment,
            self.default_profile(),
            self.encoding_bytes,
        );
        let products = sketch.products().to_vec();
        if sketch.dynamic() || !worth_measuring(&products) {
            return self.compile_with(graph, weights, planned);
        }
        let mut profiles = vec![planned];
        for workgroup in [planned.workgroup() / 2, planned.workgroup() * 2] {
            if !(32..=SETTLE_WORKGROUP).contains(&workgroup) {
                continue;
            }
            let resident = self
                .profiles()
                .into_iter()
                .filter(|profile| profile.workgroup() == workgroup)
                .min_by_key(|profile| profile.registers());
            if let Some(profile) = resident
                && !profiles.contains(&profile)
            {
                profiles.push(profile);
            }
        }
        let candidates = profiles
            .into_iter()
            .map(|profile| (profile, planned_tiles(&products, profile)))
            .collect::<Vec<_>>();
        let held = self.hold_state(weights);
        let winner = self.fastest_plan(graph, weights, &candidates);
        let (profile, _) = match winner {
            0 => candidates[0].clone(),
            challenger => self.verified(
                graph,
                weights,
                candidates[0].clone(),
                candidates[challenger].clone(),
            ),
        };
        self.restore_held_state(weights, &held);
        self.remember(graph, &Choice::of(profile, &[]), ChoiceMode::Settled);
        self.compile_chosen(graph, weights, profile, &[])
    }
}

fn worth_measuring(products: &[Product]) -> bool {
    let words = products
        .iter()
        .map(|product| {
            let elements = u64::from(product.planes())
                * u64::from(product.columns())
                * u64::from(product.depth());
            product.weight().element().payload_words(elements)
        })
        .sum::<u64>();
    words >= SETTLE_WEIGHT_WORDS
}
