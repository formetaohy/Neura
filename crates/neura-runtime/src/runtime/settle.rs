use super::choice::{Choice, ChoiceMode};
use super::tune::planned_tiles;
use super::{Program, Runtime, Weights};
use neura_graph::Graph;
use neura_plan::{Plan, Product};

const SETTLE_WORKGROUP: u32 = 256;
const SETTLE_PREFILL_WORKGROUP: u32 = 512;
const SETTLE_PREFILL_ROWS: u32 = 8;
const SETTLE_WEIGHT_WORDS: u64 = 1 << 20;
const SETTLE_WARMUP: u32 = 1;
const SETTLE_ROUNDS: u32 = 4;

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
        let ceiling = match products
            .iter()
            .all(|product| product.rows() >= SETTLE_PREFILL_ROWS)
        {
            true => SETTLE_PREFILL_WORKGROUP,
            false => SETTLE_WORKGROUP,
        };
        if sketch.dynamic() || !worth_measuring(&products) || planned.workgroup() > ceiling {
            return self.compile_with(graph, weights, planned);
        }
        let mut profiles = vec![planned];
        for workgroup in [planned.workgroup() / 2, planned.workgroup() * 2] {
            if !(32..=ceiling).contains(&workgroup) {
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
        let held = self.hold_state(weights);
        let mut measured = Vec::new();
        let mut fastest = f64::INFINITY;
        let mut winner = 0;
        for profile in profiles {
            let chosen = planned_tiles(&products, profile);
            let words = self.plan_words(graph, weights, profile, &chosen);
            if !self.heap.holds(words) {
                continue;
            }
            let program = self.compile_chosen(graph, weights, profile, &chosen);
            for _ in 0..SETTLE_WARMUP {
                self.run(&program);
            }
            let mut seconds = f64::INFINITY;
            for _ in 0..SETTLE_ROUNDS {
                seconds = seconds.min(self.run(&program).seconds());
            }
            if seconds < fastest {
                fastest = seconds;
                winner = measured.len();
            }
            measured.push((profile, chosen, program));
        }
        assert!(
            !measured.is_empty(),
            "a plan worth measuring fits the heap at one of its profiles",
        );
        let (profile, chosen, program) = measured.swap_remove(winner);
        self.restore_held_state(weights, &held);
        self.remember(graph, &Choice::of(profile, &chosen), ChoiceMode::Settled);
        program
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
