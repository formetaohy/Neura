use neura_graph::{Graph, Value};

pub struct Decoder<'g> {
    token: Value<'g>,
    position: Value<'g>,
    chosen: Value<'g>,
    states: Vec<Value<'g>>,
    end: u32,
}

impl<'g> Decoder<'g> {
    pub fn greedy(
        graph: &Graph<'g>,
        token: Value<'g>,
        position: Value<'g>,
        logits: Value<'g>,
        states: impl IntoIterator<Item = Value<'g>>,
        end: u32,
    ) -> Self {
        for (held, what) in [(token, "token"), (position, "position")] {
            assert_eq!(
                graph.shape(held).dims(),
                [1, 1, 1, 1],
                "a {what} walks {:?} where one number walks [1, 1, 1, 1]",
                graph.shape(held).dims(),
            );
        }
        let chosen = graph.argmax(logits);
        graph.retain(chosen);
        Self {
            token,
            position,
            chosen,
            states: states.into_iter().collect(),
            end,
        }
    }

    pub fn token(&self) -> Value<'g> {
        self.token
    }

    pub fn position(&self) -> Value<'g> {
        self.position
    }

    pub fn chosen(&self) -> Value<'g> {
        self.chosen
    }

    pub fn states(&self) -> &[Value<'g>] {
        &self.states
    }

    pub fn end(&self) -> u32 {
        self.end
    }
}
