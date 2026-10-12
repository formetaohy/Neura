use neura_graph::{Graph, Value};

pub struct Decoder {
    token: Value,
    position: Value,
    chosen: Value,
    states: Vec<Value>,
    end: u32,
}

impl Decoder {
    pub fn greedy(
        graph: &Graph,
        token: Value,
        position: Value,
        logits: Value,
        states: impl IntoIterator<Item = Value>,
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

    pub fn token(&self) -> Value {
        self.token
    }

    pub fn position(&self) -> Value {
        self.position
    }

    pub fn chosen(&self) -> Value {
        self.chosen
    }

    pub fn states(&self) -> &[Value] {
        &self.states
    }

    pub fn end(&self) -> u32 {
        self.end
    }
}
