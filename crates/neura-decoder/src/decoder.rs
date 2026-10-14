use neura_graph::{Graph, Value};

pub struct Decoder {
    token: Value,
    cursor: Value,
    chosen: Value,
    rows: u32,
    end: u32,
}

impl Decoder {
    pub fn greedy(graph: &Graph, token: Value, cursor: Value, logits: Value, end: u32) -> Self {
        let held = graph.shape(logits).dims();
        assert_eq!(
            [held[0], held[1]],
            [1, 1],
            "a decoder reads the logits of one plane, and value {:?} holds {} planes",
            held,
            held[0] * held[1],
        );
        let rows = held[2];
        assert_eq!(
            graph.shape(token).dims(),
            [1, 1, rows, 1],
            "a token of {:?} walks beside the {rows} rows of logits a chunk carries",
            graph.shape(token).dims(),
        );
        assert_eq!(
            graph.shape(cursor).dims(),
            [1, 1, 1, 1],
            "a cursor of {:?} names the position the first row of a chunk stands at, and {rows} rows read the count one number names",
            graph.shape(cursor).dims(),
        );
        let chosen = graph.argmax(logits);
        graph.retain(chosen);
        Self {
            token,
            cursor,
            chosen,
            rows,
            end,
        }
    }

    pub fn token(&self) -> Value {
        self.token
    }

    pub fn cursor(&self) -> Value {
        self.cursor
    }

    pub fn chosen(&self) -> Value {
        self.chosen
    }

    pub fn rows(&self) -> u32 {
        self.rows
    }

    pub fn end(&self) -> u32 {
        self.end
    }
}
