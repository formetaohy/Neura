use neura_abi::Element;
use neura_graph::{Graph, Init, Shape, Triangle, Value};

const DEFAULT_CHUNK: u32 = 64;

#[derive(Clone, Copy)]
pub struct DeltaInput {
    pub query: Value,
    pub key: Value,
    pub value: Value,
    pub beta: Value,
    pub log_decay: Value,
}

struct Chunk {
    query: Value,
    key: Value,
    value: Value,
    beta: Value,
    log_decay: Value,
    extent: u32,
}

pub struct DeltaRule {
    state: Value,
    heads: u32,
    width: u32,
    batch: u32,
    rows: u32,
    chunk: u32,
}

impl DeltaRule {
    pub fn new(graph: &Graph, heads: u32, width: u32, batch: u32, rows: u32) -> Self {
        Self::chunked(graph, heads, width, batch, rows, DEFAULT_CHUNK)
    }

    pub fn chunked(
        graph: &Graph,
        heads: u32,
        width: u32,
        batch: u32,
        rows: u32,
        chunk: u32,
    ) -> Self {
        assert!(
            heads > 0 && width > 0 && batch > 0,
            "a delta rule of {batch} planes of {heads} heads of {width} numbers carries no state",
        );
        assert!(rows > 0, "a delta rule walks no row of {rows}");
        assert!(
            rows == 1 || chunk >= 2,
            "a delta rule walks {rows} rows in chunks of {chunk}, and a chunk of fewer than two rows reaches nothing a recurrent step does not",
        );
        Self {
            state: graph.state(
                Shape::of([heads, batch, width, width]),
                Init::Zero,
                Element::Single,
            ),
            heads,
            width,
            batch,
            rows,
            chunk,
        }
    }

    pub fn forward(&self, graph: &Graph, input: DeltaInput, keep: Option<Value>) -> Value {
        let DeltaInput {
            query,
            key,
            value,
            beta,
            log_decay,
        } = input;
        self.holds(graph, "query", query);
        self.holds(graph, "key", key);
        self.holds(graph, "value", value);
        for (what, held) in [("beta", beta), ("log decay", log_decay)] {
            assert_eq!(
                graph.shape(held).dims(),
                [self.heads, self.batch, self.rows, 1],
                "a delta rule scales the {what} of {} heads over {} planes of {} rows, and value {} walks {:?}",
                self.heads,
                self.batch,
                self.rows,
                held.id(),
                graph.shape(held).dims(),
            );
        }
        let input = match keep {
            Some(keep) => self.kept(graph, input, keep),
            None => input,
        };
        if self.rows == 1 {
            return self.step(graph, input);
        }
        self.walk(graph, input)
    }

    pub fn state(&self) -> Value {
        self.state
    }

    pub fn heads(&self) -> u32 {
        self.heads
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn batch(&self) -> u32 {
        self.batch
    }

    pub fn reset(&self, graph: &Graph, reset: Value) {
        let zero = graph.fill(Shape::scalar(), 0.0);
        graph.copy_into(self.state, graph.select(reset, zero, self.state));
    }

    pub fn rows(&self) -> u32 {
        self.rows
    }

    pub fn chunk(&self) -> u32 {
        self.chunk
    }

    fn holds(&self, graph: &Graph, what: &str, value: Value) {
        assert_eq!(
            graph.shape(value).dims(),
            [self.heads, self.batch, self.rows, self.width],
            "a delta rule reads {what} of {} heads of {} planes of {} rows of {} numbers, and value {} walks {:?}",
            self.heads,
            self.batch,
            self.rows,
            self.width,
            value.id(),
            graph.shape(value).dims(),
        );
    }

    fn kept(&self, graph: &Graph, input: DeltaInput, keep: Value) -> DeltaInput {
        let shape = graph.shape(keep).dims();
        assert!(
            shape[0] == 1
                && shape[1] == self.batch
                && (shape[2] == 1 || shape[2] == self.rows)
                && shape[3] == 1,
            "a delta rule keeps the {shape:?} rows of {} planes where it reads {} rows of {} planes",
            self.batch,
            self.rows,
            self.batch,
        );
        DeltaInput {
            query: graph.mul(input.query, keep),
            key: graph.mul(input.key, keep),
            value: graph.mul(input.value, keep),
            beta: graph.mul(input.beta, keep),
            log_decay: graph.mul(input.log_decay, keep),
        }
    }

    fn step(&self, graph: &Graph, input: DeltaInput) -> Value {
        let DeltaInput {
            query,
            key,
            value,
            beta,
            log_decay,
        } = input;
        let decayed = graph.mul(self.state, graph.exp(log_decay));
        let memory = graph.matmul(key, decayed);
        let delta = graph.mul(graph.sub(value, memory), beta);
        let grown = graph.add(
            decayed,
            graph.matmul(graph.permute(key, [0, 1, 3, 2]), delta),
        );
        let attended = graph.matmul(query, grown);
        graph.copy_into(self.state, grown);
        attended
    }

    fn walk(&self, graph: &Graph, input: DeltaInput) -> Value {
        let mut state = self.state;
        let mut attended = Vec::new();
        let mut start = 0;
        while start < self.rows {
            let extent = self.chunk.min(self.rows - start);
            let rows = |held: Value| graph.slice(held, 2, start, extent);
            let held = Chunk {
                query: rows(input.query),
                key: rows(input.key),
                value: rows(input.value),
                beta: rows(input.beta),
                log_decay: rows(input.log_decay),
                extent,
            };
            let (attended_rows, next) = self.chunk_forward(graph, state, held);
            attended.push(attended_rows);
            state = next;
            start += extent;
        }
        graph.copy_into(self.state, state);
        graph.concat(&attended, 2)
    }

    fn chunk_forward(&self, graph: &Graph, state: Value, chunk: Chunk) -> (Value, Value) {
        let Chunk {
            query,
            key,
            value,
            beta,
            log_decay,
            extent,
        } = chunk;
        let triangle = graph.triangle(extent);
        let key_transposed = graph.permute(key, [0, 1, 3, 2]);
        let cumulative = graph.matmul(triangle.lower(), log_decay);
        let decay = graph.exp(cumulative);
        let difference = graph.sub(cumulative, graph.permute(cumulative, [0, 1, 3, 2]));
        let ratio = graph.exp(graph.min(difference, graph.fill(Shape::scalar(), 0.0)));
        let strict = graph.mul(ratio, triangle.strict());
        let inclusive = graph.mul(ratio, triangle.lower());
        let memory = graph.matmul(key, state);
        let update = graph.mul(graph.sub(value, graph.mul(memory, decay)), beta);
        let keys = graph.matmul(key, key_transposed);
        let lower = graph.mul(graph.mul(keys, strict), beta);
        let solved = self.solve(graph, lower, update, triangle);
        let direct = graph.mul(graph.matmul(query, state), decay);
        let scored = graph.mul(graph.matmul(query, key_transposed), inclusive);
        let attended = graph.add(direct, graph.matmul(scored, solved));
        let carried = graph.exp(graph.slice(cumulative, 2, extent - 1, 1));
        let spread = graph.exp(graph.sub(graph.slice(cumulative, 2, extent - 1, 1), cumulative));
        let grown = graph.add(
            graph.mul(state, carried),
            graph.matmul(key_transposed, graph.mul(solved, spread)),
        );
        (attended, grown)
    }

    fn solve(&self, graph: &Graph, lower: Value, update: Value, triangle: Triangle) -> Value {
        let extent = triangle.extent();
        let identity = triangle.identity();
        let scaled = graph.mul(identity, graph.fill(Shape::scalar(), 2.0));
        let system = graph.add(identity, lower);
        let mut inverse = graph.sub(identity, lower);
        for _ in 0..extent.next_power_of_two().trailing_zeros() {
            let residual = graph.sub(scaled, graph.matmul(system, inverse));
            inverse = graph.matmul(inverse, residual);
        }
        graph.matmul(inverse, update)
    }
}
