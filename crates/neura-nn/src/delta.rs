use neura_abi::Element;
use neura_graph::{Graph, Init, Shape, Triangle, Value};

const DEFAULT_CHUNK: u32 = 64;

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
    rows: u32,
    chunk: u32,
}

impl DeltaRule {
    pub fn new(graph: &Graph, heads: u32, width: u32, rows: u32) -> Self {
        Self::chunked(graph, heads, width, rows, DEFAULT_CHUNK)
    }

    pub fn chunked(graph: &Graph, heads: u32, width: u32, rows: u32, chunk: u32) -> Self {
        assert!(
            heads > 0 && width > 0,
            "a delta rule of {heads} heads of {width} numbers carries no state",
        );
        assert!(rows > 0, "a delta rule walks no row of {rows}");
        assert!(
            rows == 1 || chunk >= 2,
            "a delta rule walks {rows} rows in chunks of {chunk}, and a chunk of fewer than two rows reaches nothing a recurrent step does not",
        );
        Self {
            state: graph.state(
                Shape::of([heads, 1, width, width]),
                Init::Zero,
                Element::Single,
            ),
            heads,
            width,
            rows,
            chunk,
        }
    }

    pub fn forward(
        &self,
        graph: &Graph,
        query: Value,
        key: Value,
        value: Value,
        beta: Value,
        log_decay: Value,
    ) -> Value {
        self.holds(graph, "query", query);
        self.holds(graph, "key", key);
        self.holds(graph, "value", value);
        for (what, held) in [("beta", beta), ("log decay", log_decay)] {
            assert_eq!(
                graph.shape(held).dims(),
                [self.heads, 1, self.rows, 1],
                "a delta rule scales the {what} of {} heads over {} rows, and value {} walks {:?}",
                self.heads,
                self.rows,
                held.id(),
                graph.shape(held).dims(),
            );
        }
        if self.rows == 1 {
            return self.step(graph, query, key, value, beta, log_decay);
        }
        self.walk(graph, query, key, value, beta, log_decay)
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

    pub fn rows(&self) -> u32 {
        self.rows
    }

    pub fn chunk(&self) -> u32 {
        self.chunk
    }

    fn holds(&self, graph: &Graph, what: &str, value: Value) {
        assert_eq!(
            graph.shape(value).dims(),
            [self.heads, 1, self.rows, self.width],
            "a delta rule reads {what} of {} heads of {} rows of {} numbers, and value {} walks {:?}",
            self.heads,
            self.rows,
            self.width,
            value.id(),
            graph.shape(value).dims(),
        );
    }

    fn step(
        &self,
        graph: &Graph,
        query: Value,
        key: Value,
        value: Value,
        beta: Value,
        log_decay: Value,
    ) -> Value {
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

    fn walk(
        &self,
        graph: &Graph,
        query: Value,
        key: Value,
        value: Value,
        beta: Value,
        log_decay: Value,
    ) -> Value {
        let mut state = self.state;
        let mut attended = Vec::new();
        let mut start = 0;
        while start < self.rows {
            let extent = self.chunk.min(self.rows - start);
            let rows = |held: Value| graph.slice(held, 2, start, extent);
            let held = Chunk {
                query: rows(query),
                key: rows(key),
                value: rows(value),
                beta: rows(beta),
                log_decay: rows(log_decay),
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
