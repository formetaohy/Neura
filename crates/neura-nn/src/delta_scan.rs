use neura_abi::Element;
use neura_graph::{DeltaInput, Graph, Shape, Triangle, Value};

const CHUNK: u32 = 64;

pub struct DeltaScan {
    heads: u32,
    width: u32,
    batch: u32,
    rows: u32,
    chunk: u32,
}

pub struct Scanned {
    pub attended: Value,
    pub state: Value,
}

impl DeltaScan {
    pub fn new(heads: u32, width: u32, batch: u32, rows: u32) -> Self {
        Self::chunked(heads, width, batch, rows, CHUNK)
    }

    pub fn chunked(heads: u32, width: u32, batch: u32, rows: u32, chunk: u32) -> Self {
        assert!(
            heads > 0 && width > 0 && batch > 0,
            "a delta scan of {batch} planes of {heads} heads of {width} numbers carries no state",
        );
        assert!(rows > 0, "a delta scan walks no row of {rows}");
        assert!(
            chunk > 0 && (chunk >= 2 || rows == 1),
            "a delta scan walks {rows} rows in chunks of {chunk}, and a chunk of fewer than two rows reaches nothing a recurrent step does not",
        );
        Self {
            heads,
            width,
            batch,
            rows,
            chunk,
        }
    }

    pub fn forward(&self, graph: &Graph, state: Value, input: DeltaInput) -> Scanned {
        let DeltaInput {
            query,
            key,
            value,
            beta,
            log_decay,
        } = input;
        assert_eq!(
            graph.shape(state).dims(),
            [self.heads, self.batch, self.width, self.width],
            "a delta scan walks the square state of {} heads over {} planes of {} numbers, and value {} walks {:?}",
            self.heads,
            self.batch,
            self.width,
            state.id(),
            graph.shape(state).dims(),
        );
        for (what, held) in [("query", query), ("key", key), ("value", value)] {
            assert_eq!(
                graph.shape(held).dims(),
                [self.heads, self.batch, self.rows, self.width],
                "a delta scan walks a {what} of {} heads over {} planes of {} rows of {} numbers, and value {} walks {:?}",
                self.heads,
                self.batch,
                self.rows,
                self.width,
                held.id(),
                graph.shape(held).dims(),
            );
        }
        for (what, held) in [("beta", beta), ("log decay", log_decay)] {
            assert_eq!(
                graph.shape(held).dims(),
                [self.heads, self.batch, self.rows, 1],
                "a delta scan scales the {what} of {} heads over {} planes of {} rows, and value {} walks {:?}",
                self.heads,
                self.batch,
                self.rows,
                held.id(),
                graph.shape(held).dims(),
            );
        }
        for held in [state, query, key, value, beta, log_decay] {
            assert_eq!(
                graph.element(held),
                Element::Single,
                "a delta scan walks single precision numbers, and value {} holds {}",
                held.id(),
                graph.element(held).name(),
            );
        }
        let mut carried = state;
        let mut attended = Vec::new();
        let mut start = 0;
        while start < self.rows {
            let extent = self.chunk.min(self.rows - start);
            let rows = |held: Value| graph.slice(held, 2, start, extent);
            let chunk = Chunk {
                query: rows(query),
                key: rows(key),
                value: rows(value),
                beta: rows(beta),
                log_decay: rows(log_decay),
                extent,
            };
            let (held, grown) = self.chunk(graph, carried, chunk);
            attended.push(held);
            carried = grown;
            start += extent;
        }
        Scanned {
            attended: match attended.len() {
                1 => attended[0],
                _ => graph.concat(&attended, 2),
            },
            state: carried,
        }
    }

    fn chunk(&self, graph: &Graph, state: Value, chunk: Chunk) -> (Value, Value) {
        let Chunk {
            query,
            key,
            value,
            beta,
            log_decay,
            extent,
        } = chunk;
        let triangle = graph.triangle(extent);
        let cumulative = graph.matmul(triangle.lower(), log_decay);
        let decay = graph.exp(cumulative);
        let difference = graph.sub(cumulative, graph.permute(cumulative, [0, 1, 3, 2]));
        let ratio = graph.exp(graph.min(difference, graph.fill(Shape::scalar(), 0.0)));
        let strict = graph.mul(ratio, triangle.strict());
        let inclusive = graph.mul(ratio, triangle.lower());
        let memory = graph.matmul(key, state);
        let update = graph.mul(graph.sub(value, graph.mul(memory, decay)), beta);
        let keys = graph.matmul(key, graph.permute(key, [0, 1, 3, 2]));
        let lower = graph.mul(graph.mul(keys, strict), beta);
        let solved = solve(graph, lower, update, triangle);
        let direct = graph.mul(graph.matmul(query, state), decay);
        let scored = graph.mul(
            graph.matmul(query, graph.permute(key, [0, 1, 3, 2])),
            inclusive,
        );
        let held = graph.add(direct, graph.matmul(scored, solved));
        let last = graph.slice(cumulative, 2, extent - 1, 1);
        let carried = graph.exp(last);
        let spread = graph.exp(graph.sub(last, cumulative));
        let grown = graph.add(
            graph.mul(state, carried),
            graph.matmul(graph.permute(key, [0, 1, 3, 2]), graph.mul(solved, spread)),
        );
        (held, grown)
    }
}

struct Chunk {
    query: Value,
    key: Value,
    value: Value,
    beta: Value,
    log_decay: Value,
    extent: u32,
}

fn solve(graph: &Graph, lower: Value, update: Value, triangle: Triangle) -> Value {
    let identity = triangle.identity();
    let scaled = graph.mul(identity, graph.fill(Shape::scalar(), 2.0));
    let system = graph.add(identity, lower);
    let mut inverse = graph.sub(identity, lower);
    for _ in 0..triangle.extent().next_power_of_two().trailing_zeros() {
        let residual = graph.sub(scaled, graph.matmul(system, inverse));
        inverse = graph.matmul(inverse, residual);
    }
    graph.matmul(inverse, update)
}
