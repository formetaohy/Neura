use neura_abi::Element;
use neura_graph::{AttentionOptions, Graph, Init, Shape, Value};

pub struct Cache {
    keys: Value,
    values: Value,
    seat: Value,
    heads: u32,
    width: u32,
    capacity: u32,
    batch: u32,
    rows: u32,
}

impl Cache {
    pub fn new(
        graph: &Graph,
        heads: u32,
        width: u32,
        capacity: u32,
        batch: u32,
        rows: u32,
    ) -> Self {
        assert!(
            heads > 0 && width > 0 && batch > 0,
            "a cache of {batch} planes of {heads} heads of {width} numbers holds no key",
        );
        assert!(capacity > 0, "a cache of {capacity} positions holds no key");
        assert!(
            rows > 0 && rows <= capacity,
            "a cache of {capacity} positions takes {rows} rows at a time, and a chunk of more rows than the cache holds walks past the slots it fills",
        );
        let held = Shape::of([heads, batch, capacity, width]);
        Self {
            keys: graph.state(held, Init::Zero, Element::Single),
            values: graph.state(held, Init::Zero, Element::Single),
            seat: crate::layout::seat(graph, heads, batch, capacity, rows),
            heads,
            width,
            capacity,
            batch,
            rows,
        }
    }

    pub fn states(&self) -> [Value; 2] {
        [self.keys, self.values]
    }

    pub fn rows(&self) -> u32 {
        self.rows
    }

    pub fn batch(&self) -> u32 {
        self.batch
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    pub fn reset(&self, graph: &Graph, reset: Value) {
        let zero = graph.fill(Shape::scalar(), 0.0);
        for held in self.states() {
            graph.copy_into(held, graph.select(reset, zero, held));
        }
    }

    pub fn write(&self, graph: &Graph, cursor: Value, keys: Value, values: Value) {
        for (held, what) in [(keys, "key"), (values, "value")] {
            assert_eq!(
                graph.shape(held).dims(),
                [self.heads, self.batch, self.rows, self.width],
                "a {what} of {} heads of {} planes walks {:?} where a cache of {:?} holds {} planes of {} rows of a chunk",
                self.heads,
                self.batch,
                graph.shape(held).dims(),
                graph.shape(cursor).dims(),
                self.batch,
                self.rows,
            );
        }
        let slots = graph.add(self.seat, cursor);
        for (held, stored) in [(keys, self.keys), (values, self.values)] {
            graph.write_into(stored, slots, held);
        }
    }

    pub fn attend(&self, graph: &Graph, query: Value, cursor: Value, scale: Value) -> Value {
        let shape = graph.shape(query).dims();
        assert_eq!(
            shape[1], self.batch,
            "a query of {} planes weighs the keys a cache of {} planes writes, and {:?} walks another count",
            self.batch, self.batch, shape,
        );
        assert_eq!(
            shape[2], self.rows,
            "a query of {} rows weighs the keys a cache of {} rows writes, and {:?} walks another count",
            shape[2], self.rows, shape,
        );
        assert_eq!(
            shape[3], self.width,
            "a query of {} numbers walks {shape:?} where the cache it reads holds keys of {} numbers",
            shape[3], self.width,
        );
        assert!(
            shape[0].is_multiple_of(self.heads),
            "a query of {} heads reads a cache of {} key heads, and every key head serves a whole group of queries",
            shape[0],
            self.heads,
        );
        graph.attention(
            query,
            self.keys,
            self.values,
            AttentionOptions {
                scale: Some(scale),
                causal: true,
                origin: Some(cursor),
                segments: None,
                reach: None,
                query_segments: None,
                bias: None,
            },
        )
    }
}
