use neura_abi::Element;
use neura_graph::{AttentionOptions, Graph, Shape, Value};

pub struct Cache {
    keys: Value,
    values: Value,
    heads: u32,
    width: u32,
    capacity: u32,
}

impl Cache {
    pub fn new(graph: &Graph, heads: u32, width: u32, capacity: u32) -> Self {
        assert!(
            heads > 0 && width > 0,
            "a cache of {heads} heads of {width} numbers holds no key",
        );
        assert!(capacity > 0, "a cache of {capacity} positions holds no key",);
        Self {
            keys: graph.resident(Shape::of([heads, 1, capacity, width]), Element::Single),
            values: graph.resident(Shape::of([heads, 1, capacity, width]), Element::Single),
            heads,
            width,
            capacity,
        }
    }

    pub fn states(&self) -> [Value; 2] {
        [self.keys, self.values]
    }

    pub fn write(&self, graph: &Graph, position: Value, keys: Value, values: Value) {
        for (held, what) in [(keys, "key"), (values, "value")] {
            assert_eq!(
                graph.shape(held).dims(),
                [self.heads, 1, 1, self.width],
                "a {what} of {} heads of {} numbers walks {:?} where a cache of {} positions holds it",
                self.heads,
                self.width,
                graph.shape(held).dims(),
                self.capacity,
            );
        }
        let rows = crate::layout::seat(graph, position, self.heads, self.capacity);
        graph.write_into(self.keys, rows, keys);
        graph.write_into(self.values, rows, values);
    }

    pub fn attend(&self, graph: &Graph, query: Value, position: Value, scale: Value) -> Value {
        let shape = graph.shape(query).dims();
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
                origin: Some(position),
                segments: None,
                reach: None,
                query_segments: None,
                bias: None,
            },
        )
    }
}
