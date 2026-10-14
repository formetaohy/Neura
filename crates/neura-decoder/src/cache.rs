use neura_abi::{Element, WORD_BYTES};
use neura_graph::{AttentionOptions, Graph, Shape, Value};

pub struct Cache {
    keys: Value,
    values: Value,
    heads: u32,
    width: u32,
    capacity: u32,
    element: Element,
}

impl Cache {
    pub fn new(graph: &Graph, heads: u32, width: u32, capacity: u32) -> Self {
        Self::narrow(graph, heads, width, capacity, Element::Single)
    }

    pub fn narrow(graph: &Graph, heads: u32, width: u32, capacity: u32, element: Element) -> Self {
        assert!(
            heads > 0 && width > 0,
            "a cache of {heads} heads of {width} numbers holds no key",
        );
        assert!(capacity > 0, "a cache of {capacity} positions holds no key",);
        let quantized = element.block().is_some();
        assert!(
            (!quantized || element.packs_on_device())
                && (element.narrow() || element == Element::Single),
            "a cache of {} storage holds the keys a device writes in place, and the {} storage of a cache walks one number per word or the words a device packs",
            element.name(),
            element.name(),
        );
        let rows = Shape::of([heads, 1, capacity, width]);
        for at in rows.dims() {
            assert!(
                at > 0,
                "a cache of {:?} walks no number on an axis its shape leaves empty",
                rows.dims(),
            );
        }
        assert!(
            element.payload_words(u64::from(width)) * WORD_BYTES
                == element.payload_bytes(u64::from(width)),
            "a cache takes a written row in whole words, and a row of {width} {} numbers spans {} bytes",
            element.name(),
            element.payload_bytes(u64::from(width)),
        );
        Self {
            keys: graph.resident(rows, element),
            values: graph.resident(rows, element),
            heads,
            width,
            capacity,
            element,
        }
    }

    pub fn element(&self) -> Element {
        self.element
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
        for (held, stored) in [(keys, self.keys), (values, self.values)] {
            let packed = match self.element {
                Element::Single => held,
                block if block.block().is_some() => graph.block_quantize(held, block),
                narrow => graph.cast(held, narrow),
            };
            graph.write_into(stored, rows, packed);
        }
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
