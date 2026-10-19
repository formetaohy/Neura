use neura_abi::Element;
use neura_graph::{AttentionOptions, Graph, Init, Shape, Value};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Seat {
    Planes,
    Shared,
}

struct Shared {
    keys: Value,
    values: Value,
}

#[derive(Clone, Copy)]
struct CacheShape {
    heads: u32,
    width: u32,
    capacity: u32,
    batch: u32,
    rows: u32,
}

pub struct Cache {
    keys: Value,
    values: Value,
    shared: Option<Shared>,
    heads: u32,
    width: u32,
    capacity: u32,
    batch: u32,
    rows: u32,
    seat: Seat,
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
        Self::of(
            graph,
            CacheShape {
                heads,
                width,
                capacity,
                batch,
                rows,
            },
            false,
            Seat::Planes,
        )
    }

    pub fn with_shared(
        graph: &Graph,
        heads: u32,
        width: u32,
        capacity: u32,
        batch: u32,
        rows: u32,
    ) -> Self {
        Self::of(
            graph,
            CacheShape {
                heads,
                width,
                capacity,
                batch,
                rows,
            },
            true,
            Seat::Planes,
        )
    }

    pub fn shared(
        graph: &Graph,
        heads: u32,
        width: u32,
        capacity: u32,
        batch: u32,
        rows: u32,
    ) -> Self {
        Self::of(
            graph,
            CacheShape {
                heads,
                width,
                capacity,
                batch,
                rows,
            },
            true,
            Seat::Shared,
        )
    }

    fn of(graph: &Graph, shape: CacheShape, declares_shared: bool, seat: Seat) -> Self {
        let CacheShape {
            heads,
            width,
            capacity,
            batch,
            rows,
        } = shape;
        assert!(
            heads > 0 && width > 0 && batch > 0,
            "a cache of {batch} planes of {heads} heads of {width} numbers holds no key",
        );
        assert!(capacity > 0, "a cache of {capacity} positions holds no key");
        assert!(
            rows > 0 && rows <= capacity,
            "a cache of {capacity} positions takes {rows} rows at a time, and a chunk of more rows than the cache holds walks past the slots it fills",
        );
        assert!(
            declares_shared || seat == Seat::Planes,
            "a cache that walks a shared prefix declares the prefix it walks",
        );
        let held = Shape::of([heads, batch, capacity, width]);
        let keys = graph.state(held, Init::Zero, Element::Single);
        let values = graph.state(held, Init::Zero, Element::Single);
        let shared = declares_shared.then(|| Shared {
            keys: graph.state(
                Shape::of([heads, 1, capacity, width]),
                Init::Zero,
                Element::Single,
            ),
            values: graph.state(
                Shape::of([heads, 1, capacity, width]),
                Init::Zero,
                Element::Single,
            ),
        });
        Self {
            keys,
            values,
            shared,
            heads,
            width,
            capacity,
            batch: match seat {
                Seat::Planes => batch,
                Seat::Shared => 1,
            },
            rows,
            seat,
        }
    }

    pub fn states(&self) -> [Value; 2] {
        [self.keys, self.values]
    }

    pub fn shared_states(&self) -> Option<[Value; 2]> {
        self.shared
            .as_ref()
            .map(|shared| [shared.keys, shared.values])
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

    pub fn reset(&self, graph: &Graph, reset: Value, share: Option<Value>) {
        assert_eq!(
            self.seat,
            Seat::Planes,
            "a cache resets the planes it writes, and this cache walks the shared prefix",
        );
        let zero = graph.fill(Shape::scalar(), 0.0);
        match (&self.shared, share) {
            (Some(shared), Some(share)) => {
                let shape = graph.shape(self.keys);
                for (held, shared) in [(self.keys, shared.keys), (self.values, shared.values)] {
                    let prefix = graph.broadcast_to(graph.slice(shared, 1, 0, 1), shape);
                    let blank = graph.select(reset, zero, held);
                    graph.copy_into(held, graph.select(share, prefix, blank));
                }
            }
            (None, None) => {
                for held in self.states() {
                    graph.copy_into(held, graph.select(reset, zero, held));
                }
            }
            _ => panic!(
                "a cache resets with the prefix it declares, and this cache declares {}",
                self.shared.is_some(),
            ),
        }
    }

    pub fn clear_shared(&self, graph: &Graph, reset: Value) {
        assert_eq!(
            self.seat,
            Seat::Shared,
            "a shared cache clears the prefix it writes, and this cache walks the planes",
        );
        let zero = graph.fill(Shape::scalar(), 0.0);
        for held in self
            .shared_states()
            .expect("a shared cache declares the prefix it clears")
        {
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
        let seat = crate::layout::seat(graph, self.heads, self.batch, self.capacity, self.rows);
        let slots = graph.add(seat, cursor);
        let (stored_keys, stored_values) = match self.seat {
            Seat::Planes => (self.keys, self.values),
            Seat::Shared => match &self.shared {
                Some(shared) => (shared.keys, shared.values),
                None => panic!("a cache that walks a shared prefix declares one"),
            },
        };
        for (held, stored) in [(keys, stored_keys), (values, stored_values)] {
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
        let (keys, values) = match self.seat {
            Seat::Planes => (self.keys, self.values),
            Seat::Shared => match &self.shared {
                Some(shared) => (shared.keys, shared.values),
                None => panic!("a cache that walks a shared prefix declares one"),
            },
        };
        graph.attention(
            query,
            keys,
            values,
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
