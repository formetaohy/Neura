use neura_abi::Element;
use neura_graph::{AttentionOptions, Graph, Init, Shape, Value};
use std::cell::OnceCell;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Seat {
    Planes,
    Shared,
}

struct Shared {
    keys: Value,
    values: Value,
    visible_keys: Value,
    visible_values: Value,
}

#[derive(Clone, Copy)]
pub struct CacheShape {
    heads: u32,
    width: u32,
    capacity: u32,
    window: u32,
    batch: u32,
    rows: u32,
}

impl CacheShape {
    pub fn new(heads: u32, width: u32, capacity: u32, window: u32, batch: u32, rows: u32) -> Self {
        Self {
            heads,
            width,
            capacity,
            window,
            batch,
            rows,
        }
    }
}

pub struct Cache {
    keys: Value,
    values: Value,
    visible_keys: Value,
    visible_values: Value,
    shared: Option<Shared>,
    base: Value,
    row: Value,
    span: OnceCell<Value>,
    scratch: OnceCell<Value>,
    heads: u32,
    width: u32,
    capacity: u32,
    window: u32,
    ring: bool,
    batch: u32,
    rows: u32,
    seat: Seat,
    element: Element,
}

impl Cache {
    pub fn new(graph: &Graph, shape: CacheShape, element: Element) -> Self {
        Self::of(graph, shape, false, Seat::Planes, false, element)
    }

    pub fn windowed(graph: &Graph, shape: CacheShape, element: Element) -> Self {
        Self::of(graph, shape, false, Seat::Planes, true, element)
    }

    pub fn with_shared(graph: &Graph, shape: CacheShape, element: Element) -> Self {
        Self::of(graph, shape, true, Seat::Planes, false, element)
    }

    pub fn windowed_with_shared(graph: &Graph, shape: CacheShape, element: Element) -> Self {
        Self::of(graph, shape, true, Seat::Planes, true, element)
    }

    pub fn shared(graph: &Graph, shape: CacheShape, element: Element) -> Self {
        Self::of(graph, shape, true, Seat::Shared, false, element)
    }

    pub fn windowed_shared(graph: &Graph, shape: CacheShape, element: Element) -> Self {
        Self::of(graph, shape, true, Seat::Shared, true, element)
    }

    fn of(
        graph: &Graph,
        shape: CacheShape,
        declares_shared: bool,
        seat: Seat,
        ring: bool,
        element: Element,
    ) -> Self {
        let CacheShape {
            heads,
            width,
            capacity,
            window,
            batch,
            rows,
        } = shape;
        assert!(
            heads > 0 && width > 0 && batch > 0,
            "a cache of {batch} planes of {heads} heads of {width} numbers holds no key",
        );
        assert!(capacity > 0, "a cache of {capacity} positions holds no key");
        assert!(
            window > 0 && window <= capacity,
            "a cache of {capacity} positions keeps a window of {window} positions",
        );
        assert!(
            rows > 0 && rows <= window,
            "a cache of a {window} position window takes {rows} rows at a time, and a chunk of more rows than the window holds walks past the slots it fills",
        );
        assert!(
            declares_shared || seat == Seat::Planes,
            "a cache that walks a shared prefix declares the prefix it walks",
        );
        assert!(
            matches!(element, Element::Single | Element::Half | Element::Bfloat16),
            "a cache holds keys and values in single, half or bfloat16 storage, and {} storage carries another grid",
            element.name(),
        );
        let held = Shape::of([heads, batch, capacity, width]);
        let keys = graph.state(held, Init::Zero, element);
        let values = graph.state(held, Init::Zero, element);
        let shared = declares_shared.then(|| {
            let held = Shape::of([heads, 1, capacity, width]);
            let keys = graph.state(held, Init::Zero, element);
            let values = graph.state(held, Init::Zero, element);
            Shared {
                keys,
                values,
                visible_keys: visible(graph, keys, window, capacity),
                visible_values: visible(graph, values, window, capacity),
            }
        });
        let (visible_keys, visible_values) = match seat {
            Seat::Planes => (
                visible(graph, keys, window, capacity),
                visible(graph, values, window, capacity),
            ),
            Seat::Shared => {
                let shared = shared
                    .as_ref()
                    .expect("a shared cache declares the prefix it walks");
                (shared.visible_keys, shared.visible_values)
            }
        };
        let base = crate::layout::base(
            graph,
            heads,
            match seat {
                Seat::Planes => batch,
                Seat::Shared => 1,
            },
            capacity,
        );
        let row = crate::layout::rows(graph, rows);
        Self {
            keys,
            values,
            visible_keys,
            visible_values,
            shared,
            base,
            row,
            span: OnceCell::new(),
            scratch: OnceCell::new(),
            heads,
            width,
            capacity,
            window,
            ring,
            batch: match seat {
                Seat::Planes => batch,
                Seat::Shared => 1,
            },
            rows,
            seat,
            element,
        }
    }

    pub fn states(&self) -> [Value; 2] {
        [self.visible_keys, self.visible_values]
    }

    pub fn shared_states(&self) -> Option<[Value; 2]> {
        self.shared
            .as_ref()
            .map(|shared| [shared.visible_keys, shared.visible_values])
    }

    pub fn rows(&self) -> u32 {
        self.rows
    }

    pub fn batch(&self) -> u32 {
        self.batch
    }

    pub fn window(&self) -> u32 {
        self.window
    }

    pub fn element(&self) -> Element {
        self.element
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
                    let chosen = graph.cast(graph.select(share, prefix, blank), self.element);
                    graph.copy_into(held, chosen);
                }
            }
            (None, None) => {
                for held in [self.keys, self.values] {
                    graph.clear_where(held, reset);
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
        let shared = self
            .shared
            .as_ref()
            .expect("a shared cache declares the prefix it clears");
        for held in [shared.keys, shared.values] {
            graph.clear_where(held, reset);
        }
    }

    pub fn write(&self, graph: &Graph, cursor: Value, keys: Value, values: Value) {
        self.store(graph, cursor, keys, values, None);
    }

    pub fn write_masked(
        &self,
        graph: &Graph,
        cursor: Value,
        keys: Value,
        values: Value,
        active: Value,
    ) {
        self.store(graph, cursor, keys, values, Some(active));
    }

    fn store(
        &self,
        graph: &Graph,
        cursor: Value,
        keys: Value,
        values: Value,
        active: Option<Value>,
    ) {
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
        let cursor_shape = graph.shape(cursor);
        assert!(
            cursor_shape.elements() == 1 || cursor_shape.dims() == [1, self.batch, 1, 1],
            "a cache of {} planes reads one cursor a plane or one cursor for every plane, and value {} walks {:?}",
            self.batch,
            cursor.id(),
            cursor_shape.dims(),
        );
        let active_slots = match self.ring {
            true => graph.add(
                self.base,
                crate::layout::ring(graph, self.span(graph), self.row, cursor),
            ),
            false => graph.add(self.base, graph.add(cursor, self.row)),
        };
        let slots = match active {
            Some(active) => {
                assert_eq!(
                    graph.shape(active).dims(),
                    [1, self.batch, 1, 1],
                    "a cache of {} planes takes one active flag a plane, and value {} walks {:?}",
                    self.batch,
                    active.id(),
                    graph.shape(active).dims(),
                );
                assert!(
                    self.rows <= self.capacity - self.window,
                    "a cache of {} positions keeps a {} position window, and a masked chunk of {} rows holds only {} scratch rows",
                    self.capacity,
                    self.window,
                    self.rows,
                    self.capacity - self.window,
                );
                graph.select(active, active_slots, self.scratch(graph))
            }
            None => active_slots,
        };
        let (stored_keys, stored_values) = match self.seat {
            Seat::Planes => (self.keys, self.values),
            Seat::Shared => match &self.shared {
                Some(shared) => (shared.keys, shared.values),
                None => panic!("a cache that walks a shared prefix declares one"),
            },
        };
        let keys = graph.cast(keys, self.element);
        let values = graph.cast(values, self.element);
        for (held, stored) in [(keys, stored_keys), (values, stored_values)] {
            graph.write_into(stored, slots, held);
        }
    }

    fn span(&self, graph: &Graph) -> Value {
        *self
            .span
            .get_or_init(|| graph.fill(Shape::scalar(), self.window as f32))
    }

    fn scratch(&self, graph: &Graph) -> Value {
        *self
            .scratch
            .get_or_init(|| graph.add(self.base, graph.add(self.span(graph), self.row)))
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
            self.visible_keys,
            self.visible_values,
            AttentionOptions {
                scale: Some(scale),
                causal: true,
                origin: Some(cursor),
                segments: None,
                reach: match self.ring {
                    true => Some(self.window),
                    false => None,
                },
                query_segments: None,
                bias: None,
            },
        )
    }
}

fn visible(graph: &Graph, value: Value, window: u32, capacity: u32) -> Value {
    match window == capacity {
        true => value,
        false => graph.slice(value, 2, 0, window),
    }
}
