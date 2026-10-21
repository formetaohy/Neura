use neura_abi::{DELTA_WIDTH, Element};
use neura_graph::{DeltaInput, Graph, Init, Shape, Value};

pub struct DeltaRule {
    state: Value,
    shared: Option<Value>,
    heads: u32,
    width: u32,
    batch: u32,
    rows: u32,
    seat: Seat,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Seat {
    Planes,
    Shared,
}

impl DeltaRule {
    pub fn new(graph: &Graph, heads: u32, width: u32, batch: u32, rows: u32) -> Self {
        Self::of(graph, heads, width, batch, rows, false, Seat::Planes)
    }

    pub fn with_shared(graph: &Graph, heads: u32, width: u32, batch: u32, rows: u32) -> Self {
        Self::of(graph, heads, width, batch, rows, true, Seat::Planes)
    }

    pub fn shared(graph: &Graph, heads: u32, width: u32, batch: u32, rows: u32) -> Self {
        Self::of(graph, heads, width, batch, rows, true, Seat::Shared)
    }

    fn of(
        graph: &Graph,
        heads: u32,
        width: u32,
        batch: u32,
        rows: u32,
        declares_shared: bool,
        seat: Seat,
    ) -> Self {
        assert!(
            heads > 0 && width > 0 && batch > 0,
            "a delta rule of {batch} planes of {heads} heads of {width} numbers carries no state",
        );
        assert!(
            width <= DELTA_WIDTH,
            "a delta rule walks {width} numbers of width where its kernel holds {DELTA_WIDTH}",
        );
        assert!(rows > 0, "a delta rule walks no row of {rows}");
        assert!(
            declares_shared || seat == Seat::Planes,
            "a delta rule that walks a shared prefix declares the prefix it walks",
        );
        let planes = graph.state(
            Shape::of([heads, batch, width, width]),
            Init::Zero,
            Element::Single,
        );
        let shared = declares_shared.then(|| {
            graph.state(
                Shape::of([heads, 1, width, width]),
                Init::Zero,
                Element::Single,
            )
        });
        Self {
            state: match seat {
                Seat::Planes => planes,
                Seat::Shared => shared.expect("a shared delta rule declares its prefix"),
            },
            shared,
            heads,
            width,
            batch: match seat {
                Seat::Planes => batch,
                Seat::Shared => 1,
            },
            rows,
            seat,
        }
    }

    pub fn forward(&self, graph: &Graph, input: DeltaInput, keep: Option<Value>) -> Value {
        for (what, held) in [
            ("query", input.query),
            ("key", input.key),
            ("value", input.value),
        ] {
            assert_eq!(
                graph.shape(held).dims(),
                [self.heads, self.batch, self.rows, self.width],
                "a delta rule walks a {what} of {} heads over {} planes of {} rows of {} numbers, and value {} walks {:?}",
                self.heads,
                self.batch,
                self.rows,
                self.width,
                held.id(),
                graph.shape(held).dims(),
            );
        }
        for (what, held) in [("beta", input.beta), ("log decay", input.log_decay)] {
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
        graph.delta(self.state, input, keep)
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

    pub fn rows(&self) -> u32 {
        self.rows
    }

    pub fn reset(&self, graph: &Graph, reset: Value, share: Option<Value>) {
        assert_eq!(
            self.seat,
            Seat::Planes,
            "a delta rule resets the planes it walks, and this rule walks the shared prefix",
        );
        let zero = graph.fill(Shape::scalar(), 0.0);
        match (self.shared, share) {
            (Some(shared), Some(share)) => {
                let shape = graph.shape(self.state);
                let prefix = graph.broadcast_to(graph.slice(shared, 1, 0, 1), shape);
                let blank = graph.select(reset, zero, self.state);
                graph.copy_into(self.state, graph.select(share, prefix, blank));
            }
            (None, None) => {
                graph.copy_into(self.state, graph.select(reset, zero, self.state));
            }
            _ => panic!(
                "a delta rule resets with the prefix it declares, and this rule declares {}",
                self.shared.is_some(),
            ),
        }
    }

    pub fn clear_shared(&self, graph: &Graph, reset: Value) {
        assert_eq!(
            self.seat,
            Seat::Shared,
            "a shared delta rule clears the prefix it walks, and this rule walks the planes",
        );
        let zero = graph.fill(Shape::scalar(), 0.0);
        let shared = self
            .shared
            .expect("a shared delta rule declares the prefix it clears");
        graph.copy_into(shared, graph.select(reset, zero, shared));
    }
}
