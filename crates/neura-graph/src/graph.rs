use crate::autodiff::Recomputation;
use crate::init::Init;
use crate::shape::{Free, Shape};
use crate::window::Window;
use neura_abi::{EXACT_WALK_LIMIT, Element, Kind, MAX_RANK, NO_VALUE, StepRecord};
use neura_pointwise as op;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Weak};

static NEXT_GRAPH: AtomicU64 = AtomicU64::new(1);

pub(crate) const NORM_FLOOR: f32 = 1e-6;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct Value<'g> {
    graph: u64,
    id: u32,
    shape: Shape,
    brand: PhantomData<fn(&'g ()) -> &'g ()>,
}

impl<'g> Value<'g> {
    pub const fn id(self) -> u32 {
        self.id
    }

    pub const fn shape(self) -> Shape {
        self.shape
    }

    fn of(graph: u64, id: u32, shape: Shape) -> Self {
        Self {
            graph,
            id,
            shape,
            brand: PhantomData,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct AttentionOptions<'g> {
    pub scale: Option<Value<'g>>,
    pub causal: bool,
    pub origin: Option<Value<'g>>,
    pub segments: Option<Value<'g>>,
    pub reach: Option<u32>,
    pub query_segments: Option<Value<'g>>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Ragged<'g> {
    pub extent: Free,
    pub offsets: Value<'g>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Rows<'g> {
    pub plane: Value<'g>,
    pub position: Value<'g>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Prefixes<'g> {
    pub exclusive: Value<'g>,
    pub total: Value<'g>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Compacted<'g> {
    pub indices: Value<'g>,
    pub count: Value<'g>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Residency {
    Input,
    Parameter,
    State,
    Resident,
    Derived,
    View,
}

#[non_exhaustive]
#[derive(Clone, PartialEq, Debug)]
pub struct TaskInfo {
    pub kind: Kind,
    pub op: u32,
    pub out: u32,
    pub extra: u32,
    pub inputs: [u32; 6],
    pub origin: u32,
    pub slot: u32,
    pub literal: f32,
    pub knob: u32,
    pub window: Window,
    pub in_place: bool,
    pub axis: u32,
    pub offset: u32,
    pub segments: u32,
    pub reach: u32,
    pub queries: u32,
    pub keep: u32,
    pub prelude: Vec<StepRecord>,
    pub chain: Vec<StepRecord>,
}

impl TaskInfo {
    pub(crate) fn of(kind: Kind, op: u32, out: u32, inputs: [u32; 6]) -> Self {
        Self {
            kind,
            op,
            out,
            extra: NO_VALUE,
            inputs,
            origin: NO_VALUE,
            slot: 0,
            literal: 0.0,
            knob: NO_VALUE,
            window: Window::sliding([1, 1]),
            in_place: false,
            axis: 0,
            offset: 0,
            segments: NO_VALUE,
            reach: 0,
            queries: NO_VALUE,
            keep: 0,
            prelude: Vec::new(),
            chain: Vec::new(),
        }
    }
}

#[non_exhaustive]
#[derive(Clone)]
pub struct ValueInfo {
    pub shape: Shape,
    pub strides: [u32; 4],
    pub strides_source: Option<[u8; 4]>,
    pub storage: u32,
    pub element: Element,
    pub scale: f32,
    pub residency: Residency,
    pub requires_grad: bool,
    pub retained: bool,
    pub written_in_place: bool,
    pub recomputes: Option<u32>,
    pub seed: Option<Init>,
    pub name: Option<Arc<str>>,
}

impl ValueInfo {
    pub fn derived(shape: Shape, id: u32) -> Self {
        Self {
            shape,
            strides: shape.strides(),
            strides_source: None,
            storage: id,
            element: Element::Single,
            scale: 1.0,
            residency: Residency::Derived,
            requires_grad: false,
            retained: false,
            written_in_place: false,
            recomputes: None,
            seed: None,
            name: None,
        }
    }
}

pub(crate) struct GraphState {
    pub(crate) values: Vec<ValueInfo>,
    pub(crate) tasks: Vec<TaskInfo>,
    pub(crate) recomputations: Vec<Recomputation>,
    pub(crate) revisions: Vec<Weak<RevisionState>>,
    pub(crate) names: HashSet<Arc<str>>,
    pub(crate) authored: Vec<u32>,
    pub(crate) ragged: HashMap<u32, RaggedAxis>,
    pub(crate) prefixes: HashMap<u32, u32>,
    pub(crate) differentiated: bool,
    pub(crate) updated_in_place: bool,
    pub(crate) version: u64,
    pub(crate) frees: u32,
}

#[derive(Clone)]
pub(crate) struct RaggedAxis {
    pub(crate) planes: Shape,
    pub(crate) token: u32,
    pub(crate) queries: bool,
}

pub struct GraphSnapshot {
    values: Vec<ValueInfo>,
    tasks: Vec<TaskInfo>,
    authored: Vec<u32>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct GraphStamp {
    graph: u64,
    version: u64,
}

#[derive(Debug)]
pub(crate) struct RevisionState {
    live: AtomicBool,
}

#[derive(Clone, Debug)]
pub struct Revision {
    stamp: GraphStamp,
    state: Arc<RevisionState>,
}

impl Revision {
    pub const fn stamp(&self) -> GraphStamp {
        self.stamp
    }

    pub fn is_current(&self) -> bool {
        self.state.live.load(Ordering::Acquire)
    }
}

pub(crate) fn advance(state: &mut GraphState) {
    state.version += 1;
    for revision in state.revisions.drain(..) {
        if let Some(revision) = revision.upgrade() {
            revision.live.store(false, Ordering::Release);
        }
    }
}

impl GraphSnapshot {
    pub fn values(&self) -> &[ValueInfo] {
        &self.values
    }

    pub fn tasks(&self) -> &[TaskInfo] {
        &self.tasks
    }

    pub fn authored(&self) -> &[u32] {
        &self.authored
    }
}

pub(crate) struct Declaration<'a> {
    name: Option<&'a str>,
    shape: Shape,
    strides: [u32; 4],
    strides_source: Option<[u8; 4]>,
    storage: Option<u32>,
    residency: Residency,
    element: Element,
    scale: f32,
    seed: Option<Init>,
    requires_grad: bool,
}

impl<'a> Declaration<'a> {
    pub(crate) fn of(shape: Shape, element: Element, residency: Residency) -> Self {
        Self {
            name: None,
            shape,
            strides: shape.strides(),
            strides_source: None,
            storage: None,
            residency,
            element,
            scale: 1.0,
            seed: None,
            requires_grad: false,
        }
    }

    pub(crate) fn view(
        shape: Shape,
        strides: [u32; 4],
        strides_source: Option<[u8; 4]>,
        storage: u32,
        element: Element,
        scale: f32,
        tracked: bool,
    ) -> Self {
        Self {
            name: None,
            shape,
            strides,
            strides_source,
            storage: Some(storage),
            residency: Residency::View,
            element,
            scale,
            seed: None,
            requires_grad: tracked,
        }
    }

    fn named(mut self, name: &'a str) -> Self {
        self.name = Some(name);
        self
    }

    fn scaled(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    fn seeded(mut self, init: Init) -> Self {
        self.seed = Some(init);
        self
    }

    fn trained(mut self, tracked: bool) -> Self {
        self.requires_grad = tracked;
        self
    }
}

pub struct Graph<'g> {
    pub(crate) instance: u64,
    pub(crate) state: RefCell<GraphState>,
    brand: PhantomData<fn(&'g ()) -> &'g ()>,
}

impl<'g> Graph<'g> {
    pub fn new() -> Self {
        Self {
            state: RefCell::new(GraphState {
                values: Vec::new(),
                tasks: Vec::new(),
                recomputations: Vec::new(),
                revisions: Vec::new(),
                names: HashSet::new(),
                authored: Vec::new(),
                ragged: HashMap::new(),
                prefixes: HashMap::new(),
                differentiated: false,
                updated_in_place: false,
                version: 0,
                frees: 0,
            }),
            instance: NEXT_GRAPH.fetch_add(1, Ordering::Relaxed),
            brand: PhantomData,
        }
    }

    pub fn stamp(&self) -> GraphStamp {
        GraphStamp {
            graph: self.instance,
            version: self.state.borrow().version,
        }
    }

    pub fn revision(&self) -> Revision {
        let mut state = self.state.borrow_mut();
        state
            .revisions
            .retain(|revision| revision.strong_count() > 0);
        let revision = Revision {
            stamp: GraphStamp {
                graph: self.instance,
                version: state.version,
            },
            state: Arc::new(RevisionState {
                live: AtomicBool::new(true),
            }),
        };
        state.revisions.push(Arc::downgrade(&revision.state));
        revision
    }

    pub fn free(&self, bound: u32) -> Free {
        let mut state = self.state.borrow_mut();
        let slot = state.frees;
        state.frees += 1;
        assert!(
            state.frees < u32::from(u8::MAX),
            "a graph names at most {} free extents",
            u8::MAX,
        );
        state.authored.push(NO_VALUE);
        Free::of(slot, bound)
    }

    pub fn counted(&self, bound: u32, count: Value<'g>) -> Free {
        let count = self.own(count);
        let free = self.free(bound);
        self.author(free, count);
        free
    }

    pub fn trim(&self, value: Value<'g>, axis: u32, count: Value<'g>) -> Value<'g> {
        let value = self.own(value);
        let count = self.own(count);
        assert!(
            axis < MAX_RANK,
            "a trim names one of the {MAX_RANK} axes, and {axis} is not one of them",
        );
        let (shape, strides, strides_source, storage, element, scale, tracked) = {
            let state = self.state.borrow();
            let info = &state.values[value.id() as usize];
            assert!(
                info.shape.free(axis).is_none(),
                "axis {axis} of {:?} walks free extent {}, and a free extent already takes every length the axis holds",
                info.shape.dims(),
                info.shape.free(axis).unwrap_or_default(),
            );
            (
                info.shape,
                info.strides,
                info.strides_source,
                info.storage,
                info.element,
                info.scale,
                info.requires_grad,
            )
        };
        assert!(
            storage == value.id(),
            "a trim walks the rows of a tensor its storage lays out row by row, and value {} is a view of value {storage}",
            value.id(),
        );
        let beside = shape.dims()[..axis as usize].iter().product::<u32>();
        assert!(
            beside == 1,
            "a device authored extent cuts one walk of a tensor, and the {beside} planes beside axis {axis} of {:?} hold every stride the cut moves; a count cuts one plane, and every plane of a batch walks the offsets a ragged axis closes",
            shape.dims(),
        );
        let free = self.counted(shape.dims()[axis as usize], count);
        let mut frees = shape.frees();
        frees[axis as usize] = Some(free.slot());
        let live = self.declare(Declaration::view(
            Shape::from_axes(shape.dims(), frees),
            strides,
            strides_source,
            storage,
            element,
            scale,
            tracked,
        ));
        self.state.borrow_mut().prefixes.insert(live.id(), storage);
        live
    }

    pub(crate) fn prefix_owner(&self, value: u32) -> Option<u32> {
        self.state.borrow().prefixes.get(&value).copied()
    }

    pub(crate) fn mark_prefix(&self, view: u32, owner: u32) {
        let mut state = self.state.borrow_mut();
        assert_eq!(
            state.values[view as usize].storage, owner,
            "a view marked as a prefix walks the storage of the tensor it prefixes, and value {view} walks the storage of value {owner}",
        );
        state.prefixes.insert(view, owner);
    }

    pub(crate) fn walks_a_prefix(&self, value: u32) -> bool {
        let state = self.state.borrow();
        let info = &state.values[value as usize];
        if info.residency != Residency::View {
            return false;
        }
        let owner = &state.values[info.storage as usize];
        let walked = (0..MAX_RANK).filter_map(|axis| owner.shape.free(axis));
        let mut walked = walked.collect::<Vec<u32>>();
        walked.sort_unstable();
        (0..MAX_RANK)
            .filter_map(|axis| info.shape.free(axis))
            .any(|slot| walked.binary_search(&slot).is_err())
    }

    pub(crate) fn packed_axis(&self, shape: Shape) -> Option<(u32, u32, RaggedAxis)> {
        let state = self.state.borrow();
        (0..MAX_RANK).find_map(|at| {
            let slot = shape.free(at)?;
            state
                .ragged
                .iter()
                .find(|(_, axis)| axis.token == slot)
                .map(|(offsets, axis)| (at, *offsets, axis.clone()))
        })
    }

    pub(crate) fn mark_query_axis(&self, offsets: u32) {
        let mut state = self.state.borrow_mut();
        let axis = state
            .ragged
            .get_mut(&offsets)
            .expect("a query axis is a ragged axis the graph publishes");
        axis.queries = true;
        let token = axis.token;
        for task in &state.tasks {
            if !matches!(task.kind, Kind::Rope | Kind::RopeGrad) {
                continue;
            }
            let walked = state.values[task.out as usize].shape;
            if (0..MAX_RANK).all(|at| walked.free(at) != Some(token)) {
                continue;
            }
            panic!(
                "value {} turns the rows the query axis value {offsets} closes by their own plane's offsets, and the device places the row of a query chunk at the end of the key plane its sequence already holds; rotate the row of a chunk before the packing gathers it, so that a rotated row stands where the key it weighs stands",
                task.out,
            );
        }
    }

    pub fn author(&self, free: Free, count: Value<'g>) {
        let count = self.own(count);
        let elements = self.shape(count).elements();
        assert_eq!(
            elements,
            1,
            "a device authored extent is one number, and value {} holds {elements}",
            count.id(),
        );
        let produced = {
            let state = self.state.borrow();
            state
                .tasks
                .iter()
                .any(|task| task.out == count.id() || task.extra == count.id())
        };
        let count = if produced {
            count
        } else {
            self.materialized(count)
        };
        let mut state = self.state.borrow_mut();
        assert!(
            (free.slot() as usize) < state.authored.len(),
            "free extent {} is authored by a value the graph declares no extent for",
            free.slot(),
        );
        assert_eq!(
            state.authored[free.slot() as usize],
            NO_VALUE,
            "free extent {} already walks the count of value {}",
            free.slot(),
            state.authored[free.slot() as usize],
        );
        state.authored[free.slot() as usize] = count.id();
        advance(&mut state);
    }

    pub fn ragged(&self, bound: u32, lengths: Value<'g>) -> Ragged<'g> {
        let lengths = self.own(lengths);
        assert!(
            bound <= EXACT_WALK_LIMIT,
            "a ragged extent of {bound} numbers outruns the {EXACT_WALK_LIMIT} numbers a device sums exactly",
        );
        self.exact_walk(lengths, "a ragged axis");
        let planes = self.shape(lengths).elements();
        assert!(
            planes < u32::MAX,
            "a ragged axis of {planes} planes leaves no room for the offset that closes it",
        );
        let offsets = self.fresh(
            Shape::matrix(planes + 1, 1),
            Element::Single,
            Residency::Derived,
            false,
        );
        let total = self.fresh(Shape::scalar(), Element::Single, Residency::Derived, false);
        self.prefix(lengths, offsets, total);
        let planes = self.shape(lengths);
        let extent = self.counted(bound, total);
        self.state.borrow_mut().ragged.insert(
            offsets.id(),
            RaggedAxis {
                planes,
                token: extent.slot(),
                queries: false,
            },
        );
        Ragged { extent, offsets }
    }

    pub fn rows(&self, ragged: Ragged<'g>) -> Rows<'g> {
        let offsets = self.own(ragged.offsets);
        let axis = {
            let state = self.state.borrow();
            state.ragged.get(&offsets.id()).cloned()
        }
        .unwrap_or_else(|| {
            panic!(
                "a row map walks the planes a ragged axis closes, and value {} holds a table no ragged axis published; close the lengths of every plane with Graph::ragged",
                offsets.id(),
            )
        });
        assert_eq!(
            axis.token,
            ragged.extent.slot(),
            "a row map names the rows of the {} planes a ragged axis closes over free extent {}, and the axis hands it free extent {}",
            axis.planes.elements(),
            axis.token,
            ragged.extent.slot(),
        );
        let shape = Shape::of([1, 1, ragged.extent.bound(), 1]).freed(&[(2, ragged.extent)]);
        let plane = self.fresh(shape, Element::Single, Residency::Derived, false);
        let position = self.fresh(shape, Element::Single, Residency::Derived, false);
        let mut task = TaskInfo::of(
            Kind::Rows,
            op::NONE,
            plane.id(),
            [
                offsets.id(),
                NO_VALUE,
                NO_VALUE,
                NO_VALUE,
                NO_VALUE,
                NO_VALUE,
            ],
        );
        task.extra = position.id();
        task.segments = offsets.id();
        self.push(task);
        Rows { plane, position }
    }

    pub fn prefix_sum(&self, value: Value<'g>) -> Prefixes<'g> {
        let value = self.own(value);
        self.exact_walk(value, "a prefix sum");
        let shape = self.shape(value);
        let total = self.fresh(Shape::scalar(), Element::Single, Residency::Derived, false);
        let exclusive = self.fresh(shape, Element::Single, Residency::Derived, false);
        self.prefix(value, exclusive, total);
        Prefixes { exclusive, total }
    }

    pub fn compact(&self, mask: Value<'g>) -> Compacted<'g> {
        let mask = self.own(mask);
        let shape = self.shape(mask);
        assert_eq!(
            shape.dims()[3],
            1,
            "a compaction weighs one flag per row, and {:?} holds {} of them",
            shape.dims(),
            shape.dims()[3],
        );
        assert!(
            shape.free(3).is_none(),
            "a compaction weighs one flag per row, and axis 3 of {:?} walks free extent {}",
            shape.dims(),
            shape.free(3).unwrap_or_default(),
        );
        let prefix = self.prefix_sum(mask);
        let rows = shape.dims()[..3].iter().product::<u32>();
        let indices = self.resident(Shape::matrix(rows, 1), Element::Single);
        self.push(TaskInfo::of(
            Kind::Compact,
            op::NONE,
            indices.id(),
            [
                mask.id(),
                prefix.exclusive.id(),
                NO_VALUE,
                NO_VALUE,
                NO_VALUE,
                NO_VALUE,
            ],
        ));
        Compacted {
            indices: self.trim(indices, 2, prefix.total),
            count: prefix.total,
        }
    }

    fn exact_walk(&self, value: Value<'g>, walks: &str) {
        assert!(
            self.contiguous(value),
            "{walks} walks a tensor its storage lays out row by row, and value {} is a view",
            value.id(),
        );
        let element = self.element(value);
        assert!(
            !element.narrow() && !element.per_block(),
            "{walks} sums the exact numbers it reads, and value {} holds {} storage",
            value.id(),
            element.name(),
        );
        let elements = self.shape(value).elements();
        assert!(
            elements <= EXACT_WALK_LIMIT,
            "{walks} of {elements} numbers outruns the {EXACT_WALK_LIMIT} numbers a device sums exactly",
        );
    }

    fn prefix(&self, value: Value<'g>, offsets: Value<'g>, total: Value<'g>) {
        let mut unit = TaskInfo::of(
            Kind::PrefixChunk,
            op::NONE,
            total.id(),
            [value.id(), NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE],
        );
        unit.extra = offsets.id();
        self.push(unit);
    }

    pub fn input(&self, shape: Shape, element: Element) -> Value<'g> {
        assert!(
            !element.quantized(),
            "an input of {} storage is declared with the quantum it reconstructs by, and an input carries none; quantize the tensor that reads it instead",
            element.name(),
        );
        self.hold(Declaration::of(shape, element, Residency::Input))
    }

    pub fn gradient_input(&self, shape: Shape, element: Element) -> Value<'g> {
        assert!(
            !element.quantized(),
            "a gradient input of {} storage is declared with the quantum it reconstructs by, and a gradient input carries none; quantize the tensor that reads it instead",
            element.name(),
        );
        self.hold(Declaration::of(shape, element, Residency::Input).trained(true))
    }

    pub fn resident(&self, shape: Shape, element: Element) -> Value<'g> {
        assert!(
            !element.quantized(),
            "a resident tensor of {} storage is declared with the quantum it reconstructs by, and a resident tensor carries none; quantize the tensor that reads it instead",
            element.name(),
        );
        self.hold(Declaration::of(shape, element, Residency::Resident))
    }

    pub fn parameter(&self, shape: Shape, init: Init, element: Element) -> Value<'g> {
        assert!(
            !element.quantized(),
            "a parameter of {} storage is declared with the quantum it reconstructs by, and a parameter carries none; declare a quantized or a block quantized parameter instead",
            element.name(),
        );
        self.hold(
            Declaration::of(shape, element, Residency::Parameter)
                .seeded(init)
                .trained(true),
        )
    }

    pub fn named_parameter(
        &self,
        name: &str,
        shape: Shape,
        init: Init,
        element: Element,
    ) -> Value<'g> {
        assert!(
            !element.quantized(),
            "a parameter of {} storage is declared with the quantum it reconstructs by, and a parameter carries none; declare a quantized or a block quantized parameter instead",
            element.name(),
        );
        self.hold(
            Declaration::of(shape, element, Residency::Parameter)
                .named(name)
                .seeded(init)
                .trained(true),
        )
    }

    pub fn knob(&self, value: f32) -> Value<'g> {
        self.hold(
            Declaration::of(Shape::scalar(), Element::Single, Residency::State)
                .seeded(Init::Constant(value)),
        )
    }

    pub fn named_knob(&self, name: &str, value: f32) -> Value<'g> {
        self.hold(
            Declaration::of(Shape::scalar(), Element::Single, Residency::State)
                .named(name)
                .seeded(Init::Constant(value)),
        )
    }

    pub fn state(&self, shape: Shape, init: Init, element: Element) -> Value<'g> {
        assert!(
            !element.quantized(),
            "a training state of {} storage is declared with the quantum it reconstructs by, and a training state carries none",
            element.name(),
        );
        self.hold(Declaration::of(shape, element, Residency::State).seeded(init))
    }

    pub fn named_state(&self, name: &str, shape: Shape, init: Init, element: Element) -> Value<'g> {
        assert!(
            !element.quantized(),
            "a training state of {} storage is declared with the quantum it reconstructs by, and a training state carries none",
            element.name(),
        );
        self.hold(
            Declaration::of(shape, element, Residency::State)
                .named(name)
                .seeded(init),
        )
    }

    pub fn quantized_parameter(&self, shape: Shape, init: Init, scale: f32) -> Value<'g> {
        self.hold(
            Declaration::of(shape, Element::Int8, Residency::Parameter)
                .scaled(scale)
                .seeded(init),
        )
    }

    pub fn named_quantized_parameter(
        &self,
        name: &str,
        shape: Shape,
        init: Init,
        scale: f32,
    ) -> Value<'g> {
        self.hold(
            Declaration::of(shape, Element::Int8, Residency::Parameter)
                .named(name)
                .scaled(scale)
                .seeded(init),
        )
    }

    pub fn block_quantized_parameter(
        &self,
        shape: Shape,
        init: Init,
        element: Element,
    ) -> Value<'g> {
        assert!(
            element.per_block(),
            "a block quantized parameter of {} storage carries one quantum of every number, and a block quantized tensor declares the block its storage packs",
            element.name(),
        );
        self.hold(Declaration::of(shape, element, Residency::Parameter).seeded(init))
    }

    pub fn named_block_quantized_parameter(
        &self,
        name: &str,
        shape: Shape,
        init: Init,
        element: Element,
    ) -> Value<'g> {
        assert!(
            element.per_block(),
            "a block quantized parameter of {} storage carries one quantum of every number, and a block quantized tensor declares the block its storage packs",
            element.name(),
        );
        self.hold(
            Declaration::of(shape, element, Residency::Parameter)
                .named(name)
                .seeded(init),
        )
    }

    pub fn quantize(&self, value: Value<'g>, scale: f32) -> Value<'g> {
        let value = self.own(value);
        let out = self.quantized(self.shape(value), scale);
        self.push(TaskInfo::of(
            Kind::Unary,
            op::IDENTITY,
            out.id(),
            [value.id(), NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE],
        ));
        out
    }

    pub fn fill(&self, shape: Shape, value: f32) -> Value<'g> {
        let out = self.fresh(shape, Element::Single, Residency::Derived, false);
        let mut task = TaskInfo::of(Kind::Fill, op::NONE, out.id(), [NO_VALUE; 6]);
        task.literal = value;
        self.push(task);
        out
    }

    fn materialized(&self, value: Value<'g>) -> Value<'g> {
        let element = self.element(value);
        let out = self.unary_as(op::IDENTITY, value, element);
        self.retain(out);
        out
    }

    pub fn shape(&self, value: Value<'g>) -> Shape {
        let value = self.own(value);
        self.state.borrow().values[value.id() as usize].shape
    }

    pub fn element(&self, value: Value<'g>) -> Element {
        let value = self.own(value);
        self.state.borrow().values[value.id() as usize].element
    }

    pub fn scale(&self, value: Value<'g>) -> f32 {
        let value = self.own(value);
        let info = &self.state.borrow().values[value.id() as usize];
        assert!(
            !info.element.per_block(),
            "a {} tensor reconstructs through the quantum its storage holds of every {} blocks, and the tensor itself declares none",
            info.element.name(),
            info.element.block(),
        );
        info.scale
    }

    pub(crate) fn declared_quantum(&self, value: Value<'g>) -> f32 {
        let value = self.own(value);
        self.state.borrow().values[value.id() as usize].scale
    }

    pub fn value_count(&self) -> usize {
        self.state.borrow().values.len()
    }

    pub fn task_count(&self) -> usize {
        self.state.borrow().tasks.len()
    }

    pub fn snapshot(&self) -> GraphSnapshot {
        let state = self.state.borrow();
        GraphSnapshot {
            values: state.values.clone(),
            tasks: state.tasks.clone(),
            authored: state.authored.clone(),
        }
    }

    pub fn updates_weights(&self) -> bool {
        let state = self.state.borrow();
        state.tasks.iter().any(|task| {
            task.in_place
                && matches!(
                    state.values[task.out as usize].residency,
                    Residency::Parameter | Residency::State
                )
        })
    }

    pub(crate) fn elementwise(&self, op: u32, left: Value<'g>, right: Value<'g>) -> Value<'g> {
        let left = self.own(left);
        let right = self.own(right);
        let shape = self.shape(left).combined(self.shape(right));
        let element = self.element(left).promote(self.element(right));
        let out = self.stored(
            shape,
            element,
            self.carries(element, &[left, right]),
            Residency::Derived,
            self.tracked(&[left, right]),
        );
        self.push(TaskInfo::of(
            op::kind(op),
            op,
            out.id(),
            [
                left.id(),
                right.id(),
                NO_VALUE,
                NO_VALUE,
                NO_VALUE,
                NO_VALUE,
            ],
        ));
        out
    }

    pub(crate) fn unary(&self, op: u32, value: Value<'g>) -> Value<'g> {
        self.unary_as(op, value, self.element(value))
    }

    pub(crate) fn unary_as(&self, op: u32, value: Value<'g>, element: Element) -> Value<'g> {
        let value = self.own(value);
        let shape = self.shape(value);
        let out = self.stored(
            shape,
            element,
            self.carries(element, &[value]),
            Residency::Derived,
            self.tracked(&[value]),
        );
        self.push(TaskInfo::of(
            op::kind(op),
            op,
            out.id(),
            [value.id(), NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE],
        ));
        out
    }

    pub fn retain(&self, value: Value<'g>) {
        let value = self.own(value);
        let storage = self.state.borrow().values[value.id() as usize].storage;
        let mut state = self.state.borrow_mut();
        state.values[storage as usize].retained = true;
        advance(&mut state);
    }

    pub fn freeze(&self, values: &[Value<'g>]) {
        assert!(
            !values.is_empty(),
            "freezing no tensor leaves every parameter learning",
        );
        let ids = values
            .iter()
            .map(|value| self.own(*value).id())
            .collect::<Vec<u32>>();
        for id in &ids {
            let state = self.state.borrow();
            let info = &state.values[*id as usize];
            assert_eq!(
                info.residency,
                Residency::Parameter,
                "a frozen tensor holds a model parameter, and value {id} holds training state or a tensor of no weight store",
            );
            assert!(
                !state.tasks.iter().any(|task| reads(task, *id)),
                "a parameter is frozen before the tasks that read it are authored, and value {id} already feeds one",
            );
        }
        let mut state = self.state.borrow_mut();
        for id in ids {
            state.values[id as usize].requires_grad = false;
        }
        advance(&mut state);
    }

    pub fn detach(&self, value: Value<'g>) -> Value<'g> {
        let value = self.own(value);
        let (shape, strides, strides_source, storage, element, scale) = {
            let state = self.state.borrow();
            let info = &state.values[value.id() as usize];
            (
                info.shape,
                info.strides,
                info.strides_source,
                info.storage,
                info.element,
                info.scale,
            )
        };
        self.declare(Declaration::view(
            shape,
            strides,
            strides_source,
            storage,
            element,
            scale,
            false,
        ))
    }

    pub(crate) fn update_in_place(&self, op: u32, target: Value<'g>, operand: Value<'g>) {
        let target = self.own(target);
        let operand = self.own(operand);
        self.assert_in_place(target, operand);
        let mut task = TaskInfo::of(
            op::kind(op),
            op,
            target.id(),
            [
                target.id(),
                operand.id(),
                NO_VALUE,
                NO_VALUE,
                NO_VALUE,
                NO_VALUE,
            ],
        );
        task.in_place = true;
        self.push(task);
        self.wrote_in_place(target);
    }

    pub(crate) fn assert_in_place(&self, target: Value<'g>, operand: Value<'g>) {
        let target = self.own(target);
        let operand = self.own(operand);
        {
            let state = self.state.borrow();
            let info = &state.values[target.id() as usize];
            assert!(
                matches!(
                    info.residency,
                    Residency::Input
                        | Residency::Parameter
                        | Residency::State
                        | Residency::Resident
                ),
                "only a leaf tensor updates in place, and value {} is derived from other tasks",
                target.id(),
            );
            assert_eq!(
                info.shape.strides(),
                info.strides,
                "a tensor written in place must be stored contiguously",
            );
            let read = &state.values[operand.id() as usize];
            assert!(
                read.storage != info.storage || read.strides == info.strides,
                "an update in place reads the element it writes, and value {} walks other strides over the same storage",
                operand.id(),
            );
        }
        let combined = self.shape(target).combined(self.shape(operand));
        assert_eq!(
            combined,
            self.shape(target),
            "updating {} in place with {} would reshape it",
            self.shape(target).elements(),
            self.shape(operand).elements(),
        );
    }

    pub(crate) fn wrote_in_place(&self, target: Value<'g>) {
        let target = self.own(target);
        let mut state = self.state.borrow_mut();
        state.values[target.id() as usize].written_in_place = true;
        state.updated_in_place = true;
        advance(&mut state);
    }

    pub(crate) fn scatter(
        &self,
        kind: Kind,
        target: Value<'g>,
        indices: Value<'g>,
        updates: Value<'g>,
    ) {
        self.index_list(indices);
        assert!(
            self.contiguous(target),
            "a scatter walks a table row by row, and value {} is a view",
            target.id(),
        );
        assert!(
            self.contiguous(updates),
            "a scatter walks its updates row by row, and value {} is a view",
            updates.id(),
        );
        let indices_shape = self.shape(indices);
        let updates_shape = self.shape(updates);
        let target_shape = self.shape(target);
        for axis in 0..3 {
            assert!(
                indices_shape.meets(updates_shape, axis, axis),
                "a scatter of {:?} indices meets updates of {:?}",
                indices_shape.dims(),
                updates_shape.dims(),
            );
        }
        assert!(
            updates_shape.meets(target_shape, 3, 3),
            "a scatter of {:?} updates meets a table of {} numbers per row",
            updates_shape.dims(),
            target_shape.dims()[3],
        );
        let mut task = TaskInfo::of(
            kind,
            op::NONE,
            target.id(),
            [
                target.id(),
                indices.id(),
                updates.id(),
                NO_VALUE,
                NO_VALUE,
                NO_VALUE,
            ],
        );
        task.in_place = true;
        self.push(task);
    }

    pub fn sum_rows(&self, value: Value<'g>) -> Value<'g> {
        let value = self.own(value);
        self.assert_axis_packs_one_plane(value, MAX_RANK - 1);
        self.fold(value, MAX_RANK - 1)
    }

    pub fn sum_axis(&self, value: Value<'g>, axis: u32) -> Value<'g> {
        let value = self.own(value);
        let shape = self.shape(value);
        assert!(
            axis < MAX_RANK,
            "a fold names one of the {MAX_RANK} axes of {:?}",
            shape.dims(),
        );
        self.assert_axis_packs_one_plane(value, axis);
        self.fold(value, axis)
    }

    fn assert_axis_packs_one_plane(&self, value: Value<'g>, axis: u32) {
        let shape = self.shape(value);
        let Some(slot) = shape.free(axis) else {
            return;
        };
        let planes = {
            let state = self.state.borrow();
            state
                .ragged
                .values()
                .find(|ragged| ragged.token == slot)
                .map(|ragged| ragged.planes.elements())
        };
        if let Some(planes) = planes {
            panic!(
                "a fold over axis {axis} of {:?} walks the {planes} planes a ragged axis packs one after another, and one number stands for every sequence; sum the rows of each plane with Graph::segment_sum, and fold the planes of that result for one number per column",
                shape.dims(),
            );
        }
    }

    pub fn length(&self, value: Value<'g>, axis: u32) -> Value<'g> {
        let value = self.own(value);
        let shape = self.shape(value);
        assert!(
            axis < MAX_RANK,
            "a length names one of the {MAX_RANK} axes of {:?}",
            shape.dims(),
        );
        let dim = shape.dims()[axis as usize];
        assert!(
            dim <= EXACT_WALK_LIMIT,
            "axis {axis} of {:?} walks {dim} numbers, and a device counts the {EXACT_WALK_LIMIT} numbers a single precision word holds exactly",
            shape.dims(),
        );
        if shape.free(axis).is_none() {
            return self.fill(Shape::scalar(), dim as f32);
        }
        let out = self.fresh(Shape::scalar(), Element::Single, Residency::Derived, false);
        let mut task = TaskInfo::of(
            Kind::Length,
            op::NONE,
            out.id(),
            [value.id(), NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE],
        );
        task.slot = axis;
        self.push(task);
        out
    }

    pub fn mean_axis(&self, value: Value<'g>, axis: u32) -> Value<'g> {
        let value = self.own(value);
        let shape = self.shape(value);
        assert!(
            axis < MAX_RANK,
            "a mean names one of the {MAX_RANK} axes of {:?}",
            shape.dims(),
        );
        let summed = self.sum_axis(value, axis);
        let count = shape.dims()[axis as usize];
        if shape.free(axis).is_some() {
            return self.mul(summed, self.recip(self.length(value, axis)));
        }
        if count == 1 {
            return summed;
        }
        self.mul(summed, self.fill(shape.reduced(axis), 1.0 / count as f32))
    }

    pub fn broadcast_to(&self, value: Value<'g>, shape: Shape) -> Value<'g> {
        let value = self.own(value);
        let source = self.shape(value);
        assert!(
            source.fits_within(shape),
            "a broadcast spreads {:?} over {:?}",
            source.dims(),
            shape.dims(),
        );
        if source == shape {
            return value;
        }
        self.broadcast(value, shape)
    }

    pub(crate) fn fold(&self, value: Value<'g>, axis: u32) -> Value<'g> {
        let value = self.own(value);
        let shape = self.shape(value);
        assert!(
            axis < MAX_RANK,
            "a fold names one of the {MAX_RANK} axes of {:?}",
            shape.dims(),
        );
        if shape.free(axis).is_none() && shape.dims()[axis as usize] == 1 {
            return value;
        }
        let out = self.fresh(
            shape.reduced(axis),
            Element::Single,
            Residency::Derived,
            self.tracked(&[value]),
        );
        let mut task = TaskInfo::of(
            Kind::SumAxis,
            op::NONE,
            out.id(),
            [value.id(), NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE],
        );
        task.slot = axis;
        self.push(task);
        out
    }

    pub(crate) fn broadcast(&self, source: Value<'g>, shape: Shape) -> Value<'g> {
        let source = self.own(source);
        assert!(
            self.shape(source).fits_within(shape),
            "a broadcast spreads {:?} over {:?}",
            self.shape(source).dims(),
            shape.dims(),
        );
        let out = self.stored(
            shape,
            self.element(source),
            self.carries(self.element(source), &[source]),
            Residency::Derived,
            self.tracked(&[source]),
        );
        self.push(TaskInfo::of(
            Kind::Broadcast,
            op::NONE,
            out.id(),
            [
                source.id(),
                NO_VALUE,
                NO_VALUE,
                NO_VALUE,
                NO_VALUE,
                NO_VALUE,
            ],
        ));
        out
    }

    pub(crate) fn owner_of(&self, value: u32) -> u32 {
        self.state.borrow().values[value as usize].storage
    }

    pub fn trains(&self, value: Value<'g>) -> bool {
        let value = self.own(value);
        self.carries_gradient(value.id())
    }

    pub(crate) fn carries_gradient(&self, id: u32) -> bool {
        let state = self.state.borrow();
        let info = &state.values[id as usize];
        info.requires_grad && matches!(info.residency, Residency::Input | Residency::Parameter)
    }

    pub(crate) fn contiguous(&self, value: Value<'g>) -> bool {
        let value = self.own(value);
        let state = self.state.borrow();
        let info = &state.values[value.id() as usize];
        info.strides == info.shape.strides()
    }

    pub(crate) fn carries(&self, element: Element, sources: &[Value<'g>]) -> f32 {
        sources
            .iter()
            .find(|source| self.element(**source) == element)
            .map_or(1.0, |source| self.declared_quantum(*source))
    }

    pub(crate) fn tracked(&self, values: &[Value<'g>]) -> bool {
        let state = self.state.borrow();
        values
            .iter()
            .any(|value| state.values[value.id() as usize].requires_grad)
    }

    pub(crate) fn own(&self, value: Value<'g>) -> Value<'g> {
        assert_eq!(
            value.graph, self.instance,
            "a tensor of another graph reached this graph",
        );
        value
    }

    pub(crate) fn value_of(&self, id: u32) -> Value<'g> {
        Value::of(
            self.instance,
            id,
            self.state.borrow().values[id as usize].shape,
        )
    }

    pub fn name_of(&self, value: Value<'g>) -> Option<Arc<str>> {
        let value = self.own(value);
        let state = self.state.borrow();
        let info = &state.values[value.id() as usize];
        state.values[info.storage as usize].name.clone()
    }

    pub(crate) fn task(&self, index: usize) -> TaskInfo {
        self.state.borrow().tasks[index].clone()
    }

    fn hold(&self, declared: Declaration<'_>) -> Value<'g> {
        assert!(
            !matches!(declared.residency, Residency::Parameter | Residency::State)
                || !declared.shape.dynamic(),
            "a weight or a training state of {:?} holds one tensor of every run, and a free extent takes a length it cannot hold",
            declared.shape.dims(),
        );
        self.declare(declared)
    }

    pub(crate) fn declare(&self, declared: Declaration<'_>) -> Value<'g> {
        let Declaration {
            name,
            shape,
            strides,
            strides_source,
            storage,
            residency,
            element,
            scale,
            seed,
            requires_grad,
        } = declared;
        let mut state = self.state.borrow_mut();
        let name = name.map(|name| intern(&mut state, name));
        let id = state.values.len() as u32;
        state.values.push(ValueInfo {
            shape,
            strides,
            strides_source,
            storage: storage.unwrap_or(id),
            element,
            scale,
            residency,
            requires_grad,
            retained: false,
            written_in_place: false,
            recomputes: None,
            seed,
            name,
        });
        advance(&mut state);
        Value::of(self.instance, id, shape)
    }

    pub(crate) fn fresh(
        &self,
        shape: Shape,
        element: Element,
        residency: Residency,
        tracked: bool,
    ) -> Value<'g> {
        self.stored(shape, element, 1.0, residency, tracked)
    }

    fn quantized(&self, shape: Shape, scale: f32) -> Value<'g> {
        assert!(
            scale.is_finite() && scale > 0.0,
            "an int8 tensor of quantum {scale} reconstructs nothing",
        );
        self.stored(shape, Element::Int8, scale, Residency::Derived, false)
    }

    pub(crate) fn stored(
        &self,
        shape: Shape,
        element: Element,
        scale: f32,
        residency: Residency,
        tracked: bool,
    ) -> Value<'g> {
        assert!(
            !element.per_block(),
            "a {} tensor is a weight whose storage packs the quantum of every {} numbers, and a task writes the numbers of a tensor it does not weigh; cast the weight into exact storage first",
            element.name(),
            element.block(),
        );
        self.declare(
            Declaration::of(shape, element, residency)
                .scaled(scale)
                .trained(tracked),
        )
    }

    pub(crate) fn push(&self, task: TaskInfo) {
        let mut state = self.state.borrow_mut();
        state.tasks.push(task);
        advance(&mut state);
    }
}

impl<'g> Default for Graph<'g> {
    fn default() -> Self {
        Self::new()
    }
}

fn reads(task: &TaskInfo, value: u32) -> bool {
    task.inputs.contains(&value)
        || task.origin == value
        || task.prelude.iter().any(|step| step.operand == value)
        || task.chain.iter().any(|step| step.operand == value)
}

fn intern(state: &mut GraphState, name: &str) -> Arc<str> {
    assert!(
        !name.is_empty(),
        "a tensor is named by a non-empty name alone",
    );
    let name: Arc<str> = Arc::from(name);
    assert!(
        state.names.insert(name.clone()),
        "two tensors of this graph carry the name {name}, and a name identifies one tensor",
    );
    name
}
