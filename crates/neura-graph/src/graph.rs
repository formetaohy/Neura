use crate::autodiff::Recomputation;
use crate::init::Init;
use crate::shape::{Free, Shape};
use crate::window::Window;
use neura_abi::{Element, Kind, MAX_RANK, NO_VALUE, StepRecord};
use neura_pointwise as op;
use std::cell::RefCell;
use std::collections::HashSet;
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
    pub scale: f32,
    pub causal: bool,
    pub origin: Option<Value<'g>>,
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
    pub param: f32,
    pub window: Window,
    pub in_place: bool,
    pub axis: u32,
    pub offset: u32,
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
            param: 0.0,
            window: Window::sliding([1, 1]),
            in_place: false,
            axis: 0,
            offset: 0,
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
    pub(crate) differentiated: bool,
    pub(crate) updated_in_place: bool,
    pub(crate) version: u64,
    pub(crate) frees: u32,
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
        assert!(
            !tracked,
            "a device authored extent walks a length no gradient knows, and value {} trains",
            value.id(),
        );
        let free = self.counted(shape.dims()[axis as usize], count);
        let mut frees = shape.frees();
        frees[axis as usize] = Some(free.slot());
        self.alias(
            Shape::from_axes(shape.dims(), frees),
            strides,
            strides_source,
            storage,
            element,
            scale,
            false,
        )
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

    pub fn input(&self, shape: Shape, element: Element) -> Value<'g> {
        assert!(
            !element.quantized(),
            "an input of {} storage is declared with the quantum it reconstructs by, and an input carries none; quantize the tensor that reads it instead",
            element.name(),
        );
        self.hold(None, shape, Residency::Input, element, 1.0, None, false)
    }

    pub fn gradient_input(&self, shape: Shape, element: Element) -> Value<'g> {
        assert!(
            !element.quantized(),
            "a gradient input of {} storage is declared with the quantum it reconstructs by, and a gradient input carries none; quantize the tensor that reads it instead",
            element.name(),
        );
        self.hold(None, shape, Residency::Input, element, 1.0, None, true)
    }

    pub fn resident(&self, shape: Shape, element: Element) -> Value<'g> {
        assert!(
            !element.quantized(),
            "a resident tensor of {} storage is declared with the quantum it reconstructs by, and a resident tensor carries none; quantize the tensor that reads it instead",
            element.name(),
        );
        self.hold(None, shape, Residency::Resident, element, 1.0, None, false)
    }

    pub fn parameter(&self, shape: Shape, init: Init, element: Element) -> Value<'g> {
        assert!(
            !element.quantized(),
            "a parameter of {} storage is declared with the quantum it reconstructs by, and a parameter carries none; declare a quantized or a block quantized parameter instead",
            element.name(),
        );
        self.hold(
            None,
            shape,
            Residency::Parameter,
            element,
            1.0,
            Some(init),
            true,
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
            Some(name),
            shape,
            Residency::Parameter,
            element,
            1.0,
            Some(init),
            true,
        )
    }

    pub fn state(&self, shape: Shape, init: Init, element: Element) -> Value<'g> {
        assert!(
            !element.quantized(),
            "a training state of {} storage is declared with the quantum it reconstructs by, and a training state carries none",
            element.name(),
        );
        self.hold(
            None,
            shape,
            Residency::State,
            element,
            1.0,
            Some(init),
            false,
        )
    }

    pub fn named_state(&self, name: &str, shape: Shape, init: Init, element: Element) -> Value<'g> {
        assert!(
            !element.quantized(),
            "a training state of {} storage is declared with the quantum it reconstructs by, and a training state carries none",
            element.name(),
        );
        self.hold(
            Some(name),
            shape,
            Residency::State,
            element,
            1.0,
            Some(init),
            false,
        )
    }

    pub fn quantized_parameter(&self, shape: Shape, init: Init, scale: f32) -> Value<'g> {
        self.hold(
            None,
            shape,
            Residency::Parameter,
            Element::Int8,
            scale,
            Some(init),
            false,
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
            Some(name),
            shape,
            Residency::Parameter,
            Element::Int8,
            scale,
            Some(init),
            false,
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
        self.hold(
            None,
            shape,
            Residency::Parameter,
            element,
            1.0,
            Some(init),
            false,
        )
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
            Some(name),
            shape,
            Residency::Parameter,
            element,
            1.0,
            Some(init),
            false,
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
        task.param = value;
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
        self.alias(
            shape,
            strides,
            strides_source,
            storage,
            element,
            scale,
            false,
        )
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
        if shape.free(axis).is_none() && shape.dims()[axis as usize] == 1 {
            return value;
        }
        self.fold(value, axis)
    }

    pub fn mean_axis(&self, value: Value<'g>, axis: u32) -> Value<'g> {
        let value = self.own(value);
        let shape = self.shape(value);
        assert!(
            axis < MAX_RANK,
            "a mean names one of the {MAX_RANK} axes of {:?}",
            shape.dims(),
        );
        assert!(
            shape.free(axis).is_none(),
            "a mean over axis {axis} of {:?} weighs every length the free extent takes, and the weight a plan carries is one number",
            shape.dims(),
        );
        let summed = self.sum_axis(value, axis);
        let count = shape.dims()[axis as usize];
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
        assert!(
            shape.dims()[axis as usize] > 1,
            "folding axis {axis} of {:?} reduces a single element",
            shape.dims(),
        );
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

    #[allow(clippy::too_many_arguments)]
    fn hold(
        &self,
        name: Option<&str>,
        shape: Shape,
        residency: Residency,
        element: Element,
        scale: f32,
        seed: Option<Init>,
        requires_grad: bool,
    ) -> Value<'g> {
        assert!(
            !matches!(residency, Residency::Parameter | Residency::State) || !shape.dynamic(),
            "a weight or a training state of {:?} holds one tensor of every run, and a free extent takes a length it cannot hold",
            shape.dims(),
        );
        let mut state = self.state.borrow_mut();
        let name = name.map(|name| intern(&mut state, name));
        let id = state.values.len() as u32;
        state.values.push(ValueInfo {
            shape,
            strides: shape.strides(),
            strides_source: None,
            storage: id,
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
        let mut state = self.state.borrow_mut();
        let id = state.values.len() as u32;
        state.values.push(ValueInfo {
            shape,
            strides: shape.strides(),
            strides_source: None,
            storage: id,
            element,
            scale,
            residency,
            requires_grad: tracked,
            retained: false,
            written_in_place: false,
            recomputes: None,
            seed: None,
            name: None,
        });
        advance(&mut state);
        Value::of(self.instance, id, shape)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn alias(
        &self,
        shape: Shape,
        strides: [u32; 4],
        strides_source: Option<[u8; 4]>,
        storage: u32,
        element: Element,
        scale: f32,
        tracked: bool,
    ) -> Value<'g> {
        let mut state = self.state.borrow_mut();
        let id = state.values.len() as u32;
        state.values.push(ValueInfo {
            shape,
            strides,
            strides_source,
            storage,
            element,
            scale,
            residency: Residency::View,
            requires_grad: tracked,
            retained: false,
            written_in_place: false,
            recomputes: None,
            seed: None,
            name: None,
        });
        advance(&mut state);
        Value::of(self.instance, id, shape)
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
