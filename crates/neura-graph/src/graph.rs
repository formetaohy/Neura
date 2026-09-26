use crate::init::Init;
use crate::pool::Pool;
use crate::shape::Shape;
use crate::window::Window;
use neura_abi::{Element, Kind, MAX_RANK, NO_VALUE, StepRecord};
use neura_op as op;
use std::cell::RefCell;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_GRAPH: AtomicU64 = AtomicU64::new(1);

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
pub struct AttentionOptions {
    pub scale: f32,
    pub causal: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Residency {
    Input,
    Parameter,
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
    pub slot: u32,
    pub param: f32,
    pub window: Window,
    pub in_place: bool,
    pub prelude: Vec<StepRecord>,
    pub chain: Vec<StepRecord>,
}

impl TaskInfo {
    fn of(kind: Kind, op: u32, out: u32, inputs: [u32; 6]) -> Self {
        Self {
            kind,
            op,
            out,
            extra: NO_VALUE,
            inputs,
            slot: 0,
            param: 0.0,
            window: Window::sliding([1, 1]),
            in_place: false,
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
    pub storage: u32,
    pub element: Element,
    pub residency: Residency,
    pub requires_grad: bool,
    pub retained: bool,
    pub written_in_place: bool,
    pub seed: Option<Init>,
}

impl ValueInfo {
    pub fn derived(shape: Shape, id: u32) -> Self {
        Self {
            shape,
            strides: shape.strides(),
            storage: id,
            element: Element::Single,
            residency: Residency::Derived,
            requires_grad: false,
            retained: false,
            written_in_place: false,
            seed: None,
        }
    }
}

struct GraphState {
    values: Vec<ValueInfo>,
    tasks: Vec<TaskInfo>,
    differentiated: bool,
    updated_in_place: bool,
}

pub struct GraphSnapshot {
    values: Vec<ValueInfo>,
    tasks: Vec<TaskInfo>,
}

impl GraphSnapshot {
    pub fn values(&self) -> &[ValueInfo] {
        &self.values
    }

    pub fn tasks(&self) -> &[TaskInfo] {
        &self.tasks
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gradients<'g> {
    values: HashMap<u32, Value<'g>>,
}

impl<'g> Gradients<'g> {
    pub fn of(&self, value: Value<'g>) -> Value<'g> {
        *self.values.get(&value.id()).unwrap_or_else(|| {
            panic!(
                "no gradient reaches {:?} from the loss, and the gradient of a view reaches the tensor that owns its storage",
                value.shape(),
            )
        })
    }
}

pub struct Graph<'g> {
    instance: u64,
    state: RefCell<GraphState>,
    brand: PhantomData<fn(&'g ()) -> &'g ()>,
}

impl<'g> Graph<'g> {
    pub fn new() -> Self {
        Self {
            state: RefCell::new(GraphState {
                values: Vec::new(),
                tasks: Vec::new(),
                differentiated: false,
                updated_in_place: false,
            }),
            instance: NEXT_GRAPH.fetch_add(1, Ordering::Relaxed),
            brand: PhantomData,
        }
    }

    pub fn input(&self, shape: Shape, element: Element) -> Value<'g> {
        self.hold(shape, Residency::Input, element, None)
    }

    pub fn resident(&self, shape: Shape, element: Element) -> Value<'g> {
        self.hold(shape, Residency::Resident, element, None)
    }

    pub fn parameter(&self, shape: Shape, init: Init, element: Element) -> Value<'g> {
        self.hold(shape, Residency::Parameter, element, Some(init))
    }

    pub fn fill(&self, shape: Shape, value: f32) -> Value<'g> {
        let out = self.fresh(shape, Residency::Derived, false);
        let mut task = TaskInfo::of(Kind::Fill, op::NONE, out.id(), [NO_VALUE; 6]);
        task.param = value;
        self.push(task);
        out
    }

    pub fn matmul(&self, left: Value<'g>, right: Value<'g>) -> Value<'g> {
        let left = self.own(left);
        let right = self.own(right);
        let left_dims = left.shape().dims();
        let right_dims = right.shape().dims();
        assert_eq!(
            left_dims[3], right_dims[2],
            "a matmul of {:?} by {:?} has no shared depth",
            left_dims, right_dims,
        );
        let (left_batch, right_batch) = (left.shape().batch(), right.shape().batch());
        assert!(
            left_batch
                .iter()
                .zip(right_batch)
                .all(|(left, right)| left == &right || *left == 1 || right == 1),
            "a matmul of {:?} by {:?} carries batches that do not meet",
            left_dims,
            right_dims,
        );
        let out = self.fresh(
            Shape::of([
                left_batch[0].max(right_batch[0]),
                left_batch[1].max(right_batch[1]),
                left_dims[2],
                right_dims[3],
            ]),
            Residency::Derived,
            self.tracked(&[left, right]),
        );
        self.push(TaskInfo::of(
            Kind::Matmul,
            op::NONE,
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

    pub fn attention(
        &self,
        query: Value<'g>,
        key: Value<'g>,
        value: Value<'g>,
        attention: AttentionOptions,
    ) -> Value<'g> {
        let query = self.own(query);
        let key = self.own(key);
        let value = self.own(value);
        let query_shape = self.shape(query);
        let key_shape = self.shape(key);
        let value_shape = self.shape(value);
        assert!(
            attention.scale.is_finite() && attention.scale != 0.0,
            "an attention scaled by {} weighs every score to nothing",
            attention.scale,
        );
        assert_eq!(
            query_shape.batch(),
            key_shape.batch(),
            "an attention reads {query_shape:?} through keys of {key_shape:?}",
        );
        assert_eq!(
            key_shape.batch(),
            value_shape.batch(),
            "an attention reads keys of {key_shape:?} through values of {value_shape:?}",
        );
        assert_eq!(
            query_shape.dims()[3],
            key_shape.dims()[3],
            "an attention of width {} scores keys of width {}",
            query_shape.dims()[3],
            key_shape.dims()[3],
        );
        assert_eq!(
            key_shape.dims()[2],
            value_shape.dims()[2],
            "an attention weighs {} keys by {} values",
            key_shape.dims()[2],
            value_shape.dims()[2],
        );
        assert!(
            !attention.causal || query_shape.dims()[2] == key_shape.dims()[2],
            "a causal attention walks {} queries over {} keys, and a mask aligns them one by one",
            query_shape.dims()[2],
            key_shape.dims()[2],
        );
        let tracked = self.tracked(&[query, key, value]);
        let out = self.fresh(
            Shape::of([
                query_shape.dims()[0],
                query_shape.dims()[1],
                query_shape.dims()[2],
                value_shape.dims()[3],
            ]),
            Residency::Derived,
            tracked,
        );
        let log_sum_exp = self.fresh(
            Shape::of([
                query_shape.dims()[0],
                query_shape.dims()[1],
                query_shape.dims()[2],
                1,
            ]),
            Residency::Derived,
            false,
        );
        let mut task = TaskInfo::of(
            Kind::Attention,
            op::NONE,
            out.id(),
            [
                query.id(),
                key.id(),
                value.id(),
                NO_VALUE,
                NO_VALUE,
                NO_VALUE,
            ],
        );
        task.extra = log_sum_exp.id();
        task.param = attention.scale;
        task.slot = u32::from(attention.causal);
        self.push(task);
        out
    }

    pub fn conv2d(&self, input: Value<'g>, filter: Value<'g>, window: Window) -> Value<'g> {
        let input = self.own(input);
        let filter = self.own(filter);
        let input_dims = self.shape(input).dims();
        let filter_dims = self.shape(filter).dims();
        assert!(
            input_dims[1].is_multiple_of(filter_dims[1]),
            "a convolution reads {} channels through a filter of {}",
            input_dims[1],
            filter_dims[1],
        );
        let groups = input_dims[1] / filter_dims[1];
        assert!(
            filter_dims[0].is_multiple_of(groups),
            "a convolution of {groups} channel groups writes {} output channels",
            filter_dims[0],
        );
        assert_eq!(
            [filter_dims[2], filter_dims[3]],
            [window.reach_rows(), window.reach_columns()],
            "a window of {} by {} taps walks a filter of {} by {} taps",
            window.reach_rows(),
            window.reach_columns(),
            filter_dims[2],
            filter_dims[3],
        );
        let padded_rows = input_dims[2] + 2 * window.pad_rows();
        let padded_columns = input_dims[3] + 2 * window.pad_columns();
        assert!(
            padded_rows >= filter_dims[2] && padded_columns >= filter_dims[3],
            "a window of {} by {} taps over {:?} padded by {} by {} reaches no position",
            filter_dims[2],
            filter_dims[3],
            input_dims,
            window.pad_rows(),
            window.pad_columns(),
        );
        let out = self.fresh(
            Shape::of([
                input_dims[0],
                filter_dims[0],
                (padded_rows - filter_dims[2]) / window.stride_rows() + 1,
                (padded_columns - filter_dims[3]) / window.stride_columns() + 1,
            ]),
            Residency::Derived,
            self.tracked(&[input, filter]),
        );
        let mut task = TaskInfo::of(
            Kind::Conv2d,
            op::NONE,
            out.id(),
            [
                input.id(),
                filter.id(),
                NO_VALUE,
                NO_VALUE,
                NO_VALUE,
                NO_VALUE,
            ],
        );
        task.window = window;
        self.push(task);
        out
    }

    pub fn pool2d(&self, input: Value<'g>, window: Window, mode: Pool) -> Value<'g> {
        let input = self.own(input);
        let input_dims = self.shape(input).dims();
        let padded_rows = input_dims[2] + 2 * window.pad_rows();
        let padded_columns = input_dims[3] + 2 * window.pad_columns();
        assert!(
            padded_rows >= window.reach_rows() && padded_columns >= window.reach_columns(),
            "a window of {} by {} taps over {:?} padded by {} by {} reaches no position",
            window.reach_rows(),
            window.reach_columns(),
            input_dims,
            window.pad_rows(),
            window.pad_columns(),
        );
        let out = self.fresh(
            Shape::of([
                input_dims[0],
                input_dims[1],
                (padded_rows - window.reach_rows()) / window.stride_rows() + 1,
                (padded_columns - window.reach_columns()) / window.stride_columns() + 1,
            ]),
            Residency::Derived,
            self.tracked(&[input]),
        );
        let kind = match mode {
            Pool::Max => Kind::PoolMax2d,
            Pool::Mean => Kind::PoolMean2d,
        };
        let mut task = TaskInfo::of(
            kind,
            op::NONE,
            out.id(),
            [input.id(), NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE],
        );
        task.window = window;
        self.push(task);
        out
    }

    pub fn add(&self, left: Value<'g>, right: Value<'g>) -> Value<'g> {
        let left = self.own(left);
        let right = self.own(right);
        self.elementwise(op::ADD, left, right)
    }

    pub fn mul(&self, left: Value<'g>, right: Value<'g>) -> Value<'g> {
        let left = self.own(left);
        let right = self.own(right);
        self.elementwise(op::MUL, left, right)
    }

    pub fn sub(&self, left: Value<'g>, right: Value<'g>) -> Value<'g> {
        let left = self.own(left);
        let right = self.own(right);
        self.elementwise(op::SUB, left, right)
    }

    pub fn div(&self, left: Value<'g>, right: Value<'g>) -> Value<'g> {
        let left = self.own(left);
        let right = self.own(right);
        self.elementwise(op::DIV, left, right)
    }

    pub fn max(&self, left: Value<'g>, right: Value<'g>) -> Value<'g> {
        let left = self.own(left);
        let right = self.own(right);
        self.elementwise(op::MAXIMUM, left, right)
    }

    pub fn min(&self, left: Value<'g>, right: Value<'g>) -> Value<'g> {
        let left = self.own(left);
        let right = self.own(right);
        self.elementwise(op::MINIMUM, left, right)
    }

    pub fn relu(&self, value: Value<'g>) -> Value<'g> {
        let value = self.own(value);
        self.unary(op::RELU, value)
    }

    pub fn sqrt(&self, value: Value<'g>) -> Value<'g> {
        let value = self.own(value);
        self.unary(op::SQRT, value)
    }

    pub fn recip(&self, value: Value<'g>) -> Value<'g> {
        let value = self.own(value);
        self.unary(op::RECIP, value)
    }

    pub fn exp(&self, value: Value<'g>) -> Value<'g> {
        let value = self.own(value);
        self.unary(op::EXP, value)
    }

    pub fn log(&self, value: Value<'g>) -> Value<'g> {
        let value = self.own(value);
        self.unary(op::LOG, value)
    }

    pub fn tanh(&self, value: Value<'g>) -> Value<'g> {
        let value = self.own(value);
        self.unary(op::TANH, value)
    }

    pub fn sigmoid(&self, value: Value<'g>) -> Value<'g> {
        let value = self.own(value);
        self.unary(op::SIGMOID, value)
    }

    pub fn neg(&self, value: Value<'g>) -> Value<'g> {
        let value = self.own(value);
        self.unary(op::NEG, value)
    }

    pub fn abs(&self, value: Value<'g>) -> Value<'g> {
        let value = self.own(value);
        self.unary(op::ABS, value)
    }

    pub fn identity(&self, value: Value<'g>) -> Value<'g> {
        let value = self.own(value);
        self.unary(op::IDENTITY, value)
    }

    pub fn softmax(&self, value: Value<'g>) -> Value<'g> {
        let value = self.own(value);
        self.rows(Kind::Softmax, value)
    }

    pub fn log_softmax(&self, value: Value<'g>) -> Value<'g> {
        let value = self.own(value);
        self.rows(Kind::LogSoftmax, value)
    }

    pub fn argmax(&self, value: Value<'g>) -> Value<'g> {
        let value = self.own(value);
        self.choice(Kind::Argmax, value, NO_VALUE)
    }

    pub fn categorical(&self, logits: Value<'g>, seed: Value<'g>) -> Value<'g> {
        let logits = self.own(logits);
        let seed = self.own(seed);
        assert!(
            self.shape(seed).is_scalar(),
            "a categorical draw takes one seed, and value {} holds {} elements",
            seed.id(),
            self.shape(seed).elements(),
        );
        self.choice(Kind::Categorical, logits, seed.id())
    }

    fn choice(&self, kind: Kind, source: Value<'g>, seed: u32) -> Value<'g> {
        let source = self.own(source);
        assert!(
            self.contiguous(source),
            "a {} folds a row of a tensor stored row by row, and value {} is a view",
            kind.name(),
            source.id(),
        );
        let mut dims = self.shape(source).dims();
        dims[3] = 1;
        let out = self.fresh(Shape::of(dims), Residency::Derived, false);
        self.push(TaskInfo::of(
            kind,
            op::NONE,
            out.id(),
            [source.id(), seed, NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE],
        ));
        out
    }

    fn index_list(&self, indices: Value<'g>) {
        let indices = self.own(indices);
        let shape = self.shape(indices);
        assert_eq!(
            shape.dims()[3],
            1,
            "an index list holds one index per row, and {:?} holds {} of them",
            shape.dims(),
            shape.dims()[3],
        );
        assert!(
            self.contiguous(indices),
            "an index list is walked row by row, and value {} is a view",
            indices.id(),
        );
    }

    pub fn one_hot(&self, indices: Value<'g>, classes: u32) -> Value<'g> {
        let indices = self.own(indices);
        self.index_list(indices);
        assert!(
            classes > 0,
            "a one hot tensor of {classes} classes holds none"
        );
        let mut dims = self.shape(indices).dims();
        dims[3] = classes;
        let out = self.fresh(Shape::of(dims), Residency::Derived, false);
        self.push(TaskInfo::of(
            Kind::OneHot,
            op::NONE,
            out.id(),
            [
                indices.id(),
                NO_VALUE,
                NO_VALUE,
                NO_VALUE,
                NO_VALUE,
                NO_VALUE,
            ],
        ));
        out
    }

    pub fn gather(&self, table: Value<'g>, indices: Value<'g>) -> Value<'g> {
        let table = self.own(table);
        let indices = self.own(indices);
        self.index_list(indices);
        assert!(
            self.contiguous(table),
            "a gather walks a table row by row, and value {} is a view",
            table.id(),
        );
        let mut dims = self.shape(indices).dims();
        dims[3] = self.shape(table).dims()[3];
        let out = self.fresh(Shape::of(dims), Residency::Derived, self.tracked(&[table]));
        self.push(TaskInfo::of(
            Kind::Gather,
            op::NONE,
            out.id(),
            [
                table.id(),
                indices.id(),
                NO_VALUE,
                NO_VALUE,
                NO_VALUE,
                NO_VALUE,
            ],
        ));
        out
    }

    pub fn scatter_into(&self, target: Value<'g>, indices: Value<'g>, updates: Value<'g>) {
        let target = self.own(target);
        let indices = self.own(indices);
        let updates = self.own(updates);
        {
            let state = self.state.borrow();
            let info = &state.values[target.id() as usize];
            assert!(
                matches!(
                    info.residency,
                    Residency::Input | Residency::Parameter | Residency::Resident
                ),
                "only a leaf tensor scatters in place, and value {} is derived from other tasks",
                target.id(),
            );
        }
        self.scatter(target, indices, updates);
        self.wrote_in_place(target);
    }

    pub fn sum(&self, value: Value<'g>) -> Value<'g> {
        let value = self.own(value);
        assert!(
            self.contiguous(value),
            "a sum walks its operand element by element, and value {} is a view",
            value.id(),
        );
        let out = self.fresh(Shape::scalar(), Residency::Derived, self.tracked(&[value]));
        self.push(TaskInfo::of(
            Kind::SumChunk,
            op::NONE,
            out.id(),
            [value.id(), NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE],
        ));
        out
    }

    pub fn permute(&self, value: Value<'g>, order: [u32; 4]) -> Value<'g> {
        let value = self.own(value);
        let mut walked = 0u32;
        for axis in order {
            assert!(
                axis < MAX_RANK && walked & (1 << axis) == 0,
                "a permutation walks each of the {MAX_RANK} axes of {:?} once, and {order:?} does not",
                self.shape(value).dims(),
            );
            walked |= 1 << axis;
        }
        let (dims, strides, storage, element, tracked) = {
            let state = self.state.borrow();
            let info = &state.values[value.id() as usize];
            (
                info.shape.dims(),
                info.strides,
                info.storage,
                info.element,
                info.requires_grad,
            )
        };
        let mut permuted_dims = [1u32; MAX_RANK as usize];
        let mut permuted_strides = [0u32; MAX_RANK as usize];
        for axis in 0..MAX_RANK as usize {
            permuted_dims[axis] = dims[order[axis] as usize];
            permuted_strides[axis] = strides[order[axis] as usize];
        }
        self.alias(
            Shape::of(permuted_dims),
            permuted_strides,
            storage,
            element,
            tracked,
        )
    }

    pub fn reshape(&self, value: Value<'g>, shape: Shape) -> Value<'g> {
        let value = self.own(value);
        assert!(
            self.contiguous(value),
            "a reshape reads a tensor its storage lays out row by row, and value {} walks other strides; materialize it first",
            value.id(),
        );
        assert_eq!(
            shape.elements(),
            self.shape(value).elements(),
            "a reshape holds {} numbers where {} numbers are reshaped",
            shape.elements(),
            self.shape(value).elements(),
        );
        let (storage, element, tracked) = {
            let state = self.state.borrow();
            let info = &state.values[value.id() as usize];
            (info.storage, info.element, info.requires_grad)
        };
        self.alias(shape, shape.strides(), storage, element, tracked)
    }

    pub fn add_into(&self, target: Value<'g>, addend: Value<'g>) {
        let target = self.own(target);
        let addend = self.own(addend);
        self.update_in_place(op::ADD, target, addend);
    }

    pub fn mul_into(&self, target: Value<'g>, factor: Value<'g>) {
        let target = self.own(target);
        let factor = self.own(factor);
        self.update_in_place(op::MUL, target, factor);
    }

    pub fn copy_into(&self, target: Value<'g>, source: Value<'g>) {
        let target = self.own(target);
        let source = self.own(source);
        self.assert_in_place(target, source);
        let mut task = TaskInfo::of(
            Kind::Unary,
            op::IDENTITY,
            target.id(),
            [
                source.id(),
                NO_VALUE,
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

    pub fn shape(&self, value: Value<'g>) -> Shape {
        let value = self.own(value);
        self.state.borrow().values[value.id() as usize].shape
    }

    pub fn element(&self, value: Value<'g>) -> Element {
        let value = self.own(value);
        self.state.borrow().values[value.id() as usize].element
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
        }
    }

    pub fn updates_weights(&self) -> bool {
        let state = self.state.borrow();
        state.tasks.iter().any(|task| {
            task.in_place && state.values[task.out as usize].residency == Residency::Parameter
        })
    }

    pub fn backward(&self, loss: Value<'g>) -> Gradients<'g> {
        let loss = self.own(loss);
        {
            let state = self.state.borrow();
            assert!(
                !state.differentiated,
                "a graph is differentiated once; build another graph for a second pass",
            );
            assert!(
                !state.updated_in_place,
                "a graph is differentiated before any of its leaves is updated in place",
            );
            assert!(
                state.values[loss.id() as usize].requires_grad,
                "the loss derives from no parameter",
            );
        }
        assert!(
            self.shape(loss).is_scalar(),
            "the loss must be a scalar tensor, not {:?}",
            self.shape(loss),
        );
        self.retain(loss);
        let forward = {
            let mut state = self.state.borrow_mut();
            state.differentiated = true;
            state.tasks.len()
        };
        let mut grads: Vec<Option<u32>> = vec![None; self.value_count()];
        let seed = self.fill(self.shape(loss), 1.0);
        self.accumulate(&mut grads, loss, seed);
        for index in (0..forward).rev() {
            let task = self.task(index);
            let Some(gradient) = grads[self.owner_of(task.out) as usize] else {
                continue;
            };
            let gradient = self.value_of(gradient);
            self.backward_task(&task, gradient, &mut grads);
        }
        let mut values = HashMap::new();
        for (id, grad) in grads.into_iter().enumerate() {
            if let Some(grad) = grad {
                values.insert(id as u32, self.value_of(grad));
            }
        }
        Gradients { values }
    }

    fn backward_task(&self, task: &TaskInfo, gradient: Value<'g>, grads: &mut [Option<u32>]) {
        let gradient = self.own(gradient);
        match task.kind {
            Kind::Matmul => {
                let (left, right) = (self.value_of(task.inputs[0]), self.value_of(task.inputs[1]));
                if self.tracked(&[left]) {
                    let transposed = self.permute(right, [0, 1, 3, 2]);
                    let contribution = self.matmul(gradient, transposed);
                    self.accumulate(grads, left, contribution);
                }
                if self.tracked(&[right]) {
                    let transposed = self.permute(left, [0, 1, 3, 2]);
                    let contribution = self.matmul(transposed, gradient);
                    self.accumulate(grads, right, contribution);
                }
            }
            Kind::Binary | Kind::Unary => {
                let definition = op::of(task.op);
                for slot in 0..definition.family.operands() {
                    let operand = self.value_of(task.inputs[slot as usize]);
                    if !self.tracked(&[operand]) {
                        continue;
                    }
                    let contribution = self.partial(definition, task, slot, gradient);
                    self.accumulate(grads, operand, contribution);
                }
            }
            Kind::Softmax => {
                let source = self.value_of(task.inputs[0]);
                if self.tracked(&[source]) {
                    self.row_gradient(Kind::SoftmaxGrad, task, source, gradient, grads);
                }
            }
            Kind::LogSoftmax => {
                let source = self.value_of(task.inputs[0]);
                if self.tracked(&[source]) {
                    self.row_gradient(Kind::LogSoftmaxGrad, task, source, gradient, grads);
                }
            }
            Kind::SumChunk | Kind::SumAxis => {
                let source = self.value_of(task.inputs[0]);
                if self.tracked(&[source]) {
                    let out = self.broadcast(gradient, self.shape(source));
                    self.accumulate(grads, source, out);
                }
            }
            Kind::Attention => {
                let query = self.value_of(task.inputs[0]);
                let key = self.value_of(task.inputs[1]);
                let value = self.value_of(task.inputs[2]);
                assert_ne!(
                    task.extra, NO_VALUE,
                    "an attention carries the log sum of every row it weighs",
                );
                let operands = [
                    task.inputs[0],
                    task.inputs[1],
                    task.inputs[2],
                    gradient.id(),
                    task.out,
                    task.extra,
                ];
                let without_output = [
                    operands[0],
                    operands[1],
                    NO_VALUE,
                    operands[3],
                    NO_VALUE,
                    operands[5],
                ];
                for (operand, kind, inputs) in [
                    (query, Kind::AttentionQueryGrad, operands),
                    (key, Kind::AttentionKeyGrad, operands),
                    (value, Kind::AttentionValueGrad, without_output),
                ] {
                    if !self.tracked(&[operand]) {
                        continue;
                    }
                    let out = self.fresh(self.shape(operand), Residency::Derived, false);
                    let mut grad = TaskInfo::of(kind, op::NONE, out.id(), inputs);
                    grad.param = task.param;
                    grad.slot = task.slot;
                    self.push(grad);
                    self.accumulate(grads, operand, out);
                }
            }
            Kind::Conv2d => {
                let input = self.value_of(task.inputs[0]);
                let filter = self.value_of(task.inputs[1]);
                if self.tracked(&[input]) {
                    let out = self.fresh(self.shape(input), Residency::Derived, false);
                    let mut grad = TaskInfo::of(
                        Kind::Conv2dInputGrad,
                        op::NONE,
                        out.id(),
                        [
                            filter.id(),
                            gradient.id(),
                            NO_VALUE,
                            NO_VALUE,
                            NO_VALUE,
                            NO_VALUE,
                        ],
                    );
                    grad.window = task.window;
                    self.push(grad);
                    self.accumulate(grads, input, out);
                }
                if self.tracked(&[filter]) {
                    let out = self.fresh(self.shape(filter), Residency::Derived, false);
                    let mut grad = TaskInfo::of(
                        Kind::Conv2dWeightGrad,
                        op::NONE,
                        out.id(),
                        [
                            input.id(),
                            gradient.id(),
                            filter.id(),
                            NO_VALUE,
                            NO_VALUE,
                            NO_VALUE,
                        ],
                    );
                    grad.window = task.window;
                    self.push(grad);
                    self.accumulate(grads, filter, out);
                }
            }
            Kind::Gather => {
                let table = self.value_of(task.inputs[0]);
                let indices = self.value_of(task.inputs[1]);
                if self.tracked(&[table]) {
                    let zeros = self.fill(self.shape(table), 0.0);
                    self.scatter(zeros, indices, gradient);
                    self.accumulate(grads, table, zeros);
                }
            }
            Kind::PoolMax2d | Kind::PoolMean2d => {
                let input = self.value_of(task.inputs[0]);
                if self.tracked(&[input]) {
                    let kind = if task.kind == Kind::PoolMax2d {
                        Kind::PoolMax2dInputGrad
                    } else {
                        Kind::PoolMean2dInputGrad
                    };
                    let out = self.fresh(self.shape(input), Residency::Derived, false);
                    let mut grad = TaskInfo::of(
                        kind,
                        op::NONE,
                        out.id(),
                        [
                            input.id(),
                            gradient.id(),
                            NO_VALUE,
                            NO_VALUE,
                            NO_VALUE,
                            NO_VALUE,
                        ],
                    );
                    grad.window = task.window;
                    self.push(grad);
                    self.accumulate(grads, input, out);
                }
            }
            Kind::Fill
            | Kind::Broadcast
            | Kind::Layout
            | Kind::Partial
            | Kind::SoftmaxGrad
            | Kind::LogSoftmaxGrad
            | Kind::AttentionQueryGrad
            | Kind::AttentionKeyGrad
            | Kind::AttentionValueGrad
            | Kind::Conv2dInputGrad
            | Kind::Conv2dWeightGrad
            | Kind::PoolMax2dInputGrad
            | Kind::PoolMean2dInputGrad
            | Kind::MatmulFold
            | Kind::Pack
            | Kind::Scatter => {}
            Kind::Argmax | Kind::Categorical | Kind::OneHot => {
                panic!(
                    "the {} task yields the index of a row, and an index carries no gradient",
                    task.kind.name(),
                )
            }
        }
    }

    fn partial(
        &self,
        definition: &op::Op,
        task: &TaskInfo,
        slot: u32,
        gradient: Value<'g>,
    ) -> Value<'g> {
        let gradient = self.own(gradient);
        let op::Partial::Formula { roles, .. } = definition.partial(slot) else {
            return gradient;
        };
        let left = if roles.contains(&op::Role::Result) {
            task.out
        } else if roles.contains(&op::Role::Operand) {
            task.inputs[slot as usize]
        } else {
            NO_VALUE
        };
        let right = if roles.contains(&op::Role::Other) {
            task.inputs[slot as usize ^ 1]
        } else {
            NO_VALUE
        };
        let out = self.fresh(self.shape(gradient), Residency::Derived, false);
        let mut partial = TaskInfo::of(
            Kind::Partial,
            task.op,
            out.id(),
            [left, right, gradient.id(), NO_VALUE, NO_VALUE, NO_VALUE],
        );
        partial.slot = slot;
        self.push(partial);
        out
    }

    fn row_gradient(
        &self,
        kind: Kind,
        task: &TaskInfo,
        source: Value<'g>,
        gradient: Value<'g>,
        grads: &mut [Option<u32>],
    ) {
        let source = self.own(source);
        let gradient = self.own(gradient);
        let out = self.fresh(self.shape(source), Residency::Derived, true);
        self.push(TaskInfo::of(
            kind,
            op::NONE,
            out.id(),
            [
                task.out,
                gradient.id(),
                NO_VALUE,
                NO_VALUE,
                NO_VALUE,
                NO_VALUE,
            ],
        ));
        self.accumulate(grads, source, out);
    }

    fn rows(&self, kind: Kind, value: Value<'g>) -> Value<'g> {
        let value = self.own(value);
        assert!(
            self.contiguous(value),
            "a {} folds a row of a tensor stored row by row, and value {} is a view",
            kind.name(),
            value.id(),
        );
        let shape = self.shape(value);
        let out = self.fresh(shape, Residency::Derived, self.tracked(&[value]));
        self.push(TaskInfo::of(
            kind,
            op::NONE,
            out.id(),
            [value.id(), NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE],
        ));
        out
    }

    fn elementwise(&self, op: u32, left: Value<'g>, right: Value<'g>) -> Value<'g> {
        let left = self.own(left);
        let right = self.own(right);
        let shape = self.shape(left).combined(self.shape(right));
        let out = self.fresh(shape, Residency::Derived, self.tracked(&[left, right]));
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

    fn unary(&self, op: u32, value: Value<'g>) -> Value<'g> {
        let value = self.own(value);
        let shape = self.shape(value);
        let out = self.fresh(shape, Residency::Derived, self.tracked(&[value]));
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
        self.state.borrow_mut().values[storage as usize].retained = true;
    }

    fn update_in_place(&self, op: u32, target: Value<'g>, operand: Value<'g>) {
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

    fn assert_in_place(&self, target: Value<'g>, operand: Value<'g>) {
        let target = self.own(target);
        let operand = self.own(operand);
        {
            let state = self.state.borrow();
            let info = &state.values[target.id() as usize];
            assert!(
                matches!(
                    info.residency,
                    Residency::Input | Residency::Parameter | Residency::Resident
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

    fn wrote_in_place(&self, target: Value<'g>) {
        let target = self.own(target);
        let mut state = self.state.borrow_mut();
        state.values[target.id() as usize].written_in_place = true;
        state.updated_in_place = true;
    }

    fn scatter(&self, target: Value<'g>, indices: Value<'g>, updates: Value<'g>) {
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
        let expected = self.shape(indices).dims();
        let actual = self.shape(updates).dims();
        let width = self.shape(target).dims()[3];
        assert_eq!(
            actual,
            [expected[0], expected[1], expected[2], width],
            "a scatter of {:?} indices meets updates of {:?} where the table holds {width} numbers per row",
            expected,
            actual,
        );
        let mut task = TaskInfo::of(
            Kind::Scatter,
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

    fn reduced_to(&self, gradient: Value<'g>, value: Value<'g>) -> Value<'g> {
        let gradient = self.own(gradient);
        let value = self.own(value);
        let shape = self.shape(value);
        if self.shape(gradient) == shape {
            return gradient;
        }
        assert!(
            shape.fits_within(self.shape(gradient)),
            "a gradient of {:?} does not fold back into the {:?} it belongs to",
            self.shape(gradient).dims(),
            shape.dims(),
        );
        let mut folded = gradient;
        for axis in (0..MAX_RANK).rev() {
            if self.shape(folded).dims()[axis as usize] != shape.dims()[axis as usize] {
                folded = self.fold(folded, axis);
            }
        }
        folded
    }

    pub fn sum_rows(&self, value: Value<'g>) -> Value<'g> {
        let value = self.own(value);
        self.fold(value, MAX_RANK - 1)
    }

    fn fold(&self, value: Value<'g>, axis: u32) -> Value<'g> {
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

    fn broadcast(&self, source: Value<'g>, shape: Shape) -> Value<'g> {
        let source = self.own(source);
        assert!(
            self.shape(source).fits_within(shape),
            "a broadcast spreads {:?} over {:?}",
            self.shape(source).dims(),
            shape.dims(),
        );
        let out = self.fresh(shape, Residency::Derived, false);
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

    fn accumulate(&self, grads: &mut [Option<u32>], value: Value<'g>, contribution: Value<'g>) {
        let value = self.own(value);
        let owner = self.owner_of(value.id());
        let contribution = self.aligned(value, contribution);
        let contribution = self.reduced_to(contribution, self.value_of(owner));
        grads[owner as usize] = Some(match grads[owner as usize] {
            None => self.landed(contribution).id(),
            Some(existing) => {
                let existing = self.value_of(existing);
                self.add(existing, contribution).id()
            }
        });
    }

    fn aligned(&self, value: Value<'g>, contribution: Value<'g>) -> Value<'g> {
        let owner = self.owner_of(value.id());
        let (view_shape, view_strides, owner_shape) = {
            let state = self.state.borrow();
            let info = &state.values[value.id() as usize];
            (info.shape, info.strides, state.values[owner as usize].shape)
        };
        if view_shape.dims() == owner_shape.dims() && view_strides == owner_shape.strides() {
            return contribution;
        }
        let contribution = self.own(contribution);
        if view_strides == view_shape.strides() {
            assert!(
                self.contiguous(contribution),
                "a gradient of a tensor the storage lays out row by row arrives row by row, and value {} walks other strides",
                contribution.id(),
            );
            return self.alias(
                owner_shape,
                owner_shape.strides(),
                self.owner_of(contribution.id()),
                self.element(contribution),
                false,
            );
        }
        let out = self.fresh(owner_shape, Residency::Derived, false);
        self.push(TaskInfo::of(
            Kind::Layout,
            op::NONE,
            out.id(),
            [
                contribution.id(),
                value.id(),
                NO_VALUE,
                NO_VALUE,
                NO_VALUE,
                NO_VALUE,
            ],
        ));
        out
    }

    fn landed(&self, value: Value<'g>) -> Value<'g> {
        let value = self.own(value);
        if self.owner_of(value.id()) == value.id() {
            return value;
        }
        let row_by_row = {
            let state = self.state.borrow();
            let info = &state.values[value.id() as usize];
            info.strides == info.shape.strides()
        };
        assert!(
            row_by_row,
            "a gradient reaches value {} through a view that walks other strides than the tensor that owns its storage, and a gradient lands row by row",
            value.id(),
        );
        value
    }

    fn owner_of(&self, value: u32) -> u32 {
        self.state.borrow().values[value as usize].storage
    }

    fn contiguous(&self, value: Value<'g>) -> bool {
        let value = self.own(value);
        let state = self.state.borrow();
        let info = &state.values[value.id() as usize];
        info.strides == info.shape.strides()
    }

    fn tracked(&self, values: &[Value<'g>]) -> bool {
        let state = self.state.borrow();
        values
            .iter()
            .any(|value| state.values[value.id() as usize].requires_grad)
    }

    fn own(&self, value: Value<'g>) -> Value<'g> {
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

    fn task(&self, index: usize) -> TaskInfo {
        self.state.borrow().tasks[index].clone()
    }

    fn hold(
        &self,
        shape: Shape,
        residency: Residency,
        element: Element,
        seed: Option<Init>,
    ) -> Value<'g> {
        let mut state = self.state.borrow_mut();
        let id = state.values.len() as u32;
        state.values.push(ValueInfo {
            shape,
            strides: shape.strides(),
            storage: id,
            element,
            residency,
            requires_grad: residency == Residency::Parameter,
            retained: false,
            written_in_place: false,
            seed,
        });
        Value::of(self.instance, id, shape)
    }

    fn fresh(&self, shape: Shape, residency: Residency, tracked: bool) -> Value<'g> {
        let mut state = self.state.borrow_mut();
        let id = state.values.len() as u32;
        state.values.push(ValueInfo {
            shape,
            strides: shape.strides(),
            storage: id,
            element: Element::Single,
            residency,
            requires_grad: tracked,
            retained: false,
            written_in_place: false,
            seed: None,
        });
        Value::of(self.instance, id, shape)
    }

    fn alias(
        &self,
        shape: Shape,
        strides: [u32; 4],
        storage: u32,
        element: Element,
        tracked: bool,
    ) -> Value<'g> {
        let mut state = self.state.borrow_mut();
        let id = state.values.len() as u32;
        state.values.push(ValueInfo {
            shape,
            strides,
            storage,
            element,
            residency: Residency::View,
            requires_grad: tracked,
            retained: false,
            written_in_place: false,
            seed: None,
        });
        Value::of(self.instance, id, shape)
    }

    fn push(&self, task: TaskInfo) {
        self.state.borrow_mut().tasks.push(task);
    }
}

impl<'g> Default for Graph<'g> {
    fn default() -> Self {
        Self::new()
    }
}
