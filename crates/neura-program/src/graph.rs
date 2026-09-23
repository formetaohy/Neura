use crate::encode::Encoding;
use crate::init::Init;
use crate::layout::Layout;
use crate::shape::Shape;
use neura_abi::Kind;
use neura_abi::op;
use neura_abi::{MAX_RANK, Precision, Window};
use neura_abi::{NO_VALUE, Profile, StepRecord};
use std::cell::RefCell;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};

const ENTROPY_SEED: u32 = 0x9e37_79b9;

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

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Residency {
    Input,
    Parameter,
    Resident,
    Derived,
    View,
}

#[derive(Clone, PartialEq, Debug)]
pub(crate) struct TaskInfo {
    pub(crate) kind: Kind,
    pub(crate) op: u32,
    pub(crate) out: u32,
    pub(crate) inputs: [u32; 3],
    pub(crate) slot: u32,
    pub(crate) param: f32,
    pub(crate) window: Window,
    pub(crate) in_place: bool,
    pub(crate) chain: Vec<StepRecord>,
    pub(crate) time: u32,
}

impl TaskInfo {
    fn of(kind: Kind, op: u32, out: u32, inputs: [u32; 3]) -> Self {
        Self {
            kind,
            op,
            out,
            inputs,
            slot: 0,
            param: 0.0,
            window: Window::sliding([1, 1]),
            in_place: false,
            chain: Vec::new(),
            time: 0,
        }
    }
}

#[derive(Clone)]
pub(crate) struct ValueInfo {
    pub(crate) shape: Shape,
    pub(crate) strides: [u32; 4],
    pub(crate) storage: u32,
    pub(crate) residency: Residency,
    pub(crate) requires_grad: bool,
    pub(crate) retained: bool,
    pub(crate) written_in_place: bool,
    pub(crate) initial: Option<Vec<f32>>,
}

impl ValueInfo {
    pub(crate) fn derived(shape: Shape, id: u32) -> Self {
        Self {
            shape,
            strides: shape.strides(),
            storage: id,
            residency: Residency::Derived,
            requires_grad: false,
            retained: false,
            written_in_place: false,
            initial: None,
        }
    }
}

pub(crate) struct GraphState {
    pub(crate) values: Vec<ValueInfo>,
    pub(crate) tasks: Vec<TaskInfo>,
    pub(crate) entropy: u32,
    pub(crate) differentiated: bool,
    pub(crate) updated_in_place: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gradients<'g> {
    values: HashMap<u32, Value<'g>>,
}

impl<'g> Gradients<'g> {
    pub fn of(&self, value: Value<'g>) -> Value<'g> {
        *self
            .values
            .get(&value.id())
            .unwrap_or_else(|| panic!("no gradient reaches {:?} from the loss", value.shape()))
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
                entropy: ENTROPY_SEED,
                differentiated: false,
                updated_in_place: false,
            }),
            instance: NEXT_GRAPH.fetch_add(1, Ordering::Relaxed),
            brand: PhantomData,
        }
    }

    pub fn input(&self, shape: Shape) -> Value<'g> {
        self.hold(shape, Residency::Input, None)
    }

    pub fn resident(&self, shape: Shape) -> Value<'g> {
        self.hold(shape, Residency::Resident, None)
    }

    pub fn parameter(&self, shape: Shape, init: Init) -> Value<'g> {
        let data = {
            let mut state = self.state.borrow_mut();
            let entropy = &mut state.entropy;
            init.samples(shape.elements(), entropy)
        };
        self.hold(shape, Residency::Parameter, Some(data))
    }

    pub fn fill(&self, shape: Shape, value: f32) -> Value<'g> {
        let out = self.fresh(shape, Residency::Derived, false);
        let mut task = TaskInfo::of(Kind::Fill, op::NONE, out.id(), [NO_VALUE; 3]);
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
            [left.id(), right.id(), NO_VALUE],
        ));
        out
    }

    pub fn conv2d(&self, input: Value<'g>, filter: Value<'g>, window: Window) -> Value<'g> {
        let input = self.own(input);
        let filter = self.own(filter);
        let input_dims = self.shape(input).dims();
        let filter_dims = self.shape(filter).dims();
        assert_eq!(
            filter_dims[1], input_dims[1],
            "a convolution reads {} channels through a filter of {} of them",
            input_dims[1], filter_dims[1],
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
            [input.id(), filter.id(), NO_VALUE],
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
            [source.id(), seed, NO_VALUE],
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
            [indices.id(), NO_VALUE, NO_VALUE],
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
            [table.id(), indices.id(), NO_VALUE],
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
            [value.id(), NO_VALUE, NO_VALUE],
        ));
        out
    }

    pub fn transpose(&self, value: Value<'g>) -> Value<'g> {
        let value = self.own(value);
        let (strides, storage, tracked) = {
            let state = self.state.borrow();
            let info = &state.values[value.id() as usize];
            (info.strides, info.storage, info.requires_grad)
        };
        let mut strides = strides;
        strides.swap(2, 3);
        let mut dims = value.shape().dims();
        dims.swap(2, 3);
        self.alias(Shape::of(dims), strides, storage, tracked)
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
            [source.id(), NO_VALUE, NO_VALUE],
        );
        task.in_place = true;
        self.push(task);
        self.wrote_in_place(target);
    }

    pub fn shape(&self, value: Value<'g>) -> Shape {
        let value = self.own(value);
        self.state.borrow().values[value.id() as usize].shape
    }

    pub fn value_count(&self) -> usize {
        self.state.borrow().values.len()
    }

    pub fn task_count(&self) -> usize {
        self.state.borrow().tasks.len()
    }

    pub fn layout(&self, alignment: u64, precision: Precision) -> Layout {
        Layout::of(&self.state.borrow().values, precision, alignment)
    }

    pub fn updates_weights(&self) -> bool {
        let state = self.state.borrow();
        state.tasks.iter().any(|task| {
            task.in_place && state.values[task.out as usize].residency == Residency::Parameter
        })
    }

    pub fn encode(&self, alignment: u64, profile: Profile, precision: Precision) -> Encoding {
        let state = self.state.borrow();
        Encoding::plan(&state, profile, alignment, precision)
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
            let Some(gradient) = grads[task.out as usize] else {
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
                    let transposed = self.transpose(right);
                    let contribution = self.matmul(gradient, transposed);
                    self.accumulate(grads, left, contribution);
                }
                if self.tracked(&[right]) {
                    let transposed = self.transpose(left);
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
            Kind::Conv2d => {
                let input = self.value_of(task.inputs[0]);
                let filter = self.value_of(task.inputs[1]);
                if self.tracked(&[input]) {
                    let out = self.fresh(self.shape(input), Residency::Derived, false);
                    let mut grad = TaskInfo::of(
                        Kind::Conv2dInputGrad,
                        op::NONE,
                        out.id(),
                        [filter.id(), gradient.id(), NO_VALUE],
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
                        [input.id(), gradient.id(), filter.id()],
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
            Kind::Fill
            | Kind::Broadcast
            | Kind::Partial
            | Kind::SoftmaxGrad
            | Kind::LogSoftmaxGrad
            | Kind::Conv2dInputGrad
            | Kind::Conv2dWeightGrad
            | Kind::MatmulFold
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
            [left, right, gradient.id()],
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
            [task.out, gradient.id(), NO_VALUE],
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
            [value.id(), NO_VALUE, NO_VALUE],
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
            [left.id(), right.id(), NO_VALUE],
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
            [value.id(), NO_VALUE, NO_VALUE],
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
            [target.id(), operand.id(), NO_VALUE],
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
            [target.id(), indices.id(), updates.id()],
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
            [value.id(), NO_VALUE, NO_VALUE],
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
            [source.id(), NO_VALUE, NO_VALUE],
        ));
        out
    }

    fn accumulate(&self, grads: &mut [Option<u32>], value: Value<'g>, contribution: Value<'g>) {
        let value = self.own(value);
        let contribution = self.reduced_to(contribution, value);
        grads[value.id() as usize] = Some(match grads[value.id() as usize] {
            None => contribution.id(),
            Some(existing) => {
                let existing = self.value_of(existing);
                self.add(existing, contribution).id()
            }
        });
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

    fn hold(&self, shape: Shape, residency: Residency, initial: Option<Vec<f32>>) -> Value<'g> {
        let mut state = self.state.borrow_mut();
        let id = state.values.len() as u32;
        state.values.push(ValueInfo {
            shape,
            strides: shape.strides(),
            storage: id,
            residency,
            requires_grad: residency == Residency::Parameter,
            retained: false,
            written_in_place: false,
            initial,
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
            residency,
            requires_grad: tracked,
            retained: false,
            written_in_place: false,
            initial: None,
        });
        Value::of(self.instance, id, shape)
    }

    fn alias(&self, shape: Shape, strides: [u32; 4], storage: u32, tracked: bool) -> Value<'g> {
        let mut state = self.state.borrow_mut();
        let id = state.values.len() as u32;
        state.values.push(ValueInfo {
            shape,
            strides,
            storage,
            residency: Residency::View,
            requires_grad: tracked,
            retained: false,
            written_in_place: false,
            initial: None,
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
