use crate::encode::Encoding;
use crate::init::Init;
use crate::shape::Shape;
use neura_abi::{
    BINARY_ADD, BINARY_MUL, KIND_BINARY, KIND_BROADCAST, KIND_EXPAND, KIND_FILL, KIND_MATMUL,
    KIND_SOFTMAX, KIND_SOFTMAX_GRAD, KIND_SUM_CHUNK, KIND_SUM_TO, KIND_UNARY, KIND_UNARY_GRAD,
    MATMUL_COL_TILE, MATMUL_DEPTH_TILE, MATMUL_ROW_TILE, NO_VALUE, StepRecord, UNARY_RECIP,
    UNARY_RELU, UNARY_SQRT,
};
use std::cell::RefCell;
use std::collections::HashMap;

pub const ELEMENT_TILE: u32 = 1024;
pub const REDUCE_TILE: u32 = 4096;
pub const MATMUL_TILES_PER_TASK: u32 = 8;
pub const ROWS_PER_TASK: u32 = 8;

const ENTROPY_SEED: u32 = 0x9e37_79b9;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct Value {
    id: u32,
    shape: Shape,
}

impl Value {
    pub const fn id(self) -> u32 {
        self.id
    }

    pub const fn shape(self) -> Shape {
        self.shape
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Residency {
    Input,
    Parameter,
    Derived,
    View,
}

#[derive(Clone, PartialEq, Debug)]
pub(crate) struct TaskInfo {
    pub(crate) kind: u32,
    pub(crate) flags: u32,
    pub(crate) first: u32,
    pub(crate) count: u32,
    pub(crate) slot: u32,
    pub(crate) out: u32,
    pub(crate) inputs: [u32; 3],
    pub(crate) param: f32,
    pub(crate) work: u64,
    pub(crate) in_place: bool,
    pub(crate) chain: Vec<StepRecord>,
    pub(crate) time: u32,
}

impl TaskInfo {
    fn tiled(
        kind: u32,
        flags: u32,
        span: (u32, u32),
        out: u32,
        inputs: [u32; 3],
        param: f32,
        work: u64,
    ) -> Self {
        let (first, count) = span;
        Self {
            kind,
            flags,
            first,
            count,
            slot: 0,
            out,
            inputs,
            param,
            work,
            in_place: false,
            chain: Vec::new(),
            time: 0,
        }
    }

    pub(crate) fn reads(&self) -> impl Iterator<Item = u32> + '_ {
        self.inputs
            .iter()
            .copied()
            .chain(self.chain.iter().map(|step| step.operand))
            .filter(|value| *value != NO_VALUE)
    }
}

pub(crate) struct ValueInfo {
    pub(crate) shape: Shape,
    pub(crate) strides: [u32; 4],
    pub(crate) storage: u32,
    pub(crate) residency: Residency,
    pub(crate) requires_grad: bool,
    pub(crate) retained: bool,
    pub(crate) written_in_place: bool,
    pub(crate) partials_of: Option<(Value, u32)>,
    pub(crate) initial: Option<Vec<f32>>,
}

pub(crate) struct GraphState {
    pub(crate) values: Vec<ValueInfo>,
    pub(crate) tasks: Vec<TaskInfo>,
    pub(crate) entropy: u32,
    pub(crate) differentiated: bool,
    pub(crate) updated_in_place: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gradients {
    values: HashMap<u32, Value>,
}

impl Gradients {
    pub fn of(&self, value: Value) -> Value {
        *self
            .values
            .get(&value.id())
            .unwrap_or_else(|| panic!("no gradient reaches {:?} from the loss", value.shape()))
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

pub struct Graph {
    state: RefCell<GraphState>,
}

impl Graph {
    pub fn new() -> Self {
        Self {
            state: RefCell::new(GraphState {
                values: Vec::new(),
                tasks: Vec::new(),
                entropy: ENTROPY_SEED,
                differentiated: false,
                updated_in_place: false,
            }),
        }
    }

    pub fn input(&self, shape: Shape) -> Value {
        self.hold(shape, Residency::Input, None)
    }

    pub fn parameter(&self, shape: Shape, init: Init) -> Value {
        let data = {
            let mut state = self.state.borrow_mut();
            let entropy = &mut state.entropy;
            init.samples(shape.elements(), entropy)
        };
        self.hold(shape, Residency::Parameter, Some(data))
    }

    pub fn fill(&self, shape: Shape, value: f32) -> Value {
        let out = self.fresh(shape, Residency::Derived, false);
        for (first, count) in spans(shape.elements(), ELEMENT_TILE) {
            self.push(TaskInfo::tiled(
                KIND_FILL,
                0,
                (first, count),
                out.id(),
                [NO_VALUE; 3],
                value,
                u64::from(count),
            ));
        }
        out
    }

    pub fn matmul(&self, left: Value, right: Value) -> Value {
        let (rows, depth) = left.shape().as_matrix().unwrap_or_else(|| {
            panic!(
                "a matmul left operand must be a matrix, not {:?}",
                left.shape()
            )
        });
        let (right_rows, columns) = right.shape().as_matrix().unwrap_or_else(|| {
            panic!(
                "a matmul right operand must be a matrix, not {:?}",
                right.shape()
            )
        });
        assert_eq!(
            depth,
            right_rows,
            "a matmul of {:?} by {:?} has no shared depth",
            left.shape(),
            right.shape(),
        );
        let out = self.fresh(
            Shape::matrix(rows, columns),
            Residency::Derived,
            self.tracked(&[left, right]),
        );
        let tiles = rows.div_ceil(MATMUL_ROW_TILE) * columns.div_ceil(MATMUL_COL_TILE);
        for (first, count) in spans(tiles, MATMUL_TILES_PER_TASK) {
            let work = u64::from(count * MATMUL_ROW_TILE * MATMUL_COL_TILE * MATMUL_DEPTH_TILE);
            self.push(TaskInfo::tiled(
                KIND_MATMUL,
                0,
                (first, count),
                out.id(),
                [left.id(), right.id(), NO_VALUE],
                0.0,
                work,
            ));
        }
        out
    }

    pub fn add(&self, left: Value, right: Value) -> Value {
        self.elementwise(BINARY_ADD, left, right)
    }

    pub fn mul(&self, left: Value, right: Value) -> Value {
        self.elementwise(BINARY_MUL, left, right)
    }

    pub fn relu(&self, value: Value) -> Value {
        self.unary(UNARY_RELU, value)
    }

    pub fn sqrt(&self, value: Value) -> Value {
        self.unary(UNARY_SQRT, value)
    }

    pub fn recip(&self, value: Value) -> Value {
        self.unary(UNARY_RECIP, value)
    }

    pub fn softmax(&self, value: Value) -> Value {
        assert!(
            self.contiguous(value),
            "a softmax folds a row of a tensor stored row by row, and value {} is a view",
            value.id(),
        );
        let shape = self.shape(value);
        let out = self.fresh(shape, Residency::Derived, self.tracked(&[value]));
        for (first, count) in spans(shape.rows(), ROWS_PER_TASK) {
            self.push(TaskInfo::tiled(
                KIND_SOFTMAX,
                0,
                (first, count),
                out.id(),
                [value.id(), NO_VALUE, NO_VALUE],
                0.0,
                u64::from(count * shape.columns()),
            ));
        }
        out
    }

    pub fn sum(&self, value: Value) -> Value {
        assert!(
            self.contiguous(value),
            "a sum walks its operand element by element, and value {} is a view",
            value.id(),
        );
        let elements = self.shape(value).elements();
        let chunks = elements.div_ceil(REDUCE_TILE);
        assert!(
            chunks <= REDUCE_TILE,
            "a sum over {elements} elements leaves {chunks} partial sums, more than the {REDUCE_TILE} one more level folds",
        );
        let partials = self.fresh(Shape::vector(chunks), Residency::Derived, false);
        self.state.borrow_mut().values[partials.id() as usize].partials_of =
            Some((value, REDUCE_TILE));
        let mut tasks = Vec::new();
        for (slot, (first, count)) in spans(elements, REDUCE_TILE).enumerate() {
            let mut task = TaskInfo::tiled(
                KIND_SUM_CHUNK,
                0,
                (first, count),
                partials.id(),
                [value.id(), NO_VALUE, NO_VALUE],
                0.0,
                u64::from(count),
            );
            task.slot = slot as u32;
            tasks.push(task);
        }
        let out = self.fresh(Shape::scalar(), Residency::Derived, self.tracked(&[value]));
        for (first, count) in spans(chunks, REDUCE_TILE) {
            tasks.push(TaskInfo::tiled(
                KIND_SUM_CHUNK,
                0,
                (first, count),
                out.id(),
                [partials.id(), NO_VALUE, NO_VALUE],
                0.0,
                u64::from(count),
            ));
        }
        for task in tasks {
            self.push(task);
        }
        out
    }

    pub fn transpose(&self, value: Value) -> Value {
        let (rows, columns) = value
            .shape()
            .as_matrix()
            .unwrap_or_else(|| panic!("only a matrix transposes, not {:?}", value.shape()));
        let (strides, storage, tracked) = {
            let state = self.state.borrow();
            let info = &state.values[value.id() as usize];
            (info.strides, info.storage, info.requires_grad)
        };
        let mut strides = strides;
        strides.swap(2, 3);
        self.alias(Shape::matrix(columns, rows), strides, storage, tracked)
    }

    pub fn add_into(&self, target: Value, addend: Value) {
        self.update_in_place(BINARY_ADD, target, addend);
    }

    pub fn mul_into(&self, target: Value, factor: Value) {
        self.update_in_place(BINARY_MUL, target, factor);
    }

    pub fn shape(&self, value: Value) -> Shape {
        self.state.borrow().values[value.id() as usize].shape
    }

    pub fn value_count(&self) -> usize {
        self.state.borrow().values.len()
    }

    pub fn task_count(&self) -> usize {
        self.state.borrow().tasks.len()
    }

    pub fn encode(&self, alignment: u64, capacity: u64) -> Encoding {
        let state = self.state.borrow();
        Encoding::plan(&state, alignment, capacity)
    }

    pub fn backward(&self, loss: Value) -> Gradients {
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
        let mut walked: Vec<bool> = vec![false; self.value_count()];
        let seed = self.fill(self.shape(loss), 1.0);
        grads[loss.id() as usize] = Some(seed.id());
        for index in (0..forward).rev() {
            let task = self.task(index);
            if task.in_place || walked[task.out as usize] {
                continue;
            }
            walked[task.out as usize] = true;
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

    fn backward_task(&self, task: &TaskInfo, gradient: Value, grads: &mut [Option<u32>]) {
        match task.kind {
            KIND_MATMUL => {
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
            KIND_BINARY => {
                let (left, right) = (self.value_of(task.inputs[0]), self.value_of(task.inputs[1]));
                match task.flags {
                    BINARY_ADD => {
                        for operand in [left, right] {
                            if self.tracked(&[operand]) {
                                let contribution = self.reduce_to(gradient, operand);
                                self.accumulate(grads, operand, contribution);
                            }
                        }
                    }
                    BINARY_MUL => {
                        for (operand, other) in [(left, right), (right, left)] {
                            if self.tracked(&[operand]) {
                                let product = self.mul(gradient, other);
                                let contribution = self.reduce_to(product, operand);
                                self.accumulate(grads, operand, contribution);
                            }
                        }
                    }
                    other => panic!("binary code {other} has no gradient rule"),
                }
            }
            KIND_UNARY => {
                let source = self.value_of(task.inputs[0]);
                if self.tracked(&[source]) {
                    let out = self.fresh(
                        self.shape(source),
                        Residency::Derived,
                        self.tracked(&[source]),
                    );
                    for (first, count) in spans(out.shape().elements(), ELEMENT_TILE) {
                        self.push(TaskInfo::tiled(
                            KIND_UNARY_GRAD,
                            task.flags,
                            (first, count),
                            out.id(),
                            [task.out, gradient.id(), NO_VALUE],
                            0.0,
                            u64::from(count),
                        ));
                    }
                    self.accumulate(grads, source, out);
                }
            }
            KIND_SOFTMAX => {
                let source = self.value_of(task.inputs[0]);
                if self.tracked(&[source]) {
                    let shape = self.shape(source);
                    let out = self.fresh(shape, Residency::Derived, true);
                    for (first, count) in spans(shape.rows(), ROWS_PER_TASK) {
                        self.push(TaskInfo::tiled(
                            KIND_SOFTMAX_GRAD,
                            0,
                            (first, count),
                            out.id(),
                            [task.out, gradient.id(), NO_VALUE],
                            0.0,
                            u64::from(count * shape.columns()),
                        ));
                    }
                    self.accumulate(grads, source, out);
                }
            }
            KIND_SUM_CHUNK => {
                let source = self.value_of(task.inputs[0]);
                let partials_of = self.state.borrow().values[source.id() as usize].partials_of;
                let Some((summed, chunk)) = partials_of else {
                    return;
                };
                if !self.tracked(&[summed]) {
                    return;
                }
                let chunk_grads = self.broadcast(gradient, self.shape(source), 0);
                let out = self.fresh(self.shape(summed), Residency::Derived, true);
                for (first, count) in spans(out.shape().elements(), ELEMENT_TILE) {
                    self.push(TaskInfo::tiled(
                        KIND_EXPAND,
                        chunk,
                        (first, count),
                        out.id(),
                        [chunk_grads.id(), NO_VALUE, NO_VALUE],
                        0.0,
                        u64::from(count),
                    ));
                }
                self.accumulate(grads, summed, out);
            }
            KIND_FILL | KIND_BROADCAST | KIND_EXPAND | KIND_SUM_TO | KIND_UNARY_GRAD
            | KIND_SOFTMAX_GRAD => {}
            other => panic!("kind {other} has no gradient rule"),
        }
    }

    fn elementwise(&self, flags: u32, left: Value, right: Value) -> Value {
        let shape = self.shape(left).combined(self.shape(right));
        let out = self.fresh(shape, Residency::Derived, self.tracked(&[left, right]));
        for (first, count) in spans(shape.elements(), ELEMENT_TILE) {
            self.push(TaskInfo::tiled(
                KIND_BINARY,
                flags,
                (first, count),
                out.id(),
                [left.id(), right.id(), NO_VALUE],
                0.0,
                u64::from(count),
            ));
        }
        out
    }

    fn unary(&self, flags: u32, value: Value) -> Value {
        let shape = self.shape(value);
        let out = self.fresh(shape, Residency::Derived, self.tracked(&[value]));
        for (first, count) in spans(shape.elements(), ELEMENT_TILE) {
            self.push(TaskInfo::tiled(
                KIND_UNARY,
                flags,
                (first, count),
                out.id(),
                [value.id(), NO_VALUE, NO_VALUE],
                0.0,
                u64::from(count),
            ));
        }
        out
    }

    pub fn retain(&self, value: Value) {
        let storage = self.state.borrow().values[value.id() as usize].storage;
        self.state.borrow_mut().values[storage as usize].retained = true;
    }

    fn update_in_place(&self, flags: u32, target: Value, operand: Value) {
        {
            let state = self.state.borrow();
            let info = &state.values[target.id() as usize];
            assert!(
                matches!(info.residency, Residency::Input | Residency::Parameter),
                "only a leaf tensor updates in place, and value {} is derived from other tasks",
                target.id(),
            );
            assert_eq!(
                info.shape.strides(),
                info.strides,
                "a parameter updated in place must be stored contiguously",
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
        for (first, count) in spans(self.shape(target).elements(), ELEMENT_TILE) {
            let mut task = TaskInfo::tiled(
                KIND_BINARY,
                flags,
                (first, count),
                target.id(),
                [target.id(), operand.id(), NO_VALUE],
                0.0,
                u64::from(count),
            );
            task.in_place = true;
            self.push(task);
        }
        let mut state = self.state.borrow_mut();
        state.values[target.id() as usize].written_in_place = true;
        state.updated_in_place = true;
    }

    fn reduce_to(&self, gradient: Value, target: Value) -> Value {
        if self.shape(gradient) == self.shape(target) {
            return gradient;
        }
        let shape = self.shape(target);
        assert!(
            shape.fits_within(self.shape(gradient)),
            "a gradient of {} elements cannot fold back into {} elements",
            self.shape(gradient).elements(),
            shape.elements(),
        );
        let replicas = replicas(&shape, self.shape(gradient));
        let out = self.fresh(shape, Residency::Derived, false);
        for (first, count) in spans(shape.elements(), ELEMENT_TILE) {
            self.push(TaskInfo::tiled(
                KIND_SUM_TO,
                0,
                (first, count),
                out.id(),
                [gradient.id(), NO_VALUE, NO_VALUE],
                0.0,
                u64::from(count) * replicas,
            ));
        }
        out
    }

    fn broadcast(&self, source: Value, shape: Shape, slot: u32) -> Value {
        let out = self.fresh(shape, Residency::Derived, false);
        for (first, count) in spans(shape.elements(), ELEMENT_TILE) {
            let mut task = TaskInfo::tiled(
                KIND_BROADCAST,
                0,
                (first, count),
                out.id(),
                [source.id(), NO_VALUE, NO_VALUE],
                0.0,
                u64::from(count),
            );
            task.slot = slot;
            self.push(task);
        }
        out
    }

    fn accumulate(&self, grads: &mut [Option<u32>], value: Value, contribution: Value) {
        grads[value.id() as usize] = Some(match grads[value.id() as usize] {
            None => contribution.id(),
            Some(existing) => {
                let existing = self.value_of(existing);
                self.add(existing, contribution).id()
            }
        });
    }

    fn contiguous(&self, value: Value) -> bool {
        let state = self.state.borrow();
        let info = &state.values[value.id() as usize];
        info.strides == info.shape.strides()
    }

    fn tracked(&self, values: &[Value]) -> bool {
        let state = self.state.borrow();
        values
            .iter()
            .any(|value| state.values[value.id() as usize].requires_grad)
    }

    pub(crate) fn value_of(&self, id: u32) -> Value {
        Value {
            id,
            shape: self.state.borrow().values[id as usize].shape,
        }
    }

    fn task(&self, index: usize) -> TaskInfo {
        self.state.borrow().tasks[index].clone()
    }

    fn hold(&self, shape: Shape, residency: Residency, initial: Option<Vec<f32>>) -> Value {
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
            partials_of: None,
            initial,
        });
        Value { id, shape }
    }

    fn fresh(&self, shape: Shape, residency: Residency, tracked: bool) -> Value {
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
            partials_of: None,
            initial: None,
        });
        Value { id, shape }
    }

    fn alias(&self, shape: Shape, strides: [u32; 4], storage: u32, tracked: bool) -> Value {
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
            partials_of: None,
            initial: None,
        });
        Value { id, shape }
    }

    fn push(&self, task: TaskInfo) {
        self.state.borrow_mut().tasks.push(task);
    }
}

fn spans(units: u32, per_task: u32) -> impl Iterator<Item = (u32, u32)> {
    let mut first = 0;
    std::iter::from_fn(move || {
        if first >= units {
            return None;
        }
        let count = per_task.min(units - first);
        let span = (first, count);
        first += count;
        Some(span)
    })
}

fn replicas(target: &Shape, source: Shape) -> u64 {
    let target = target.dims();
    let source = source.dims();
    (0..4)
        .map(|axis| u64::from(source[axis]) / u64::from(target[axis]))
        .product()
}
