use crate::graph::{Graph, NORM_FLOOR, Residency, TaskInfo, Value, ValueInfo, advance};
use crate::shape::Shape;
use neura_abi::{Element, Kind, MAX_RANK, NO_VALUE};
use neura_op as op;
use std::collections::{BTreeMap, HashMap};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gradients<'g> {
    values: BTreeMap<u32, (Value<'g>, Value<'g>)>,
}

impl<'g> Gradients<'g> {
    pub fn of(&self, value: Value<'g>) -> Value<'g> {
        self.values
            .get(&value.id())
            .map(|(_, gradient)| *gradient)
            .unwrap_or_else(|| {
                panic!(
                    "no gradient reaches {:?} from the loss, and the gradient of a view reaches the tensor that owns its storage",
                    value.shape(),
                )
            })
    }

    pub fn clip(&self, graph: &Graph<'g>, threshold: f32) -> Self {
        assert!(
            threshold.is_finite() && threshold > 0.0,
            "a clip of {threshold} rescales a gradient set to nothing",
        );
        let mut squared = graph.fill(Shape::scalar(), 0.0);
        for (id, (_, gradient)) in self.values.iter() {
            if !graph.carries_gradient(*id) {
                continue;
            }
            squared = graph.add(squared, graph.sum(graph.mul(*gradient, *gradient)));
        }
        let factor = graph.min(
            graph.fill(Shape::scalar(), 1.0),
            graph.mul(
                graph.fill(Shape::scalar(), threshold),
                graph
                    .recip(graph.add(graph.sqrt(squared), graph.fill(Shape::scalar(), NORM_FLOOR))),
            ),
        );
        Self {
            values: self
                .values
                .iter()
                .map(|(id, (parameter, gradient))| {
                    (*id, (*parameter, graph.mul(*gradient, factor)))
                })
                .collect(),
        }
    }
}

pub(crate) struct Recomputation {
    first: usize,
    last: usize,
    values: Vec<u32>,
}

impl Recomputation {
    fn holds(&self, task: usize) -> bool {
        self.first <= task && task < self.last
    }
}

pub(crate) struct Recomputing {
    indices: HashMap<u32, usize>,
    copies: HashMap<u32, u32>,
    views: HashMap<u32, u32>,
    materialized: Vec<bool>,
    differentiated: Vec<bool>,
}

impl Recomputing {
    fn prepare<'g>(&mut self, graph: &Graph<'g>, task: &TaskInfo) {
        for written in [task.out, task.extra] {
            if written == NO_VALUE || self.copies.contains_key(&written) {
                continue;
            }
            let copy = graph.copy_of(written);
            self.copies.insert(written, copy);
        }
    }

    fn resolved<'g>(&mut self, graph: &Graph<'g>, task: &TaskInfo) -> TaskInfo {
        let mut resolved = task.clone();
        resolved.out = self.resolve(graph, task.out);
        resolved.extra = self.resolve(graph, task.extra);
        resolved.origin = self.resolve(graph, task.origin);
        for (input, value) in resolved.inputs.iter_mut().zip(task.inputs) {
            *input = self.resolve(graph, value);
        }
        for (step, source) in resolved.prelude.iter_mut().zip(&task.prelude) {
            step.operand = self.resolve(graph, source.operand);
        }
        for (step, source) in resolved.chain.iter_mut().zip(&task.chain) {
            step.operand = self.resolve(graph, source.operand);
        }
        resolved
    }

    fn resolve<'g>(&mut self, graph: &Graph<'g>, value: u32) -> u32 {
        if value == NO_VALUE {
            return value;
        }
        if let Some(copy) = self.copies.get(&value) {
            return *copy;
        }
        let view = {
            let state = graph.state.borrow();
            let info = &state.values[value as usize];
            (info.residency == Residency::View).then_some((
                info.storage,
                info.shape,
                info.strides,
                info.element,
                info.scale,
                info.requires_grad,
            ))
        };
        let Some((storage, shape, strides, element, scale, tracked)) = view else {
            return value;
        };
        let Some(copy) = self.copies.get(&storage).copied() else {
            return value;
        };
        if let Some(view) = self.views.get(&value) {
            return *view;
        }
        let view = graph.alias(shape, strides, copy, element, scale, tracked);
        self.views.insert(value, view.id());
        view.id()
    }

    fn restore(&self, grads: &mut [Option<u32>]) {
        for (original, copy) in &self.copies {
            if let Some(gradient) = grads[*copy as usize].take() {
                grads[*original as usize] = Some(gradient);
            }
        }
    }
}

impl<'g> Graph<'g> {
    pub fn recompute(&self, region: impl FnOnce(&Graph<'g>) -> Value<'g>) -> Value<'g> {
        let first = self.state.borrow().tasks.len();
        assert!(
            !self.state.borrow().differentiated,
            "a graph declares its recompute regions before it differentiates them",
        );
        let out = self.own(region(self));
        let mut state = self.state.borrow_mut();
        let last = state.tasks.len();
        assert!(
            last > first,
            "a recompute region holds no task the device could run again",
        );
        assert!(
            state
                .recomputations
                .iter()
                .all(|held| held.first >= last || held.last <= first),
            "a recompute region opens inside another, and one block recomputes as one region",
        );
        let mut values = Vec::new();
        for task in &state.tasks[first..last] {
            assert!(
                !task.in_place,
                "a recompute region updates a tensor in place, and a second run of it would update that tensor twice",
            );
            for value in [task.out, task.extra] {
                if value == NO_VALUE {
                    continue;
                }
                let storage = state.values[value as usize].storage;
                assert_eq!(
                    storage, value,
                    "a recompute region writes a tensor of another storage",
                );
                if !values.contains(&storage) {
                    values.push(storage);
                }
            }
        }
        assert!(
            values.contains(&state.values[out.id() as usize].storage),
            "a recompute region returns the tensor its own tasks write",
        );
        state.recomputations.push(Recomputation {
            first,
            last,
            values,
        });
        out
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
        let mut recomputing = self.recomputing();
        for index in (0..forward).rev() {
            if let Some(region) = self.region_of_task(index) {
                if !recomputing.differentiated[region] {
                    recomputing.differentiated[region] = true;
                    self.recompute_region(region, &mut recomputing, &mut grads);
                }
                continue;
            }
            let task = self.task(index);
            let Some(gradient) = grads[self.owner_of(task.out) as usize] else {
                continue;
            };
            self.materialize(&task, &mut recomputing, &mut grads);
            let task = recomputing.resolved(self, &task);
            self.widen(&mut grads);
            let gradient = self.value_of(gradient);
            self.backward_task(&task, gradient, &mut grads);
        }
        recomputing.restore(&mut grads);
        let mut values = BTreeMap::new();
        for (id, grad) in grads.into_iter().enumerate() {
            if let Some(grad) = grad {
                values.insert(id as u32, (self.value_of(id as u32), self.value_of(grad)));
            }
        }
        Gradients { values }
    }

    fn recomputing(&self) -> Recomputing {
        let state = self.state.borrow();
        let mut indices = HashMap::new();
        for (region, recomputation) in state.recomputations.iter().enumerate() {
            for value in &recomputation.values {
                indices.insert(*value, region);
            }
        }
        Recomputing {
            indices,
            copies: HashMap::new(),
            views: HashMap::new(),
            materialized: vec![false; state.recomputations.len()],
            differentiated: vec![false; state.recomputations.len()],
        }
    }

    fn region_of_task(&self, task: usize) -> Option<usize> {
        self.state
            .borrow()
            .recomputations
            .iter()
            .position(|recomputation| recomputation.holds(task))
    }

    fn materialize(
        &self,
        task: &TaskInfo,
        recomputing: &mut Recomputing,
        grads: &mut Vec<Option<u32>>,
    ) {
        let mut opened = Vec::new();
        let reads = task
            .inputs
            .iter()
            .copied()
            .chain([task.origin])
            .chain(task.prelude.iter().map(|step| step.operand))
            .chain(task.chain.iter().map(|step| step.operand));
        for value in reads {
            if value == NO_VALUE {
                continue;
            }
            let storage = self.owner_of(value);
            let Some(region) = recomputing.indices.get(&storage).copied() else {
                continue;
            };
            if recomputing.materialized[region]
                || !self.state.borrow().values[storage as usize].requires_grad
                || opened.contains(&region)
            {
                continue;
            }
            opened.push(region);
        }
        for region in opened {
            self.materialize_region(region, recomputing, grads);
        }
    }

    fn materialize_region(
        &self,
        region: usize,
        recomputing: &mut Recomputing,
        grads: &mut Vec<Option<u32>>,
    ) {
        let (first, last, values) = {
            let state = self.state.borrow();
            let recomputation = &state.recomputations[region];
            (
                recomputation.first,
                recomputation.last,
                recomputation.values.clone(),
            )
        };
        for index in first..last {
            let task = self.task(index);
            recomputing.prepare(self, &task);
            let copy = recomputing.resolved(self, &task);
            self.push(copy);
        }
        recomputing.materialized[region] = true;
        self.widen(grads);
        for value in values {
            if let Some(gradient) = grads[value as usize].take() {
                grads[recomputing.copies[&value] as usize] = Some(gradient);
            }
        }
    }

    fn recompute_region(
        &self,
        region: usize,
        recomputing: &mut Recomputing,
        grads: &mut Vec<Option<u32>>,
    ) {
        if !recomputing.materialized[region] {
            let carried = self.state.borrow().recomputations[region]
                .values
                .iter()
                .any(|value| grads[*value as usize].is_some());
            if !carried {
                return;
            }
            self.materialize_region(region, recomputing, grads);
        }
        let (first, last) = {
            let state = self.state.borrow();
            let recomputation = &state.recomputations[region];
            (recomputation.first, recomputation.last)
        };
        for index in (first..last).rev() {
            let task = self.task(index);
            let task = recomputing.resolved(self, &task);
            self.widen(grads);
            let Some(gradient) = grads[self.owner_of(task.out) as usize] else {
                continue;
            };
            let gradient = self.value_of(gradient);
            self.backward_task(&task, gradient, grads);
        }
    }

    fn widen(&self, grads: &mut Vec<Option<u32>>) {
        let width = self.value_count();
        if grads.len() < width {
            grads.resize(width, None);
        }
    }

    fn copy_of(&self, value: u32) -> u32 {
        let mut state = self.state.borrow_mut();
        let info = state.values[value as usize].clone();
        let id = state.values.len() as u32;
        state.values.push(ValueInfo {
            storage: id,
            retained: false,
            recomputes: Some(value),
            ..info
        });
        advance(&mut state);
        id
    }

    fn backward_task(&self, task: &TaskInfo, gradient: Value<'g>, grads: &mut [Option<u32>]) {
        let gradient = self.own(gradient);
        match task.kind {
            Kind::Broadcast => {
                let source = self.value_of(task.inputs[0]);
                if self.tracked(&[source]) {
                    self.accumulate(grads, source, gradient);
                }
            }
            Kind::Concat => {
                let source = self.value_of(task.inputs[0]);
                if self.tracked(&[source]) {
                    let out = self.fresh(
                        self.shape(source),
                        Element::Single,
                        Residency::Derived,
                        false,
                    );
                    let mut grad = TaskInfo::of(
                        Kind::Slice,
                        op::NONE,
                        out.id(),
                        [
                            gradient.id(),
                            NO_VALUE,
                            NO_VALUE,
                            NO_VALUE,
                            NO_VALUE,
                            NO_VALUE,
                        ],
                    );
                    grad.axis = task.axis;
                    grad.offset = task.offset;
                    self.push(grad);
                    self.accumulate(grads, source, out);
                }
            }
            Kind::Slice => {
                let source = self.value_of(task.inputs[0]);
                if self.tracked(&[source]) {
                    let zeros = self.fill(self.shape(source), 0.0);
                    let mut grad = TaskInfo::of(
                        Kind::Concat,
                        op::NONE,
                        zeros.id(),
                        [
                            gradient.id(),
                            NO_VALUE,
                            NO_VALUE,
                            NO_VALUE,
                            NO_VALUE,
                            NO_VALUE,
                        ],
                    );
                    grad.axis = task.axis;
                    grad.offset = task.offset;
                    self.push(grad);
                    self.accumulate(grads, source, zeros);
                }
            }
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
                    let out = self.fresh(
                        self.shape(operand),
                        Element::Single,
                        Residency::Derived,
                        false,
                    );
                    let mut grad = TaskInfo::of(kind, op::NONE, out.id(), inputs);
                    grad.origin = task.origin;
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
                    let out = self.fresh(
                        self.shape(input),
                        Element::Single,
                        Residency::Derived,
                        false,
                    );
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
                    let out = self.fresh(
                        self.shape(filter),
                        Element::Single,
                        Residency::Derived,
                        false,
                    );
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
                    self.scatter(Kind::Scatter, zeros, indices, gradient);
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
                    let out = self.fresh(
                        self.shape(input),
                        Element::Single,
                        Residency::Derived,
                        false,
                    );
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
            | Kind::Convert
            | Kind::Scatter
            | Kind::ScatterWrite => {}
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
        let out = self.fresh(
            self.shape(gradient),
            Element::Single,
            Residency::Derived,
            false,
        );
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
        let out = self.fresh(
            self.shape(source),
            Element::Single,
            Residency::Derived,
            true,
        );
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

    pub(crate) fn rows(&self, kind: Kind, value: Value<'g>) -> Value<'g> {
        let value = self.own(value);
        assert!(
            self.contiguous(value),
            "a {} folds a row of a tensor stored row by row, and value {} is a view",
            kind.name(),
            value.id(),
        );
        let shape = self.shape(value);
        let out = self.stored(
            shape,
            self.element(value),
            self.carries(self.element(value), &[value]),
            Residency::Derived,
            self.tracked(&[value]),
        );
        self.push(TaskInfo::of(
            kind,
            op::NONE,
            out.id(),
            [value.id(), NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE],
        ));
        out
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
                self.scale(contribution),
                false,
            );
        }
        let element = self.element(contribution);
        let out = self.stored(
            owner_shape,
            element,
            self.carries(element, &[contribution]),
            Residency::Derived,
            false,
        );
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
}
