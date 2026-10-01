use crate::graph::{AttentionOptions, Graph, Residency, TaskInfo, Value};
use crate::pool::Pool;
use crate::shape::Shape;
use crate::window::Window;
use neura_abi::{Element, Kind, MAX_RANK, NO_VALUE};
use neura_op as op;

impl<'g> Graph<'g> {
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
        let element = self.element(left).promote(self.element(right));
        let out = self.stored(
            Shape::of([
                left_batch[0].max(right_batch[0]),
                left_batch[1].max(right_batch[1]),
                left_dims[2],
                right_dims[3],
            ]),
            element,
            self.carries(element, &[left, right]),
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
        attention: AttentionOptions<'g>,
    ) -> Value<'g> {
        let query = self.own(query);
        let key = self.own(key);
        let value = self.own(value);
        let query_shape = self.shape(query);
        let key_shape = self.shape(key);
        let value_shape = self.shape(value);
        let origin = attention.origin.map(|origin| self.own(origin));
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
        if let Some(origin) = origin {
            let positions = Shape::of([query_shape.dims()[0], query_shape.dims()[1], 1, 1]);
            assert!(
                self.shape(origin).fits_within(positions),
                "a cursor holds one position per {:?} plane, and value {} walks {:?}",
                positions.dims(),
                origin.id(),
                self.shape(origin).dims(),
            );
            assert!(
                query_shape.dims()[2] <= key_shape.dims()[2],
                "a cursor walks {} queries over {} keys, and the last query of a block reads every key before it",
                query_shape.dims()[2],
                key_shape.dims()[2],
            );
        }
        assert!(
            !attention.causal || origin.is_some() || query_shape.dims()[2] == key_shape.dims()[2],
            "a causal attention walks {} queries over {} keys, and a cursor is what aligns them",
            query_shape.dims()[2],
            key_shape.dims()[2],
        );
        let tracked = self.tracked(&[query, key, value]);
        let element = self
            .element(query)
            .promote(self.element(key))
            .promote(self.element(value));
        let out = self.stored(
            Shape::of([
                query_shape.dims()[0],
                query_shape.dims()[1],
                query_shape.dims()[2],
                value_shape.dims()[3],
            ]),
            element,
            self.carries(element, &[query, key, value]),
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
            Element::Single,
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
        task.origin = origin.map_or(NO_VALUE, |origin| origin.id());
        task.param = attention.scale;
        task.slot = u32::from(attention.causal);
        self.push(task);
        out
    }

    pub fn rope(&self, value: Value<'g>, origin: Option<Value<'g>>, base: f32) -> Value<'g> {
        let value = self.own(value);
        let shape = self.shape(value);
        let width = shape.dims()[3];
        assert!(
            width.is_multiple_of(2),
            "a rotation pairs every number of a row with the number half a row away, and {:?} holds {width} numbers per row",
            shape.dims(),
        );
        assert!(
            base.is_finite() && base > 1.0,
            "a rotary base of {base} spreads every position over the same angle",
        );
        let element = self.element(value);
        assert!(
            !element.quantized(),
            "a rotation places the numbers of {} storage on no grid, and the storage a tensor carries already holds the grid it reconstructs by",
            element.name(),
        );
        let origin = origin.map(|origin| self.own(origin));
        if let Some(origin) = origin {
            let positions = Shape::of([shape.dims()[0], shape.dims()[1], 1, 1]);
            assert!(
                self.shape(origin).fits_within(positions),
                "a cursor holds one position per {:?} plane, and value {} walks {:?}",
                positions.dims(),
                origin.id(),
                self.shape(origin).dims(),
            );
        }
        let out = self.stored(
            shape,
            element,
            self.carries(element, &[value]),
            Residency::Derived,
            self.tracked(&[value]),
        );
        let mut task = TaskInfo::of(
            Kind::Rope,
            op::NONE,
            out.id(),
            [value.id(), NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE],
        );
        task.origin = origin.map_or(NO_VALUE, |origin| origin.id());
        task.param = base;
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
        let element = self.element(input).promote(self.element(filter));
        let out = self.stored(
            Shape::of([
                input_dims[0],
                filter_dims[0],
                (padded_rows - filter_dims[2]) / window.stride_rows() + 1,
                (padded_columns - filter_dims[3]) / window.stride_columns() + 1,
            ]),
            element,
            self.carries(element, &[input, filter]),
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
        let out = self.stored(
            Shape::of([
                input_dims[0],
                input_dims[1],
                (padded_rows - window.reach_rows()) / window.stride_rows() + 1,
                (padded_columns - window.reach_columns()) / window.stride_columns() + 1,
            ]),
            self.element(input),
            self.carries(self.element(input), &[input]),
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

    pub fn cast(&self, value: Value<'g>, element: Element) -> Value<'g> {
        let value = self.own(value);
        assert!(
            !element.quantized(),
            "a cast into {} storage quantizes the numbers it copies, and only a weight holds the quantum it was declared with",
            element.name(),
        );
        if self.element(value) == element {
            return value;
        }
        self.unary_as(op::IDENTITY, value, element)
    }

    pub fn sin(&self, value: Value<'g>) -> Value<'g> {
        let value = self.own(value);
        self.unary(op::SIN, value)
    }

    pub fn cos(&self, value: Value<'g>) -> Value<'g> {
        let value = self.own(value);
        self.unary(op::COS, value)
    }

    pub fn floor(&self, value: Value<'g>) -> Value<'g> {
        let value = self.own(value);
        self.unary(op::FLOOR, value)
    }

    pub fn gelu(&self, value: Value<'g>) -> Value<'g> {
        let value = self.own(value);
        self.unary(op::GELU, value)
    }

    pub fn silu(&self, value: Value<'g>) -> Value<'g> {
        let value = self.own(value);
        self.unary(op::SILU, value)
    }

    pub fn pow(&self, base: Value<'g>, exponent: Value<'g>) -> Value<'g> {
        let base = self.own(base);
        let exponent = self.own(exponent);
        self.elementwise(op::POW, base, exponent)
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
        let out = self.fresh(Shape::of(dims), Element::Single, Residency::Derived, false);
        self.push(TaskInfo::of(
            kind,
            op::NONE,
            out.id(),
            [source.id(), seed, NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE],
        ));
        out
    }

    pub(crate) fn index_list(&self, indices: Value<'g>) {
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
        let out = self.fresh(Shape::of(dims), Element::Single, Residency::Derived, false);
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
        let out = self.stored(
            Shape::of(dims),
            self.element(table),
            self.carries(self.element(table), &[table]),
            Residency::Derived,
            self.tracked(&[table]),
        );
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

    pub fn concat(&self, values: &[Value<'g>], axis: u32) -> Value<'g> {
        assert!(
            !values.is_empty(),
            "a concatenation joins at least one tensor",
        );
        assert!(
            axis < MAX_RANK,
            "a concatenation names one of the {MAX_RANK} axes",
        );
        let values = values
            .iter()
            .map(|value| self.own(*value))
            .collect::<Vec<_>>();
        let first = self.shape(values[0]);
        let element = self.element(values[0]);
        let scale = self.scale(values[0]);
        let mut dims = first.dims();
        dims[axis as usize] = 0;
        for value in &values {
            let shape = self.shape(*value);
            assert_eq!(
                self.element(*value),
                element,
                "a concatenation joins {} numbers with {} numbers",
                self.element(*value).name(),
                element.name(),
            );
            assert_eq!(
                self.scale(*value),
                scale,
                "a concatenation joins numbers reconstructed by {} with numbers reconstructed by {scale}",
                self.scale(*value),
            );
            for (index, (left, right)) in first.dims().iter().zip(shape.dims()).enumerate() {
                assert!(
                    index as u32 == axis || left == &right,
                    "a concatenation along axis {axis} meets {:?} and {:?}",
                    first.dims(),
                    shape.dims(),
                );
            }
            dims[axis as usize] += shape.dims()[axis as usize];
        }
        let widened = element.narrow();
        let out = self.stored(
            Shape::of(dims),
            if widened { Element::Single } else { element },
            if widened { 1.0 } else { scale },
            Residency::Derived,
            self.tracked(&values),
        );
        let mut offset = 0;
        for value in &values {
            let mut task = TaskInfo::of(
                Kind::Concat,
                op::NONE,
                out.id(),
                [value.id(), NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE],
            );
            task.axis = axis;
            task.offset = offset;
            self.push(task);
            offset += self.shape(*value).dims()[axis as usize];
        }
        if !widened {
            return out;
        }
        match element {
            Element::Int8 => self.quantize(out, scale),
            other => self.cast(out, other),
        }
    }

    pub fn slice(&self, value: Value<'g>, axis: u32, start: u32, length: u32) -> Value<'g> {
        let value = self.own(value);
        let shape = self.shape(value);
        assert!(
            axis < MAX_RANK
                && length > 0
                && start
                    .checked_add(length)
                    .is_some_and(|end| end <= shape.dims()[axis as usize]),
            "a slice of {length} numbers from {start} along axis {axis} reaches beyond {:?}",
            shape.dims(),
        );
        let mut dims = shape.dims();
        dims[axis as usize] = length;
        let out = self.stored(
            Shape::of(dims),
            self.element(value),
            self.scale(value),
            Residency::Derived,
            self.tracked(&[value]),
        );
        let mut task = TaskInfo::of(
            Kind::Slice,
            op::NONE,
            out.id(),
            [value.id(), NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE],
        );
        task.axis = axis;
        task.offset = start;
        self.push(task);
        out
    }

    pub fn scatter_into(&self, target: Value<'g>, indices: Value<'g>, updates: Value<'g>) {
        self.rows_into(Kind::Scatter, target, indices, updates);
    }

    pub fn write_into(&self, target: Value<'g>, indices: Value<'g>, updates: Value<'g>) {
        self.rows_into(Kind::ScatterWrite, target, indices, updates);
    }

    fn rows_into(&self, kind: Kind, target: Value<'g>, indices: Value<'g>, updates: Value<'g>) {
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
                "only a leaf tensor takes rows in place, and value {} is derived from other tasks",
                target.id(),
            );
        }
        self.scatter(kind, target, indices, updates);
        self.wrote_in_place(target);
    }

    pub fn sum(&self, value: Value<'g>) -> Value<'g> {
        let value = self.own(value);
        assert!(
            self.contiguous(value),
            "a sum walks its operand element by element, and value {} is a view",
            value.id(),
        );
        let out = self.fresh(
            Shape::scalar(),
            Element::Single,
            Residency::Derived,
            self.tracked(&[value]),
        );
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
        let (dims, strides, storage, element, scale, tracked) = {
            let state = self.state.borrow();
            let info = &state.values[value.id() as usize];
            (
                info.shape.dims(),
                info.strides,
                info.storage,
                info.element,
                info.scale,
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
            scale,
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
        let (storage, element, scale, tracked) = {
            let state = self.state.borrow();
            let info = &state.values[value.id() as usize];
            (info.storage, info.element, info.scale, info.requires_grad)
        };
        self.alias(shape, shape.strides(), storage, element, scale, tracked)
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
}
