use crate::graph::{AttentionOptions, Graph, Residency, TaskInfo, Value};
use crate::pool::Pool;
use crate::shape::Shape;
use crate::window::Window;
use neura_abi::{EXACT_WALK_LIMIT, Element, Kind, MAX_RANK, NO_VALUE};
use neura_pointwise as op;

impl<'g> Graph<'g> {
    pub fn matmul(&self, left: Value<'g>, right: Value<'g>) -> Value<'g> {
        let left = self.own(left);
        let right = self.own(right);
        let left_shape = self.shape(left);
        let right_shape = self.shape(right);
        let (batch_rows, rows_free) = left_shape.combining_axis(right_shape, 0, 0);
        let (batch_columns, columns_free) = left_shape.combining_axis(right_shape, 1, 1);
        let (_, _) = left_shape.meeting(right_shape, 3, 2);
        let element = self.element(left).promote(self.element(right));
        let out = self.stored(
            Shape::from_axes(
                [
                    batch_rows,
                    batch_columns,
                    left_shape.dims()[2],
                    right_shape.dims()[3],
                ],
                [
                    rows_free,
                    columns_free,
                    left_shape.free(2),
                    right_shape.free(3),
                ],
            ),
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

    pub fn grouped_matmul(
        &self,
        left: Value<'g>,
        weights: Value<'g>,
        segments: Value<'g>,
    ) -> Value<'g> {
        let left = self.own(left);
        let weights = self.own(weights);
        let segments = self.own(segments);
        let axis = {
            let state = self.state.borrow();
            state.ragged.get(&segments.id()).cloned()
        }
        .unwrap_or_else(|| {
            panic!(
                "a grouped product walks the segments a ragged axis closes, and value {} holds a table no ragged axis published; close the lengths of every plane with Graph::ragged",
                segments.id(),
            )
        });
        let left_shape = self.shape(left);
        let weights_shape = self.shape(weights);
        let groups = axis.planes.elements();
        assert_eq!(
            left_shape.free(2),
            Some(axis.token),
            "a grouped product walks the rows of every segment through the token axis its offsets close, and the rows of {left_shape:?} walk free extent {:?} where the ragged axis closes free extent {}",
            left_shape.free(2),
            axis.token,
        );
        assert!(
            left_shape.dims()[0] == 1 && left_shape.dims()[1] == 1,
            "a grouped product packs the rows of every segment into one axis, and {:?} holds {} planes of rows",
            left_shape.dims(),
            left_shape.dims()[0] * left_shape.dims()[1],
        );
        assert!(
            self.contiguous(left),
            "a grouped product walks the rows a ragged axis packs one after another, and value {} strides every row by a layout the offsets do not close",
            left.id(),
        );
        assert_eq!(
            weights_shape.dims()[0],
            groups,
            "a grouped product weighs the {groups} segments a ragged axis closes, and the weights of {weights_shape:?} hold {} planes",
            weights_shape.dims()[0],
        );
        assert!(
            weights_shape.dims()[1] == 1
                && weights_shape.free(0).is_none()
                && weights_shape.free(1).is_none()
                && weights_shape.free(2).is_none()
                && weights_shape.free(3).is_none(),
            "a grouped product segments its rows alone, and the weights of {weights_shape:?} walk a length the binding rules",
        );
        let depth = left_shape.dims()[3];
        assert!(
            left_shape.meets(weights_shape, 3, 2),
            "a grouped product of depth {depth} weighs pairs of depth {} against every row of {:?}",
            weights_shape.dims()[2],
            left_shape.dims(),
        );
        let columns = weights_shape.dims()[3];
        let element = self.element(left).promote(self.element(weights));
        let out = self.stored(
            Shape::from_axes(
                [1, 1, left_shape.dims()[2], columns],
                [None, None, left_shape.free(2), None],
            ),
            element,
            self.carries(element, &[left, weights]),
            Residency::Derived,
            self.tracked(&[left, weights]),
        );
        let mut unit = TaskInfo::of(
            Kind::Matmul,
            op::NONE,
            out.id(),
            [
                left.id(),
                weights.id(),
                NO_VALUE,
                NO_VALUE,
                NO_VALUE,
                NO_VALUE,
            ],
        );
        unit.segments = segments.id();
        self.push(unit);
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
        let segments = attention.segments.map(|segments| self.own(segments));
        let reach = attention.reach;
        if let Some(reach) = reach {
            assert!(
                reach > 0,
                "an attention that reaches back over {reach} keys weighs no key at all",
            );
            assert!(
                reach <= EXACT_WALK_LIMIT,
                "an attention that reaches back over {reach} keys positions them past the {EXACT_WALK_LIMIT} keys a device counts exactly",
            );
            assert!(
                attention.causal,
                "an attention that reaches back over {reach} keys reads the keys its causal mask leaves behind, and an attention without a mask weighs every key",
            );
        }
        let tracked = self.tracked(&[query, key, value]);
        if let Some(offsets) = segments {
            let axis = {
                let state = self.state.borrow();
                state.ragged.get(&offsets.id()).cloned()
            }
            .unwrap_or_else(|| {
                panic!(
                    "a segmented attention walks the offsets a ragged axis closes, and value {} holds a table no ragged axis published; close the lengths of every plane with Graph::ragged",
                    offsets.id(),
                )
            });
            assert_eq!(
                key_shape.free(2),
                Some(axis.token),
                "a segmented attention packs the keys of every plane into the token axis its offsets close, and the keys of {key_shape:?} walk free extent {:?} where the ragged axis closes free extent {}",
                key_shape.free(2),
                axis.token,
            );
            assert_eq!(
                value_shape.free(2),
                Some(axis.token),
                "a segmented attention reads every key through the value its plane packs beside it, and the values of {value_shape:?} walk free extent {:?} where the ragged axis closes free extent {}",
                value_shape.free(2),
                axis.token,
            );
            let planes = query_shape.dims()[0] * query_shape.dims()[1];
            assert!(
                axis.planes.domain().meets(&query_shape.plane_domain()),
                "a segmented attention of {planes} planes walks the segments a ragged axis closes over {:?} of {} planes, and the query walks {:?}: the device reads a segment by the very plane the query walks, so one segment closes each plane and no more",
                axis.planes.dims(),
                axis.planes.elements(),
                query_shape.dims(),
            );
            assert!(
                axis.planes.elements() == 1 || query_shape.free(2) != Some(axis.token),
                "a segmented attention weighs the {:?} queries of a plane against the keys its own offsets close, and the ragged axis of {} planes packs the free extent {} that the token axis of {query_shape:?} walks: the rows a packed query axis holds belong to different planes",
                query_shape.dims()[2],
                axis.planes.elements(),
                axis.token,
            );
            assert!(
                key_shape.dims()[0] == 1
                    && key_shape.dims()[1] == 1
                    && value_shape.dims()[0] == 1
                    && value_shape.dims()[1] == 1,
                "a segmented attention packs the keys and values of every plane into their token axis, and keys of {key_shape:?} walk values of {value_shape:?}",
            );
        }
        assert_eq!(
            key_shape.dims()[0],
            value_shape.dims()[0],
            "an attention weighs {} key heads by {} value heads, and a key head shares the values of its own head",
            key_shape.dims()[0],
            value_shape.dims()[0],
        );
        assert!(
            query_shape.free(0).is_none() && key_shape.free(0).is_none(),
            "an attention of {} query heads reads {} key heads, and the group every key head serves holds every length a free extent takes",
            query_shape.dims()[0],
            key_shape.dims()[0],
        );
        assert!(
            query_shape.meets(key_shape, 3, 3),
            "an attention of width {} scores keys of width {}",
            query_shape.dims()[3],
            key_shape.dims()[3],
        );
        assert!(
            key_shape.meets(value_shape, 3, 3),
            "an attention scores keys of width {} through values of width {}",
            key_shape.dims()[3],
            value_shape.dims()[3],
        );
        assert!(
            key_shape.meets(value_shape, 2, 2),
            "an attention weighs {} keys by {} values",
            key_shape.dims()[2],
            value_shape.dims()[2],
        );
        let planes = if segments.is_some() {
            Shape::of([query_shape.dims()[0], query_shape.dims()[1], 1, 1])
        } else {
            assert_eq!(
                query_shape.dims()[1],
                key_shape.dims()[1],
                "an attention reads {query_shape:?} through keys of {key_shape:?}",
            );
            assert_eq!(
                key_shape.dims()[1],
                value_shape.dims()[1],
                "an attention reads keys of {key_shape:?} through values of {value_shape:?}",
            );
            assert!(
                query_shape.dims()[0].is_multiple_of(key_shape.dims()[0]),
                "an attention of {} query heads reads {} key heads, and every key head serves a whole group of queries",
                query_shape.dims()[0],
                key_shape.dims()[0],
            );
            assert!(
                query_shape.free(2) == key_shape.free(2)
                    || (query_shape.free(2).is_none() && key_shape.free(2).is_none()),
                "an attention walks {:?} queries over {:?} keys, and a free extent of one meets the bound of the other",
                query_shape.dims(),
                key_shape.dims(),
            );
            assert!(
                query_shape.free(1) == key_shape.free(1)
                    && key_shape.free(1) == value_shape.free(1),
                "an attention reads keys of {key_shape:?} through values of {value_shape:?}, and a free extent of one meets the bound of the other",
            );
            Shape::of([key_shape.dims()[0], key_shape.dims()[1], 1, 1])
        };
        if let Some(origin) = origin {
            assert!(
                self.shape(origin).fits_within(planes),
                "a cursor holds one position per {:?} plane, and value {} walks {:?}",
                planes.dims(),
                origin.id(),
                self.shape(origin).dims(),
            );
            assert!(
                query_shape.meets(key_shape, 2, 2) || query_shape.dims()[2] < key_shape.dims()[2],
                "a cursor walks {} queries over {} keys, and the last query of a block reads every key before it",
                query_shape.dims()[2],
                key_shape.dims()[2],
            );
            assert!(
                segments.is_none() || query_shape.dims()[2] <= key_shape.dims()[2],
                "a cursor walks {} queries over a packed key axis of {} tokens",
                query_shape.dims()[2],
                key_shape.dims()[2],
            );
        } else {
            assert!(
                !attention.causal || segments.is_none(),
                "a segmented attention places the key of every plane by its offsets, and a causal mask of the queries needs the cursor that tells them where their keys end",
            );
        }
        assert!(
            !attention.causal || origin.is_some() || query_shape.dims()[2] == key_shape.dims()[2],
            "a causal attention walks {} queries over {} keys, and a cursor is what aligns them",
            query_shape.dims()[2],
            key_shape.dims()[2],
        );
        let element = self
            .element(query)
            .promote(self.element(key))
            .promote(self.element(value));
        let out = self.stored(
            Shape::from_axes(
                [
                    query_shape.dims()[0],
                    query_shape.dims()[1],
                    query_shape.dims()[2],
                    value_shape.dims()[3],
                ],
                [
                    query_shape.free(0),
                    query_shape.free(1),
                    query_shape.free(2),
                    value_shape.free(3),
                ],
            ),
            element,
            self.carries(element, &[query, key, value]),
            Residency::Derived,
            tracked,
        );
        let log_sum_exp = self.fresh(
            Shape::from_axes(
                [
                    query_shape.dims()[0],
                    query_shape.dims()[1],
                    query_shape.dims()[2],
                    1,
                ],
                [
                    query_shape.free(0),
                    query_shape.free(1),
                    query_shape.free(2),
                    None,
                ],
            ),
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
        task.segments = segments.map_or(NO_VALUE, |offsets| offsets.id());
        task.param = attention.scale;
        task.slot = u32::from(attention.causal);
        task.reach = reach.unwrap_or(0);
        self.push(task);
        out
    }

    pub fn rope(&self, value: Value<'g>, origin: Option<Value<'g>>, base: f32) -> Value<'g> {
        let value = self.own(value);
        let shape = self.shape(value);
        let width = shape.dims()[3];
        assert!(
            shape.free(3).is_none(),
            "a rotation pairs every number of a row with the number half a row away, and axis 3 of {:?} walks every width a free extent takes",
            shape.dims(),
        );
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
        let input_shape = self.shape(input);
        let filter_shape = self.shape(filter);
        let input_dims = input_shape.dims();
        let filter_dims = filter_shape.dims();
        assert!(
            input_shape.free(1).is_none() && filter_shape.free(1).is_none(),
            "a convolution of {} channels reads {} channels of a filter, and a free extent of the pair hands every length the group every channel serves",
            input_dims[1],
            filter_dims[1],
        );
        assert!(
            input_shape.free(2).is_none() && input_shape.free(3).is_none(),
            "a window walks the rows and columns of {:?}, and a free extent of either hands every length the taps reach",
            input_dims,
        );
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
            Shape::from_axes(
                [
                    input_dims[0],
                    filter_dims[0],
                    (padded_rows - filter_dims[2]) / window.stride_rows() + 1,
                    (padded_columns - filter_dims[3]) / window.stride_columns() + 1,
                ],
                [input_shape.free(0), filter_shape.free(0), None, None],
            ),
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
        let input_shape = self.shape(input);
        let input_dims = input_shape.dims();
        assert!(
            input_shape.free(2).is_none() && input_shape.free(3).is_none(),
            "a window walks the rows and columns of {:?}, and a free extent of either hands every length the taps reach",
            input_dims,
        );
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
            Shape::from_axes(
                [
                    input_dims[0],
                    input_dims[1],
                    (padded_rows - window.reach_rows()) / window.stride_rows() + 1,
                    (padded_columns - window.reach_columns()) / window.stride_columns() + 1,
                ],
                [input_shape.free(0), input_shape.free(1), None, None],
            ),
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
        self.rowwise(Kind::Softmax, value)
    }

    pub fn log_softmax(&self, value: Value<'g>) -> Value<'g> {
        let value = self.own(value);
        self.rowwise(Kind::LogSoftmax, value)
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
        let shape = self.shape(source);
        let out = self.fresh(
            shape.fixed_axis(3, 1),
            Element::Single,
            Residency::Derived,
            false,
        );
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
        assert!(
            shape.free(3).is_none() && shape.dims()[3] == 1,
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
        let out = self.fresh(
            self.shape(indices).fixed_axis(3, classes),
            Element::Single,
            Residency::Derived,
            false,
        );
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
        let mut frees = self.shape(indices).frees();
        frees[3] = self.shape(table).free(3);
        let out = self.stored(
            Shape::from_axes(dims, frees),
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
        let mut frees = first.frees();
        assert!(
            first.free(axis).is_none(),
            "a concatenation along axis {axis} joins {:?} with the tensors beside it, and a free extent of the axis holds every length the sum takes",
            first.dims(),
        );
        frees[axis as usize] = None;
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
            for index in 0..MAX_RANK as usize {
                let walked = index as u32;
                assert!(
                    walked == axis || first.meets(shape, walked, walked),
                    "a concatenation along axis {axis} meets {:?} and {:?}",
                    first.dims(),
                    shape.dims(),
                );
            }
            dims[axis as usize] += shape.dims()[axis as usize];
        }
        let widened = element.narrow();
        let out = self.stored(
            Shape::from_axes(dims, frees),
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
        let out = self.stored(
            shape.fixed_axis(axis, length),
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
                    Residency::Input
                        | Residency::Parameter
                        | Residency::State
                        | Residency::Resident
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
        let (shape, strides, strides_source, storage, element, scale, tracked) = {
            let state = self.state.borrow();
            let info = &state.values[value.id() as usize];
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
        let mut permuted_dims = [1u32; MAX_RANK as usize];
        let mut permuted_frees = [None; MAX_RANK as usize];
        let mut permuted_strides = [0u32; MAX_RANK as usize];
        let mut permuted_source = [0u8; MAX_RANK as usize];
        let source = strides_source.unwrap_or([0, 1, 2, 3]);
        for axis in 0..MAX_RANK as usize {
            let walked = order[axis] as usize;
            permuted_dims[axis] = shape.dims()[walked];
            permuted_frees[axis] = shape.free(walked as u32);
            permuted_strides[axis] = strides[walked];
            permuted_source[axis] = source[walked];
        }
        self.alias(
            Shape::from_axes(permuted_dims, permuted_frees),
            permuted_strides,
            Some(permuted_source),
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
        assert_eq!(
            shape.frees(),
            self.shape(value).frees(),
            "a reshape of {:?} into {:?} holds every free extent where it stood, since a reshape that merges one with the axes around it walks a length no binding names",
            self.shape(value).dims(),
            shape.dims(),
        );
        let (storage, element, scale, tracked) = {
            let state = self.state.borrow();
            let info = &state.values[value.id() as usize];
            (info.storage, info.element, info.scale, info.requires_grad)
        };
        self.alias(
            shape,
            shape.strides(),
            None,
            storage,
            element,
            scale,
            tracked,
        )
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
