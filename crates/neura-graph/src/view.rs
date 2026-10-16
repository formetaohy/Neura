use crate::graph::{Declaration, Graph, Value};
use crate::layout::Layout;
use crate::shape::Shape;
use neura_abi::{Element, MAX_RANK};

pub(crate) struct Source {
    storage: u32,
    strides: [u32; 4],
    origin: u64,
    element: Element,
    scale: f32,
    tracked: bool,
}

impl Graph {
    fn view_source(&self, value: u32) -> Option<Source> {
        let state = self.state.borrow();
        let info = &state.values[value as usize];
        if info.element != Element::Single {
            return None;
        }
        if info.shape.dynamic() || state.values[info.storage as usize].shape.dynamic() {
            return None;
        }
        Some(Source {
            storage: info.storage,
            strides: info.strides,
            origin: info.layout.origin(),
            element: info.element,
            scale: info.scale,
            tracked: info.requires_grad,
        })
    }

    pub(crate) fn slice_view(
        &self,
        value: Value,
        axis: u32,
        start: u32,
        length: u32,
    ) -> Option<Value> {
        let source = self.view_source(value.id())?;
        let shape = self.shape(value).fixed_axis(axis, length);
        let origin = source.origin + u64::from(start) * u64::from(source.strides[axis as usize]);
        Some(self.declare(Declaration::viewed(
            shape,
            Layout::strided(shape.dims(), source.strides, origin),
            source.storage,
            source.element,
            source.scale,
            source.tracked,
        )))
    }

    pub(crate) fn concat_view(&self, values: &[Value], axis: u32, shape: Shape) -> Option<Value> {
        let axis = axis as usize;
        let first = self.view_source(values[0].id())?;
        if values.len() == 1 {
            return Some(values[0]);
        }
        let mut starts = Vec::with_capacity(values.len());
        let mut origins = Vec::with_capacity(values.len());
        let mut length = 0u32;
        let mut step = None;
        for value in values {
            let source = self.view_source(value.id())?;
            if source.storage != first.storage {
                return None;
            }
            for walked in 0..MAX_RANK as usize {
                if walked != axis && source.strides[walked] != first.strides[walked] {
                    return None;
                }
            }
            let held = self.shape(*value).dims()[axis];
            if held > 1 {
                match step {
                    Some(stride) if stride != source.strides[axis] => return None,
                    _ => step = Some(source.strides[axis]),
                }
            }
            starts.push(length);
            origins.push(source.origin);
            length += held;
        }
        let stride = match step {
            Some(stride) if stride > 0 => stride,
            Some(_) => return None,
            None => {
                let step = i64::try_from(origins[1]).ok()? - i64::try_from(origins[0]).ok()?;
                if step <= 0 {
                    return None;
                }
                u32::try_from(step).ok()?
            }
        };
        let base = i64::try_from(origins[0]).ok()?;
        for (start, origin) in starts.iter().zip(&origins) {
            let reached = base + i64::from(*start) * i64::from(stride);
            if i64::try_from(*origin).ok()? != reached {
                return None;
            }
        }
        let mut strides = first.strides;
        strides[axis] = stride;
        Some(self.declare(Declaration::viewed(
            shape,
            Layout::strided(shape.dims(), strides, base as u64),
            first.storage,
            first.element,
            first.scale,
            first.tracked,
        )))
    }
}
