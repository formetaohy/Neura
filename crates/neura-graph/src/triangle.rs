use crate::graph::{Graph, Value};
use crate::shape::Shape;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Triangle {
    lower: Value,
    strict: Value,
    identity: Value,
    extent: u32,
}

impl Triangle {
    pub const fn lower(self) -> Value {
        self.lower
    }

    pub const fn strict(self) -> Value {
        self.strict
    }

    pub const fn identity(self) -> Value {
        self.identity
    }

    pub const fn extent(self) -> u32 {
        self.extent
    }
}

impl Graph {
    pub fn triangle(&self, extent: u32) -> Triangle {
        assert!(extent > 0, "a triangle of {extent} rows holds no row");
        if let Some(triangle) = self.state.borrow().triangles.get(&extent) {
            return *triangle;
        }
        let triangle = self.build_triangle(extent);
        self.state.borrow_mut().triangles.insert(extent, triangle);
        triangle
    }

    fn build_triangle(&self, extent: u32) -> Triangle {
        let lower = self.triangular_rows(extent, true);
        let strict = self.triangular_rows(extent, false);
        let identity = self.sub(lower, strict);
        Triangle {
            lower,
            strict,
            identity,
            extent,
        }
    }

    fn triangular_rows(&self, extent: u32, diagonal: bool) -> Value {
        let mut rows = Vec::with_capacity(extent as usize);
        for row in 0..extent {
            let upper = if diagonal { row + 1 } else { row };
            let mut parts = Vec::with_capacity(2);
            if upper > 0 {
                parts.push(self.fill(Shape::of([1, 1, 1, upper]), 1.0));
            }
            if upper < extent {
                parts.push(self.fill(Shape::of([1, 1, 1, extent - upper]), 0.0));
            }
            rows.push(match parts.len() {
                1 => parts[0],
                _ => self.concat(&parts, 3),
            });
        }
        self.concat(&rows, 2)
    }
}
