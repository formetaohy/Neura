use neura::gguf::GgufFile;
use neura::{Graph, Init, Shape, Value};

pub fn tensor<'g>(graph: &Graph<'g>, source: &GgufFile, name: &str) -> Value<'g> {
    let held = source
        .tensor(name)
        .unwrap_or_else(|| panic!("this gguf holds no tensor named {name}"));
    let shape = match held.rank {
        1 => Shape::vector(held.shape[0] as u32),
        2 => Shape::matrix(held.shape[1] as u32, held.shape[0] as u32),
        rank => panic!("the tensor {name} walks {rank} dimensions and holds no row layout"),
    };
    if held.element.block().is_some() {
        graph.named_block_quantized_parameter(name, shape, Init::Zero, held.element)
    } else {
        graph.named_parameter(name, shape, Init::Zero, held.element)
    }
}

pub fn rms<'g>(
    graph: &Graph<'g>,
    value: Value<'g>,
    weight: Value<'g>,
    floor: Value<'g>,
) -> Value<'g> {
    let squares = graph.mul(value, value);
    let mean = graph.mean_axis(squares, 3);
    let scale = graph.recip(graph.sqrt(graph.add(mean, floor)));
    graph.mul(graph.mul(value, scale), weight)
}

pub fn l2<'g>(graph: &Graph<'g>, value: Value<'g>, floor: Value<'g>) -> Value<'g> {
    let squares = graph.mul(value, value);
    let total = graph.sum_axis(squares, 3);
    let scale = graph.recip(graph.sqrt(graph.add(total, floor)));
    graph.mul(value, scale)
}

pub fn softplus<'g>(graph: &Graph<'g>, value: Value<'g>) -> Value<'g> {
    let threshold = graph.fill(Shape::scalar(), 20.0);
    let raised = graph.log(graph.add(graph.fill(Shape::scalar(), 1.0), graph.exp(value)));
    graph.select(graph.greater(value, threshold), value, raised)
}

pub fn flatten<'g>(graph: &Graph<'g>, value: Value<'g>) -> Value<'g> {
    let elements = graph.shape(value).elements();
    graph.reshape(value, Shape::of([1, 1, 1, elements]))
}

pub fn heads<'g>(graph: &Graph<'g>, value: Value<'g>, count: u32, width: u32) -> Value<'g> {
    let parts = (0..count)
        .map(|head| graph.slice(value, 3, head * width, width))
        .collect::<Vec<Value<'g>>>();
    graph.concat(&parts, 0)
}

pub fn interleaved<'g>(
    graph: &Graph<'g>,
    value: Value<'g>,
    count: u32,
    width: u32,
    offset: u32,
) -> Value<'g> {
    let parts = (0..count)
        .map(|head| graph.slice(value, 3, head * width * 2 + offset, width))
        .collect::<Vec<Value<'g>>>();
    graph.concat(&parts, 0)
}

pub fn partial_rope<'g>(
    graph: &Graph<'g>,
    value: Value<'g>,
    position: Value<'g>,
    base: Value<'g>,
    width: u32,
) -> Value<'g> {
    let written = graph.shape(value).dims()[3];
    let front = graph.slice(value, 3, 0, width);
    let turned = graph.rope(front, Some(position), Some(base));
    if written == width {
        return turned;
    }
    let rest = graph.slice(value, 3, width, written - width);
    graph.concat(&[turned, rest], 3)
}

pub fn seat<'g>(graph: &Graph<'g>, position: Value<'g>, count: u32, capacity: u32) -> Value<'g> {
    let rows = (0..count)
        .map(|head| {
            graph.add(
                position,
                graph.fill(Shape::scalar(), (head * capacity) as f32),
            )
        })
        .collect::<Vec<Value<'g>>>();
    graph.concat(&rows, 0)
}
