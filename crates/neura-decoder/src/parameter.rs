use neura_abi::Element;
use neura_gguf::{Archive, Tensor};
use neura_graph::{Graph, Init, Shape, Value};

pub fn matrix(graph: &Graph, source: &impl Archive, name: &str, rows: u32, columns: u32) -> Value {
    let held = held(source, name);
    assert_eq!(
        (held.rank, held.shape[0], held.shape[1]),
        (2, u64::from(columns), u64::from(rows)),
        "the tensor {name} of this gguf walks {} dimensions of {:?} where {rows} rows of {columns} numbers walk [1, 1, {rows}, {columns}]",
        held.rank,
        held.shape,
    );
    declare(graph, name, Shape::matrix(rows, columns), held.element)
}

pub fn vector(graph: &Graph, source: &impl Archive, name: &str, width: u32) -> Value {
    let held = held(source, name);
    assert_eq!(
        (held.rank, held.shape[0]),
        (1, u64::from(width)),
        "the tensor {name} of this gguf walks {} dimensions of {:?} where {width} numbers walk [1, 1, 1, {width}]",
        held.rank,
        held.shape,
    );
    declare(graph, name, Shape::vector(width), held.element)
}

fn held(source: &impl Archive, name: &str) -> Tensor {
    source
        .tensor(name)
        .unwrap_or_else(|| panic!("this gguf holds no tensor named {name}"))
}

fn declare(graph: &Graph, name: &str, shape: Shape, element: Element) -> Value {
    if element.block().is_some() {
        graph.named_block_quantized_parameter(name, shape, Init::Zero, element)
    } else {
        graph.named_parameter(name, shape, Init::Zero, element)
    }
}
