use neura_abi::Element;
use neura_gpu::WARM_PROGRAMS;
use neura_graph::{Graph, Init, Shape, Value};
use neura_runtime::{Program, Runtime};

#[path = "support/input.rs"]
mod input;
#[path = "support/reference.rs"]
mod reference;
#[path = "support/mod.rs"]
mod support;

use input::random;
use reference::matmul_reference;
use support::{assert_close, open};

const ELEMENTS: [Element; 3] = [Element::Single, Element::Half, Element::Fp8E4M3];

type Mix = fn(&Graph<'static>, Value<'static>) -> Value<'static>;

fn mixes() -> [Mix; 5] {
    [
        |_graph, out| out,
        |graph, out| graph.relu(out),
        |graph, out| graph.sum_rows(out),
        |graph, out| graph.softmax(out),
        |graph, out| graph.add(out, graph.fill(Shape::matrix(ROWS, COLUMNS), 0.5)),
    ]
}

const ROWS: u32 = 64;
const DEPTH: u32 = 32;
const COLUMNS: u32 = 64;

struct Built {
    program: Program,
    out: Value<'static>,
    left_data: Vec<f32>,
    right_data: Vec<f32>,
}

fn compiled(runtime: &Runtime, element: Element, mix: Mix) -> Built {
    let graph: Graph<'static> = Graph::new();
    let left = graph.input(Shape::matrix(ROWS, DEPTH), Element::Single);
    let right = graph.parameter(Shape::matrix(DEPTH, COLUMNS), Init::Zero, element);
    let out = mix(&graph, graph.matmul(left, right));
    graph.retain(out);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let left_data = random(ROWS * DEPTH, 5);
    let right_data = random(DEPTH * COLUMNS, 9);
    runtime.write(&program, left, &left_data);
    runtime.write(&program, right, &right_data);
    runtime.run(&program);
    Built {
        program,
        out,
        left_data,
        right_data,
    }
}

fn product(built: &Built) -> Vec<f32> {
    matmul_reference(&built.left_data, &built.right_data, ROWS, DEPTH, COLUMNS)
}

fn churn(runtime: &Runtime) -> Vec<Built> {
    let mut built = Vec::new();
    for element in ELEMENTS {
        for mix in mixes() {
            built.push(compiled(runtime, element, mix));
            assert!(
                runtime.declared_kernels() <= WARM_PROGRAMS,
                "{} programs are alive, and the device holds {} of them",
                built.len(),
                runtime.declared_kernels(),
            );
            assert!(
                runtime.assembled_kernels() <= WARM_PROGRAMS,
                "{} programs are alive, and the device assembled {} of them",
                built.len(),
                runtime.assembled_kernels(),
            );
        }
    }
    built
}

#[test]
fn a_runtime_keeps_a_bounded_set_of_device_programs() {
    let runtime = open();
    assert_eq!(runtime.declared_kernels(), 0);
    assert_eq!(runtime.assembled_kernels(), 0);
    let shapes = ELEMENTS.len() * mixes().len();
    assert!(
        shapes > WARM_PROGRAMS + 2,
        "a device keeps {WARM_PROGRAMS} programs beside the live ones, and {shapes} shapes cannot outrun that",
    );
    let built = churn(&runtime);
    assert_eq!(
        runtime.resident_plans(),
        built.len(),
        "every live program holds the plan it was compiled from",
    );
    let expected = product(&built[0]);
    assert_close(
        &runtime.read(&built[0].program, built[0].out),
        &expected,
        1e-4,
    );
    drop(built);
    assert!(
        runtime.declared_kernels() <= WARM_PROGRAMS,
        "a runtime that dropped every program holds {} of them",
        runtime.declared_kernels(),
    );
    assert!(
        runtime.assembled_kernels() <= WARM_PROGRAMS,
        "a runtime that dropped every program assembled {} of them",
        runtime.assembled_kernels(),
    );
    let again = compiled(&runtime, Element::Single, mixes()[0]);
    assert_close(
        &runtime.read(&again.program, again.out),
        &product(&again),
        1e-4,
    );
}

#[test]
fn a_program_a_cache_let_go_still_computes_its_numbers() {
    let runtime = open();
    let kept = compiled(&runtime, Element::Single, mixes()[0]);
    let expected = product(&kept);
    let churned = churn(&runtime);
    assert_close(&runtime.read(&kept.program, kept.out), &expected, 1e-4);
    for built in &churned {
        runtime.run(&built.program);
    }
    runtime.run(&kept.program);
    assert_close(&runtime.read(&kept.program, kept.out), &expected, 1e-4);
}
