use neura_abi::Element;
use neura_graph::{Graph, Init, Shape};
use neura_profile::{MatmulStrategy, MatmulTile, Profile};
use neura_runtime::Runtime;

#[path = "support/reference.rs"]
mod reference;

use reference::{matmul_reference, random};

const WORKGROUP: u32 = 256;

fn open() -> Runtime {
    pollster::block_on(Runtime::open(neura_runtime::RuntimeRequest {
        gpu: neura_runtime::GpuRequest {
            backends: neura_runtime::Backends::VULKAN,
            ..Default::default()
        },
        readback_bytes: 4 << 20,
        ..Default::default()
    }))
    .unwrap_or_else(|error| panic!("no Vulkan device runs the tests: {error}"))
}

fn cooperative_profile(runtime: &Runtime) -> Option<Profile> {
    let capability = runtime.capability().cooperative_matrix?;
    let subgroups = WORKGROUP / capability.subgroup;
    let subgroup_columns = (subgroups / 2).max(1);
    let subgroup_rows = subgroups / subgroup_columns;
    let streamed = MatmulTile::new(MatmulStrategy::Streamed, 1, WORKGROUP, 8, 1, WORKGROUP);
    let cooperative = MatmulTile::cooperative(
        subgroup_rows * capability.rows,
        subgroup_columns * capability.columns,
        capability.depth,
        subgroup_rows,
        subgroup_columns,
    );
    Some(Profile::of(&[streamed, cooperative]))
}

fn scalar_profile(runtime: &Runtime) -> Profile {
    *Profile::derive(runtime.budget(), None)
        .last()
        .expect("a profile without cooperative tiles")
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let allowed = tolerance * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= allowed,
            "element {index} came back as {actual} where {expected} was expected",
        );
    }
}

fn product(runtime: &Runtime, profile: Profile, rows: u32, depth: u32, columns: u32) -> Vec<f32> {
    let graph = Graph::new();
    let left = graph.parameter(Shape::matrix(rows, depth), Init::Zero, Element::Single);
    let right = graph.parameter(Shape::matrix(depth, columns), Init::Zero, Element::Single);
    let out = graph.matmul(left, right);
    let weights = runtime.weights(&graph);
    let program = runtime.compile_with(&graph, &weights, profile);
    let left_data = random(rows * depth, 11);
    let right_data = random(depth * columns, 29);
    runtime.write(&program, left, &left_data);
    runtime.write(&program, right, &right_data);
    runtime.run(&program);
    let product = runtime.read(&program, out);
    assert_close(
        &product,
        &matmul_reference(&left_data, &right_data, rows, depth, columns),
        2e-2,
    );
    product
}

#[test]
fn a_cooperative_product_matches_a_cpu_reference() {
    let runtime = open();
    let Some(cooperative) = cooperative_profile(&runtime) else {
        return;
    };
    let scalar = scalar_profile(&runtime);
    for (rows, depth, columns) in [(64, 64, 64), (128, 96, 160), (37, 48, 19)] {
        let gathered = product(&runtime, cooperative, rows, depth, columns);
        let plain = product(&runtime, scalar, rows, depth, columns);
        assert_close(&gathered, &plain, 2e-2);
    }
}

#[test]
fn a_cooperative_profile_carries_cooperative_tiles() {
    let runtime = open();
    let Some(cooperative) = cooperative_profile(&runtime) else {
        return;
    };
    let graph = Graph::new();
    let left = graph.parameter(Shape::matrix(96, 64), Init::Zero, Element::Single);
    let right = graph.parameter(Shape::matrix(64, 96), Init::Zero, Element::Single);
    let out = graph.matmul(left, right);
    let weights = runtime.weights(&graph);
    let gathered = runtime.compile_with(&graph, &weights, cooperative);
    let plain = runtime.compile_with(&graph, &weights, scalar_profile(&runtime));
    assert!(
        gathered
            .tiles()
            .iter()
            .any(|tile| matches!(tile.strategy(), MatmulStrategy::Cooperative)),
        "a cooperative profile carries a cooperative tile",
    );
    assert!(
        plain
            .tiles()
            .iter()
            .all(|tile| !matches!(tile.strategy(), MatmulStrategy::Cooperative)),
        "a profile without cooperative tiles carries none",
    );
    let left_data = random(96 * 64, 7);
    let right_data = random(64 * 96, 13);
    runtime.write(&gathered, left, &left_data);
    runtime.write(&gathered, right, &right_data);
    runtime.run(&gathered);
    let product = runtime.read(&gathered, out);
    assert_close(
        &product,
        &matmul_reference(&left_data, &right_data, 96, 64, 96),
        2e-2,
    );
}
