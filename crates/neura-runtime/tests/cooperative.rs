use neura_abi::Element;
use neura_graph::{Graph, Init, Shape};
use neura_plan::Product;
use neura_profile::{CooperativeMatrix, CooperativeTile, MatmulStrategy, MatmulTile, Profile};
use neura_runtime::Runtime;

#[path = "support/input.rs"]
mod input;
#[path = "support/reference.rs"]
mod reference;

use input::random;
use reference::matmul_reference;

const WORKGROUP: u32 = 256;

fn open() -> Runtime {
    Runtime::open(neura_runtime::RuntimeRequest {
        gpu: neura_gpu::GpuRequest {
            backends: neura_gpu::Backends::VULKAN,
            ..Default::default()
        },
        memory: neura_runtime::MemoryRequest {
            readback_bytes: 4 << 20,
            ..Default::default()
        },
    })
    .unwrap_or_else(|error| panic!("no Vulkan device runs the tests: {error}"))
}

fn cooperative_tile(runtime: &Runtime, fragments: (u32, u32)) -> Option<MatmulTile> {
    let capability = runtime.capability().cooperative_matrix?;
    let fragment = CooperativeMatrix::new(
        capability.subgroup,
        capability.rows,
        capability.columns,
        capability.depth,
    );
    let subgroups = WORKGROUP / capability.subgroup;
    if subgroups == 0 {
        return None;
    }
    let subgroups_rows = (subgroups / fragments.1).max(1);
    let tile = CooperativeTile::new(
        fragment,
        (subgroups_rows, subgroups / subgroups_rows),
        fragments,
        fragment.depth(),
    );
    (tile.threads() == WORKGROUP).then_some(MatmulTile::cooperative(tile))
}

fn cooperative_tiles(runtime: &Runtime) -> Option<Vec<MatmulTile>> {
    let tiles = [(2u32, 2u32), (2, 1), (1, 1)]
        .into_iter()
        .filter_map(|fragments| cooperative_tile(runtime, fragments))
        .collect::<Vec<_>>();
    (!tiles.is_empty()).then_some(tiles)
}

fn cooperative_profile(tile: MatmulTile) -> Profile {
    let streamed = MatmulTile::new(MatmulStrategy::Streamed, 1, WORKGROUP, 8, 1, WORKGROUP);
    Profile::of(&[streamed, tile])
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
    let Some(tiles) = cooperative_tiles(&runtime) else {
        return;
    };
    let scalar = scalar_profile(&runtime);
    for tile in tiles {
        for (rows, depth, columns) in [(64, 64, 64), (160, 96, 128)] {
            let gathered = product(&runtime, cooperative_profile(tile), rows, depth, columns);
            let plain = product(&runtime, scalar, rows, depth, columns);
            assert_close(&gathered, &plain, 2e-2);
        }
    }
}

#[test]
fn a_product_shortlists_one_tile_of_every_strategy_its_profile_offers() {
    let runtime = open();
    let Some(tile) = cooperative_tile(&runtime, (2, 2)) else {
        return;
    };
    let profile = cooperative_profile(tile);
    let product = Product::of(1, 512, 512, 512);
    let shortlist = product.shortlist(profile);
    assert_eq!(shortlist.len(), 2, "a product shortlists two tiles");
    assert!(
        shortlist
            .iter()
            .any(|tile| tile.strategy() == MatmulStrategy::Cooperative),
        "a product shortlists no cooperative tile its profile offers",
    );
    assert!(
        shortlist
            .iter()
            .any(|tile| tile.strategy() != MatmulStrategy::Cooperative),
        "a product shortlists no plain tile its profile offers",
    );
    assert!(
        shortlist.iter().all(|tile| profile.tiles().contains(tile)),
        "a product shortlists a tile its profile does not offer",
    );
    assert_eq!(product.planned(profile), shortlist[0]);
    assert_eq!(product.gathered(profile), Some(shortlist[1]));
}

#[test]
fn a_product_walks_the_cooperative_tile_a_measured_choice_names() {
    let runtime = open();
    let Some(tiles) = cooperative_tiles(&runtime) else {
        return;
    };
    let tile = tiles[0];
    let graph = Graph::new();
    let left = graph.parameter(Shape::matrix(96, 64), Init::Zero, Element::Single);
    let right = graph.parameter(Shape::matrix(64, 96), Init::Zero, Element::Single);
    let out = graph.matmul(left, right);
    let weights = runtime.weights(&graph);
    let product = Product::of(1, 96, 96, 64);
    let gathered = runtime.compile_chosen(
        &graph,
        &weights,
        cooperative_profile(tile),
        &[(product, tile)],
    );
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
    assert!(
        gathered
            .matmul_geometries()
            .iter()
            .any(|(used, _)| *used == tile),
        "a product of 96 rows, 64 of depth and 96 columns walks the cooperative tile {tile:?} a measured plan asks for",
    );
    let left_data = random(96 * 64, 7);
    let right_data = random(64 * 96, 13);
    runtime.write(&gathered, left, &left_data);
    runtime.write(&gathered, right, &right_data);
    runtime.run(&gathered);
    let measured = runtime.read(&gathered, out);
    assert_close(
        &measured,
        &matmul_reference(&left_data, &right_data, 96, 64, 96),
        2e-2,
    );
}
