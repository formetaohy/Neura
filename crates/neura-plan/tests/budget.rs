use neura_abi::Element;
use neura_graph::{AttentionOptions, Graph, Init, Shape};
use neura_plan::{Contiguous, DEFAULT_ENCODING_BYTES, Plan, Product, Weight};
use neura_profile::{CooperativeMatrix, CooperativeTile, MatmulTile, Profile};

#[test]
fn a_workgroup_holds_the_attention_beside_the_panels_its_cooperative_tiles_stage() {
    let cooperative =
        CooperativeTile::new(CooperativeMatrix::new(32, 16, 16, 16), (4, 8), (1, 1), 16);
    let profile = Profile::within(&[MatmulTile::cooperative(cooperative)], 40);
    let graph = Graph::new();
    let query = graph.parameter(Shape::of([1, 1, 64, 256]), Init::Zero, Element::Single);
    let key = graph.parameter(Shape::of([1, 1, 1500, 256]), Init::Zero, Element::Single);
    let value = graph.parameter(Shape::of([1, 1, 1500, 256]), Init::Zero, Element::Single);
    let attended = graph.attention(
        query,
        key,
        value,
        AttentionOptions {
            bias: None,
            scale: None,
            causal: false,
            origin: None,
            segments: None,
            reach: None,
            query_segments: None,
        },
    );
    let left = graph.parameter(Shape::of([1, 1, 64, 256]), Init::Zero, Element::Single);
    let stored = graph.parameter(Shape::of([1, 1, 256, 256]), Init::Zero, Element::Single);
    let right = graph.permute(stored, [0, 1, 3, 2]);
    let product_out = graph.matmul(left, right);
    graph.retain(graph.add(attended, product_out));
    let product = Product::of(
        1,
        64,
        256,
        256,
        Weight::of(Element::Single, Contiguous::Reduction),
    );
    let plan = Plan::chosen(
        &graph,
        256,
        profile,
        &[(product, MatmulTile::cooperative(cooperative))],
        DEFAULT_ENCODING_BYTES,
        0,
    );
    let geometry = profile.geometry(&plan.walked_tiles().collect::<Vec<_>>(), plan.attention());
    let declared = geometry.workgroup_bytes(plan.kinds());
    assert!(
        declared <= geometry.budget(),
        "a workgroup of {} kinds declares {declared} bytes of scratch where its profile offers {}",
        plan.kinds().len(),
        geometry.budget(),
    );
}
