use neura_abi::Element;
use neura_graph::{Graph, Init, Shape, Value};
use neura_runtime::{MemoryRequest, Runtime, RuntimeRequest};

fn runtime() -> Runtime {
    Runtime::open(RuntimeRequest {
        memory: MemoryRequest {
            heap_bytes: 1 << 28,
            readback_bytes: 1 << 20,
            ..Default::default()
        },
        ..Default::default()
    })
    .expect("a device that runs the tests")
}

fn held() -> Vec<f32> {
    (0..12).map(|at| at as f32 * 0.25 - 0.5).collect()
}

fn composed_rms(graph: &Graph, value: Value, weight: Value, floor: Value) -> Value {
    let squares = graph.mul(value, value);
    let mean = graph.mean_axis(squares, 3);
    let scale = graph.recip(graph.sqrt(graph.add(mean, floor)));
    graph.mul(graph.mul(value, scale), weight)
}

fn composed_l2(graph: &Graph, value: Value, floor: Value) -> Value {
    let squares = graph.mul(value, value);
    let total = graph.sum_axis(squares, 3);
    let scale = graph.recip(graph.sqrt(graph.add(total, floor)));
    graph.mul(value, scale)
}

#[test]
fn a_fused_norm_walks_the_rows_the_composed_norms_walk() {
    let runtime = runtime();
    for view in [false, true] {
        let graph = Graph::new();
        let source = graph.input(Shape::of([1, 1, 4, 3]), Element::Single);
        let value = if view {
            graph.permute(source, [0, 1, 3, 2])
        } else {
            source
        };
        let columns = graph.shape(value).dims()[3];
        let weight = graph.parameter(
            Shape::vector(columns),
            Init::Uniform {
                low: 0.5,
                high: 1.5,
            },
            Element::Single,
        );
        let floor = graph.fill(Shape::scalar(), 1e-5);
        let fused = graph.rms_norm(value, weight, floor);
        let l2 = graph.l2_norm(value, floor);
        let scale = composed_rms(&graph, value, weight, floor);
        let plain = composed_l2(&graph, value, floor);
        graph.retain(fused);
        graph.retain(l2);
        graph.retain(scale);
        graph.retain(plain);
        let weights = runtime.weights(&graph);
        let program = runtime.compile(&graph, &weights);
        runtime.write(&program, source, &held());
        runtime.run(&program);
        runtime.context().drain();
        for (fused, composed) in [(fused, scale), (l2, plain)] {
            let fused = runtime.read(&program, fused);
            let composed = runtime.read(&program, composed);
            for (left, right) in fused.iter().zip(&composed) {
                assert!(
                    (left - right).abs() < 1e-5,
                    "view {view}: {left} != {right}",
                );
            }
        }
    }
}

#[test]
fn a_fused_norm_descends_the_rows_it_normalizes() {
    let runtime = runtime();
    for view in [false, true] {
        let graph = Graph::new();
        let source = graph.gradient_input(Shape::of([1, 1, 4, 3]), Element::Single);
        let value = if view {
            graph.permute(source, [0, 1, 3, 2])
        } else {
            source
        };
        let columns = graph.shape(value).dims()[3];
        let weight = graph.parameter(
            Shape::vector(columns),
            Init::Uniform {
                low: 0.5,
                high: 1.5,
            },
            Element::Single,
        );
        let floor = graph.fill(Shape::scalar(), 1e-5);
        let loss = graph.sum(graph.rms_norm(value, weight, floor));
        let gradients = graph.backward(loss);
        let input = gradients.of(source);
        let weighted = gradients.of(weight);
        graph.retain(input);
        graph.retain(weighted);

        let reference = Graph::new();
        let source_ref = reference.gradient_input(Shape::of([1, 1, 4, 3]), Element::Single);
        let value_ref = if view {
            reference.permute(source_ref, [0, 1, 3, 2])
        } else {
            source_ref
        };
        let weight_ref = reference.parameter(
            Shape::vector(columns),
            Init::Uniform {
                low: 0.5,
                high: 1.5,
            },
            Element::Single,
        );
        let floor_ref = reference.fill(Shape::scalar(), 1e-5);
        let loss_ref = reference.sum(composed_rms(&reference, value_ref, weight_ref, floor_ref));
        let gradients_ref = reference.backward(loss_ref);
        let input_ref = gradients_ref.of(source_ref);
        let weighted_ref = gradients_ref.of(weight_ref);
        reference.retain(input_ref);
        reference.retain(weighted_ref);

        let weights = runtime.weights(&graph);
        let program = runtime.compile(&graph, &weights);
        let reference_weights = runtime.weights(&reference);
        let reference_program = runtime.compile(&reference, &reference_weights);
        runtime.write(&program, source, &held());
        runtime.write(&reference_program, source_ref, &held());
        runtime.run(&program);
        runtime.run(&reference_program);
        runtime.context().drain();
        for (fused, composed) in [(input, input_ref), (weighted, weighted_ref)] {
            let fused = runtime.read(&program, fused);
            let composed = runtime.read(&reference_program, composed);
            for (left, right) in fused.iter().zip(&composed) {
                assert!(
                    (left - right).abs() < 1e-4,
                    "view {view}: {left} != {right}",
                );
            }
        }
    }
}

#[test]
fn a_fused_l2_norm_descends_the_rows_it_scales() {
    let runtime = runtime();
    let graph = Graph::new();
    let source = graph.gradient_input(Shape::of([1, 1, 4, 3]), Element::Single);
    let floor = graph.fill(Shape::scalar(), 1e-5);
    let loss = graph.sum(graph.l2_norm(source, floor));
    let gradients = graph.backward(loss);
    let input = gradients.of(source);
    graph.retain(input);

    let reference = Graph::new();
    let source_ref = reference.gradient_input(Shape::of([1, 1, 4, 3]), Element::Single);
    let floor_ref = reference.fill(Shape::scalar(), 1e-5);
    let loss_ref = reference.sum(composed_l2(&reference, source_ref, floor_ref));
    let gradients_ref = reference.backward(loss_ref);
    let input_ref = gradients_ref.of(source_ref);
    reference.retain(input_ref);

    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let reference_weights = runtime.weights(&reference);
    let reference_program = runtime.compile(&reference, &reference_weights);
    runtime.write(&program, source, &held());
    runtime.write(&reference_program, source_ref, &held());
    runtime.run(&program);
    runtime.run(&reference_program);
    runtime.context().drain();
    let fused = runtime.read(&program, input);
    let composed = runtime.read(&reference_program, input_ref);
    for (left, right) in fused.iter().zip(&composed) {
        assert!((left - right).abs() < 1e-4, "{left} != {right}");
    }
}
