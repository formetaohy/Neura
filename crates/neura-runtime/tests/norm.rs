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
        let shape = match view {
            true => Shape::of([1, 1, 3, 4]),
            false => Shape::of([1, 1, 4, 3]),
        };
        let source = graph.gradient_input(shape, Element::Single);
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
        let shape = match view {
            true => Shape::of([1, 1, 3, 4]),
            false => Shape::of([1, 1, 4, 3]),
        };
        let source_ref = reference.gradient_input(shape, Element::Single);
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

fn composed_layer(graph: &Graph, value: Value, weight: Value, shift: Value, floor: Value) -> Value {
    let mean = graph.mean_axis(value, 3);
    let centered = graph.sub(value, mean);
    let spread = graph.mean_axis(graph.mul(centered, centered), 3);
    let deviation = graph.sqrt(graph.add(spread, floor));
    let sharpened = graph.mul(centered, graph.recip(deviation));
    graph.add(graph.mul(sharpened, weight), shift)
}

fn composed_group(
    graph: &Graph,
    value: Value,
    weight: Value,
    shift: Value,
    floor: Value,
    groups: u32,
) -> Value {
    let dims = graph.shape(value).dims();
    let grouped = graph.reshape(
        value,
        Shape::of([dims[0] * groups, dims[1] / groups, dims[2], dims[3]]),
    );
    let mean = graph.mean_axis(graph.mean_axis(graph.mean_axis(grouped, 3), 2), 1);
    let centered = graph.sub(grouped, mean);
    let spread = graph.mean_axis(
        graph.mean_axis(graph.mean_axis(graph.mul(centered, centered), 3), 2),
        1,
    );
    let deviation = graph.sqrt(graph.add(spread, floor));
    let normalized = graph.mul(centered, graph.recip(deviation));
    let whole = graph.reshape(normalized, Shape::of(dims));
    graph.add(graph.mul(whole, weight), shift)
}

fn compared(
    runtime: &Runtime,
    program: &neura_runtime::Program,
    left: Value,
    right: &neura_runtime::Program,
    right_value: Value,
) {
    let left = runtime.read(program, left);
    let right = runtime.read(right, right_value);
    for (left, right) in left.iter().zip(&right) {
        assert!((left - right).abs() < 1e-4, "{left} != {right}");
    }
}

#[test]
fn a_fused_layer_norm_walks_and_descends_the_rows_it_centers() {
    let runtime = runtime();
    let held: Vec<f32> = (0..12)
        .map(|at| (at as f32 * 0.37).sin() * 1.5 + 0.2)
        .collect();
    for view in [false, true] {
        let graph = Graph::new();
        let shape = match view {
            true => Shape::of([1, 1, 3, 4]),
            false => Shape::of([1, 1, 4, 3]),
        };
        let source = graph.gradient_input(shape, Element::Single);
        let value = if view {
            graph.permute(source, [0, 1, 3, 2])
        } else {
            source
        };
        let weight = graph.parameter(
            Shape::vector(3),
            Init::Uniform {
                low: 0.5,
                high: 1.5,
            },
            Element::Single,
        );
        let shift = graph.parameter(
            Shape::vector(3),
            Init::Uniform {
                low: -0.5,
                high: 0.5,
            },
            Element::Single,
        );
        let floor = graph.fill(Shape::scalar(), 1e-5);
        let loss = graph.add(
            graph.sum(graph.layer_norm(value, weight, shift, floor)),
            graph.sum(source),
        );
        let gradients = graph.backward(loss);
        let source_gradient = gradients.of(source);
        let weight_gradient = gradients.of(weight);
        let shift_gradient = gradients.of(shift);
        graph.retain(source_gradient);
        graph.retain(weight_gradient);
        graph.retain(shift_gradient);

        let reference = Graph::new();
        let shape = match view {
            true => Shape::of([1, 1, 3, 4]),
            false => Shape::of([1, 1, 4, 3]),
        };
        let source_ref = reference.gradient_input(shape, Element::Single);
        let value_ref = if view {
            reference.permute(source_ref, [0, 1, 3, 2])
        } else {
            source_ref
        };
        let weight_ref = reference.parameter(
            Shape::vector(3),
            Init::Uniform {
                low: 0.5,
                high: 1.5,
            },
            Element::Single,
        );
        let shift_ref = reference.parameter(
            Shape::vector(3),
            Init::Uniform {
                low: -0.5,
                high: 0.5,
            },
            Element::Single,
        );
        let floor_ref = reference.fill(Shape::scalar(), 1e-5);
        let loss_ref = reference.add(
            reference.sum(composed_layer(
                &reference, value_ref, weight_ref, shift_ref, floor_ref,
            )),
            reference.sum(source_ref),
        );
        let gradients_ref = reference.backward(loss_ref);
        let source_gradient_ref = gradients_ref.of(source_ref);
        let weight_gradient_ref = gradients_ref.of(weight_ref);
        let shift_gradient_ref = gradients_ref.of(shift_ref);
        reference.retain(source_gradient_ref);
        reference.retain(weight_gradient_ref);
        reference.retain(shift_gradient_ref);

        let weights = runtime.weights(&graph);
        let program = runtime.compile(&graph, &weights);
        let reference_weights = runtime.weights(&reference);
        let reference_program = runtime.compile(&reference, &reference_weights);
        runtime.write(&program, source, &held);
        runtime.write(&reference_program, source_ref, &held);
        runtime.run(&program);
        runtime.run(&reference_program);
        runtime.context().drain();
        compared(
            &runtime,
            &program,
            source_gradient,
            &reference_program,
            source_gradient_ref,
        );
        compared(
            &runtime,
            &program,
            weight_gradient,
            &reference_program,
            weight_gradient_ref,
        );
        compared(
            &runtime,
            &program,
            shift_gradient,
            &reference_program,
            shift_gradient_ref,
        );
    }
}

#[test]
fn a_fused_group_norm_walks_and_descends_the_groups_it_centers() {
    let runtime = runtime();
    let held: Vec<f32> = (0..48)
        .map(|at| (at as f32 * 0.21).cos() * 1.2 - 0.1)
        .collect();
    let graph = Graph::new();
    let source = graph.gradient_input(Shape::of([2, 4, 2, 3]), Element::Single);
    let weight = graph.parameter(
        Shape::of([1, 4, 1, 1]),
        Init::Uniform {
            low: 0.5,
            high: 1.5,
        },
        Element::Single,
    );
    let shift = graph.parameter(
        Shape::of([1, 4, 1, 1]),
        Init::Uniform {
            low: -0.5,
            high: 0.5,
        },
        Element::Single,
    );
    let floor = graph.fill(Shape::scalar(), 1e-5);
    let loss = graph.sum(graph.group_norm(source, weight, shift, floor, 2));
    let gradients = graph.backward(loss);
    let source_gradient = gradients.of(source);
    let weight_gradient = gradients.of(weight);
    let shift_gradient = gradients.of(shift);
    graph.retain(source_gradient);
    graph.retain(weight_gradient);
    graph.retain(shift_gradient);

    let reference = Graph::new();
    let source_ref = reference.gradient_input(Shape::of([2, 4, 2, 3]), Element::Single);
    let weight_ref = reference.parameter(
        Shape::of([1, 4, 1, 1]),
        Init::Uniform {
            low: 0.5,
            high: 1.5,
        },
        Element::Single,
    );
    let shift_ref = reference.parameter(
        Shape::of([1, 4, 1, 1]),
        Init::Uniform {
            low: -0.5,
            high: 0.5,
        },
        Element::Single,
    );
    let floor_ref = reference.fill(Shape::scalar(), 1e-5);
    let loss_ref = reference.sum(composed_group(
        &reference, source_ref, weight_ref, shift_ref, floor_ref, 2,
    ));
    let gradients_ref = reference.backward(loss_ref);
    let source_gradient_ref = gradients_ref.of(source_ref);
    let weight_gradient_ref = gradients_ref.of(weight_ref);
    let shift_gradient_ref = gradients_ref.of(shift_ref);
    reference.retain(source_gradient_ref);
    reference.retain(weight_gradient_ref);
    reference.retain(shift_gradient_ref);

    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let reference_weights = runtime.weights(&reference);
    let reference_program = runtime.compile(&reference, &reference_weights);
    runtime.write(&program, source, &held);
    runtime.write(&reference_program, source_ref, &held);
    runtime.run(&program);
    runtime.run(&reference_program);
    runtime.context().drain();
    compared(
        &runtime,
        &program,
        source_gradient,
        &reference_program,
        source_gradient_ref,
    );
    compared(
        &runtime,
        &program,
        weight_gradient,
        &reference_program,
        weight_gradient_ref,
    );
    compared(
        &runtime,
        &program,
        shift_gradient,
        &reference_program,
        shift_gradient_ref,
    );

    let dynamic = Graph::new();
    let batch = dynamic.free(2);
    let source = dynamic.input(
        Shape::of([2, 4, 2, 3]).freed(&[(0, batch)]),
        Element::Single,
    );
    let weight = dynamic.parameter(
        Shape::of([1, 4, 1, 1]),
        Init::Uniform {
            low: 0.5,
            high: 1.5,
        },
        Element::Single,
    );
    let shift = dynamic.parameter(
        Shape::of([1, 4, 1, 1]),
        Init::Uniform {
            low: -0.5,
            high: 0.5,
        },
        Element::Single,
    );
    let floor = dynamic.fill(Shape::scalar(), 1e-5);
    let output = dynamic.group_norm(source, weight, shift, floor, 2);
    dynamic.retain(output);
    let weights = runtime.weights(&dynamic);
    let program = runtime.compile(&dynamic, &weights);
    runtime.bind(&program, &[1]);
    let held: Vec<f32> = (0..24).map(|at| (at as f32 * 0.31).sin() - 0.2).collect();
    runtime.write(&program, source, &held);
    runtime.run(&program);
    runtime.context().drain();
    let walked = runtime.read(&program, output);

    let fixed = Graph::new();
    let source = fixed.input(Shape::of([1, 4, 2, 3]), Element::Single);
    let weight = fixed.parameter(
        Shape::of([1, 4, 1, 1]),
        Init::Uniform {
            low: 0.5,
            high: 1.5,
        },
        Element::Single,
    );
    let shift = fixed.parameter(
        Shape::of([1, 4, 1, 1]),
        Init::Uniform {
            low: -0.5,
            high: 0.5,
        },
        Element::Single,
    );
    let floor = fixed.fill(Shape::scalar(), 1e-5);
    let out = fixed.group_norm(source, weight, shift, floor, 2);
    fixed.retain(out);
    let weights = runtime.weights(&fixed);
    let program = runtime.compile(&fixed, &weights);
    runtime.write(&program, source, &held);
    runtime.run(&program);
    runtime.context().drain();
    let expected = runtime.read(&program, out);
    for (walked, expected) in walked.iter().zip(&expected) {
        assert!((walked - expected).abs() < 1e-5, "{walked} != {expected}");
    }
}

#[test]
fn a_fused_norm_opens_with_the_residual_it_reads() {
    let runtime = runtime();
    let held: Vec<f32> = (0..12).map(|at| at as f32 * 0.25 - 0.5).collect();
    let build = |joined: bool| {
        let graph = Graph::new();
        let a = graph.input(Shape::of([1, 1, 4, 3]), Element::Single);
        let b = graph.input(Shape::of([1, 1, 4, 3]), Element::Single);
        let weight = graph.parameter(
            Shape::vector(3),
            Init::Uniform {
                low: 0.5,
                high: 1.5,
            },
            Element::Single,
        );
        let floor = graph.fill(Shape::scalar(), 1e-5);
        let sum = graph.add(a, b);
        if joined {
            graph.retain(sum);
        }
        let out = graph.rms_norm(sum, weight, floor);
        graph.retain(out);
        (graph, a, b, out)
    };
    let (fused, a, b, out) = build(false);
    let (formed, a_ref, b_ref, out_ref) = build(true);
    let weights = runtime.weights(&fused);
    let program = runtime.compile(&fused, &weights);
    let reference_weights = runtime.weights(&formed);
    let reference_program = runtime.compile(&formed, &reference_weights);
    runtime.write(&program, a, &held);
    runtime.write(&program, b, &held);
    runtime.write(&reference_program, a_ref, &held);
    runtime.write(&reference_program, b_ref, &held);
    runtime.run(&program);
    runtime.run(&reference_program);
    runtime.context().drain();
    assert!(
        program.task_count() < reference_program.task_count(),
        "a fused norm holds {} tasks where the formed residual holds {}",
        program.task_count(),
        reference_program.task_count(),
    );
    compared(&runtime, &program, out, &reference_program, out_ref);
}

#[test]
fn a_sectioned_layer_norm_folds_and_applies_the_band_it_walks() {
    let runtime = runtime();
    let held: Vec<f32> = (0..12288)
        .map(|at| (at as f32 * 0.017).sin() + 0.3)
        .collect();
    let graph = Graph::new();
    let source = graph.gradient_input(Shape::of([1, 1, 3, 4096]), Element::Single);
    let weight = graph.parameter(
        Shape::vector(4096),
        Init::Uniform {
            low: 0.5,
            high: 1.5,
        },
        Element::Single,
    );
    let shift = graph.parameter(
        Shape::vector(4096),
        Init::Uniform {
            low: -0.5,
            high: 0.5,
        },
        Element::Single,
    );
    let floor = graph.fill(Shape::scalar(), 1e-5);
    let loss = graph.sum(graph.layer_norm(source, weight, shift, floor));
    let gradients = graph.backward(loss);
    let source_gradient = gradients.of(source);
    let weight_gradient = gradients.of(weight);
    let shift_gradient = gradients.of(shift);
    graph.retain(source_gradient);
    graph.retain(weight_gradient);
    graph.retain(shift_gradient);

    let reference = Graph::new();
    let source_ref = reference.gradient_input(Shape::of([1, 1, 3, 4096]), Element::Single);
    let weight_ref = reference.parameter(
        Shape::vector(4096),
        Init::Uniform {
            low: 0.5,
            high: 1.5,
        },
        Element::Single,
    );
    let shift_ref = reference.parameter(
        Shape::vector(4096),
        Init::Uniform {
            low: -0.5,
            high: 0.5,
        },
        Element::Single,
    );
    let floor_ref = reference.fill(Shape::scalar(), 1e-5);
    let loss_ref = reference.sum(composed_layer(
        &reference, source_ref, weight_ref, shift_ref, floor_ref,
    ));
    let gradients_ref = reference.backward(loss_ref);
    let source_gradient_ref = gradients_ref.of(source_ref);
    let weight_gradient_ref = gradients_ref.of(weight_ref);
    let shift_gradient_ref = gradients_ref.of(shift_ref);
    reference.retain(source_gradient_ref);
    reference.retain(weight_gradient_ref);
    reference.retain(shift_gradient_ref);

    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let reference_weights = runtime.weights(&reference);
    let reference_program = runtime.compile(&reference, &reference_weights);
    runtime.write(&program, source, &held);
    runtime.write(&reference_program, source_ref, &held);
    runtime.run(&program);
    runtime.run(&reference_program);
    runtime.context().drain();
    compared(
        &runtime,
        &program,
        source_gradient,
        &reference_program,
        source_gradient_ref,
    );
    compared(
        &runtime,
        &program,
        weight_gradient,
        &reference_program,
        weight_gradient_ref,
    );
    compared(
        &runtime,
        &program,
        shift_gradient,
        &reference_program,
        shift_gradient_ref,
    );
}

#[test]
fn a_sectioned_group_norm_folds_the_group_it_centers() {
    let runtime = runtime();
    let held: Vec<f32> = (0..4096)
        .map(|at| (at as f32 * 0.071).cos() - 0.2)
        .collect();
    let graph = Graph::new();
    let source = graph.input(Shape::of([1, 4, 8, 128]), Element::Single);
    let weight = graph.parameter(
        Shape::of([1, 4, 1, 1]),
        Init::Uniform {
            low: 0.5,
            high: 1.5,
        },
        Element::Single,
    );
    let shift = graph.parameter(
        Shape::of([1, 4, 1, 1]),
        Init::Uniform {
            low: -0.5,
            high: 0.5,
        },
        Element::Single,
    );
    let floor = graph.fill(Shape::scalar(), 1e-5);
    let fused = graph.group_norm(source, weight, shift, floor, 2);
    graph.retain(fused);

    let reference = Graph::new();
    let source_ref = reference.input(Shape::of([1, 4, 8, 128]), Element::Single);
    let weight_ref = reference.parameter(
        Shape::of([1, 4, 1, 1]),
        Init::Uniform {
            low: 0.5,
            high: 1.5,
        },
        Element::Single,
    );
    let shift_ref = reference.parameter(
        Shape::of([1, 4, 1, 1]),
        Init::Uniform {
            low: -0.5,
            high: 0.5,
        },
        Element::Single,
    );
    let floor_ref = reference.fill(Shape::scalar(), 1e-5);
    let composed = composed_group(&reference, source_ref, weight_ref, shift_ref, floor_ref, 2);
    reference.retain(composed);

    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let reference_weights = runtime.weights(&reference);
    let reference_program = runtime.compile(&reference, &reference_weights);
    runtime.write(&program, source, &held);
    runtime.write(&reference_program, source_ref, &held);
    runtime.run(&program);
    runtime.run(&reference_program);
    runtime.context().drain();
    compared(&runtime, &program, fused, &reference_program, composed);
}

#[test]
fn a_sectioned_rms_norm_opens_with_the_residual_it_reads() {
    let runtime = runtime();
    let held: Vec<f32> = (0..8192)
        .map(|at| (at as f32 * 0.019).sin() * 1.4 - 0.3)
        .collect();
    let build = |joined: bool| {
        let graph = Graph::new();
        let a = graph.input(Shape::of([1, 1, 2, 4096]), Element::Single);
        let b = graph.input(Shape::of([1, 1, 2, 4096]), Element::Single);
        let weight = graph.parameter(
            Shape::vector(4096),
            Init::Uniform {
                low: 0.5,
                high: 1.5,
            },
            Element::Single,
        );
        let floor = graph.fill(Shape::scalar(), 1e-5);
        let sum = graph.add(a, b);
        if joined {
            graph.retain(sum);
        }
        let out = graph.rms_norm(sum, weight, floor);
        graph.retain(out);
        (graph, a, b, out)
    };
    let (fused, a, b, out) = build(false);
    let (formed, a_ref, b_ref, out_ref) = build(true);
    let weights = runtime.weights(&fused);
    let program = runtime.compile(&fused, &weights);
    let reference_weights = runtime.weights(&formed);
    let reference_program = runtime.compile(&formed, &reference_weights);
    runtime.write(&program, a, &held);
    runtime.write(&program, b, &held);
    runtime.write(&reference_program, a_ref, &held);
    runtime.write(&reference_program, b_ref, &held);
    runtime.run(&program);
    runtime.run(&reference_program);
    runtime.context().drain();
    assert!(
        program.task_count() < reference_program.task_count(),
        "a fused norm holds {} tasks where the formed residual holds {}",
        program.task_count(),
        reference_program.task_count(),
    );
    compared(&runtime, &program, out, &reference_program, out_ref);
}

#[test]
fn a_sectioned_rms_norm_descends_the_rows_it_scales() {
    let runtime = runtime();
    let held: Vec<f32> = (0..8192)
        .map(|at| (at as f32 * 0.013).sin() * 1.2 + 0.1)
        .collect();
    let graph = Graph::new();
    let source = graph.gradient_input(Shape::of([1, 1, 2, 4096]), Element::Single);
    let weight = graph.parameter(
        Shape::vector(4096),
        Init::Uniform {
            low: 0.5,
            high: 1.5,
        },
        Element::Single,
    );
    let floor = graph.fill(Shape::scalar(), 1e-5);
    let loss = graph.sum(graph.rms_norm(source, weight, floor));
    let gradients = graph.backward(loss);
    let source_gradient = gradients.of(source);
    let weight_gradient = gradients.of(weight);
    graph.retain(source_gradient);
    graph.retain(weight_gradient);

    let reference = Graph::new();
    let source_ref = reference.gradient_input(Shape::of([1, 1, 2, 4096]), Element::Single);
    let weight_ref = reference.parameter(
        Shape::vector(4096),
        Init::Uniform {
            low: 0.5,
            high: 1.5,
        },
        Element::Single,
    );
    let floor_ref = reference.fill(Shape::scalar(), 1e-5);
    let loss_ref = reference.sum(composed_rms(&reference, source_ref, weight_ref, floor_ref));
    let gradients_ref = reference.backward(loss_ref);
    let source_gradient_ref = gradients_ref.of(source_ref);
    let weight_gradient_ref = gradients_ref.of(weight_ref);
    reference.retain(source_gradient_ref);
    reference.retain(weight_gradient_ref);

    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let reference_weights = runtime.weights(&reference);
    let reference_program = runtime.compile(&reference, &reference_weights);
    runtime.write(&program, source, &held);
    runtime.write(&reference_program, source_ref, &held);
    runtime.run(&program);
    runtime.run(&reference_program);
    runtime.context().drain();
    compared(
        &runtime,
        &program,
        source_gradient,
        &reference_program,
        source_gradient_ref,
    );
    compared(
        &runtime,
        &program,
        weight_gradient,
        &reference_program,
        weight_gradient_ref,
    );
}

#[test]
fn a_norm_gradient_leaves_the_path_another_task_reads() {
    let runtime = runtime();
    let held: Vec<f32> = (0..12).map(|at| at as f32 * 0.25 - 0.5).collect();
    let build = |reference: bool| {
        let graph = Graph::new();
        let x = graph.gradient_input(Shape::of([1, 1, 4, 3]), Element::Single);
        let source = graph.relu(x);
        let weight = graph.parameter(
            Shape::vector(3),
            Init::Uniform {
                low: 0.5,
                high: 1.5,
            },
            Element::Single,
        );
        let shift = graph.parameter(
            Shape::vector(3),
            Init::Uniform {
                low: -0.5,
                high: 0.5,
            },
            Element::Single,
        );
        let floor = graph.fill(Shape::scalar(), 1e-5);
        let normed = if reference {
            composed_layer(&graph, source, weight, shift, floor)
        } else {
            graph.layer_norm(source, weight, shift, floor)
        };
        let loss = graph.add(graph.sum(normed), graph.sum(source));
        let gradients = graph.backward(loss);
        let dx = gradients.of(x);
        let dw = gradients.of(weight);
        let db = gradients.of(shift);
        graph.retain(dx);
        graph.retain(dw);
        graph.retain(db);
        (graph, x, dx, dw, db)
    };
    let (fused, x, dx, dw, db) = build(false);
    let (reference, x_ref, dx_ref, dw_ref, db_ref) = build(true);
    let weights = runtime.weights(&fused);
    let program = runtime.compile(&fused, &weights);
    let reference_weights = runtime.weights(&reference);
    let reference_program = runtime.compile(&reference, &reference_weights);
    runtime.write(&program, x, &held);
    runtime.write(&reference_program, x_ref, &held);
    runtime.run(&program);
    runtime.run(&reference_program);
    runtime.context().drain();
    compared(&runtime, &program, dx, &reference_program, dx_ref);
    compared(&runtime, &program, dw, &reference_program, dw_ref);
    compared(&runtime, &program, db, &reference_program, db_ref);
}
