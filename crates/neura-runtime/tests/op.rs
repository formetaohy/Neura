use neura_abi::Element;
use neura_graph::{Graph, Init, Shape, Value};
use neura_op::{self as op, OPS};

#[path = "support/mod.rs"]
mod support;

use support::{assert_close, open};

struct Probe {
    op: u32,
    build: for<'g> fn(&'g Graph<'g>, Value<'g>, Value<'g>) -> Value<'g>,
    apply: fn(f32, f32) -> f32,
    partial: fn(f32, f32) -> (f32, f32),
}

const ELEMENTS: u32 = 4;

fn observations() -> Vec<f32> {
    (0..ELEMENTS)
        .map(|index| 0.5 + 0.25 * index as f32)
        .collect()
}

fn others() -> Vec<f32> {
    (0..ELEMENTS)
        .map(|index| 1.5 - 0.25 * index as f32)
        .collect()
}

fn weights() -> Vec<f32> {
    (0..ELEMENTS)
        .map(|index| -1.0 + 0.5 * index as f32)
        .collect()
}

fn probes() -> Vec<Probe> {
    vec![
        Probe {
            op: op::ADD,
            build: |graph, left, right| graph.add(left, right),
            apply: |a, b| a + b,
            partial: |_a, _b| (1.0, 1.0),
        },
        Probe {
            op: op::MUL,
            build: |graph, left, right| graph.mul(left, right),
            apply: |a, b| a * b,
            partial: |a, b| (b, a),
        },
        Probe {
            op: op::SUB,
            build: |graph, left, right| graph.sub(left, right),
            apply: |a, b| a - b,
            partial: |_a, _b| (1.0, -1.0),
        },
        Probe {
            op: op::DIV,
            build: |graph, left, right| graph.div(left, right),
            apply: |a, b| a / b,
            partial: |a, b| (1.0 / b, -a / (b * b)),
        },
        Probe {
            op: op::MAXIMUM,
            build: |graph, left, right| graph.max(left, right),
            apply: |a, b| a.max(b),
            partial: |a, b| (f32::from(a > b), f32::from(b > a)),
        },
        Probe {
            op: op::MINIMUM,
            build: |graph, left, right| graph.min(left, right),
            apply: |a, b| a.min(b),
            partial: |a, b| (f32::from(a < b), f32::from(b < a)),
        },
        Probe {
            op: op::RELU,
            build: |graph, left, _right| graph.relu(left),
            apply: |a, _b| a.max(0.0),
            partial: |a, _b| (f32::from(a > 0.0), 0.0),
        },
        Probe {
            op: op::SQRT,
            build: |graph, left, _right| graph.sqrt(left),
            apply: |a, _b| a.sqrt(),
            partial: |a, _b| (0.5 / a.sqrt(), 0.0),
        },
        Probe {
            op: op::RECIP,
            build: |graph, left, _right| graph.recip(left),
            apply: |a, _b| 1.0 / a,
            partial: |a, _b| (-1.0 / (a * a), 0.0),
        },
        Probe {
            op: op::EXP,
            build: |graph, left, _right| graph.exp(left),
            apply: |a, _b| a.exp(),
            partial: |a, _b| (a.exp(), 0.0),
        },
        Probe {
            op: op::LOG,
            build: |graph, left, _right| graph.log(left),
            apply: |a, _b| a.ln(),
            partial: |a, _b| (1.0 / a, 0.0),
        },
        Probe {
            op: op::TANH,
            build: |graph, left, _right| graph.tanh(left),
            apply: |a, _b| a.tanh(),
            partial: |a, _b| (1.0 - a.tanh() * a.tanh(), 0.0),
        },
        Probe {
            op: op::SIGMOID,
            build: |graph, left, _right| graph.sigmoid(left),
            apply: |a, _b| 1.0 / (1.0 + (-a).exp()),
            partial: |a, _b| {
                let s = 1.0 / (1.0 + (-a).exp());
                (s * (1.0 - s), 0.0)
            },
        },
        Probe {
            op: op::NEG,
            build: |graph, left, _right| graph.neg(left),
            apply: |a, _b| -a,
            partial: |_a, _b| (-1.0, 0.0),
        },
        Probe {
            op: op::ABS,
            build: |graph, left, _right| graph.abs(left),
            apply: |a, _b| a.abs(),
            partial: |a, _b| (if a > 0.0 { 1.0 } else { -1.0 }, 0.0),
        },
        Probe {
            op: op::IDENTITY,
            build: |graph, left, _right| graph.identity(left),
            apply: |a, _b| a,
            partial: |_a, _b| (1.0, 0.0),
        },
        Probe {
            op: op::SIN,
            build: |graph, left, _right| graph.sin(left),
            apply: |a, _b| a.sin(),
            partial: |a, _b| (a.cos(), 0.0),
        },
        Probe {
            op: op::COS,
            build: |graph, left, _right| graph.cos(left),
            apply: |a, _b| a.cos(),
            partial: |a, _b| (-a.sin(), 0.0),
        },
        Probe {
            op: op::POW,
            build: |graph, left, right| graph.pow(left, right),
            apply: |a, b| a.powf(b),
            partial: |a, b| (b * a.powf(b - 1.0), a.powf(b) * a.ln()),
        },
        Probe {
            op: op::FLOOR,
            build: |graph, left, _right| graph.floor(left),
            apply: |a, _b| a.floor(),
            partial: |_a, _b| (0.0, 0.0),
        },
        Probe {
            op: op::GELU,
            build: |graph, left, _right| graph.gelu(left),
            apply: |a, _b| 0.5 * a * (1.0 + (0.797_884_6 * (a + 0.044715 * a * a * a)).tanh()),
            partial: |a, _b| {
                let slope = (0.797_884_6 * (a + 0.044715 * a * a * a)).tanh();
                (
                    0.5 * (1.0 + slope)
                        + 0.5
                            * a
                            * (1.0 - slope * slope)
                            * 0.797_884_6
                            * (1.0 + 3.0 * 0.044715 * a * a),
                    0.0,
                )
            },
        },
        Probe {
            op: op::SILU,
            build: |graph, left, _right| graph.silu(left),
            apply: |a, _b| {
                let s = 1.0 / (1.0 + (-a).exp());
                a * s
            },
            partial: |a, _b| {
                let s = 1.0 / (1.0 + (-a).exp());
                (s * (1.0 + a * (1.0 - s)), 0.0)
            },
        },
    ]
}

#[test]
fn every_declared_op_runs_and_differentiates_on_the_device() {
    let runtime = open();
    let table = probes();
    assert_eq!(table.len(), OPS.len(), "every declared op carries a probe",);
    for definition in OPS {
        let reached = table
            .iter()
            .filter(|probe| probe.op == definition.code)
            .count();
        assert_eq!(reached, 1, "the {} op carries one probe", definition.name);
    }
    for probe in &table {
        let definition = op::of(probe.op);
        let graph = Graph::new();
        let left = graph.parameter(Shape::vector(ELEMENTS), Init::Zero, Element::Single);
        let right = graph.parameter(Shape::vector(ELEMENTS), Init::Zero, Element::Single);
        let out = (probe.build)(&graph, left, right);
        graph.retain(out);
        let store = runtime.weights(&graph);
        let program = runtime.compile(&graph, &store);
        let (first, second) = (observations(), others());
        runtime.write(&program, left, &first);
        runtime.write(&program, right, &second);
        runtime.run(&program);
        let expected = first
            .iter()
            .zip(&second)
            .map(|(a, b)| (probe.apply)(*a, *b))
            .collect::<Vec<_>>();
        assert_close(&runtime.read(&program, out), &expected, 1e-4);

        let gradient = Graph::new();
        let left = gradient.parameter(Shape::vector(ELEMENTS), Init::Zero, Element::Single);
        let right = gradient.parameter(Shape::vector(ELEMENTS), Init::Zero, Element::Single);
        let scales = gradient.parameter(Shape::vector(ELEMENTS), Init::Zero, Element::Single);
        let out = (probe.build)(&gradient, left, right);
        let loss = gradient.sum(gradient.mul(out, scales));
        let gradients = gradient.backward(loss);
        gradient.retain(gradients.of(left));
        let binary = definition.family == op::Family::Binary;
        if binary {
            gradient.retain(gradients.of(right));
        }
        let store = runtime.weights(&gradient);
        let program = runtime.compile(&gradient, &store);
        let scale = weights();
        runtime.write(&program, left, &first);
        runtime.write(&program, right, &second);
        runtime.write(&program, scales, &scale);
        runtime.run(&program);
        let (left_slope, right_slope) = first
            .iter()
            .zip(&second)
            .zip(&scale)
            .map(|((a, b), weight)| {
                let (over_left, over_right) = (probe.partial)(*a, *b);
                (over_left * weight, over_right * weight)
            })
            .unzip::<f32, f32, Vec<f32>, Vec<f32>>();
        assert_close(
            &runtime.read(&program, gradients.of(left)),
            &left_slope,
            1e-4,
        );
        if binary {
            assert_close(
                &runtime.read(&program, gradients.of(right)),
                &right_slope,
                1e-4,
            );
        }
    }
}
