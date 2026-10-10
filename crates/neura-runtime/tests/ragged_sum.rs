use neura_abi::Element;
use neura_gpu::{Backend, PREFERENCE};
use neura_graph::{Graph, Shape, Value};
use neura_runtime::{Program, Runtime};

#[path = "support/backend.rs"]
mod backend;

#[path = "support/mod.rs"]
mod support;

use support::{assert_close, open};

const WIDTH: u32 = 4;
const PLANES: u32 = 4;
const BOUND: u32 = 16;
const TALL: u32 = 1024;

fn refuses(action: impl FnOnce()) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(action)).is_err()
}

fn data(count: u32, seed: u32) -> Vec<f32> {
    let mut entropy = seed | 1;
    (0..count)
        .map(|_| {
            entropy ^= entropy << 13;
            entropy ^= entropy >> 17;
            entropy ^= entropy << 5;
            (entropy >> 8) as f32 / 16_777_216.0 - 0.5
        })
        .collect()
}

fn offsets(lengths: &[f32]) -> Vec<f32> {
    let mut offsets = Vec::with_capacity(lengths.len() + 1);
    let mut running = 0.0f32;
    offsets.push(running);
    for length in lengths {
        running += length;
        offsets.push(running);
    }
    offsets
}

fn expected(lengths: &[f32], cache: &[f32], width: u32) -> Vec<f32> {
    let offsets = offsets(lengths);
    let mut out = vec![0.0f32; lengths.len() * width as usize];
    for plane in 0..lengths.len() {
        for row in offsets[plane] as usize..offsets[plane + 1] as usize {
            for column in 0..width as usize {
                out[plane * width as usize + column] += cache[row * width as usize + column];
            }
        }
    }
    out
}

struct Sums {
    runtime: Runtime,
    graph: Graph<'static>,
    lengths: Value<'static>,
    cache: Value<'static>,
    offsets: Value<'static>,
    total: Value<'static>,
    planes: u32,
    bound: u32,
    width: u32,
}

impl Sums {
    fn of(backend: Backend, planes: u32, bound: u32, width: u32) -> Self {
        Self::over(backend::open_with(backend), planes, bound, width)
    }

    fn over(runtime: Runtime, planes: u32, bound: u32, width: u32) -> Self {
        let graph: Graph<'static> = Graph::new();
        let lengths = graph.input(Shape::vector(planes), Element::Single);
        let ragged = graph.ragged(bound, lengths);
        let cache = graph.resident(
            Shape::of([1, 1, bound, width]).freed(&[(2, ragged.extent)]),
            Element::Single,
        );
        let total = graph.segment_sum(cache, ragged);
        graph.retain(total);
        graph.retain(ragged.offsets);
        Self {
            runtime,
            graph,
            lengths,
            cache,
            offsets: ragged.offsets,
            total,
            planes,
            bound,
            width,
        }
    }

    fn compile(&self) -> Program {
        let weights = self.runtime.weights(&self.graph);
        self.runtime.compile(&self.graph, &weights)
    }

    fn step(&self, program: &Program, lengths: &[f32]) -> Vec<f32> {
        assert_eq!(lengths.len(), self.planes as usize);
        let cache = data(self.bound * self.width, 17);
        self.runtime.write(program, self.lengths, lengths);
        self.runtime.write(program, self.cache, &cache);
        self.runtime.run(program);
        let produced = self.runtime.read(program, self.total);
        assert_close(
            &self.runtime.read(program, self.offsets),
            &offsets(lengths),
            0.0,
        );
        assert_close(&produced, &expected(lengths, &cache, self.width), 1e-4);
        produced
    }
}

#[test]
fn a_plane_sums_the_rows_a_ragged_axis_closes() {
    for &backend in PREFERENCE {
        for lengths in [
            [3.0, 0.0, 5.0, 2.0],
            [16.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ] {
            let sums = Sums::of(backend, PLANES, BOUND, WIDTH);
            let program = sums.compile();
            sums.step(&program, &lengths);
        }
    }
}

#[test]
fn a_plane_sums_the_rows_of_a_long_axis_in_chunks() {
    let sums = Sums::over(open(), PLANES, TALL, 1);
    let program = sums.compile();
    sums.step(&program, &[600.0, 1.0, 0.0, 400.0]);
}

#[test]
fn a_plane_sums_a_length_a_device_counts() {
    for &backend in PREFERENCE {
        let graph: Graph<'static> = Graph::new();
        let mask = graph.input(Shape::of([PLANES, 1, BOUND, 1]), Element::Single);
        let counts = graph.sum_axis(mask, 2);
        let ragged = graph.ragged(BOUND, counts);
        let cache = graph.resident(
            Shape::of([1, 1, BOUND, WIDTH]).freed(&[(2, ragged.extent)]),
            Element::Single,
        );
        let total = graph.segment_sum(cache, ragged);
        graph.retain(total);
        let runtime = backend::open_with(backend);
        let weights = runtime.weights(&graph);
        let program = runtime.compile(&graph, &weights);
        let lengths = [3.0f32, 0.0, 5.0, 2.0];
        let spans = offsets(&lengths);
        let mut flags = vec![0.0f32; (PLANES * BOUND) as usize];
        for plane in 0..PLANES as usize {
            for row in spans[plane] as usize..spans[plane + 1] as usize {
                flags[plane * BOUND as usize + row] = 1.0;
            }
        }
        let image = data(BOUND * WIDTH, 17);
        runtime.write(&program, mask, &flags);
        runtime.write(&program, cache, &image);
        runtime.run(&program);
        assert_close(
            &runtime.read(&program, total),
            &expected(&lengths, &image, WIDTH),
            1e-4,
        );
    }
}

#[test]
fn a_plane_sums_a_binding_that_narrows_the_plane_axis() {
    for &backend in PREFERENCE {
        let graph: Graph<'static> = Graph::new();
        let live = graph.free(PLANES);
        let lengths = graph.input(Shape::of([1, PLANES]).freed(&[(3, live)]), Element::Single);
        let ragged = graph.ragged(BOUND, lengths);
        let cache = graph.resident(
            Shape::of([1, 1, BOUND, WIDTH]).freed(&[(2, ragged.extent)]),
            Element::Single,
        );
        let total = graph.segment_sum(cache, ragged);
        graph.retain(total);
        let runtime = backend::open_with(backend);
        let weights = runtime.weights(&graph);
        let program = runtime.compile(&graph, &weights);
        let image = data(BOUND * WIDTH, 17);
        for (bound, widths) in [
            (PLANES, [4.0f32, 4.0, 4.0, 4.0]),
            (2, [4.0f32, 4.0, 0.0, 0.0]),
        ] {
            runtime.bind(&program, &[bound]);
            runtime.write(&program, lengths, &widths[..bound as usize]);
            runtime.write(&program, cache, &image);
            runtime.run(&program);
            assert_close(
                &runtime.read(&program, total),
                &expected(&widths[..bound as usize], &image, WIDTH),
                1e-4,
            );
        }
    }
}

#[test]
fn a_plane_sums_a_batch_the_lengths_lay_out_by_head() {
    for &backend in PREFERENCE {
        for broad in [true, false] {
            let heads = 2u32;
            let batch = 3u32;
            let graph: Graph<'static> = Graph::new();
            let live = graph.free(batch);
            let lengths = graph.input(
                Shape::of([heads, batch, 1, 1]).freed(&[(1, live)]),
                Element::Single,
            );
            let ragged = graph.ragged(BOUND, lengths);
            let cache = graph.resident(
                Shape::of([1, 1, BOUND, WIDTH]).freed(&[(2, ragged.extent)]),
                Element::Single,
            );
            let total = graph.segment_sum(cache, ragged);
            graph.retain(total);
            let runtime = backend::open_with(backend);
            let weights = runtime.weights(&graph);
            let program = runtime.compile(&graph, &weights);
            let planes = if broad { batch } else { batch - 1 };
            let spans = [2.0f32, 3.0, 1.0, 2.0, 0.0, 1.0];
            let image = data(BOUND * WIDTH, 17);
            runtime.bind(&program, &[planes]);
            runtime.write(&program, lengths, &spans[..(heads * planes) as usize]);
            runtime.write(&program, cache, &image);
            runtime.run(&program);
            assert_close(
                &runtime.read(&program, total),
                &expected(&spans[..(heads * planes) as usize], &image, WIDTH),
                1e-4,
            );
        }
    }
}

#[test]
fn a_per_plane_sum_trains_the_rows_that_reach_it() {
    for &backend in PREFERENCE {
        let graph: Graph<'static> = Graph::new();
        let lengths = graph.input(Shape::vector(PLANES), Element::Single);
        let ragged = graph.ragged(BOUND, lengths);
        let cache = graph.gradient_input(
            Shape::of([1, 1, BOUND, WIDTH]).freed(&[(2, ragged.extent)]),
            Element::Single,
        );
        let total = graph.segment_sum(cache, ragged);
        let weight = graph.input(Shape::of([1, PLANES, 1, WIDTH]), Element::Single);
        let loss = graph.sum(graph.mul(total, weight));
        let collected = graph.backward(loss);
        let gradient = collected.of(cache);
        graph.retain(gradient);
        graph.retain(total);
        let runtime = backend::open_with(backend);
        let weights = runtime.weights(&graph);
        let program = runtime.compile(&graph, &weights);
        let lengths_data = [3.0f32, 0.0, 5.0, 2.0];
        let spans = offsets(&lengths_data);
        let cache_data = data(BOUND * WIDTH, 17);
        let weight_data = data(PLANES * WIDTH, 29);
        runtime.write(&program, lengths, &lengths_data);
        runtime.write(&program, cache, &cache_data);
        runtime.write(&program, weight, &weight_data);
        runtime.run(&program);
        let mut expected = vec![0.0f32; *spans.last().expect("a plane") as usize * WIDTH as usize];
        for plane in 0..PLANES as usize {
            for row in spans[plane] as usize..spans[plane + 1] as usize {
                for column in 0..WIDTH as usize {
                    expected[row * WIDTH as usize + column] =
                        weight_data[plane * WIDTH as usize + column];
                }
            }
        }
        assert_close(&runtime.read(&program, gradient), &expected, 1e-5);
    }
}

#[test]
fn a_per_plane_sum_walks_the_planes_a_device_counts() {
    for &backend in PREFERENCE {
        let graph: Graph<'static> = Graph::new();
        let flags = graph.input(Shape::of([1, 1, 1, PLANES]), Element::Single);
        let count = graph.sum(flags);
        let live = graph.counted(PLANES, count);
        let lengths = graph.input(
            Shape::of([1, 1, 1, PLANES]).freed(&[(3, live)]),
            Element::Single,
        );
        let ragged = graph.ragged(BOUND, lengths);
        let cache = graph.resident(
            Shape::of([1, 1, BOUND, WIDTH]).freed(&[(2, ragged.extent)]),
            Element::Single,
        );
        let total = graph.segment_sum(cache, ragged);
        graph.retain(total);
        let runtime = backend::open_with(backend);
        let weights = runtime.weights(&graph);
        let program = runtime.compile(&graph, &weights);
        let lengths_data = [3.0f32, 0.0, 5.0, 1.0];
        let image = data(BOUND * WIDTH, 17);
        runtime.write(&program, lengths, &lengths_data);
        runtime.write(&program, cache, &image);
        for (plane_flags, planes) in [
            ([1.0f32, 1.0, 1.0, 1.0], PLANES),
            ([1.0, 1.0, 0.0, 0.0], 2),
            ([0.0, 0.0, 0.0, 0.0], 0),
        ] {
            runtime.write(&program, flags, &plane_flags);
            runtime.run(&program);
            assert_close(
                &runtime.read(&program, total),
                &expected(&lengths_data[..planes as usize], &image, WIDTH),
                1e-4,
            );
        }
    }
}

#[test]
fn a_per_plane_sum_folds_the_packs_it_fused() {
    for &backend in PREFERENCE {
        let graph: Graph<'static> = Graph::new();
        let lengths = graph.input(Shape::vector(PLANES), Element::Single);
        let ragged = graph.ragged(BOUND, lengths);
        let cache = graph.gradient_input(
            Shape::of([1, 1, BOUND, WIDTH]).freed(&[(2, ragged.extent)]),
            Element::Single,
        );
        let factor = graph.gradient_input(Shape::scalar(), Element::Single);
        let scales = graph.mul(cache, factor);
        let total = graph.segment_sum(scales, ragged);
        let loss = graph.sum(graph.mul(total, factor));
        let collected = graph.backward(loss);
        let gradient = collected.of(cache);
        graph.retain(total);
        graph.retain(gradient);
        let runtime = backend::open_with(backend);
        let weights = runtime.weights(&graph);
        let program = runtime.compile(&graph, &weights);
        let lengths_data = [3.0f32, 0.0, 5.0, 2.0];
        let scale = [-1.5f32];
        let image = data(BOUND * WIDTH, 17);
        runtime.write(&program, lengths, &lengths_data);
        runtime.write(&program, cache, &image);
        runtime.write(&program, factor, &scale);
        runtime.run(&program);
        let scaled = image
            .iter()
            .map(|value| value * scale[0])
            .collect::<Vec<f32>>();
        assert_close(
            &runtime.read(&program, total),
            &expected(&lengths_data, &scaled, WIDTH),
            1e-4,
        );
        let spans = offsets(&lengths_data);
        let mut expected_gradient =
            vec![0.0f32; *spans.last().expect("a plane") as usize * WIDTH as usize];
        for plane in 0..PLANES as usize {
            for row in spans[plane] as usize..spans[plane + 1] as usize {
                for column in 0..WIDTH as usize {
                    expected_gradient[row * WIDTH as usize + column] = scale[0] * scale[0];
                }
            }
        }
        assert_close(&runtime.read(&program, gradient), &expected_gradient, 1e-4);
    }
}

#[test]
fn a_per_plane_sum_carries_the_width_a_binding_rules() {
    for &backend in PREFERENCE {
        let graph: Graph<'static> = Graph::new();
        let widths = graph.free(WIDTH);
        let lengths = graph.input(Shape::vector(PLANES), Element::Single);
        let ragged = graph.ragged(BOUND, lengths);
        let cache = graph.resident(
            Shape::of([1, 1, BOUND, WIDTH]).freed(&[(2, ragged.extent), (3, widths)]),
            Element::Single,
        );
        let total = graph.segment_sum(cache, ragged);
        graph.retain(total);
        let runtime = backend::open_with(backend);
        let weights = runtime.weights(&graph);
        let program = runtime.compile(&graph, &weights);
        let lengths_data = [3.0f32, 0.0, 5.0, 2.0];
        for width in [WIDTH, 2, 1] {
            runtime.bind(&program, &[width]);
            let image = data(BOUND * width, 17);
            runtime.write(&program, lengths, &lengths_data);
            runtime.write(&program, cache, &image);
            runtime.run(&program);
            assert_close(
                &runtime.read(&program, total),
                &expected(&lengths_data, &image, width),
                1e-4,
            );
        }
    }
}

#[test]
fn a_fold_over_a_ragged_axis_refuses_to_mix_its_planes() {
    let graph: Graph<'static> = Graph::new();
    let lengths = graph.input(Shape::vector(PLANES), Element::Single);
    let ragged = graph.ragged(BOUND, lengths);
    let cache = graph.resident(
        Shape::of([1, 1, BOUND, WIDTH]).freed(&[(2, ragged.extent)]),
        Element::Single,
    );
    assert!(refuses(|| {
        graph.sum_axis(cache, 2);
    }));
}

#[test]
fn a_per_plane_sum_names_the_axis_it_walks() {
    let graph: Graph<'static> = Graph::new();
    let lengths = graph.input(Shape::vector(PLANES), Element::Single);
    let ragged = graph.ragged(BOUND, lengths);
    let bare = graph.input(Shape::matrix(PLANES + 1, 1), Element::Single);
    let cache = graph.resident(
        Shape::of([1, 1, BOUND, WIDTH]).freed(&[(2, ragged.extent)]),
        Element::Single,
    );
    assert!(refuses(|| {
        graph.segment_sum(
            cache,
            neura_graph::Ragged {
                extent: ragged.extent,
                offsets: bare,
            },
        );
    }));
    let plain = graph.resident(Shape::of([1, 1, BOUND, WIDTH]), Element::Single);
    assert!(refuses(|| {
        graph.segment_sum(plain, ragged);
    }));
}
