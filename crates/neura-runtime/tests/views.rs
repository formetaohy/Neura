use neura_program::{Graph, Init, Shape, Value};
use neura_runtime::Precision;

#[path = "support/reference.rs"]
mod reference;
#[path = "support/mod.rs"]
mod support;

use reference::{matmul_reference, random};
use support::{assert_close, open};

fn softmax_rows(values: &[f32], columns: u32) -> Vec<f32> {
    let mut out = values.to_vec();
    for (index, row) in out.chunks_mut(columns as usize).enumerate() {
        let _ = index;
        let largest = row.iter().copied().fold(f32::MIN, f32::max);
        let total: f32 = row.iter().map(|x| (x - largest).exp()).sum();
        for value in row.iter_mut() {
            *value = (*value - largest).exp() / total;
        }
    }
    out
}

fn weight<'g>(graph: &Graph<'g>, rows: u32, columns: u32) -> Value<'g> {
    graph.parameter(
        Shape::matrix(rows, columns),
        Init::Uniform {
            low: -0.5,
            high: 0.5,
        },
    )
}

#[test]
fn a_reshape_feeds_one_product_from_a_planned_tensor() {
    let runtime = open();
    let graph = Graph::new();
    let input = graph.input(Shape::of([2, 3, 8]));
    let reshaped = graph.reshape(input, [6, 8]);
    let dense = weight(&graph, 8, 4);
    let out = graph.matmul(reshaped, dense);
    graph.retain(out);
    let weights = runtime.weights(&graph, Precision::Single);
    let program = runtime.compile(&graph, &weights);
    let data = random(2 * 3 * 8, 11);
    runtime.write(&program, input, &data);
    runtime.run(&program);
    let dense_data = runtime.read(&program, dense);
    let expected = matmul_reference(&data, &dense_data, 6, 8, 4);
    assert_close(&runtime.read(&program, out), &expected, 1e-5);
}

#[test]
fn a_swap_axes_normalizes_rows_the_its_layout_holds() {
    let runtime = open();
    let graph = Graph::new();
    let input = graph.input(Shape::of([2, 2, 3, 4]));
    let swapped = graph.swap_axes(input, 1, 2);
    let out = graph.softmax(swapped);
    graph.retain(out);
    let weights = runtime.weights(&graph, Precision::Single);
    let program = runtime.compile(&graph, &weights);
    let data = random(2 * 2 * 3 * 4, 13);
    runtime.write(&program, input, &data);
    runtime.run(&program);
    let rows = 2 * 3 * 2;
    let mut row_major = vec![0.0f32; rows as usize * 4];
    for x in 0..2usize {
        for y in 0..2usize {
            for z in 0..3usize {
                for w in 0..4usize {
                    let source = ((x * 2 + y) * 3 + z) * 4 + w;
                    let target = ((x * 3 + z) * 2 + y) * 4 + w;
                    row_major[target] = data[source];
                }
            }
        }
    }
    let expected = softmax_rows(&row_major, 4);
    assert_close(&runtime.read(&program, out), &expected, 1e-6);
}

#[test]
fn a_slice_walks_the_rows_it_names() {
    let runtime = open();
    let graph = Graph::new();
    let input = graph.input(Shape::matrix(5, 8));
    let region = graph.slice(input, 2, 1, 3);
    let dense = weight(&graph, 8, 3);
    let out = graph.matmul(region, dense);
    graph.retain(out);
    let weights = runtime.weights(&graph, Precision::Single);
    let program = runtime.compile(&graph, &weights);
    let data = random(5 * 8, 17);
    runtime.write(&program, input, &data);
    runtime.run(&program);
    let dense_data = runtime.read(&program, dense);
    let expected = matmul_reference(&data[8..8 + 3 * 8], &dense_data, 3, 8, 3);
    assert_close(&runtime.read(&program, out), &expected, 1e-5);
}

#[test]
fn a_slice_of_columns_normalizes_the_region_it_names() {
    let runtime = open();
    let graph = Graph::new();
    let input = graph.input(Shape::matrix(4, 8));
    let region = graph.slice(input, 3, 2, 5);
    let out = graph.softmax(region);
    graph.retain(out);
    let weights = runtime.weights(&graph, Precision::Single);
    let program = runtime.compile(&graph, &weights);
    let data = random(4 * 8, 19);
    runtime.write(&program, input, &data);
    runtime.run(&program);
    let mut expected = vec![0.0f32; 4 * 5];
    for (row, values) in expected.chunks_mut(5).enumerate() {
        let source = &data[row * 8 + 2..row * 8 + 7];
        let largest = source.iter().copied().fold(f32::MIN, f32::max);
        let total: f32 = source.iter().map(|x| (x - largest).exp()).sum();
        for (column, value) in source.iter().enumerate() {
            values[column] = (value - largest).exp() / total;
        }
    }
    assert_close(&runtime.read(&program, out), &expected, 1e-6);
}

#[test]
fn a_concat_stitches_two_tensors_along_one_axis() {
    let runtime = open();
    let graph = Graph::new();
    let left = graph.input(Shape::of([2, 3, 4, 5]));
    let right = graph.input(Shape::of([2, 3, 6, 5]));
    let joined = graph.concat(2, left, right);
    let out = graph.sum(joined);
    graph.retain(joined);
    let weights = runtime.weights(&graph, Precision::Single);
    let program = runtime.compile(&graph, &weights);
    let left_data = random(2 * 3 * 4 * 5, 23);
    let right_data = random(2 * 3 * 6 * 5, 29);
    runtime.write(&program, left, &left_data);
    runtime.write(&program, right, &right_data);
    runtime.run(&program);
    let total: f32 = left_data.iter().chain(right_data.iter()).sum();
    assert_close(&runtime.read(&program, out), &[total], 1e-4);
    let joined_data = runtime.read(&program, joined);
    let mut expected = vec![0.0f32; (2 * 3 * 10 * 5) as usize];
    for x in 0..2usize {
        for y in 0..3usize {
            for z in 0..10usize {
                for w in 0..5usize {
                    let at = ((x * 3 + y) * 10 + z) * 5 + w;
                    expected[at] = if z < 4 {
                        left_data[((x * 3 + y) * 4 + z) * 5 + w]
                    } else {
                        right_data[((x * 3 + y) * 6 + z - 4) * 5 + w]
                    };
                }
            }
        }
    }
    assert_close(&joined_data, &expected, 1e-6);
}

#[test]
fn a_concat_hands_every_part_the_gradient_its_region_holds() {
    let runtime = open();
    let graph = Graph::new();
    let left = graph.parameter(Shape::matrix(6, 4), Init::Zero);
    let right = graph.parameter(Shape::matrix(3, 4), Init::Zero);
    let joined = graph.concat(2, left, right);
    let dense = weight(&graph, 4, 2);
    let out = graph.matmul(joined, dense);
    let loss = graph.sum(out);
    let gradients = graph.backward(loss);
    let weights = runtime.weights(&graph, Precision::Single);
    let program = runtime.compile(&graph, &weights);
    let left_data = random(6 * 4, 31);
    let right_data = random(3 * 4, 37);
    runtime.write(&program, left, &left_data);
    runtime.write(&program, right, &right_data);
    runtime.run(&program);
    let dense_data = runtime.read(&program, dense);
    let joined_data = [left_data, right_data].concat();
    let product = matmul_reference(&joined_data, &dense_data, 9, 4, 2);
    let upstream = vec![1.0f32; product.len()];
    let mut joined_gradient = vec![0.0f32; joined_data.len()];
    for row in 0..9usize {
        for column in 0..2usize {
            for depth in 0..4usize {
                joined_gradient[row * 4 + depth] +=
                    upstream[row * 2 + column] * dense_data[depth * 2 + column];
            }
        }
    }
    assert_close(
        &runtime.read(&program, gradients.of(left)),
        &joined_gradient[..6 * 4],
        1e-5,
    );
    assert_close(
        &runtime.read(&program, gradients.of(right)),
        &joined_gradient[6 * 4..],
        1e-5,
    );
}

#[test]
fn a_gradient_reaches_the_whole_parameter_one_of_whose_views_served() {
    let runtime = open();
    let graph = Graph::new();
    let dense = graph.parameter(
        Shape::matrix(8, 6),
        Init::Uniform {
            low: -0.5,
            high: 0.5,
        },
    );
    let region = graph.slice(dense, 3, 2, 4);
    let input = graph.input(Shape::matrix(5, 8));
    let out = graph.matmul(input, region);
    let loss = graph.sum(out);
    let gradients = graph.backward(loss);
    let weights = runtime.weights(&graph, Precision::Single);
    let program = runtime.compile(&graph, &weights);
    let input_data = random(5 * 8, 41);
    runtime.write(&program, input, &input_data);
    runtime.run(&program);
    let expected = [1.0f32; 5 * 4];
    let mut dense_gradient = vec![0.0f32; 8 * 6];
    for row in 0..5usize {
        for column in 0..4usize {
            for depth in 0..8usize {
                dense_gradient[depth * 6 + column + 2] +=
                    expected[row * 4 + column] * input_data[row * 8 + depth];
            }
        }
    }
    assert_close(
        &runtime.read(&program, gradients.of(dense)),
        &dense_gradient,
        1e-4,
    );
}

#[test]
fn two_slices_of_one_tensor_pile_their_gradients_onto_one_store() {
    let runtime = open();
    let graph = Graph::new();
    let input = graph.parameter(Shape::matrix(8, 6), Init::Zero);
    let head = graph.slice(input, 2, 0, 3);
    let tail = graph.slice(input, 2, 5, 3);
    let first = graph.sum(head);
    let second = graph.sum(tail);
    let loss = graph.add(first, second);
    let gradients = graph.backward(loss);
    let weights = runtime.weights(&graph, Precision::Single);
    let program = runtime.compile(&graph, &weights);
    let data = random(8 * 6, 43);
    runtime.write(&program, input, &data);
    runtime.run(&program);
    let mut expected = vec![0.0f32; 8 * 6];
    for row in 0..3usize {
        for column in 0..6usize {
            expected[row * 6 + column] = 1.0;
            expected[(row + 5) * 6 + column] = 1.0;
        }
    }
    assert_close(
        &runtime.read(&program, gradients.of(input)),
        &expected,
        1e-6,
    );
}

#[test]
fn a_gather_reads_the_rows_of_the_table_its_view_names() {
    let runtime = open();
    let graph = Graph::new();
    let table = graph.parameter(
        Shape::matrix(12, 6),
        Init::Uniform {
            low: -0.5,
            high: 0.5,
        },
    );
    let region = graph.slice(table, 2, 4, 6);
    let indices = graph.input(Shape::matrix(4, 1));
    let out = graph.gather(region, indices);
    graph.retain(out);
    let weights = runtime.weights(&graph, Precision::Single);
    let program = runtime.compile(&graph, &weights);
    let table_data = random(12 * 6, 47);
    runtime.write(&program, table, &table_data);
    runtime.write(&program, indices, &[2.0, 5.0, 0.0, 3.0]);
    runtime.run(&program);
    let mut expected = vec![0.0f32; 4 * 6];
    for (row, chosen) in [2.0f32, 5.0, 0.0, 3.0].into_iter().enumerate() {
        let at = (4 + chosen as usize) * 6;
        expected[row * 6..(row + 1) * 6].copy_from_slice(&table_data[at..at + 6]);
    }
    assert_close(&runtime.read(&program, out), &expected, 1e-6);
}

#[test]
fn a_scatter_lands_on_the_rows_its_view_names() {
    let runtime = open();
    let graph = Graph::new();
    let table = graph.parameter(Shape::matrix(10, 4), Init::Zero);
    let region = graph.slice(table, 2, 3, 5);
    let indices = graph.input(Shape::matrix(2, 1));
    let updates = graph.input(Shape::matrix(2, 4));
    graph.scatter_into(region, indices, updates);
    graph.retain(table);
    let weights = runtime.weights(&graph, Precision::Single);
    let program = runtime.compile(&graph, &weights);
    runtime.write(&program, indices, &[1.0, 4.0]);
    let updates_data = random(2 * 4, 53);
    runtime.write(&program, updates, &updates_data);
    runtime.run(&program);
    let mut expected = vec![0.0f32; 10 * 4];
    for (row, chosen) in [1usize, 4].into_iter().enumerate() {
        expected[(chosen + 3) * 4..(chosen + 3) * 4 + 4]
            .copy_from_slice(&updates_data[row * 4..(row + 1) * 4]);
    }
    assert_close(&runtime.read(&program, table), &expected, 1e-6);
}

#[test]
fn a_product_reads_one_parameter_through_two_reshapes() {
    let runtime = open();
    let graph = Graph::new();
    let input = graph.input(Shape::matrix(6, 8));
    let wide = weight(&graph, 8, 8);
    let half_a = graph.slice(wide, 3, 0, 4);
    let half_b = graph.slice(wide, 3, 4, 4);
    let first = graph.matmul(input, half_a);
    let second = graph.matmul(input, half_b);
    let joined = graph.concat(1, first, second);
    let loss = graph.sum(joined);
    let gradients = graph.backward(loss);
    let weights = runtime.weights(&graph, Precision::Single);
    let program = runtime.compile(&graph, &weights);
    let input_data = random(6 * 8, 59);
    runtime.write(&program, input, &input_data);
    runtime.run(&program);
    let mut expected = vec![0.0f32; 8 * 8];
    for depth in 0..8usize {
        for column in 0..8usize {
            let mut total = 0.0;
            for row in 0..6usize {
                total += input_data[row * 8 + depth];
            }
            expected[depth * 8 + column] = total;
        }
    }
    assert_close(&runtime.read(&program, gradients.of(wide)), &expected, 1e-4);
}

#[test]
fn a_gather_reference_rides_the_views_of_its_table() {
    let runtime = open();
    let graph = Graph::new();
    let table = graph.input(Shape::matrix(6, 4));
    let indices = graph.input(Shape::matrix(3, 1));
    let out = graph.gather(table, indices);
    graph.retain(out);
    let weights = runtime.weights(&graph, Precision::Single);
    let program = runtime.compile(&graph, &weights);
    let table_data = random(6 * 4, 61);
    runtime.write(&program, table, &table_data);
    runtime.write(&program, indices, &[5.0, 0.0, 3.0]);
    runtime.run(&program);
    let mut expected = vec![0.0f32; 3 * 4];
    for (row, chosen) in [5usize, 0, 3].into_iter().enumerate() {
        expected[row * 4..(row + 1) * 4].copy_from_slice(&table_data[chosen * 4..(chosen + 1) * 4]);
    }
    assert_close(&runtime.read(&program, out), &expected, 1e-6);
}
