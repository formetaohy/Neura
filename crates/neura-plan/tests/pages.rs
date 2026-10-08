use neura_abi::{Element, PAGE_WORDS};
use neura_graph::{Graph, Init, Shape};
use neura_plan::Plan;
use neura_profile::{Budget, Profile};
use std::collections::BTreeSet;

const ALIGNMENT: u64 = 256;
const WIDTH: u32 = 256;
const ROWS: u32 = 16;
const PAGES: u32 = WIDTH * WIDTH / PAGE_WORDS as u32;

fn plan(graph: &Graph) -> Plan {
    Plan::of(graph, ALIGNMENT, Profile::derive(Budget::BASELINE, None)[0])
}

#[test]
fn a_plan_names_the_weight_pages_of_every_task_it_schedules() {
    let graph = Graph::new();
    let weight = graph.named_parameter(
        "weight",
        Shape::matrix(WIDTH, WIDTH),
        Init::Zero,
        Element::Single,
    );
    let data = graph.input(Shape::matrix(ROWS, WIDTH), Element::Single);
    graph.retain(graph.matmul(data, weight));
    let upward = graph.fill(Shape::matrix(WIDTH, WIDTH), 0.25);
    graph.add_into(weight, upward);
    let plan = plan(&graph);
    assert_eq!(plan.store_words(), u64::from(WIDTH) * u64::from(WIDTH));
    let tasks = plan.weight_pages();
    assert_eq!(tasks.len(), plan.task_count() as usize);
    let mut widest = 0;
    let mut walked = BTreeSet::new();
    for task in tasks {
        assert!(
            task.pages().windows(2).all(|pair| pair[0] < pair[1]),
            "the pages of a task stand in order and without repeats",
        );
        for write in task.writes() {
            assert!(
                task.pages().contains(write),
                "a task that writes weight page {write} holds it resident",
            );
        }
        if task.pages().is_empty() {
            continue;
        }
        widest = widest.max(task.pages().len());
        walked.extend(task.pages().iter().copied());
    }
    assert_eq!(
        walked,
        (0..PAGES).collect::<BTreeSet<_>>(),
        "the tasks of a plan walk every page of the weight between them",
    );
    let (tile, carried) = plan.matmul_geometries()[0];
    let row_blocks = ROWS.div_ceil(tile.rows());
    let column_blocks = WIDTH.div_ceil(tile.columns());
    let splits = carried / (row_blocks * column_blocks);
    let depth_blocks = WIDTH.div_ceil(tile.depth());
    let slice_rows = depth_blocks.div_ceil(splits) * tile.depth();
    let rows_per_page = PAGE_WORDS as u32 / WIDTH;
    let reach = slice_rows.div_ceil(rows_per_page) + 1;
    assert!(
        widest as u32 <= reach,
        "a product walks {widest} pages of a weight whose tile walks {slice_rows} rows of it, and {reach} pages a depth slice of {slice_rows} rows can reach; {PAGES} means every tile of the product claimed the whole weight",
    );
}

#[test]
fn a_plan_without_weights_names_no_weight_page() {
    let graph = Graph::new();
    let data = graph.input(Shape::matrix(ROWS, WIDTH), Element::Single);
    graph.retain(graph.relu(graph.mul(data, data)));
    let plan = plan(&graph);
    assert_eq!(plan.store_words(), 0);
    assert!(
        plan.weight_pages()
            .iter()
            .all(|task| task.pages().is_empty() && task.writes().is_empty()),
        "a graph without a parameter holds no weight page",
    );
}
