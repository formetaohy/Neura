use neura_abi::Element;
use neura_compiler::{Compiler, ast};

pub(crate) fn install(compiler: &mut Compiler, elements: &[Element]) {
    device::define(compiler);
    for element in elements
        .iter()
        .copied()
        .filter(|element| element.narrow() && !element.sidecar())
    {
        let name = format!("scatter_words_{}", element.name());
        match element.block() {
            Some(block) => compiler.specialize(
                "template_scatter_words_block",
                &name,
                &[
                    ("SCATTER_BLOCK", block.elements),
                    ("SCATTER_BYTES", block.bytes),
                ],
            ),
            None => compiler.specialize(
                "template_scatter_words_packed",
                &name,
                &[("SCATTER_PER_WORD", element.elements_per_word() as u32)],
            ),
        }
        compiler.insert_case(
            "scatter_write_words",
            ast::Arm {
                pattern: ast::Pattern::Integer(element.code()),
                body: vec![ast::Statement::Return(Some(ast::Expression::call(
                    name,
                    vec![ast::Expression::name("width")],
                )))],
            },
        );
        compiler.insert_case(
            "run_scatter_write_by_element",
            ast::Arm {
                pattern: ast::Pattern::Integer(element.code()),
                body: vec![ast::Statement::Expression(ast::Expression::call(
                    "template_run_scatter_write_words",
                    vec![ast::Expression::name("task"), ast::Expression::name("lid")],
                ))],
            },
        );
    }
    compiler.insert_case(
        "run_scatter_write_by_element",
        ast::Arm {
            pattern: ast::Pattern::Integer(Element::Single.code()),
            body: vec![ast::Statement::Expression(ast::Expression::call(
                "run_scatter_write_element",
                vec![ast::Expression::name("task"), ast::Expression::name("lid")],
            ))],
        },
    );
    compiler.select("scatter_write_words", "scatter_words");
    compiler.select("run_scatter_write_by_element", "run_scatter_write");
}

#[neura_compiler::module]
mod device {
    fn run_scatter(task: Task, lid: u32) {
        let into = values[task.a];
        let indices = values[task.b];
        let updates = values[task.c];
        let width = into.dims.w;
        let rows = into.dims.x * into.dims.y * into.dims.z;
        for column in stride(lid, width, WORKGROUP_SIZE) {
            for row in stride(task.first, task.first + task.count, 1u32) {
                let chosen = whole_index(fetch(indices, row), rows, kind::SCATTER, refusal::INDEX);
                let address = chosen * width + column;
                publish(
                    into,
                    address,
                    fetch(into, address) + fetch(updates, row * width + column),
                );
            }
        }
    }

    fn run_compact(task: Task, lid: u32) {
        let mask = values[task.a];
        let positions = values[task.b];
        let indices = values[task.out];
        let rows = indices.dims.x * indices.dims.y * indices.dims.z;
        for row in stride(task.first + lid, task.first + task.count, WORKGROUP_SIZE) {
            let flag = fetch(mask, row);
            if flag != 0.0 && flag != 1.0 {
                refuse(kind::COMPACT, refusal::MASK, 0u32);
            }
            if flag == 1.0 {
                let at = whole_index(fetch(positions, row), rows, kind::COMPACT, refusal::INDEX);
                publish(indices, at, f32(row));
            }
        }
    }

    fn run_scatter_write_by_element(task: Task, lid: u32) {
        match values[task.a].element {
            _ => refuse(
                kind::SCATTER_WRITE,
                refusal::ELEMENT,
                values[task.a].element,
            ),
        }
    }

    fn run_scatter_write_element(task: Task, lid: u32) {
        let into = values[task.a];
        let indices = values[task.b];
        let updates = values[task.c];
        let width = into.dims.w;
        let rows = into.dims.x * into.dims.y * into.dims.z;
        for column in stride(lid, width, WORKGROUP_SIZE) {
            for row in stride(task.first, task.first + task.count, 1u32) {
                let chosen = whole_index(
                    fetch(indices, row),
                    rows,
                    kind::SCATTER_WRITE,
                    refusal::INDEX,
                );
                publish(
                    into,
                    chosen * width + column,
                    fetch(updates, row * width + column),
                );
            }
        }
    }

    fn template_run_scatter_write_words(task: Task, lid: u32) {
        let into = values[task.a];
        let indices = values[task.b];
        let updates = values[task.c];
        let width = into.dims.w;
        let words = scatter_words(values[task.a].element, width);
        let rows = into.dims.x * into.dims.y * into.dims.z;
        for word in stride(lid, words, WORKGROUP_SIZE) {
            for row in stride(task.first, task.first + task.count, 1u32) {
                let chosen = whole_index(
                    fetch(indices, row),
                    rows,
                    kind::SCATTER_WRITE,
                    refusal::INDEX,
                );
                publish_word(
                    into,
                    chosen * words + word,
                    bitcast_u32(peek(word_of(updates, row * words + word))),
                );
            }
        }
    }

    fn template_scatter_words_block(width: u32) -> u32 {
        let bytes = (width + SCATTER_BLOCK - 1u32) / SCATTER_BLOCK * SCATTER_BYTES;
        return (bytes + 3u32) / 4u32;
    }

    fn template_scatter_words_packed(width: u32) -> u32 {
        return (width + SCATTER_PER_WORD - 1u32) / SCATTER_PER_WORD;
    }

    fn scatter_write_words(element: u32, width: u32) -> u32 {
        match element {
            _ => {
                refuse(kind::SCATTER_WRITE, refusal::ELEMENT, element);
                return 0u32;
            }
        }
    }
}
