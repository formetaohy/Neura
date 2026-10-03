mod attention;
#[path = "../device/attention.rs"]
mod attention_device;
#[path = "../device/choice.rs"]
mod choice;
#[path = "../device/conv.rs"]
mod conv;
mod convert;
#[path = "../device/convert.rs"]
mod convert_device;
mod element;
#[path = "../device/layout.rs"]
mod layout;
mod matmul;
#[path = "../device/matmul.rs"]
mod matmul_device;
mod op;
#[path = "../device/op.rs"]
mod op_device;
#[path = "../device/pointwise.rs"]
mod pointwise;
#[path = "../device/pool.rs"]
mod pool;
mod reduce;
#[path = "../device/reduce.rs"]
mod reduce_device;
mod rope;
#[path = "../device/rope.rs"]
mod rope_device;
#[path = "../device/scatter.rs"]
mod scatter;
#[path = "../device/select.rs"]
mod select;
#[path = "../device/softmax.rs"]
mod softmax;

use neura_abi::{Element, Kind, Module};
use neura_compiler::{Compiler, ir};
use neura_profile::Geometry;

pub fn define(compiler: &mut Compiler, kinds: &[Kind], elements: &[Element], geometry: &Geometry) {
    let words = geometry.scratch_words(kinds);
    if words > 0 {
        compiler.workgroup("scratch", "f32", words);
    }
    substrate(compiler, elements);
    for module in Module::ALL {
        if kinds.iter().any(|kind| kind.carries(*module)) {
            install(compiler, *module, elements, geometry);
        }
    }
    for kind in Kind::ALL {
        compiler.insert_case(
            "run_task",
            ir::Arm {
                pattern: ir::Pattern::Constant(kind.symbol().to_owned()),
                body: vec![ir::Statement::Expression(ir::Expression::call(
                    kind.entry(),
                    vec![ir::Expression::name("task"), ir::Expression::name("lid")],
                ))],
            },
        );
    }
    assert!(
        compiler.workgroup_bytes() <= geometry.budget(),
        "a device program of {} kinds declares {} bytes of workgroup scratch beyond the {} its profile offers",
        kinds.len(),
        compiler.workgroup_bytes(),
        geometry.budget(),
    );
}

fn substrate(compiler: &mut Compiler, elements: &[Element]) {
    element::define(compiler, elements);
    pointwise::define(compiler);
    op::define(compiler);
    rope::define(compiler);
}

fn install(compiler: &mut Compiler, module: Module, elements: &[Element], geometry: &Geometry) {
    match module {
        Module::Matmul => matmul_device::define(compiler),
        Module::MatmulTiles => {
            compiler.constant("SCRATCH_MATMUL_RIGHT", geometry.matmul_right());
            matmul::specialize(compiler, geometry);
        }
        Module::Attention => attention::define(compiler, geometry),
        Module::Reduce => reduce::define(compiler),
        Module::Softmax => softmax::define(compiler),
        Module::Choice => {
            compiler.constant("SCRATCH_CHOICE", geometry.choice());
            choice::define(compiler);
        }
        Module::Select => select::define(compiler),
        Module::Conv => conv::define(compiler),
        Module::Scatter => scatter::define(compiler),
        Module::Layout => layout::define(compiler),
        Module::Pool => pool::define(compiler),
        Module::Convert => convert::define(compiler, elements),
    }
}
