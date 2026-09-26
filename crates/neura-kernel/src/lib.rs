mod attention;
#[path = "../device/attention.rs"]
mod attention_device;
#[path = "../device/choice.rs"]
mod choice;
#[path = "../device/conv.rs"]
mod conv;
mod element;
#[path = "../device/layout.rs"]
mod layout;
mod matmul;
#[path = "../device/matmul.rs"]
mod matmul_device;
mod op;
#[path = "../device/op.rs"]
mod op_device;
mod pack;
#[path = "../device/pack.rs"]
mod pack_device;
#[path = "../device/pointwise.rs"]
mod pointwise;
#[path = "../device/pool.rs"]
mod pool;
#[path = "../device/reduce.rs"]
mod reduce;
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
    let declared = geometry.declared_shared_bytes(kinds);
    assert!(
        declared <= geometry.shared_bytes(),
        "a device program of {} kinds declares {declared} bytes of workgroup scratch beyond the {} its profile offers",
        kinds.len(),
        geometry.shared_bytes(),
    );
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
        compiler.workgroup_bytes() <= geometry.shared_bytes(),
        "a device program of {} kinds declares {} bytes of workgroup scratch beyond the {} its profile offers",
        kinds.len(),
        compiler.workgroup_bytes(),
        geometry.shared_bytes(),
    );
}

fn substrate(compiler: &mut Compiler, elements: &[Element]) {
    element::define(compiler, elements);
    pointwise::define(compiler);
    op::define(compiler);
}

fn install(compiler: &mut Compiler, module: Module, elements: &[Element], geometry: &Geometry) {
    match module {
        Module::Matmul => matmul_device::define(compiler),
        Module::MatmulTiles => {
            let (left, right) = geometry.stage_lengths();
            compiler.workgroup("matmul_left", "f32", left);
            compiler.workgroup("matmul_right", "f32", right);
            matmul::specialize(compiler, geometry);
        }
        Module::Attention => attention::define(compiler, geometry),
        Module::Reduce => {
            compiler.workgroup("reduction_scratch", "f32", geometry.workgroup());
            reduce::define(compiler);
        }
        Module::Softmax => softmax::define(compiler),
        Module::Choice => {
            compiler.workgroup("choice_index", "u32", geometry.workgroup());
            choice::define(compiler);
        }
        Module::Select => select::define(compiler),
        Module::Conv => conv::define(compiler),
        Module::Scatter => scatter::define(compiler),
        Module::Layout => layout::define(compiler),
        Module::Pool => pool::define(compiler),
        Module::Pack => pack::define(compiler, elements),
    }
}
