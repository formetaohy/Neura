mod attention;
mod choice;
mod conv;
mod convert;
mod element;
mod layout;
mod matmul;
mod op;
mod pointwise;
mod pool;
mod reduce;
mod rope;
mod scatter;
mod select;
mod softmax;

use neura_abi::{DeviceModule, Element, Kind};
use neura_compiler::{Compiler, ir};
use neura_profile::Geometry;

pub fn define(compiler: &mut Compiler, kinds: &[Kind], elements: &[Element], geometry: &Geometry) {
    let words = geometry.scratch_words(kinds);
    if words > 0 {
        compiler.workgroup("scratch", "f32", words);
    }
    let half = geometry.half_stage_words();
    if half > 0 {
        compiler.workgroup("scratch_half", "f16", half / 2);
    }
    substrate(compiler, elements);
    for module in DeviceModule::ALL {
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
    element::install(compiler, elements);
    pointwise::install(compiler);
    op::install(compiler);
    rope::install(compiler);
}

fn install(
    compiler: &mut Compiler,
    module: DeviceModule,
    elements: &[Element],
    geometry: &Geometry,
) {
    match module {
        DeviceModule::Matmul => matmul::install(compiler),
        DeviceModule::MatmulTiles => matmul::install_tiles(compiler, geometry),
        DeviceModule::Attention => attention::install(compiler, geometry),
        DeviceModule::Reduce => reduce::install(compiler),
        DeviceModule::Softmax => softmax::install(compiler),
        DeviceModule::Choice => choice::install(compiler, geometry),
        DeviceModule::Select => select::install(compiler),
        DeviceModule::Conv => conv::install(compiler),
        DeviceModule::Scatter => scatter::install(compiler),
        DeviceModule::Layout => layout::install(compiler),
        DeviceModule::Pool => pool::install(compiler),
        DeviceModule::Convert => convert::install(compiler, elements),
    }
}
