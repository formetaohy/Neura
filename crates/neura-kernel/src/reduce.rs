use neura_abi::Features;
use neura_compiler::Compiler;

pub fn define(compiler: &mut Compiler, features: Features) {
    super::reduce_device::define(compiler);
    let (sum, max) = if features.contains(Features::SUBGROUP) {
        ("workgroup_sum_warp", "workgroup_max_warp")
    } else {
        ("workgroup_sum_tree", "workgroup_max_tree")
    };
    compiler.select(sum, "workgroup_sum");
    compiler.select(max, "workgroup_max");
}
