use neura_compiler::Compiler;

pub fn define(compiler: &mut Compiler) {
    super::reduce_device::define(compiler);
    compiler.select("workgroup_sum_tree", "workgroup_sum");
    compiler.select("workgroup_max_tree", "workgroup_max");
}
