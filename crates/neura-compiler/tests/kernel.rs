use neura_compiler::{
    Backend, BindingSpec, Read, ReadWrite, ShaderTranslation, Uvec3, abi::ValueRecord, kernel,
};

#[kernel(workgroup_size = 64)]
fn by_group(group: Uvec3, output: ReadWrite<u32>) {
    output[group.x] = group.x;
}

#[kernel(workgroup_size = 64)]
fn inspect(lid: u32, values: Read<ValueRecord>, output: ReadWrite<u32>) {
    output[lid] = values[lid].dims[3];
}

#[kernel(workgroup_size = 64)]
fn doubled(lid: u32, input: Read<u32>, output: ReadWrite<u32>) {
    output[lid] = input[lid] * 2u32;
}

#[test]
fn annotated_workgroup_position_has_a_real_rust_type() {
    let program = by_group();
    assert_eq!(program.module().entry_points[0].function.arguments.len(), 1);
    assert!(matches!(
        program.translate(Backend::Vulkan),
        ShaderTranslation::Spirv(_)
    ));
    assert!(matches!(
        program.translate(Backend::Dx12),
        ShaderTranslation::Hlsl { .. }
    ));
    assert!(matches!(
        program.translate(Backend::Metal),
        ShaderTranslation::Msl { .. }
    ));
}

#[test]
fn rust_record_fields_use_the_host_abi_on_all_backends() {
    let program = inspect();
    assert_eq!(program.reflected()[0].name, "values");
    for backend in [Backend::Vulkan, Backend::Dx12, Backend::Metal] {
        match program.translate(backend) {
            ShaderTranslation::Spirv(words) => assert_eq!(words[0], 0x0723_0203),
            ShaderTranslation::Hlsl { source, .. } => assert!(source.contains("register(t0)")),
            ShaderTranslation::Msl { source, .. } => assert!(source.contains("[[buffer(0)]]")),
        }
    }
}

#[test]
fn annotated_rust_kernel_generates_every_native_backend() {
    let program = doubled();
    let cached = doubled();
    assert!(std::ptr::eq(program.module(), cached.module()));
    assert_eq!(program.workgroup_size(), 64);
    assert_eq!(program.reflected()[0].name, "input");
    assert_eq!(program.bindings()[0], BindingSpec::storage(0));
    assert_eq!(program.bindings()[1], BindingSpec::writable_storage(1));
    let ShaderTranslation::Spirv(words) = program.translate(Backend::Vulkan) else {
        panic!("Vulkan needs SPIR-V");
    };
    assert_eq!(words[0], 0x0723_0203);
    let ShaderTranslation::Hlsl { source, .. } = program.translate(Backend::Dx12) else {
        panic!("D3D12 needs HLSL");
    };
    assert!(source.contains("register(u1)"));
    let ShaderTranslation::Msl { source, .. } = program.translate(Backend::Metal) else {
        panic!("Metal needs MSL");
    };
    assert!(source.contains("[[buffer(1)]]"));
}
