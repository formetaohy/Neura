use neura_compiler::{BindingSpec, Read, ReadWrite, Uvec3, abi::ValueRecord, kernel};
use neura_shader::{Backend, Barrier, Instruction, ShaderTranslation, Space, Type};

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
    assert_eq!(program.module().entry().arguments.len(), 1);
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

#[kernel(workgroup_size = 64)]
fn loops(lid: u32, input: Read<f32>, output: ReadWrite<f32>) {
    let mut total = 0.0f32;
    for index in stride(lid, 256u32, WORKGROUP_SIZE) {
        total += input[index];
    }
    for lane in unroll(0u32, 4u32, 1u32) {
        total += input[lid + lane * 64u32];
    }
    output[lid] = total;
}

#[kernel(workgroup_size = 8)]
mod staging {
    workgroup!(stage: [f32; 8]);

    fn twice(value: f32) -> f32 {
        value * 2.0f32
    }

    #[kernel]
    fn main(lid: u32, input: Read<f32>, output: ReadWrite<f32>) {
        stage[lid] = twice(input[lid]);
        workgroup_barrier();
        output[lid] = stage[(lid + 7u32) % 8u32];
    }
}

fn holds(body: &[Instruction], predicate: fn(&Instruction) -> bool) -> bool {
    body.iter().any(|instruction| {
        predicate(instruction)
            || match instruction {
                Instruction::If { accept, reject, .. } => {
                    holds(accept, predicate) || holds(reject, predicate)
                }
                Instruction::Switch { cases, default, .. } => {
                    cases.iter().any(|(_, body)| holds(body, predicate))
                        || holds(default, predicate)
                }
                Instruction::Loop { body, continuing } => {
                    holds(body, predicate) || holds(continuing, predicate)
                }
                Instruction::Block(body) => holds(body, predicate),
                _ => false,
            }
    })
}

#[test]
fn a_kernel_walks_a_loop_form_the_device_declares() {
    let program = loops();
    assert!(holds(
        &program.module().entry().body,
        |instruction| matches!(instruction, Instruction::Loop { .. }),
    ));
    assert!(holds(
        &program.module().entry().body,
        |instruction| matches!(instruction, Instruction::Block(_)),
    ));
    for backend in [Backend::Vulkan, Backend::Dx12, Backend::Metal] {
        let _ = program.translate(backend);
    }
}

#[test]
fn a_kernel_module_shares_workgroup_memory_and_calls_a_helper() {
    let program = staging();
    assert_eq!(program.label(), "staging");
    assert_eq!(program.workgroup_size(), 8);
    assert_eq!(program.bindings()[0], BindingSpec::storage(0));
    assert_eq!(program.bindings()[1], BindingSpec::writable_storage(1));
    let stage = program
        .module()
        .globals()
        .iter()
        .find(|global| global.name == "stage")
        .expect("the module declares its workgroup array");
    assert_eq!(stage.space, Space::WorkGroup);
    let Type::Array { count, .. } = program.module().ty(stage.ty) else {
        panic!("workgroup memory is an array");
    };
    assert_eq!(*count, Some(8));
    assert!(holds(
        &program.module().entry().body,
        |instruction| matches!(instruction, Instruction::Call { .. }),
    ));
    assert!(holds(
        &program.module().entry().body,
        |instruction| matches!(instruction, Instruction::Barrier(Barrier::WorkGroup)),
    ));
    for backend in [Backend::Vulkan, Backend::Dx12, Backend::Metal] {
        let _ = program.translate(backend);
    }
}

#[kernel(workgroup_size = 64)]
mod branches {
    fn positive(value: f32) -> f32 {
        if value > 0.0f32 { value } else { 0.0f32 }
    }

    #[kernel]
    fn main(lid: u32, input: Read<f32>, output: ReadWrite<f32>) {
        output[lid] = positive(input[lid]);
    }
}

#[test]
fn a_helper_returns_the_value_of_the_branch_it_walks() {
    let program = branches();
    let positive = program
        .module()
        .functions()
        .iter()
        .find(|function| function.name == "positive")
        .expect("the module declares its helper");
    assert!(holds(&positive.body, |instruction| matches!(
        instruction,
        Instruction::Return { value: Some(_) }
    )));
    assert!(holds(
        &program.module().entry().body,
        |instruction| matches!(instruction, Instruction::Call { .. })
    ));
    for backend in [Backend::Vulkan, Backend::Dx12, Backend::Metal] {
        let _ = program.translate(backend);
    }
}
