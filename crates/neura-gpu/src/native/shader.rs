use crate::pipeline::{BindingKind, ComputeProgram};
use naga::back::{hlsl, msl, spv};
use naga::front::wgsl;
use naga::valid::{Capabilities, ModuleInfo, ValidationFlags, Validator};
use naga::{AddressSpace, Module, ShaderStage, StorageAccess};

fn validated(program: &ComputeProgram) -> (Module, ModuleInfo, usize) {
    let module = wgsl::parse_str(program.source()).unwrap_or_else(|error| {
        panic!(
            "{} contains invalid WGSL:\n{}",
            program.label(),
            error.emit_to_string(program.source())
        )
    });
    let info = Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .unwrap_or_else(|error| {
            panic!(
                "{} contains an invalid compute shader: {error}",
                program.label()
            )
        });
    let entry = module
        .entry_points
        .iter()
        .position(|entry| entry.name == program.entry() && entry.stage == ShaderStage::Compute)
        .unwrap_or_else(|| {
            panic!(
                "{} has no compute entry point named {}",
                program.label(),
                program.entry()
            )
        });
    let declared = module
        .global_variables
        .iter()
        .filter_map(|(_, variable)| variable.binding.map(|binding| (binding, variable.space)))
        .collect::<Vec<_>>();
    assert_eq!(
        declared.len(),
        program.bindings().len(),
        "a compute program's declared resources and bound buffers differ"
    );
    for (binding, space) in declared {
        let spec = program
            .bindings()
            .get(binding.binding as usize)
            .unwrap_or_else(|| panic!("shader binding {} is not supplied", binding.binding));
        assert_eq!(binding.group, 0, "compute shaders use one storage group");
        assert_eq!(binding.binding, spec.binding);
        let AddressSpace::Storage { access } = space else {
            panic!("compute binding {} is not storage", binding.binding);
        };
        let kind = if access.contains(StorageAccess::STORE) {
            BindingKind::ReadWriteStorage
        } else {
            BindingKind::ReadOnlyStorage
        };
        assert_eq!(
            kind, spec.kind,
            "compute binding {} has the wrong access",
            binding.binding
        );
    }
    (module, info, entry)
}

pub(crate) fn spirv(program: &ComputeProgram) -> Vec<u32> {
    let (module, info, _) = validated(program);
    let mut options = spv::Options {
        fake_missing_bindings: false,
        ..Default::default()
    };
    for spec in program.bindings() {
        options.binding_map.insert(
            naga::ResourceBinding {
                group: 0,
                binding: spec.binding,
            },
            spv::BindingInfo {
                descriptor_set: 0,
                binding: spec.binding,
                binding_array_size: None,
            },
        );
    }
    crate::native::workgroup::isolate(
        spv::write_vec(
            &module,
            &info,
            &options,
            Some(&spv::PipelineOptions {
                shader_stage: ShaderStage::Compute,
                entry_point: program.entry().to_owned(),
            }),
        )
        .unwrap_or_else(|error| panic!("{} failed SPIR-V translation: {error}", program.label())),
    )
}

pub(crate) fn hlsl(program: &ComputeProgram) -> (String, String) {
    let (module, info, index) = validated(program);
    let mut options = hlsl::Options {
        shader_model: hlsl::ShaderModel::V6_0,
        fake_missing_bindings: false,
        ..Default::default()
    };
    for spec in program.bindings() {
        options.binding_map.insert(
            naga::ResourceBinding {
                group: 0,
                binding: spec.binding,
            },
            hlsl::BindTarget {
                register: spec.binding,
                ..Default::default()
            },
        );
    }
    let pipeline = hlsl::PipelineOptions {
        entry_point: Some((ShaderStage::Compute, program.entry().to_owned())),
    };
    let mut source = String::new();
    let reflection = hlsl::Writer::new(&mut source, &options, &pipeline)
        .write(&module, &info, None)
        .unwrap_or_else(|error| panic!("{} failed HLSL translation: {error}", program.label()));
    let entry = reflection.entry_point_names[index]
        .as_ref()
        .unwrap_or_else(|error| panic!("{} has no HLSL entry: {error}", program.label()))
        .clone();
    (source, entry)
}

pub(crate) fn msl(program: &ComputeProgram) -> (String, String, Vec<u32>) {
    let (module, info, index) = validated(program);
    let mut resources = msl::EntryPointResources {
        sizes_buffer: Some(crate::pipeline::METAL_SIZE_BUFFER_SLOT),
        ..Default::default()
    };
    let size_bindings = module
        .global_variables
        .iter()
        .filter(|(_, variable)| {
            module.types[variable.ty]
                .inner
                .needs_host_buffer_byte_size(&module.types)
        })
        .map(|(_, variable)| {
            let binding = variable.binding.expect("a runtime-sized buffer is bound");
            assert_eq!(binding.group, 0);
            binding.binding
        })
        .collect::<Vec<_>>();
    for spec in program.bindings() {
        resources.resources.insert(
            naga::ResourceBinding {
                group: 0,
                binding: spec.binding,
            },
            msl::BindTarget {
                buffer: Some(spec.binding as u8),
                mutable: spec.kind == BindingKind::ReadWriteStorage,
                ..Default::default()
            },
        );
    }
    let mut options = msl::Options {
        lang_version: (2, 3),
        fake_missing_bindings: false,
        ..Default::default()
    };
    options
        .per_entry_point_map
        .insert(program.entry().to_owned(), resources);
    let (source, reflection) = msl::write_string(
        &module,
        &info,
        &options,
        &msl::PipelineOptions {
            entry_point: Some((ShaderStage::Compute, program.entry().to_owned())),
            ..Default::default()
        },
    )
    .unwrap_or_else(|error| panic!("{} failed MSL translation: {error}", program.label()));
    let entry = reflection.entry_point_names[index]
        .as_ref()
        .unwrap_or_else(|error| panic!("{} has no MSL entry: {error}", program.label()))
        .clone();
    (source, entry, size_bindings)
}
