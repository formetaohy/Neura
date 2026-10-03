use neura_abi::{Element, Kind, RECORDS, WORD_BYTES};
use neura_pointwise::OPS;
use neura_profile::{AttentionTile, Budget, CLAIM_BYTES, Geometry, Profile};
use neura_shader::{
    Backend, BindingKind, ComputeProgram, Instruction, ShaderTranslation, Space, Type,
};

const DEVICE: Budget = Budget::of(1024, 48 << 10);

fn profiles() -> Vec<Profile> {
    Profile::derive(DEVICE, None)
}
use neura_megakernel::{BINDINGS, Megakernel};
use std::collections::{BTreeSet, HashSet};
use std::sync::OnceLock;

fn all(profile: usize) -> &'static Megakernel {
    static PROGRAMS: OnceLock<Vec<Megakernel>> = OnceLock::new();
    let programs = PROGRAMS.get_or_init(|| {
        profiles()
            .iter()
            .map(|profile| {
                Megakernel::assemble(
                    Kind::ALL,
                    Element::ALL,
                    Geometry::of(
                        profile.workgroup(),
                        profile.shared_bytes(),
                        profile.tiles(),
                        &[],
                    ),
                )
            })
            .collect()
    });
    &programs[profile]
}

fn selected(profile: Profile, kinds: &[Kind], elements: &[Element]) -> Megakernel {
    Megakernel::assemble(
        kinds,
        elements,
        Geometry::of(
            profile.workgroup(),
            profile.shared_bytes(),
            profile.tiles(),
            &[],
        ),
    )
}

fn functions(program: &ComputeProgram) -> BTreeSet<&str> {
    program
        .module()
        .functions()
        .iter()
        .map(|function| function.name.as_str())
        .collect()
}

fn switch_cases(program: &ComputeProgram, name: &str) -> (Vec<u32>, bool) {
    let function = program
        .module()
        .functions()
        .iter()
        .find(|function| function.name == name)
        .unwrap_or_else(|| panic!("the Rust device function {name} was compiled"));
    function
        .body
        .iter()
        .find_map(|instruction| match instruction {
            Instruction::Switch { cases, default, .. } => Some((
                cases.iter().map(|(value, _)| *value).collect::<Vec<_>>(),
                !default.is_empty(),
            )),
            _ => None,
        })
        .unwrap_or_else(|| panic!("the Rust device function {name} dispatches"))
}

#[test]
fn rust_source_compiles_into_three_native_shader_formats() {
    {
        let program = all(0).program();
        let ShaderTranslation::Spirv(words) = program.translate(Backend::Vulkan) else {
            panic!("Vulkan requires SPIR-V");
        };
        assert_eq!(words[0], 0x0723_0203);
        assert_eq!(words[1], 0x0001_0300, "a device program asks SPIR-V 1.3");
        let ShaderTranslation::Hlsl { source, entry } = program.translate(Backend::Dx12) else {
            panic!("D3D12 requires HLSL");
        };
        assert!(!entry.is_empty());
        assert!(source.contains("register(u2)"));
        assert!(source.contains("register(u4)"));
        assert!(source.contains("register(t5)"));
        let ShaderTranslation::Msl {
            source,
            entry,
            size_bindings,
        } = program.translate(Backend::Metal)
        else {
            panic!("Metal requires MSL");
        };
        assert!(!entry.is_empty());
        assert!(size_bindings.is_empty());
        assert!(source.contains("[[buffer(2)]]"));
        assert!(source.contains("[[buffer(7)]]"));
    }
}

#[test]
fn every_task_compiles_for_every_native_backend() {
    {
        let program = selected(profiles()[0], Kind::ALL, Element::ALL).program();
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
}

#[test]
fn every_profile_compiles_the_rust_abi_and_bindings() {
    {
        for (index, profile) in profiles().iter().enumerate() {
            let kernel = all(index);
            let program = kernel.program();
            assert_eq!(kernel.workgroup_size(), profile.workgroup());
            assert_eq!(kernel.geometry().tiles(), profile.tiles());
            assert_eq!(kernel.bindings().len(), BINDINGS.len());
            assert_eq!(program.bindings().len(), BINDINGS.len());
            for (binding, reflected) in BINDINGS.iter().zip(kernel.bindings()) {
                assert_eq!(binding.name, reflected.name);
                assert_eq!(binding.binding, reflected.binding);
                assert_eq!(binding.kind, reflected.kind);
            }
            assert_eq!(program.reflected()[2].kind, BindingKind::ReadWriteStorage);
            for layout in RECORDS {
                let ty = program
                    .module()
                    .types()
                    .iter()
                    .find(|ty| matches!(ty, Type::Struct { name, .. } if name == layout.name))
                    .expect("all records originate in Rust");
                let Type::Struct { members, span, .. } = ty else {
                    panic!("a host record is a device struct");
                };
                assert_eq!(*span, layout.size);
                for (member, host) in members.iter().zip(layout.fields) {
                    assert_eq!(member.name, host.name);
                    assert_eq!(member.offset, host.offset);
                }
            }
        }
    }
}

#[test]
fn specialization_includes_only_reachable_rust_functions() {
    let kernel = selected(profiles()[0], &[Kind::Fill], &[Element::Single]);
    let program = kernel.program();
    let reachable = functions(&program);
    assert!(reachable.contains("run_fill"));
    for absent in [
        "run_matmul",
        "run_conv2d",
        "run_softmax",
        "run_scatter",
        "workgroup_choice",
        "fetch_half",
    ] {
        assert!(
            !reachable.contains(absent),
            "a fill kernel compiles unused function {absent}"
        );
    }
    let scratch = program
        .module()
        .globals()
        .iter()
        .filter(|global| global.space == Space::WorkGroup)
        .map(|global| global.name.clone())
        .collect::<Vec<_>>();
    assert_eq!(
        scratch,
        vec!["claim".to_owned()],
        "a fill kernel declares no workgroup scratch beside the megakernel's own claim",
    );
    let all = all(0).program();
    assert!(program.spirv().len() < all.spirv().len());
    assert_eq!(kernel.kinds(), &[Kind::Fill]);
}

#[test]
fn the_rust_dispatcher_only_accepts_its_selected_task_kinds() {
    let program = selected(
        profiles()[0],
        &[Kind::Fill, Kind::Binary],
        &[Element::Single],
    )
    .program();
    let (codes, has_default) = switch_cases(&program, "run_task");
    assert_eq!(codes.len(), 2);
    assert!(has_default, "a task dispatcher covers its default");
    assert_eq!(codes[0], Kind::Binary.code());
    assert_eq!(codes[1], Kind::Fill.code());
}

#[test]
fn every_declared_kind_reaches_the_device_body_it_names() {
    let program = selected(profiles()[0], Kind::ALL, Element::ALL).program();
    let reachable = functions(&program);
    for kind in Kind::ALL {
        assert!(
            reachable.contains(kind.entry()),
            "the {} kind names a device body {} that no module installs",
            kind.name(),
            kind.entry(),
        );
    }
    let (dispatched, _) = switch_cases(&program, "run_task");
    let dispatched = dispatched.into_iter().collect::<HashSet<_>>();
    assert_eq!(dispatched.len(), Kind::COUNT as usize);
    for kind in Kind::ALL {
        assert!(
            dispatched.contains(&kind.code()),
            "the task dispatcher carries no arm for the {} kind",
            kind.name(),
        );
    }
}

#[test]
fn the_rust_operation_dispatcher_contains_every_declared_code() {
    let program = selected(
        profiles()[0],
        &[Kind::Binary, Kind::Partial],
        &[Element::Single],
    )
    .program();
    let (defined, _) = switch_cases(&program, "op_apply");
    let defined = defined.into_iter().collect::<HashSet<_>>();
    assert_eq!(defined.len(), OPS.len());
    for op in OPS {
        assert!(defined.contains(&op.code));
    }
}

#[test]
fn every_tensor_element_compiles_only_the_loads_it_reads() {
    let single = selected(profiles()[0], &[Kind::Unary], &[Element::Single]).program();
    let half = selected(
        profiles()[0],
        &[Kind::Unary],
        &[Element::Single, Element::Half],
    )
    .program();
    assert_ne!(single, half);
    assert!(!functions(&single).contains("fetch_half"));
    assert!(functions(&half).contains("fetch_half"));
    assert!(!functions(&half).contains("fetch_bfloat16"));
    let quantized = selected(
        profiles()[0],
        &[Kind::Unary],
        &[Element::Single, Element::Int8],
    )
    .program();
    let reachable = functions(&quantized);
    assert!(reachable.contains("fetch_int8"));
    assert!(!reachable.contains("fetch_half"));
    assert!(!reachable.contains("fetch_bfloat16"));
}

#[test]
fn a_convert_carries_only_the_elements_it_packs() {
    let half = selected(
        profiles()[0],
        &[Kind::Convert],
        &[Element::Single, Element::Half],
    )
    .program();
    let reachable = functions(&half);
    assert!(reachable.contains("run_convert"));
    assert!(reachable.contains("run_convert_half"));
    assert!(!reachable.contains("run_convert_bfloat16"));
    let both = selected(
        profiles()[0],
        &[Kind::Convert],
        &[Element::Half, Element::Bfloat16],
    )
    .program();
    let reachable = functions(&both);
    assert!(reachable.contains("run_convert_half"));
    assert!(reachable.contains("run_convert_bfloat16"));
    assert!(!reachable.contains("run_convert_int8"));
    let quantized = selected(
        profiles()[0],
        &[Kind::Convert],
        &[Element::Single, Element::Int8],
    )
    .program();
    let reachable = functions(&quantized);
    assert!(reachable.contains("run_convert_int8"));
    assert!(!reachable.contains("run_convert_half"));
}

#[test]
fn matrix_specialization_contains_every_tile_in_the_profile() {
    for (index, profile) in profiles().iter().enumerate() {
        let kernel = all(index);
        let program = kernel.program();
        let names = functions(&program);
        for (tile, shape) in profile.tiles().iter().enumerate() {
            assert!(names.contains(format!("run_matmul_{tile}").as_str()));
            if shape.strategy() == neura_profile::MatmulStrategy::Staged {
                assert!(names.contains(format!("matmul_load_{tile}").as_str()));
            } else {
                assert!(
                    !names.contains(format!("matmul_load_{tile}").as_str()),
                    "a streamed product stages nothing through the workgroup it never loads",
                );
            }
        }
    }
}

fn workgroup_bytes(program: &ComputeProgram) -> u64 {
    program
        .module()
        .globals()
        .iter()
        .filter(|global| global.space == Space::WorkGroup)
        .map(|global| u64::from(program.module().size(global.ty)))
        .sum()
}

#[test]
fn workgroup_allocation_fits_the_advertised_profile() {
    for (index, profile) in profiles().iter().enumerate() {
        let program = all(index).program();
        let used = workgroup_bytes(&program);
        assert!(used > 0);
        assert!(
            used <= profile.shared_bytes(),
            "{profile:?} reserves {used} workgroup bytes"
        );
    }
}

#[test]
fn a_device_program_shares_one_scratch_pool_between_its_bodies() {
    let profile = *profiles().last().expect("a profile");
    let attention = [AttentionTile::new(4, 8)];
    let geometry = Geometry::of(
        profile.workgroup(),
        profile.shared_bytes(),
        profile.tiles(),
        &attention,
    );
    let kernel = Megakernel::assemble(Kind::ALL, Element::ALL, geometry.clone());
    let used = workgroup_bytes(&kernel.program());
    assert_eq!(
        used,
        geometry.scratch_bytes(Kind::ALL) + CLAIM_BYTES,
        "a device program declares the pool its widest body stages beside the word its workgroups claim with",
    );
    let partitioned = geometry.staging_bytes()
        + 2 * u64::from(attention[0].stage_words()) * WORD_BYTES
        + 2 * u64::from(profile.workgroup()) * WORD_BYTES;
    assert!(
        used < partitioned,
        "a device program lets every body stage from one pool instead of partitioning {partitioned} bytes among them",
    );
}

#[test]
fn the_same_rust_specialization_produces_the_same_device_program() {
    let first = selected(profiles()[0], &[Kind::Fill], &[Element::Single]).program();
    let second = selected(profiles()[0], &[Kind::Fill], &[Element::Single]).program();
    assert_eq!(first, second);
    assert_eq!(first.spirv(), second.spirv());
}

#[test]
fn attention_specialization_contains_every_tile_of_its_geometry() {
    let profile = *profiles().last().expect("a profile");
    let attention = [AttentionTile::new(4, 8), AttentionTile::new(2, 16)];
    let kernel = Megakernel::assemble(
        &[
            Kind::Attention,
            Kind::AttentionQueryGrad,
            Kind::AttentionKeyGrad,
            Kind::AttentionValueGrad,
        ],
        &[Element::Single],
        Geometry::of(profile.workgroup(), profile.shared_bytes(), &[], &attention),
    );
    let program = kernel.program();
    let names = functions(&program);
    for (index, tile) in attention.iter().enumerate() {
        for name in [
            format!("run_attention_{index}"),
            format!("run_attention_query_grad_{index}"),
            format!("run_attention_key_grad_{index}"),
            format!("run_attention_value_grad_{index}"),
            format!("stage_attention_{index}"),
        ] {
            assert!(names.contains(name.as_str()), "{name} is missing");
        }
        assert_eq!(
            kernel.geometry().attention_tile(index as u32),
            *tile,
            "a device program carries the tile its geometry names",
        );
    }
    assert_eq!(
        kernel.geometry().scratch_bytes(&[Kind::Attention]),
        2 * u64::from(
            attention
                .iter()
                .map(|tile| tile.stage_words())
                .max()
                .expect("an attention stage"),
        ) * WORD_BYTES,
        "a device program stages the widest attention tile it carries",
    );
}
