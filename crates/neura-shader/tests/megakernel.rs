use naga::{AddressSpace, Expression, MathFunction, Statement, SwitchValue, TypeInner};
use neura_abi::{Element, Kind, RECORDS};
use neura_compiler::{Backend, BindingKind, ComputeProgram, ShaderTranslation};
use neura_op::OPS;
use neura_profile::{AttentionTile, Budget, Geometry, Profile};

const DEVICE: Budget = Budget::of(1024, 48 << 10);

fn profiles() -> Vec<Profile> {
    Profile::derive(DEVICE)
}
use neura_shader::{BINDINGS, Megakernel};
use std::collections::{BTreeSet, HashSet};
use std::sync::OnceLock;

fn all(profile: usize) -> &'static Megakernel {
    static PROGRAMS: OnceLock<Vec<Megakernel>> = OnceLock::new();
    &PROGRAMS.get_or_init(|| {
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
    })[profile]
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
        .functions
        .iter()
        .map(|(_, function)| {
            function
                .name
                .as_deref()
                .expect("a Rust device function has a name")
        })
        .collect()
}

#[test]
fn rust_source_compiles_into_three_native_shader_formats() {
    let program = all(0).program();
    let ShaderTranslation::Spirv(words) = program.translate(Backend::Vulkan) else {
        panic!("Vulkan requires SPIR-V");
    };
    assert_eq!(words[0], 0x0723_0203);
    assert!(words.len() > 100);
    let ShaderTranslation::Hlsl { source, entry } = program.translate(Backend::Dx12) else {
        panic!("D3D12 requires HLSL");
    };
    assert!(!entry.is_empty());
    assert!(source.contains("register(u2)"));
    assert!(source.contains("register(t4)"));
    let ShaderTranslation::Msl {
        source,
        entry,
        size_bindings,
    } = program.translate(Backend::Metal)
    else {
        panic!("Metal requires MSL");
    };
    assert!(!entry.is_empty());
    assert!(size_bindings.contains(&2));
    assert!(source.contains("[[buffer(30)]]"));
    assert!(source.contains("[[buffer(2)]]"));
    assert!(source.contains("[[buffer(7)]]"));
}

#[test]
fn every_element_compiles_every_task_for_every_native_backend() {
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

#[test]
fn every_profile_compiles_the_rust_abi_and_bindings() {
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
        assert!(program.bindings()[4].dynamic_offset);
        assert_eq!(program.reflected()[2].kind, BindingKind::ReadWriteStorage);
        for layout in RECORDS {
            let (_, ty) = program
                .module()
                .types
                .iter()
                .find(|(_, ty)| ty.name.as_deref() == Some(layout.name))
                .expect("all records originate in Rust");
            let TypeInner::Struct { members, span } = &ty.inner else {
                panic!("a host record is a device struct");
            };
            assert_eq!(*span, layout.size);
            for (member, host) in members.iter().zip(layout.fields) {
                assert_eq!(member.name.as_deref(), Some(host.name));
                assert_eq!(member.offset, host.offset);
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
    assert!(
        !program
            .module()
            .global_variables
            .iter()
            .any(|(_, global)| global.space == AddressSpace::WorkGroup)
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
    let function = program
        .module()
        .functions
        .iter()
        .find(|(_, function)| function.name.as_deref() == Some("run_task"))
        .expect("the Rust task dispatcher was compiled")
        .1;
    let cases = function
        .body
        .iter()
        .find_map(|statement| {
            if let Statement::Switch { cases, .. } = statement {
                Some(cases)
            } else {
                None
            }
        })
        .expect("a task dispatcher branches over kinds");
    assert_eq!(cases.len(), 3);
    assert_eq!(cases[0].value, SwitchValue::U32(Kind::Binary.code()));
    assert_eq!(cases[1].value, SwitchValue::U32(Kind::Fill.code()));
    assert_eq!(cases[2].value, SwitchValue::Default);
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
    let cases = program
        .module()
        .functions
        .iter()
        .find(|(_, function)| function.name.as_deref() == Some("run_task"))
        .expect("the Rust task dispatcher was compiled")
        .1
        .body
        .iter()
        .find_map(|statement| {
            if let Statement::Switch { cases, .. } = statement {
                Some(cases)
            } else {
                None
            }
        })
        .expect("a task dispatcher branches over kinds");
    let dispatched = cases
        .iter()
        .filter_map(|case| match case.value {
            SwitchValue::U32(code) => Some(code),
            _ => None,
        })
        .collect::<HashSet<_>>();
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
    let function = program
        .module()
        .functions
        .iter()
        .find(|(_, function)| function.name.as_deref() == Some("op_apply"))
        .expect("the Rust op dispatcher was compiled")
        .1;
    let cases = function
        .body
        .iter()
        .find_map(|statement| {
            if let Statement::Switch { cases, .. } = statement {
                Some(cases)
            } else {
                None
            }
        })
        .expect("the operation dispatcher matches every code");
    let defined = cases
        .iter()
        .filter(|case| case.value != SwitchValue::Default)
        .map(|case| case.value)
        .collect::<HashSet<_>>();
    assert_eq!(defined.len(), OPS.len());
    for op in OPS {
        assert!(defined.contains(&SwitchValue::U32(op.code)));
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
    assert!(
        !single
            .module()
            .functions
            .iter()
            .flat_map(|(_, function)| function.expressions.iter())
            .any(|(_, expr)| matches!(
                expr,
                Expression::Math {
                    fun: MathFunction::Unpack2x16float,
                    ..
                }
            ))
    );
    assert!(
        half.module()
            .functions
            .iter()
            .flat_map(|(_, function)| function.expressions.iter())
            .any(|(_, expr)| matches!(
                expr,
                Expression::Math {
                    fun: MathFunction::Unpack2x16float,
                    ..
                }
            ))
    );
    assert!(!functions(&half).contains("fetch_bfloat16"));
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
}

#[test]
fn matrix_specialization_contains_every_tile_in_the_profile() {
    for (index, profile) in profiles().iter().enumerate() {
        let kernel = all(index);
        let program = kernel.program();
        let names = functions(&program);
        for tile in 0..profile.tiles().len() {
            assert!(names.contains(format!("run_matmul_{tile}").as_str()));
            assert!(names.contains(format!("matmul_load_{tile}").as_str()));
        }
    }
}

#[test]
fn workgroup_allocation_fits_the_advertised_profile() {
    for (index, profile) in profiles().iter().enumerate() {
        let program = all(index).program();
        let used = program
            .module()
            .global_variables
            .iter()
            .filter(|(_, variable)| variable.space == AddressSpace::WorkGroup)
            .map(
                |(_, variable)| match &program.module().types[variable.ty].inner {
                    TypeInner::Array {
                        stride,
                        size: naga::ArraySize::Constant(count),
                        ..
                    } => u64::from(*stride) * u64::from(count.get()),
                    other => panic!("workgroup buffer {other:?} has no static size"),
                },
            )
            .sum::<u64>();
        assert!(used > 0);
        assert!(
            used <= profile.shared_bytes(),
            "{profile:?} reserves {used} workgroup bytes"
        );
    }
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
    let staged = program
        .module()
        .global_variables
        .iter()
        .filter(|(_, variable)| variable.name.as_deref() == Some("attention_left"))
        .map(
            |(_, variable)| match &program.module().types[variable.ty].inner {
                TypeInner::Array {
                    stride,
                    size: naga::ArraySize::Constant(count),
                    ..
                } => u64::from(*stride) * u64::from(count.get()),
                other => panic!("an attention stage {other:?} has no static size"),
            },
        )
        .sum::<u64>();
    assert_eq!(
        staged,
        u64::from(
            attention
                .iter()
                .map(|tile| tile.stage_words())
                .max()
                .expect("an attention stage"),
        ) * 4,
        "a device program stages the widest attention tile it carries",
    );
}
