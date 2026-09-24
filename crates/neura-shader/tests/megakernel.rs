use naga::AddressSpace;
use neura_abi::Kind;
use neura_gpu::{Backend, ShaderTranslation};
use neura_op::OPS;
use neura_precision::Precision;
use neura_profile::{Geometry, PROFILES, Profile};
use neura_shader::{BINDINGS, Megakernel, reflect};
use std::collections::BTreeSet;

fn assemble(profile: Profile) -> Megakernel {
    assemble_with(profile, Precision::Single)
}

fn assemble_with(profile: Profile, weights: Precision) -> Megakernel {
    Megakernel::assemble(Kind::ALL, Geometry::of(profile), weights)
}

fn assemble_carrying(profile: Profile, kinds: &[Kind]) -> Megakernel {
    Megakernel::assemble(kinds, Geometry::of(profile), Precision::Single)
}

#[test]
fn every_native_compiler_lowers_the_complete_compute_vocabulary() {
    let kernel = assemble(PROFILES[0]);
    let program = kernel.program();
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
fn every_profile_assembles_a_program_that_declares_the_framework_bindings() {
    for profile in PROFILES {
        let kernel = assemble(*profile);
        assert!(kernel.source().contains("@compute"));
        assert!(
            kernel
                .source()
                .contains(&format!("@workgroup_size({}u)", profile.workgroup()))
                || kernel.source().contains("@workgroup_size(WORKGROUP_SIZE)")
        );
        let reflected = reflect(kernel.source());
        assert_eq!(reflected.len(), BINDINGS.len());
        for declared in BINDINGS {
            let binding = reflected
                .iter()
                .find(|binding| binding.binding == declared.binding)
                .expect("every declared binding is reflected");
            assert_eq!(binding.name, declared.name);
            assert_eq!(binding.kind, declared.kind);
        }
        assert_eq!(kernel.bindings(), reflected.as_slice());
        assert_eq!(kernel.workgroup_size(), profile.workgroup());
        assert_eq!(kernel.geometry().tiles(), profile.tiles());
    }
}

#[test]
fn every_profile_declares_its_own_device_constants() {
    for profile in PROFILES {
        let kernel = assemble(*profile);
        let declarations = Geometry::of(*profile).declarations();
        for declaration in declarations.lines() {
            assert!(
                kernel.source().contains(declaration),
                "a device program misses {declaration}",
            );
        }
    }
}

#[test]
fn the_program_declares_the_taxonomy_it_dispatches() {
    let kernel = assemble(PROFILES[0]);
    for kind in Kind::ALL {
        assert!(
            kernel.source().contains(&format!(
                "const {}: u32 = {}u;",
                kind.constant(),
                kind.code()
            )),
            "a device program never declares the {} task",
            kind.name(),
        );
    }
    for op in OPS {
        assert!(
            kernel
                .source()
                .contains(&format!("const {}: u32 = {}u;", op.constant, op.code)),
            "a device program never declares the {} op",
            op.name,
        );
    }
}

#[test]
fn every_declared_kind_has_a_body() {
    let covered = Kind::ALL
        .iter()
        .map(|kind| neura_kernel::body(*kind))
        .collect::<BTreeSet<_>>();
    assert_eq!(
        covered.len(),
        Kind::ALL.len(),
        "every kind the tape can name needs a body of its own the megakernel can run",
    );
}

#[test]
fn the_megakernel_dispatches_every_kind_by_its_declared_constant() {
    for profile in PROFILES {
        let kernel = assemble(*profile);
        for kind in Kind::ALL {
            assert!(
                kernel
                    .source()
                    .contains(&format!("case {}:", kind.constant())),
                "the megakernel never dispatches the {} task",
                kind.name(),
            );
            assert!(
                kernel.source().contains(&format!(
                    "case {}: {{ {}(task, lid); }}",
                    kind.constant(),
                    neura_kernel::body(*kind),
                )),
                "the megakernel never runs the {} body",
                kind.name(),
            );
        }
    }
}

#[test]
fn the_device_applies_every_declared_op() {
    let kernel = assemble(PROFILES[0]);
    let head = "fn run_partial(task: Task, lid: u32) {";
    let rest = &kernel.source()[kernel.source().find(head).expect(head) + head.len()..];
    let partials = &rest[..rest
        .find(
            "
fn ",
        )
        .expect("a partial kernel ends")];
    for op in OPS {
        assert!(
            kernel
                .source()
                .contains(&format!("case {}: {{ return {}; }}", op.constant, op.apply)),
            "the device never applies the {} op",
            op.name,
        );
        for slot in 0..op.family.operands() {
            let ordinal = op.code * 2 + slot;
            let arm = format!("case {ordinal}u: {{");
            let Some(formula) = op.partial(slot).formula() else {
                assert!(
                    !partials.contains(&arm),
                    "the device carries a partial for {} operand {slot} the host never asks for",
                    op.name,
                );
                continue;
            };
            let start = partials.find(&arm).unwrap_or_else(|| {
                panic!(
                    "the device never differentiates {} over operand {slot}",
                    op.name
                )
            }) + arm.len();
            let case = &partials[start..];
            let case = &case[..case.find(" }").expect("every partial case ends")];
            assert!(
                case.contains(&format!("result = {formula};")),
                "the device misses the {formula} of {} over operand {slot}",
                op.name,
            );
            for role in op.partial(slot).roles() {
                assert!(
                    case.contains(&format!("let {} = fetch(", role.name())),
                    "the device never hands {} the {} its partial reads",
                    op.name,
                    role.name(),
                );
            }
        }
        assert_eq!(
            op.partials.len() as u32,
            op.family.operands(),
            "the {} op declares a partial for every operand it reads",
            op.name,
        );
    }
}

#[test]
fn a_profile_generates_a_body_for_every_tile_it_carries() {
    for profile in PROFILES {
        let kernel = assemble(*profile);
        let staging = profile.shared_bytes() - u64::from(profile.workgroup()) * 4;
        assert!(staging > 0);
        assert!(
            kernel
                .source()
                .contains("matmul_left: array<f32, MATMUL_LEFT_STAGE>"),
            "a profile of {profile:?} stages no left tile",
        );
        assert!(
            kernel
                .source()
                .contains("matmul_right: array<f32, MATMUL_RIGHT_STAGE>"),
            "a profile of {profile:?} stages no right tile",
        );
        assert!(
            kernel
                .source()
                .contains("fn run_matmul(task: Task, lid: u32)")
        );
        for (geometry, tile) in profile.tiles().iter().enumerate() {
            assert!(
                kernel.source().contains(&format!(
                    "case {geometry}u: {{ run_matmul_{geometry}(task, lid); }}"
                )),
                "the device never dispatches geometry {geometry} of {profile:?}",
            );
            assert!(
                kernel
                    .source()
                    .contains(&format!("fn run_matmul_{geometry}(task: Task, lid: u32)")),
                "a profile of {profile:?} carries no body for geometry {geometry}",
            );
            for register in (0..tile.registers()).step_by(tile.register_columns() as usize) {
                assert!(
                    kernel
                        .source()
                        .contains(&format!("var acc{register} = 0.0;")),
                    "geometry {geometry} of {profile:?} carries no register block",
                );
            }
        }
        assert!(kernel.source().contains(&format!(
            "default: {{ refuse({}, task.geometry); }}",
            Kind::Matmul.constant(),
        )));
        assert!(
            kernel.source().matches("workgroupBarrier()").count() >= profile.tiles().len() * 2,
            "a profile of {profile:?} stages its tiles without a barrier",
        );
    }
}

#[test]
fn a_program_carries_every_tile_its_profile_offers() {
    let profile = PROFILES[PROFILES.len() - 1];
    let kernel = assemble(profile);
    for geometry in 0..profile.tiles().len() {
        assert!(
            kernel
                .source()
                .contains(&format!("fn run_matmul_{geometry}(task: Task, lid: u32)"))
                && kernel.source().contains(&format!(
                    "case {geometry}u: {{ run_matmul_{geometry}(task, lid); }}"
                ))
                && kernel
                    .source()
                    .contains(&format!("const MATMUL_ROWS_{geometry}")),
            "a program that serves every shape carries geometry {geometry} of {profile:?}",
        );
    }
    assert!(
        kernel
            .source()
            .contains("matmul_left: array<f32, MATMUL_LEFT_STAGE>")
    );
}

#[test]
fn a_program_carries_only_the_kinds_its_plan_names() {
    let profile = PROFILES[PROFILES.len() - 1];
    let kernel = assemble_carrying(profile, &[Kind::Matmul]);
    assert_eq!(kernel.kinds(), &[Kind::Matmul]);
    let dispatch = kernel
        .source()
        .split_once("fn run_task(index: u32, lid: u32)")
        .expect("the assembled program dispatches its kinds")
        .1;
    assert!(dispatch.contains(&format!(
        "case {}: {{ run_matmul(task, lid); }}",
        Kind::Matmul.constant(),
    )));
    assert!(!dispatch.contains(&format!("case {}:", Kind::Binary.constant())));
    assert!(!dispatch.contains(&format!("case {}:", Kind::MatmulFold.constant())));
    for absent in [
        "run_conv2d",
        "run_softmax",
        "run_scatter",
        "run_gather",
        "run_argmax",
        "reduction_scratch",
        "choice_index",
    ] {
        assert!(
            !kernel.source().contains(absent),
            "a program that runs one kind carries {absent}",
        );
    }
    let whole = assemble(profile);
    assert!(
        kernel.source().len() < whole.source().len(),
        "a program of one kind holds {} bytes where the whole vocabulary holds {}",
        kernel.source().len(),
        whole.source().len(),
    );
}

#[test]
fn the_program_identity_is_stable_for_every_profile() {
    for profile in PROFILES {
        let first = assemble(*profile);
        let second = assemble(*profile);
        assert_eq!(first.source(), second.source());
        assert_eq!(first.program(), second.program());
    }
}

#[test]
fn two_profiles_are_two_programs() {
    let mut seen = Vec::new();
    for profile in PROFILES {
        let program = assemble(*profile).program();
        assert!(
            !seen.contains(&program),
            "two profiles were handed one device program",
        );
        seen.push(program);
    }
}

#[test]
fn the_program_hands_the_device_one_group_of_six_buffers() {
    let kernel = assemble(PROFILES[0]);
    let program = kernel.program();
    assert_eq!(program.bindings().len(), BINDINGS.len());
    assert!(
        program
            .bindings()
            .iter()
            .filter(|binding| binding.dynamic_offset)
            .count()
            == 1
    );
}

#[test]
fn half_weights_unpack_where_single_weights_load() {
    let single = assemble_with(PROFILES[0], Precision::Single);
    let half = assemble_with(PROFILES[0], Precision::Half);
    assert!(single.source().contains(
        "fn fetch(value: Value, offset: u32) -> f32 {
    return heap[base_of(value) + value.base + offset];"
    ));
    assert!(!single.source().contains("unpack2x16float"));
    assert!(
        half.source()
            .contains("heap[placement.weights + (element >> 1u)]")
    );
    assert!(half.source().contains("unpack2x16float"));
    for kernel in [&single, &half] {
        assert!(
            kernel
                .source()
                .contains("fn publish(value: Value, offset: u32, data: f32)")
        );
        assert!(
            kernel
                .source()
                .contains("fn base_of(value: Value) -> u32 {"),
            "every device program is handed the stores it addresses instead of carrying them",
        );
    }
}

#[test]
fn every_kernel_body_carries_its_chain() {
    let kernel = assemble(PROFILES[0]);
    let chained = kernel.source().matches("chained(task,").count();
    assert!(
        chained >= Kind::ALL.len() - 3,
        "every elementwise body ends in its chain, and a reduction does not",
    );
}

#[test]
fn the_entry_hands_each_workgroup_one_segment_of_its_dispatch() {
    let kernel = assemble(PROFILES[0]);
    let entry = kernel
        .source()
        .split_once("fn main(")
        .expect("the assembled program carries its entry point")
        .1;
    assert!(entry.contains("@builtin(workgroup_id) group: vec3<u32>"));
    assert!(entry.contains("segments[bounds.first_segment + group.x]"));
    assert!(entry.contains("storageBarrier()"));
    assert!(
        !entry.contains("atomicAdd") && !entry.contains("workgroupBarrier()"),
        "the entry claims no work of its own: a claim on the device lands in control flow no backend proves uniform",
    );
    assert!(
        !entry.contains("var<workgroup>"),
        "the entry carries no workgroup state between the segments it runs",
    );
}

#[test]
fn the_device_program_reads_a_lane_by_name_and_never_by_runtime_index() {
    for profile in PROFILES {
        let kernel = assemble(*profile);
        let module =
            naga::front::wgsl::parse_str(kernel.source()).expect("the assembled program parses");
        for (_, function) in module.functions.iter() {
            let context = naga::proc::ResolveContext::with_locals(
                &module,
                &function.local_variables,
                &function.arguments,
            );
            let mut resolved = Vec::with_capacity(function.expressions.len());
            for (handle, expression) in function.expressions.iter() {
                let resolution = context
                    .resolve(expression, |base| {
                        resolved
                            .get(base.index())
                            .ok_or(naga::proc::ResolveError::InvalidAccess {
                                expr: base,
                                indexed: true,
                            })
                    })
                    .expect("every expression of a validated program resolves");
                if let naga::Expression::Access { base, .. } = expression {
                    assert!(
                        !matches!(
                            resolved[base.index()].inner_with(&module.types),
                            naga::TypeInner::Vector { .. }
                        ),
                        "expression {handle:?} of {function:?} indexes a vector by a runtime lane, and the dx12 compiler lowers a program of this size that carries one to no code at all",
                        function = function.name,
                    );
                }
                resolved.push(resolution);
            }
        }
    }
}

fn workgroup_bytes(source: &str) -> u64 {
    let module = naga::front::wgsl::parse_str(source).expect("the assembled program parses");
    module
        .global_variables
        .iter()
        .filter(|(_, variable)| variable.space == AddressSpace::WorkGroup)
        .map(|(_, variable)| type_bytes(&module, variable.ty))
        .sum()
}

fn type_bytes(module: &naga::Module, ty: naga::Handle<naga::Type>) -> u64 {
    match &module.types[ty].inner {
        naga::TypeInner::Scalar(scalar) => u64::from(scalar.width),
        naga::TypeInner::Vector { size, scalar } => {
            let lanes = match size {
                naga::VectorSize::Bi => 2u64,
                naga::VectorSize::Tri => 3,
                naga::VectorSize::Quad => 4,
            };
            lanes * u64::from(scalar.width)
        }
        naga::TypeInner::Array { base, size, .. } => match size {
            naga::ArraySize::Constant(count) => u64::from(count.get()) * type_bytes(module, *base),
            other => panic!("a workgroup array of a {other:?} size holds no bytes to promise"),
        },
        naga::TypeInner::Struct { span, .. } => u64::from(*span),
        other => panic!("a workgroup binding of {other:?} holds no bytes to promise"),
    }
}

#[test]
fn the_profile_carries_the_workgroup_memory_its_program_declares() {
    for profile in PROFILES {
        let geometry = Geometry::of(*profile);
        let kernel = Megakernel::assemble(Kind::ALL, geometry, Precision::Single);
        let declared = workgroup_bytes(kernel.source());
        assert!(
            declared > 0,
            "{profile:?} declares no workgroup memory at all",
        );
        assert!(
            profile.shared_bytes() >= declared,
            "{profile:?} promises {} workgroup bytes where its program declares {declared}",
            profile.shared_bytes(),
        );
    }
}

#[test]
fn the_reductions_and_choices_keep_a_scratch_of_their_own() {
    let kernel = assemble(PROFILES[0]);
    assert!(workgroup_bytes(kernel.source()) >= 2 * u64::from(PROFILES[0].workgroup()) * 4);
}
