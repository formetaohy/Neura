use naga::AddressSpace;
use neura_abi::op::OPS;
use neura_abi::{Geometry, Kind, PROFILES, Precision, Profile};
use neura_shader::{BINDINGS, Megakernel, reflect};
use std::collections::BTreeSet;

fn assemble(profile: Profile) -> Megakernel {
    assemble_with(profile, Precision::Single)
}

fn assemble_with(profile: Profile, weights: Precision) -> Megakernel {
    Megakernel::assemble(Geometry::of(profile, profile.ladder()), weights)
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
        assert_eq!(kernel.geometry().tiles(), profile.ladder());
    }
}

#[test]
fn every_profile_declares_its_own_device_constants() {
    for profile in PROFILES {
        let kernel = assemble(*profile);
        let declarations = Geometry::of(*profile, profile.ladder()).declarations();
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
        .map(|kind| neura_kernels::body(*kind))
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
                    neura_kernels::body(*kind),
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
        for (geometry, tile) in profile.ladder().iter().enumerate() {
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
            kernel.source().matches("workgroupBarrier()").count() >= profile.ladder().len() * 2,
            "a profile of {profile:?} stages its tiles without a barrier",
        );
    }
}

#[test]
fn a_program_carries_only_the_tiles_it_is_given() {
    let profile = PROFILES[PROFILES.len() - 1];
    let geometry = Geometry::of(profile, &profile.ladder()[..1]);
    let kernel = Megakernel::assemble(geometry.clone(), Precision::Single);
    assert!(
        kernel
            .source()
            .contains("fn run_matmul_0(task: Task, lid: u32)")
    );
    assert!(
        !kernel
            .source()
            .contains("fn run_matmul_1(task: Task, lid: u32)")
    );
    assert!(!kernel.source().contains("const MATMUL_ROWS_1"));
    assert!(
        kernel
            .source()
            .contains("case 0u: { run_matmul_0(task, lid); }")
    );
    assert!(!kernel.source().contains("case 1u:"));
    assert_eq!(kernel.geometry(), &geometry);
    assert!(
        kernel
            .source()
            .contains("matmul_left: array<f32, MATMUL_LEFT_STAGE>")
            && !kernel.source().contains("MATMUL_LEFT_STAGE_1")
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
fn the_devices_bound_every_tensor_the_tape_names() {
    let kernel = assemble(PROFILES[0]);
    assert!(kernel.source().contains("atomicAdd(&cursor["));
    assert!(kernel.source().contains("bounds.task_count"));
    assert!(kernel.source().contains("bounds.first_task"));
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
        let geometry = Geometry::of(*profile, profile.ladder());
        let kernel = Megakernel::assemble(geometry, Precision::Single);
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
    let kernel = Megakernel::assemble(Geometry::of(PROFILES[0], &[]), Precision::Single);
    assert!(workgroup_bytes(kernel.source()) >= 2 * u64::from(PROFILES[0].workgroup()) * 4);
}
