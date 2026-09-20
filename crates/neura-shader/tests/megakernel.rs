use neura_abi::{Geometry, KIND_COUNT, PROFILES, Profile};
use neura_shader::{BINDINGS, Megakernel, reflect};
use std::collections::BTreeSet;

fn assemble(profile: Profile) -> Megakernel {
    Megakernel::assemble(Geometry::of(profile, profile.ladder()))
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
fn every_declared_kind_has_a_body() {
    let covered = neura_kernels::KERNELS
        .iter()
        .map(|kernel| kernel.kind)
        .collect::<BTreeSet<_>>();
    let declared = (0..neura_kernels::kind_count()).collect::<BTreeSet<_>>();
    assert_eq!(
        covered, declared,
        "every kind the tape can name needs a body the megakernel can run",
    );
    assert_eq!(declared.len() as u32, KIND_COUNT);
}

#[test]
fn the_megakernel_dispatches_every_kind_by_its_declared_constant() {
    for profile in PROFILES {
        let kernel = assemble(*profile);
        for body in neura_kernels::KERNELS {
            assert!(
                kernel
                    .source()
                    .contains(&format!("case {}:", body.constant)),
                "the megakernel never dispatches {}",
                body.body,
            );
            assert!(
                kernel
                    .source()
                    .contains(&format!("{}(task, lid)", body.body)),
                "the megakernel never calls {}",
                body.body,
            );
        }
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
        assert!(
            kernel
                .source()
                .contains("default: { refuse(KIND_MATMUL, task.geometry); }")
        );
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
    let kernel = Megakernel::assemble(geometry.clone());
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
    assert_eq!(program.bindings().len(), 6);
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
fn the_device_chain_applies_every_declared_chain_op() {
    let kernel = assemble(PROFILES[0]);
    assert_eq!(
        neura_kernels::CHAIN_OPS.len() as u32,
        neura_abi::CHAIN_COUNT
    );
    for op in neura_kernels::CHAIN_OPS {
        assert!(
            kernel.source().contains(&format!("case {op}:")),
            "the device chain never applies {op}",
        );
    }
}

#[test]
fn every_kernel_body_carries_its_chain() {
    let kernel = assemble(PROFILES[0]);
    let chained = kernel.source().matches("chained(task,").count();
    assert!(
        chained >= neura_kernels::KERNELS.len() - 1,
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
