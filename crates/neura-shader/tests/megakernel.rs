use neura_abi::{KIND_COUNT, SCHEDULES, Schedule};
use neura_shader::{BINDINGS, Megakernel, reflect};
use std::collections::BTreeSet;

fn assemble(schedule: Schedule) -> Megakernel {
    Megakernel::assemble(schedule)
}

#[test]
fn every_schedule_assembles_a_program_that_declares_the_framework_bindings() {
    for schedule in SCHEDULES {
        let kernel = assemble(*schedule);
        assert!(kernel.source().contains("@compute"));
        assert!(
            kernel
                .source()
                .contains(&format!("@workgroup_size({}u)", schedule.workgroup()))
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
        assert_eq!(kernel.workgroup_size(), schedule.workgroup());
        assert_eq!(kernel.schedule(), *schedule);
    }
}

#[test]
fn every_schedule_declares_its_own_device_constants() {
    for schedule in SCHEDULES {
        let kernel = assemble(*schedule);
        for declaration in schedule.declarations().lines() {
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
    for schedule in SCHEDULES {
        let kernel = assemble(*schedule);
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
fn a_schedule_sizes_the_matmul_tile_it_generates() {
    for schedule in SCHEDULES {
        let kernel = assemble(*schedule);
        let matmul = schedule.matmul();
        assert!(
            kernel
                .source()
                .contains("array<f32, 2u * MATMUL_ROW_TILE * MATMUL_DEPTH_TILE>"),
            "a matmul of {schedule:?} stages no left tile",
        );
        assert!(
            kernel
                .source()
                .contains("array<f32, 2u * MATMUL_DEPTH_TILE * MATMUL_COL_TILE>"),
            "a matmul of {schedule:?} stages no right tile",
        );
        for register in (0..matmul.registers()).step_by(matmul.register_columns() as usize) {
            assert!(
                kernel
                    .source()
                    .contains(&format!("var acc{register} = 0.0;")),
                "a matmul of {schedule:?} carries no register block",
            );
        }
        assert!(
            kernel.source().matches("workgroupBarrier()").count() >= 2,
            "a matmul of {schedule:?} stages its tiles without a barrier",
        );
    }
}

#[test]
fn the_program_identity_is_stable_for_every_schedule() {
    for schedule in SCHEDULES {
        let first = assemble(*schedule);
        let second = assemble(*schedule);
        assert_eq!(first.source(), second.source());
        assert_eq!(first.program(), second.program());
    }
}

#[test]
fn two_schedules_are_two_programs() {
    let mut seen = Vec::new();
    for schedule in SCHEDULES {
        let program = assemble(*schedule).program();
        assert!(
            !seen.contains(&program),
            "two schedules were handed one device program",
        );
        seen.push(program);
    }
}

#[test]
fn the_program_hands_the_device_one_group_of_six_buffers() {
    let kernel = assemble(SCHEDULES[0]);
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
    let kernel = assemble(SCHEDULES[0]);
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
    let kernel = assemble(SCHEDULES[0]);
    let chained = kernel.source().matches("chained(task,").count();
    assert!(
        chained >= neura_kernels::KERNELS.len() - 1,
        "every elementwise body ends in its chain, and a reduction does not",
    );
}

#[test]
fn the_devices_bound_every_tensor_the_tape_names() {
    let kernel = assemble(SCHEDULES[0]);
    assert!(kernel.source().contains("atomicAdd(&cursor["));
    assert!(kernel.source().contains("bounds.task_count"));
    assert!(kernel.source().contains("bounds.first_task"));
}
