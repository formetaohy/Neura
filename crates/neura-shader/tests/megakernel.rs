use neura_shader::{BINDINGS, Megakernel, reflect};
use std::collections::BTreeSet;

#[test]
fn the_device_program_parses_and_declares_the_framework_bindings() {
    let kernel = Megakernel::assemble();
    assert!(kernel.source().contains("@compute"));
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
}

#[test]
fn the_megakernel_dispatches_every_kind_by_its_declared_constant() {
    let kernel = Megakernel::assemble();
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

#[test]
fn the_program_identity_is_stable() {
    let first = Megakernel::assemble();
    let second = Megakernel::assemble();
    assert_eq!(first.source(), second.source());
    assert_eq!(first.program(), second.program());
}

#[test]
fn the_program_hands_the_device_one_group_of_six_buffers() {
    let kernel = Megakernel::assemble();
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
    let kernel = Megakernel::assemble();
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
    let kernel = Megakernel::assemble();
    let chained = kernel.source().matches("chained(task,").count();
    assert_eq!(
        chained,
        neura_kernels::KERNELS.len() - 1,
        "every elementwise body ends in its chain, and a reduction does not",
    );
}

#[test]
fn the_devices_bound_every_tensor_the_tape_names() {
    let kernel = Megakernel::assemble();
    assert_eq!(kernel.workgroup_size(), 64);
    assert!(kernel.source().contains("atomicAdd(&cursor["));
    assert!(kernel.source().contains("bounds.task_count"));
    assert!(kernel.source().contains("bounds.first_task"));
}
