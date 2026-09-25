use naga::Statement;
use neura_compiler::{BindingSpec, Compiler, ComputeProgram};

mod unrolled {
    #[neura_compiler::module]
    mod source {
        #[neura_compiler::kernel]
        fn main(lid: u32) {
            for row in unroll(0u32, 4u32, 1u32) {
                let register = row * 4u32 + 1u32;
                output[register] = register;
            }
        }
    }
}

mod registers {
    #[neura_compiler::module]
    mod source {
        #[neura_compiler::kernel]
        fn main(lid: u32) {
            let mut registers = scalar_array(0.0, 4u32);
            for lane in unroll(0u32, 4u32, 1u32) {
                registers[lane] = f32(lane);
            }
            output[lid] = registers[2u32];
        }
    }
}

mod dynamic_register {
    #[neura_compiler::module]
    mod source {
        #[neura_compiler::kernel]
        fn main(lid: u32) {
            let mut registers = scalar_array(0.0, 4u32);
            registers[lid] = 1.0;
            output[lid] = registers[0u32];
        }
    }
}

mod boolean {
    #[neura_compiler::module]
    mod source {
        fn side_effect() -> bool {
            output[0u32] = 9u32;
            return true;
        }
        #[neura_compiler::kernel]
        fn main(lid: u32) {
            let first = lid == 0u32 && side_effect();
            let second = lid == 0u32 || side_effect();
            if first && second {
                output[1u32] = 1u32;
            }
        }
    }
}

mod divergent {
    #[neura_compiler::module]
    mod source {
        #[neura_compiler::kernel]
        fn main(lid: u32) {
            if lid == 0u32 {
                workgroup_barrier();
            }
            output[lid] = 0u32;
        }
    }
}

mod divergent_call {
    #[neura_compiler::module]
    mod source {
        fn synchronize() {
            workgroup_barrier();
        }
        #[neura_compiler::kernel]
        fn main(lid: u32) {
            if lid == 0u32 {
                synchronize();
            }
            output[lid] = 1u32;
        }
    }
}

mod uniform_call {
    #[neura_compiler::module]
    mod source {
        fn synchronize(lid: u32) {
            workgroup_barrier();
            if lid == 0u32 {
                output[0u32] = 1u32;
            }
        }
        #[neura_compiler::kernel]
        fn main(lid: u32) {
            synchronize(lid);
        }
    }
}

mod loop_update {
    #[neura_compiler::module]
    mod source {
        #[neura_compiler::kernel]
        fn main(lid: u32) {
            let mut value = 0u32;
            for turn in stride(0u32, 2u32, 1u32) {
                if value != 0u32 {
                    workgroup_barrier();
                }
                value = lid;
            }
            output[lid] = value;
        }
    }
}

mod loop_divergence {
    #[neura_compiler::module]
    mod source {
        #[neura_compiler::kernel]
        fn main(lid: u32) {
            for turn in stride(lid, 2u32, 1u32) {
                workgroup_barrier();
            }
            output[lid] = 1u32;
        }
    }
}

mod readonly {
    #[neura_compiler::module]
    mod source {
        #[neura_compiler::kernel]
        fn main(lid: u32) {
            input[lid] = 1u32;
        }
    }
}

mod recursive {
    #[neura_compiler::module]
    mod source {
        fn recursion() {
            recursion();
        }
        #[neura_compiler::kernel]
        fn main(lid: u32) {
            recursion();
            output[lid] = 1u32;
        }
    }
}

mod unbound {
    #[neura_compiler::module]
    mod source {
        #[neura_compiler::kernel]
        fn main(lid: u32) {
            output[lid] = missing;
        }
    }
}

fn compiled_u32(label: &str, define: fn(&mut Compiler)) -> ComputeProgram {
    let mut compiler = Compiler::new();
    compiler.storage_array("output", "u32", BindingSpec::writable_storage(0));
    define(&mut compiler);
    compiler.finish(label, "main", 64)
}

fn compiled_f32(label: &str, define: fn(&mut Compiler)) -> ComputeProgram {
    let mut compiler = Compiler::new();
    compiler.storage_array("output", "f32", BindingSpec::writable_storage(0));
    define(&mut compiler);
    compiler.finish(label, "main", 64)
}

#[test]
fn compile_time_unrolling_folds_register_indices_into_literals() {
    let program = compiled_u32("unrolled", unrolled::define);
    let function = &program.module().entry_points[0].function;
    assert!(function.local_variables.is_empty());
    assert_eq!(
        function
            .body
            .iter()
            .filter(|statement| matches!(statement, Statement::Block(_)))
            .count(),
        4
    );
}

#[test]
fn scalar_registers_are_named_naga_locals_instead_of_indexed_arrays() {
    let program = compiled_f32("scalar registers", registers::define);
    let function = &program.module().entry_points[0].function;
    assert_eq!(function.local_variables.len(), 4);
    for (index, (_, local)) in function.local_variables.iter().enumerate() {
        assert_eq!(
            local.name.as_deref(),
            Some(format!("registers_{index}").as_str())
        );
        assert!(matches!(
            program.module().types[local.ty].inner,
            naga::TypeInner::Scalar(_)
        ));
    }
}

#[test]
#[should_panic(expected = "a scalar register requires a compile-time index")]
fn scalar_registers_refuse_runtime_indices() {
    compiled_f32("nonconstant register", dynamic_register::define);
}

#[test]
fn rust_boolean_short_circuit_keeps_device_side_effects_in_its_branch() {
    let program = compiled_u32("short circuit", boolean::define);
    let body = &program.module().entry_points[0].function.body;
    let branches = body
        .iter()
        .filter_map(|statement| match statement {
            Statement::If { accept, reject, .. } => Some((accept, reject)),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert!(
        branches[0]
            .0
            .iter()
            .any(|statement| matches!(statement, Statement::Call { .. }))
    );
    assert!(branches[0].1.is_empty());
    assert!(branches[1].0.is_empty());
    assert!(
        branches[1]
            .1
            .iter()
            .any(|statement| matches!(statement, Statement::Call { .. }))
    );
}

#[test]
#[should_panic(expected = "a workgroup barrier is reached by different invocations")]
fn a_nonuniform_workgroup_barrier_is_rejected() {
    compiled_u32("divergent barrier", divergent::define);
}

#[test]
#[should_panic(expected = "through a Rust device call")]
fn a_barrier_cannot_be_hidden_behind_a_divergent_call() {
    compiled_u32("divergent call", divergent_call::define);
}

#[test]
fn a_uniform_barrier_accepts_lane_specific_data_in_a_callee() {
    let program = compiled_u32("uniform call", uniform_call::define);
    assert!(program.spirv().len() > 5);
}

#[test]
#[should_panic(expected = "a workgroup barrier")]
fn uniformity_is_checked_again_at_the_next_loop_iteration() {
    compiled_u32("loop uniformity", loop_update::define);
}

#[test]
#[should_panic(expected = "a workgroup barrier")]
fn divergent_loop_counts_cannot_hide_a_barrier() {
    compiled_u32("loop counts", loop_divergence::define);
}

#[test]
#[should_panic(expected = "invalid device program")]
fn a_readonly_binding_cannot_be_written() {
    let mut compiler = Compiler::new();
    compiler.storage_array("input", "u32", BindingSpec::storage(0));
    readonly::define(&mut compiler);
    compiler.finish("readonly output", "main", 64);
}

#[test]
#[should_panic(expected = "recursive Rust device functions")]
fn recursive_device_calls_are_refused() {
    compiled_u32("recursive", recursive::define);
}

#[test]
#[should_panic(expected = "Rust device name missing is not defined")]
fn an_unbound_rust_name_fails_at_its_use() {
    compiled_u32("unbound", unbound::define);
}
