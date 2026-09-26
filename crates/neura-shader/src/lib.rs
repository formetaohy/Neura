#[path = "../device/core.rs"]
mod core;

use neura_abi::{Element, Kind, NO_VALUE, RECORDS, Refusal, TENSOR, refusal, store, strategy};
use neura_compiler::{BindingKind, BindingSpec, Compiler, ComputeProgram, ShaderBinding};
use neura_profile::Geometry;

pub use core::ENTRY;

pub const TASKS: u32 = 0;
pub const VALUES: u32 = 1;
pub const HEAP: u32 = 2;
pub const REFUSAL: u32 = 3;
pub const BOUNDS: u32 = 4;
pub const STEPS: u32 = 5;
pub const PLACEMENT: u32 = 6;
pub const SEGMENTS: u32 = 7;

pub struct KernelBinding {
    pub binding: u32,
    pub kind: BindingKind,
    pub dynamic_offset: bool,
    pub name: &'static str,
    pub element: &'static str,
    pub array: bool,
}

pub const BINDINGS: &[KernelBinding] = &[
    KernelBinding {
        binding: TASKS,
        kind: BindingKind::ReadOnlyStorage,
        dynamic_offset: false,
        name: "tasks",
        element: "Task",
        array: true,
    },
    KernelBinding {
        binding: VALUES,
        kind: BindingKind::ReadOnlyStorage,
        dynamic_offset: false,
        name: "values",
        element: "Value",
        array: true,
    },
    KernelBinding {
        binding: HEAP,
        kind: BindingKind::ReadWriteStorage,
        dynamic_offset: false,
        name: "heap",
        element: "f32",
        array: true,
    },
    KernelBinding {
        binding: REFUSAL,
        kind: BindingKind::ReadWriteStorage,
        dynamic_offset: false,
        name: "refusal",
        element: "AtomicU32",
        array: true,
    },
    KernelBinding {
        binding: BOUNDS,
        kind: BindingKind::ReadOnlyStorage,
        dynamic_offset: true,
        name: "bounds",
        element: "Bounds",
        array: false,
    },
    KernelBinding {
        binding: STEPS,
        kind: BindingKind::ReadOnlyStorage,
        dynamic_offset: false,
        name: "steps",
        element: "Step",
        array: true,
    },
    KernelBinding {
        binding: PLACEMENT,
        kind: BindingKind::ReadOnlyStorage,
        dynamic_offset: false,
        name: "placement",
        element: "Placement",
        array: false,
    },
    KernelBinding {
        binding: SEGMENTS,
        kind: BindingKind::ReadOnlyStorage,
        dynamic_offset: false,
        name: "segments",
        element: "Segment",
        array: true,
    },
];

pub struct Megakernel {
    kinds: Vec<Kind>,
    geometry: Geometry,
    program: ComputeProgram,
}

impl Megakernel {
    pub fn assemble(kinds: &[Kind], elements: &[Element], geometry: Geometry) -> Self {
        assert!(
            !kinds.is_empty(),
            "a device program with no task has nothing to run"
        );
        assert!(
            !elements.is_empty(),
            "a device program with no tensor has nothing to address"
        );
        let mut compiler = Compiler::new();
        for record in RECORDS {
            compiler.record(*record);
        }
        compiler.constant("WORKGROUP_SIZE", geometry.workgroup());
        compiler.constant("NO_VALUE", NO_VALUE);
        compiler.constant("refusal::TENSOR", TENSOR);
        compiler.constant("refusal::KIND_BITS", refusal::KIND_BITS);
        compiler.constant("refusal::CODE_BITS", refusal::CODE_BITS);
        for refusal in Refusal::ALL {
            compiler.constant(
                &format!("refusal::{}", refusal.name().to_uppercase()),
                refusal.code(),
            );
        }
        compiler.constant("store::TENSORS", store::TENSORS);
        compiler.constant("store::WEIGHTS", store::WEIGHTS);
        for element in Element::ALL {
            compiler.constant(element.symbol(), element.code());
        }
        for (name, value) in [
            ("strategy::THREAD_ROW", strategy::THREAD_ROW),
            ("strategy::WORKGROUP_ROW", strategy::WORKGROUP_ROW),
            ("strategy::THREAD_ELEMENT", strategy::THREAD_ELEMENT),
            ("strategy::WEIGHT_CHUNK", strategy::WEIGHT_CHUNK),
            ("strategy::WEIGHT_FOLD", strategy::WEIGHT_FOLD),
        ] {
            compiler.constant(name, value);
        }
        for kind in Kind::ALL {
            compiler.constant(kind.symbol(), kind.code());
        }
        for binding in BINDINGS {
            let spec = BindingSpec {
                binding: binding.binding,
                kind: binding.kind,
                dynamic_offset: binding.dynamic_offset,
            };
            if binding.array {
                compiler.storage_array(binding.name, binding.element, spec);
            } else {
                compiler.storage_record(binding.name, binding.element, spec);
            }
        }
        core::define(&mut compiler);
        neura_kernel::define(&mut compiler, kinds, elements, &geometry);
        let enabled = kinds
            .iter()
            .map(|kind| kind.symbol().to_owned())
            .collect::<Vec<_>>();
        compiler.retain_cases("run_task", &enabled);
        let program = compiler.finish(
            &format!(
                "neura megakernel {} threads over {} tiles and {} kinds",
                geometry.workgroup(),
                geometry.tiles().len(),
                kinds.len()
            ),
            ENTRY,
            geometry.workgroup(),
        );
        for (expected, reflected) in BINDINGS.iter().zip(program.reflected()) {
            assert_eq!(
                expected.name, reflected.name,
                "a device binding has an incorrect name"
            );
        }
        Self {
            kinds: kinds.to_vec(),
            geometry,
            program,
        }
    }

    pub fn kinds(&self) -> &[Kind] {
        &self.kinds
    }

    pub fn geometry(&self) -> &Geometry {
        &self.geometry
    }

    pub fn bindings(&self) -> &[ShaderBinding] {
        self.program.reflected()
    }

    pub fn workgroup_size(&self) -> u32 {
        self.program.workgroup_size()
    }

    pub fn program(&self) -> ComputeProgram {
        self.program.clone()
    }
}
