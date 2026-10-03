mod authored;
mod element;
mod pointwise;
mod scheduler;
mod substrate;
mod task;

use neura_abi::{
    DeviceModule, Element, FP4_BLOCK, INT4_BLOCK, Kind, NO_VALUE, RECORDS, Refusal, TENSOR,
    measure, progress, refusal, split, store, strategy,
};
use neura_compiler::{Compiler, ast};
use neura_profile::{CLAIM_BYTES, Geometry};
use neura_shader::{BindingKind, BindingSpec, ComputeProgram, ShaderBinding};

pub const TASKS: u32 = 0;
pub const VALUES: u32 = 1;
pub const HEAP: u32 = 2;
pub const REFUSAL: u32 = 3;
pub const PROGRESS: u32 = 4;
pub const STEPS: u32 = 5;
pub const PLACEMENT: u32 = 6;
pub const SEGMENTS: u32 = 7;
pub const EXTENTS: u32 = 8;
pub const MEASURES: u32 = 9;
pub const PATCHES: u32 = 10;
pub const PATCH_LIST: u32 = 11;

#[derive(Clone, Copy)]
pub struct KernelBinding {
    pub binding: u32,
    pub kind: BindingKind,
    pub name: &'static str,
    pub element: &'static str,
    pub array: bool,
}

pub const BINDINGS: &[KernelBinding] = &[
    KernelBinding {
        binding: TASKS,
        kind: BindingKind::ReadOnlyStorage,
        name: "tasks",
        element: "Task",
        array: true,
    },
    KernelBinding {
        binding: VALUES,
        kind: BindingKind::ReadOnlyStorage,
        name: "values",
        element: "Value",
        array: true,
    },
    KernelBinding {
        binding: HEAP,
        kind: BindingKind::ReadWriteStorage,
        name: "heap",
        element: "f32",
        array: true,
    },
    KernelBinding {
        binding: REFUSAL,
        kind: BindingKind::ReadWriteStorage,
        name: "refusal",
        element: "AtomicU32",
        array: true,
    },
    KernelBinding {
        binding: PROGRESS,
        kind: BindingKind::ReadWriteStorage,
        name: "progress",
        element: "AtomicU32",
        array: true,
    },
    KernelBinding {
        binding: STEPS,
        kind: BindingKind::ReadOnlyStorage,
        name: "steps",
        element: "Step",
        array: true,
    },
    KernelBinding {
        binding: PLACEMENT,
        kind: BindingKind::ReadOnlyStorage,
        name: "placement",
        element: "Placement",
        array: false,
    },
    KernelBinding {
        binding: SEGMENTS,
        kind: BindingKind::ReadOnlyStorage,
        name: "segments",
        element: "Segment",
        array: true,
    },
];

pub const AUTHORED_BINDINGS: &[KernelBinding] = &[
    KernelBinding {
        binding: EXTENTS,
        kind: BindingKind::ReadWriteStorage,
        name: "extents",
        element: "u32",
        array: true,
    },
    KernelBinding {
        binding: MEASURES,
        kind: BindingKind::ReadOnlyStorage,
        name: "measures",
        element: "Measure",
        array: true,
    },
    KernelBinding {
        binding: PATCHES,
        kind: BindingKind::ReadOnlyStorage,
        name: "patches",
        element: "Patch",
        array: true,
    },
    KernelBinding {
        binding: PATCH_LIST,
        kind: BindingKind::ReadOnlyStorage,
        name: "patch_list",
        element: "u32",
        array: true,
    },
];

pub fn bindings(authored: bool) -> Vec<KernelBinding> {
    BINDINGS
        .iter()
        .map(|binding| {
            let patched = authored && matches!(binding.binding, TASKS | VALUES);
            KernelBinding {
                kind: if patched {
                    BindingKind::TableStorage
                } else {
                    binding.kind
                },
                ..*binding
            }
        })
        .chain(AUTHORED_BINDINGS.iter().copied().filter(|_| authored))
        .collect()
}

pub struct Kernel {
    kinds: Vec<Kind>,
    geometry: Geometry,
    program: ComputeProgram,
}

impl Kernel {
    pub fn assemble(
        kinds: &[Kind],
        elements: &[Element],
        geometry: Geometry,
        authored: bool,
    ) -> Self {
        assert!(
            !kinds.is_empty(),
            "a device program with no task has nothing to run"
        );
        assert!(
            !elements.is_empty(),
            "a device program with no tensor has nothing to address"
        );
        let mut compiler = Compiler::empty();
        for record in RECORDS {
            compiler.record(*record);
        }
        compiler.constant("WORKGROUP_SIZE", geometry.workgroup());
        compiler.constant("NO_VALUE", NO_VALUE);
        compiler.constant("NO_SLOT", neura_abi::NO_SLOT);
        compiler.constant("INT4_BLOCK", INT4_BLOCK);
        compiler.constant("FP4_BLOCK", FP4_BLOCK);
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
            ("strategy::FRAME", strategy::FRAME),
            ("strategy::INDEX", strategy::INDEX),
        ] {
            compiler.constant(name, value);
        }
        for (name, value) in [
            ("measure::ELEMENTS", measure::ELEMENTS),
            ("measure::ROWS", measure::ROWS),
            ("measure::WORDS", measure::WORDS),
            ("measure::TOKENS", measure::TOKENS),
            ("measure::TILES", measure::TILES),
            ("split::RANGE", split::RANGE),
            ("split::UNIFORM", split::UNIFORM),
            ("split::PLANE", split::PLANE),
        ] {
            compiler.constant(name, value);
        }
        for (name, value) in [
            ("progress::CURSOR", progress::CURSOR),
            ("progress::FRONTIER", progress::FRONTIER),
            ("progress::SEGMENTS", progress::SEGMENTS),
            ("progress::WAVES", progress::WAVES),
            ("progress::COUNTERS", progress::COUNTERS),
            ("progress::WAVE_STRIDE", progress::WAVE_STRIDE),
        ] {
            compiler.constant(name, value);
        }
        for kind in Kind::ALL {
            compiler.constant(kind.symbol(), kind.code());
        }
        for binding in bindings(authored) {
            let spec = BindingSpec {
                binding: binding.binding,
                kind: binding.kind,
            };
            if binding.array {
                compiler.storage_array(binding.name, binding.element, spec);
            } else {
                compiler.storage_record(binding.name, binding.element, spec);
            }
        }
        compiler.workgroup_bytes("claim", "u32", CLAIM_BYTES);
        let scratch = geometry.scratch_bytes(kinds);
        if scratch > 0 {
            compiler.workgroup_bytes("scratch", "f32", scratch);
        }
        let half = geometry.scratch_half_bytes();
        if half > 0 {
            compiler.workgroup_bytes("scratch_half", "f16", half);
        }
        authored::install(&mut compiler, authored);
        scheduler::install(&mut compiler);
        substrate::install(&mut compiler);
        element::install(&mut compiler, elements);
        pointwise::install(&mut compiler);
        task::rope::install(&mut compiler);
        for module in DeviceModule::ALL {
            if kinds.iter().any(|kind| kind.carries(*module)) {
                install(&mut compiler, *module, elements, &geometry);
            }
        }
        for kind in Kind::ALL {
            compiler.insert_case(
                "run_task",
                ast::Arm {
                    pattern: ast::Pattern::Constant(kind.symbol().to_owned()),
                    body: vec![ast::Statement::Expression(ast::Expression::call(
                        kind.entry(),
                        vec![ast::Expression::name("task"), ast::Expression::name("lid")],
                    ))],
                },
            );
        }
        let enabled = kinds
            .iter()
            .map(|kind| kind.symbol().to_owned())
            .collect::<Vec<_>>();
        compiler.retain_cases("run_task", &enabled);
        let declared = compiler.declared_workgroup_bytes();
        assert_eq!(
            declared,
            geometry.workgroup_bytes(kinds),
            "a device program declares {declared} bytes of workgroup scratch where its geometry accounts for {}",
            geometry.workgroup_bytes(kinds),
        );
        assert!(
            declared <= geometry.budget(),
            "a device program of {} kinds declares {declared} bytes of workgroup scratch beyond the {} its profile offers",
            kinds.len(),
            geometry.budget(),
        );
        let program = compiler.finish(
            &format!(
                "neura kernel {} threads over {} tiles and {} kinds",
                geometry.workgroup(),
                geometry.walked().len(),
                kinds.len()
            ),
            scheduler::ENTRY,
            geometry.workgroup(),
        );
        for (expected, reflected) in bindings(authored).iter().zip(program.reflected()) {
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

fn install(
    compiler: &mut Compiler,
    module: DeviceModule,
    elements: &[Element],
    geometry: &Geometry,
) {
    match module {
        DeviceModule::Matmul => task::matmul::install(compiler),
        DeviceModule::MatmulTiles => task::matmul::install_tiles(compiler, geometry),
        DeviceModule::Attention => task::attention::install(compiler, geometry),
        DeviceModule::Reduce => task::reduce::install(compiler),
        DeviceModule::Softmax => task::softmax::install(compiler),
        DeviceModule::Choice => task::choice::install(compiler, geometry),
        DeviceModule::Select => task::select::install(compiler),
        DeviceModule::Conv => task::conv::install(compiler),
        DeviceModule::Scatter => task::scatter::install(compiler),
        DeviceModule::Layout => task::layout::install(compiler),
        DeviceModule::Pool => task::pool::install(compiler),
        DeviceModule::Convert => task::convert::install(compiler, elements),
        DeviceModule::Scan => task::scan::install(compiler),
    }
}
