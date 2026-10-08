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

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Banks {
    count: u32,
    shift: u32,
}

impl Banks {
    pub const SINGLE: Self = Self { count: 1, shift: 0 };

    pub fn of(count: u32, shift: u32) -> Self {
        assert!(count >= 1, "a heap spans no bank");
        if count == 1 {
            return Self::SINGLE;
        }
        assert!(
            shift < 32,
            "a bank of {} words beyond the device word address splits no address",
            1u64 << shift,
        );
        Self { count, shift }
    }

    pub const fn count(self) -> u32 {
        self.count
    }

    pub const fn shift(self) -> u32 {
        self.shift
    }
}

pub const BINDINGS_WITHOUT_HEAP: u32 = 7;

pub const fn refusal(banks: Banks) -> u32 {
    HEAP + banks.count()
}

pub const fn progress(banks: Banks) -> u32 {
    refusal(banks) + 1
}

pub const fn steps(banks: Banks) -> u32 {
    progress(banks) + 1
}

pub const fn placement(banks: Banks) -> u32 {
    steps(banks) + 1
}

pub const fn segments(banks: Banks) -> u32 {
    placement(banks) + 1
}

pub const fn extents(banks: Banks) -> u32 {
    segments(banks) + 1
}

pub const fn measures(banks: Banks) -> u32 {
    extents(banks) + 1
}

pub const fn patches(banks: Banks) -> u32 {
    measures(banks) + 1
}

pub const fn patch_list(banks: Banks) -> u32 {
    patches(banks) + 1
}

pub const fn bank_ceiling(slots: u32) -> u32 {
    slots.saturating_sub(BINDINGS_WITHOUT_HEAP)
}

const _: () = assert!(segments(Banks::SINGLE) + 1 == BINDINGS_WITHOUT_HEAP + 1);

pub struct KernelBinding {
    pub binding: u32,
    pub kind: BindingKind,
    pub name: String,
    pub element: &'static str,
    pub array: bool,
}

fn binding(
    binding: u32,
    kind: BindingKind,
    name: &str,
    element: &'static str,
    array: bool,
) -> KernelBinding {
    KernelBinding {
        binding,
        kind,
        name: name.to_owned(),
        element,
        array,
    }
}

pub fn bindings(authored: bool, banks: Banks) -> Vec<KernelBinding> {
    let tables = if authored {
        BindingKind::TableStorage
    } else {
        BindingKind::ReadOnlyStorage
    };
    let mut list = vec![
        binding(TASKS, tables, "tasks", "Task", true),
        binding(VALUES, tables, "values", "Value", true),
    ];
    for bank in 0..banks.count() {
        list.push(binding(
            HEAP + bank,
            BindingKind::ReadWriteStorage,
            &format!("heap{bank}"),
            "f32",
            true,
        ));
    }
    list.push(binding(
        refusal(banks),
        BindingKind::ReadWriteStorage,
        "refusal",
        "AtomicU32",
        true,
    ));
    list.push(binding(
        progress(banks),
        BindingKind::ReadWriteStorage,
        "progress",
        "AtomicU32",
        true,
    ));
    list.push(binding(
        steps(banks),
        BindingKind::ReadOnlyStorage,
        "steps",
        "Step",
        true,
    ));
    list.push(binding(
        placement(banks),
        BindingKind::ReadOnlyStorage,
        "placement",
        "Placement",
        false,
    ));
    list.push(binding(
        segments(banks),
        BindingKind::ReadOnlyStorage,
        "segments",
        "Segment",
        true,
    ));
    if authored {
        list.push(binding(
            extents(banks),
            BindingKind::ReadWriteStorage,
            "extents",
            "u32",
            true,
        ));
        list.push(binding(
            measures(banks),
            BindingKind::ReadOnlyStorage,
            "measures",
            "Measure",
            true,
        ));
        list.push(binding(
            patches(banks),
            BindingKind::ReadOnlyStorage,
            "patches",
            "Patch",
            true,
        ));
        list.push(binding(
            patch_list(banks),
            BindingKind::ReadOnlyStorage,
            "patch_list",
            "u32",
            true,
        ));
    }
    let slots = if authored {
        patch_list(banks) + 1
    } else {
        segments(banks) + 1
    };
    assert_eq!(
        list.len() as u32,
        slots,
        "a program binds the {slots} storage buffers its slots name",
    );
    list
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
        banks: Banks,
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
        compiler.constant("EXACT_WALK_LIMIT", neura_abi::EXACT_WALK_LIMIT);
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
            ("split::SEGMENT", split::SEGMENT),
            ("split::RAGGED", split::RAGGED),
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
        for binding in bindings(authored, banks) {
            let spec = BindingSpec {
                binding: binding.binding,
                kind: binding.kind,
            };
            if binding.array {
                compiler.storage_array(&binding.name, binding.element, spec);
            } else {
                compiler.storage_record(&binding.name, binding.element, spec);
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
        substrate::install(&mut compiler, banks);
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
        for (expected, reflected) in bindings(authored, banks).iter().zip(program.reflected()) {
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
        DeviceModule::MatmulWeight => task::matmul_weight::install(compiler, geometry),
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
        DeviceModule::Rows => task::rows::install(compiler),
    }
}
