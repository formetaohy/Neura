mod reflection;

use neura_abi::{
    CURSOR_WAVE_BASE, Geometry, Kind, Precision, TAPE_WGSL, kind, op, store, strategy,
};
use neura_gpu::{BindingKind, BindingSpec, ComputeProgram};
use std::fmt::Write as _;
use std::sync::Arc;

pub use reflection::{ShaderBinding, describe, reflect};

pub const ENTRY: &str = "main";
pub const GROUP: u32 = 0;
pub const TASKS: u32 = 0;
pub const VALUES: u32 = 1;
pub const HEAP: u32 = 2;
pub const CURSOR: u32 = 3;
pub const BOUNDS: u32 = 4;
pub const STEPS: u32 = 5;
pub const PLACEMENT: u32 = 6;
pub const SEGMENTS: u32 = 7;

pub struct KernelBinding {
    pub binding: u32,
    pub kind: BindingKind,
    pub dynamic_offset: bool,
    pub name: &'static str,
}

pub const BINDINGS: &[KernelBinding] = &[
    KernelBinding {
        binding: TASKS,
        kind: BindingKind::ReadOnlyStorage,
        dynamic_offset: false,
        name: "tasks",
    },
    KernelBinding {
        binding: VALUES,
        kind: BindingKind::ReadOnlyStorage,
        dynamic_offset: false,
        name: "values",
    },
    KernelBinding {
        binding: HEAP,
        kind: BindingKind::ReadWriteStorage,
        dynamic_offset: false,
        name: "heap",
    },
    KernelBinding {
        binding: CURSOR,
        kind: BindingKind::ReadWriteStorage,
        dynamic_offset: false,
        name: "cursor",
    },
    KernelBinding {
        binding: BOUNDS,
        kind: BindingKind::ReadOnlyStorage,
        dynamic_offset: true,
        name: "bounds",
    },
    KernelBinding {
        binding: STEPS,
        kind: BindingKind::ReadOnlyStorage,
        dynamic_offset: false,
        name: "steps",
    },
    KernelBinding {
        binding: PLACEMENT,
        kind: BindingKind::ReadOnlyStorage,
        dynamic_offset: false,
        name: "placement",
    },
    KernelBinding {
        binding: SEGMENTS,
        kind: BindingKind::ReadOnlyStorage,
        dynamic_offset: false,
        name: "segments",
    },
];

pub struct Megakernel {
    geometry: Geometry,
    source: Arc<str>,
    bindings: Vec<ShaderBinding>,
}

impl Megakernel {
    pub fn assemble(geometry: Geometry, weights: Precision) -> Self {
        let mut source = String::from(TAPE_WGSL);
        source.push('\n');
        source.push_str(&kind::declarations());
        source.push_str(&op::declarations());
        source.push_str(&strategy::declarations());
        source.push_str(&store::declarations());
        source.push_str(&geometry.declarations());
        for fragment in neura_kernels::fragments(geometry.clone(), weights) {
            source.push_str(&fragment);
            source.push('\n');
        }
        source.push_str(&dispatch());
        source.push_str(&task_loop());
        let source = Arc::<str>::from(source);
        let bindings = reflect(&source);
        assert_declared(&bindings);
        Self {
            geometry,
            source,
            bindings,
        }
    }

    pub fn geometry(&self) -> &Geometry {
        &self.geometry
    }

    pub fn source(&self) -> &str {
        &self.source
    }

    pub fn bindings(&self) -> &[ShaderBinding] {
        &self.bindings
    }

    pub fn workgroup_size(&self) -> u32 {
        self.geometry.workgroup()
    }

    pub fn program(&self) -> ComputeProgram {
        let specs = BINDINGS
            .iter()
            .map(|binding| BindingSpec {
                binding: binding.binding,
                kind: binding.kind,
                dynamic_offset: binding.dynamic_offset,
            })
            .collect::<Vec<_>>();
        ComputeProgram::new(
            &format!(
                "neura megakernel {} threads over {} tiles",
                self.geometry.workgroup(),
                self.geometry.tiles().len(),
            ),
            self.source.clone(),
            ENTRY,
            &specs,
        )
    }
}

fn dispatch() -> String {
    let mut out = String::from(
        "fn run_task(index: u32, lid: u32) {\n    let task = tasks[index];\n    switch (task.kind) {\n",
    );
    for kind in Kind::ALL {
        writeln!(
            out,
            "        case {}: {{ {}(task, lid); }}",
            kind.constant(),
            neura_kernels::body(*kind),
        )
        .unwrap();
    }
    out.push_str("        default: { refuse(task.kind, 0u); }\n    }\n}\n");
    out
}

fn task_loop() -> String {
    format!(
        "
var<workgroup> claimed_segment: u32;

@compute @workgroup_size(WORKGROUP_SIZE)
fn {ENTRY}(@builtin(local_invocation_index) lid: u32) {{
    loop {{
        if (lid == 0u) {{
            claimed_segment = atomicAdd(&cursor[{CURSOR_WAVE_BASE} + bounds.wave], 1u);
        }}
        workgroupBarrier();
        if (claimed_segment >= bounds.segment_count) {{
            break;
        }}
        let segment = segments[bounds.first_segment + claimed_segment];
        for (var index = segment.first; index < segment.first + segment.count; index = index + 1u) {{
            run_task(index, lid);
            storageBarrier();
        }}
    }}
}}
"
    )
}

fn assert_declared(bindings: &[ShaderBinding]) {
    assert_eq!(
        bindings.len(),
        BINDINGS.len(),
        "the device program declares {} bindings where the framework hands it {}:\n{}",
        bindings.len(),
        BINDINGS.len(),
        describe(bindings),
    );
    for declared in BINDINGS {
        let reflected = bindings
            .iter()
            .find(|binding| binding.group == GROUP && binding.binding == declared.binding)
            .unwrap_or_else(|| {
                panic!(
                    "the device program does not declare binding {} of group {GROUP}:\n{}",
                    declared.binding,
                    describe(bindings),
                )
            });
        assert_eq!(
            reflected.name, declared.name,
            "binding {} of group {GROUP} is declared as {:?} where the framework binds {:?}",
            declared.binding, reflected.name, declared.name,
        );
        assert_eq!(
            reflected.kind, declared.kind,
            "binding {} ({}) is declared as {:?} where the framework binds {:?}",
            declared.binding, declared.name, reflected.kind, declared.kind,
        );
    }
}
