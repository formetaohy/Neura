mod reflection;

use neura_abi::{CURSOR_WAVE_BASE, TAPE_WGSL, WORKGROUP_SIZE};
use neura_gpu::{BindingKind, BindingSpec, ComputeProgram};
use std::fmt::Write as _;
use std::sync::Arc;

pub use reflection::{ShaderBinding, describe, reflect};

pub const ENTRY: &str = "main";
pub const GROUP: u32 = 0;
pub const TASKS: u32 = 0;
pub const VALUES: u32 = 1;
pub const ARENA: u32 = 2;
pub const CURSOR: u32 = 3;
pub const BOUNDS: u32 = 4;

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
        binding: ARENA,
        kind: BindingKind::ReadWriteStorage,
        dynamic_offset: false,
        name: "arena",
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
];

pub struct Megakernel {
    source: Arc<str>,
    bindings: Vec<ShaderBinding>,
}

impl Megakernel {
    pub fn assemble() -> Self {
        let mut source = String::from(TAPE_WGSL);
        source.push('\n');
        for fragment in neura_kernels::FRAGMENTS {
            source.push_str(fragment);
            source.push('\n');
        }
        source.push_str(&neura_kernels::reductions());
        source.push('\n');
        source.push_str(&dispatch());
        source.push_str(&task_loop());
        let source = Arc::<str>::from(source);
        let bindings = reflect(&source);
        assert_declared(&bindings);
        Self { source, bindings }
    }

    pub fn source(&self) -> &str {
        &self.source
    }

    pub fn bindings(&self) -> &[ShaderBinding] {
        &self.bindings
    }

    pub fn workgroup_size(&self) -> u32 {
        WORKGROUP_SIZE
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
        ComputeProgram::new("neura megakernel", self.source.clone(), ENTRY, &specs)
    }
}

fn dispatch() -> String {
    let mut out = String::from(
        "fn run_task(index: u32, lid: u32) {\n    let task = tasks[index];\n    switch (task.kind) {\n",
    );
    for kernel in neura_kernels::KERNELS {
        writeln!(
            out,
            "        case {}: {{ {}(task, lid); }}",
            kernel.constant, kernel.body,
        )
        .unwrap();
    }
    out.push_str("        default: { refuse(task.kind, 0u); }\n    }\n}\n");
    out
}

fn task_loop() -> String {
    format!(
        "
var<workgroup> claimed_task: u32;

@compute @workgroup_size(WORKGROUP_SIZE)
fn {ENTRY}(@builtin(local_invocation_index) lid: u32) {{
    loop {{
        if (lid == 0u) {{
            claimed_task = atomicAdd(&cursor[{CURSOR_WAVE_BASE} + bounds.wave], 1u);
        }}
        workgroupBarrier();
        if (claimed_task >= bounds.task_count) {{
            break;
        }}
        run_task(bounds.first_task + claimed_task, lid);
        workgroupBarrier();
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
