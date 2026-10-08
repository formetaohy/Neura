use crate::context::Device;
use crate::pipeline::PipelineHandle;
use neura_shader::ComputeProgram;
use std::collections::HashMap;
use std::sync::Arc;

pub const WARM_PROGRAMS: usize = 8;

struct WarmProgram {
    handle: PipelineHandle,
    declared: u64,
}

pub(crate) struct PipelineLibrary {
    entries: HashMap<Arc<ComputeProgram>, WarmProgram>,
    clock: u64,
}

impl PipelineLibrary {
    pub(crate) fn new() -> Self {
        Self {
            entries: HashMap::new(),
            clock: 0,
        }
    }

    pub(crate) fn declare(&mut self, device: &Device, program: ComputeProgram) -> PipelineHandle {
        self.clock += 1;
        let program = Arc::new(program);
        if let Some(warm) = self.entries.get_mut(&program) {
            warm.declared = self.clock;
            return warm.handle.clone();
        }
        let handle = PipelineHandle::new(device, program.clone());
        self.entries.insert(
            program,
            WarmProgram {
                handle: handle.clone(),
                declared: self.clock,
            },
        );
        self.retain_warm();
        handle
    }

    fn retain_warm(&mut self) {
        if self.entries.len() <= WARM_PROGRAMS {
            return;
        }
        let mut declared = self
            .entries
            .values()
            .map(|warm| warm.declared)
            .collect::<Vec<u64>>();
        declared.sort_unstable();
        let oldest = declared[self.entries.len() - WARM_PROGRAMS];
        self.entries.retain(|_, warm| warm.declared >= oldest);
    }

    pub(crate) fn declared(&self) -> usize {
        self.entries.len()
    }
}
