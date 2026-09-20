use crate::pipeline::{ComputeProgram, PipelineHandle};
use std::collections::HashMap;
use std::sync::Arc;
use wgpu::Device;

pub(crate) struct PipelineLibrary {
    entries: HashMap<Arc<ComputeProgram>, PipelineHandle>,
}

impl PipelineLibrary {
    pub(crate) fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    pub(crate) fn declare(&mut self, device: &Device, program: ComputeProgram) -> PipelineHandle {
        let program = Arc::new(program);
        if let Some(handle) = self.entries.get(&program) {
            return handle.clone();
        }
        let handle = PipelineHandle::new(device, program.clone());
        self.entries.insert(program, handle.clone());
        handle
    }

    pub(crate) fn declared(&self) -> usize {
        self.entries.len()
    }
}
