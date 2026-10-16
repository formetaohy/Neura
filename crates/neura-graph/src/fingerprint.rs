use crate::graph::{Residency, TaskInfo, ValueInfo};
use neura_abi::Digest;
use neura_abi::MAX_RANK;

pub(crate) fn of(values: &[ValueInfo], tasks: &[TaskInfo]) -> u64 {
    let mut digest = Digest::new();
    digest.length(values.len());
    for value in values {
        for axis in 0..MAX_RANK {
            digest.word(value.shape.dims()[axis as usize]);
            digest.word(value.shape.free(axis).unwrap_or(u32::MAX));
        }
        for stride in value.strides {
            digest.word(stride);
        }
        for rank in value.layout.rank() {
            digest.word(u32::from(rank));
        }
        digest.word(value.layout.origin() as u32);
        digest.word((value.layout.origin() >> 32) as u32);
        digest.word(value.element.code());
        digest.single(value.scale);
        digest.word(residency(value.residency));
        digest.flag(value.requires_grad);
        digest.flag(value.retained);
        digest.flag(value.written_in_place);
        digest.word(value.recomputes.unwrap_or(u32::MAX));
    }
    digest.length(tasks.len());
    for task in tasks {
        digest.word(task.kind.code());
        digest.word(task.op);
        digest.word(task.out);
        digest.word(task.extra);
        for input in task.inputs {
            digest.word(input);
        }
        digest.word(task.origin);
        digest.word(task.slot);
        digest.single(task.literal);
        digest.word(task.knob);
        digest.word(task.window.reach_rows());
        digest.word(task.window.reach_columns());
        digest.word(task.window.stride_rows());
        digest.word(task.window.stride_columns());
        digest.word(task.window.pad_rows());
        digest.word(task.window.pad_columns());
        digest.word(task.window.dilation_rows());
        digest.word(task.window.dilation_columns());
        digest.flag(task.in_place);
        digest.word(task.axis);
        digest.word(task.offset);
        digest.word(task.segments);
        digest.word(task.reach);
        digest.word(task.queries);
        digest.word(task.keep);
        digest.word(task.bias);
        for steps in [&task.prelude, &task.chain] {
            digest.length(steps.len());
            for step in steps {
                digest.word(step.op);
                digest.word(step.operand);
                digest.word(step.swapped);
            }
        }
    }
    digest.finish()
}

fn residency(residency: Residency) -> u32 {
    match residency {
        Residency::Input => 0,
        Residency::Parameter => 1,
        Residency::State => 2,
        Residency::Resident => 3,
        Residency::Derived => 4,
        Residency::View => 5,
    }
}
