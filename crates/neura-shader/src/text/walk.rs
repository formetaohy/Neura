use crate::{Address, Instruction, Module, ValueId};
use std::collections::{HashMap, HashSet};

pub(super) fn usage(module: &Module) -> Vec<Vec<u32>> {
    let mut used: Vec<HashSet<u32>> = module
        .functions()
        .iter()
        .map(|function| {
            let mut used = HashSet::new();
            globals(&function.body, &mut used);
            used
        })
        .collect();
    loop {
        let mut grown = false;
        for index in 0..module.functions().len() {
            let mut calls = Vec::new();
            calls_of(&module.functions()[index].body, &mut calls);
            for callee in calls {
                let callee = used[callee as usize].clone();
                for global in callee {
                    grown |= used[index].insert(global);
                }
            }
        }
        if !grown {
            break;
        }
    }
    used.into_iter()
        .map(|used| {
            let mut sorted = used.into_iter().collect::<Vec<_>>();
            sorted.sort_unstable();
            sorted
        })
        .collect()
}

pub(super) fn globals(body: &[Instruction], used: &mut HashSet<u32>) {
    for instruction in body {
        if let Instruction::Address {
            address: Address::Global(index),
            ..
        } = instruction
        {
            used.insert(*index);
        }
        nested(instruction, &mut |block| globals(block, used));
    }
}

pub(super) fn calls_of(body: &[Instruction], calls: &mut Vec<u32>) {
    for instruction in body {
        if let Instruction::Call { function, .. } = instruction {
            calls.push(*function);
        }
        nested(instruction, &mut |block| calls_of(block, calls));
    }
}

pub(super) fn nested(instruction: &Instruction, visit: &mut impl FnMut(&[Instruction])) {
    match instruction {
        Instruction::If { accept, reject, .. } => {
            visit(accept);
            visit(reject);
        }
        Instruction::Switch { cases, default, .. } => {
            for (_, body) in cases {
                visit(body);
            }
            visit(default);
        }
        Instruction::Loop { body, continuing } => {
            visit(body);
            visit(continuing);
        }
        Instruction::Block(body) => visit(body),
        _ => {}
    }
}

pub(super) fn argument_value(body: &[Instruction], index: usize) -> Option<ValueId> {
    body.iter().find_map(|instruction| match instruction {
        Instruction::Argument {
            index: present,
            result,
        } if *present as usize == index => Some(*result),
        _ => None,
    })
}

pub(super) fn terminates(body: &[Instruction]) -> bool {
    match body.last() {
        Some(Instruction::Return { .. } | Instruction::Break | Instruction::Continue) => true,
        Some(Instruction::If { accept, reject, .. }) => terminates(accept) && terminates(reject),
        Some(Instruction::Switch { cases, default, .. }) => {
            cases.iter().all(|(_, body)| terminates(body)) && terminates(default)
        }
        _ => false,
    }
}

pub(super) fn collect<'m>(
    body: &'m [Instruction],
    declarations: &mut HashMap<ValueId, &'m Instruction>,
) {
    for instruction in body {
        if let Some(result) = instruction.result() {
            declarations.insert(result, instruction);
        }
        match instruction {
            Instruction::If { accept, reject, .. } => {
                collect(accept, declarations);
                collect(reject, declarations);
            }
            Instruction::Switch { cases, default, .. } => {
                for (_, body) in cases {
                    collect(body, declarations);
                }
                collect(default, declarations);
            }
            Instruction::Loop { body, continuing } => {
                collect(body, declarations);
                collect(continuing, declarations);
            }
            Instruction::Block(body) => collect(body, declarations),
            _ => {}
        }
    }
}
