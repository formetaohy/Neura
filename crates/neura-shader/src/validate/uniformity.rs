use super::collect;
use crate::instruction::{Address, Instruction};
use crate::module::{Access, Function, Module, Space};
use crate::ty::ValueId;
use std::collections::HashMap;

#[derive(Clone, Copy)]
struct Effect {
    result_uniform: bool,
    barrier: bool,
}

#[derive(Clone)]
struct State {
    locals: Vec<bool>,
    values: HashMap<ValueId, bool>,
    control_uniform: bool,
    returned_nonuniform: bool,
    loop_exit_nonuniform: bool,
    return_uniform: bool,
    barrier: bool,
}

impl State {
    fn new(function: &Function) -> Self {
        Self {
            locals: vec![false; function.locals.len()],
            values: HashMap::new(),
            control_uniform: true,
            returned_nonuniform: false,
            loop_exit_nonuniform: false,
            return_uniform: true,
            barrier: false,
        }
    }

    fn merge(&mut self, other: &Self) {
        for (left, right) in self.locals.iter_mut().zip(&other.locals) {
            *left &= *right;
        }
        for (value, uniform) in &other.values {
            *self.values.entry(*value).or_insert(true) &= *uniform;
        }
        self.returned_nonuniform |= other.returned_nonuniform;
        self.loop_exit_nonuniform |= other.loop_exit_nonuniform;
        self.return_uniform &= other.return_uniform;
        self.barrier |= other.barrier;
    }
}

enum Root {
    Local(u32),
    Global(u32),
}

pub(super) struct Analysis<'m> {
    module: &'m Module,
    declarations: HashMap<ValueId, &'m Instruction>,
    effects: HashMap<(u32, Vec<bool>), Effect>,
}

impl<'m> Analysis<'m> {
    pub(super) fn new(module: &'m Module) -> Self {
        let mut declarations = HashMap::new();
        for function in module.functions() {
            collect(&function.body, &mut declarations);
        }
        Self {
            module,
            declarations,
            effects: HashMap::new(),
        }
    }

    pub(super) fn run(&mut self) {
        let entry = self.module.entry_index();
        let arguments = self.module.functions()[entry as usize]
            .arguments
            .iter()
            .map(|argument| {
                argument
                    .builtin
                    .expect("a device entry argument is a builtin")
                    .uniform()
            })
            .collect::<Vec<_>>();
        self.function(entry, &arguments);
    }

    fn callee(&mut self, function: u32, arguments: &[bool]) -> Effect {
        let key = (function, arguments.to_vec());
        if let Some(effect) = self.effects.get(&key) {
            return *effect;
        }
        let effect = self.function(function, arguments);
        self.effects.insert(key, effect);
        effect
    }

    fn function(&mut self, index: u32, arguments: &[bool]) -> Effect {
        let function = &self.module.functions()[index as usize];
        assert_eq!(function.arguments.len(), arguments.len());
        let mut state = State::new(function);
        self.block(arguments, &function.body, &mut state);
        Effect {
            result_uniform: state.return_uniform,
            barrier: state.barrier,
        }
    }

    fn uniform(&mut self, arguments: &[bool], state: &State, value: ValueId) -> bool {
        if self.module.constant_of(value).is_some() {
            return true;
        }
        let Some(instruction) = self.declarations.get(&value).copied() else {
            return false;
        };
        match instruction {
            Instruction::Argument { index, .. } => arguments[*index as usize],
            Instruction::Address { .. } => true,
            Instruction::Load { pointer, .. } => {
                if !self.uniform(arguments, state, *pointer) {
                    return false;
                }
                match self.root(*pointer) {
                    Some(Root::Local(index)) => state.locals[index as usize],
                    Some(Root::Global(index)) => {
                        let global = self.module.global(index);
                        global.space == Space::Storage && global.access == Access::Read
                    }
                    None => false,
                }
            }
            Instruction::Access { base, index, .. } => {
                self.uniform(arguments, state, *base) && self.uniform(arguments, state, *index)
            }
            Instruction::AccessIndex { base, .. } => self.uniform(arguments, state, *base),
            Instruction::Compose { constituents, .. } => constituents
                .iter()
                .all(|value| self.uniform(arguments, state, *value)),
            Instruction::Unary { value, .. }
            | Instruction::Convert { value, .. }
            | Instruction::Bitcast { value, .. }
            | Instruction::MatrixFill { value, .. }
            | Instruction::MatrixExtract { value, .. } => self.uniform(arguments, state, *value),
            Instruction::Binary { left, right, .. } => {
                self.uniform(arguments, state, *left) && self.uniform(arguments, state, *right)
            }
            Instruction::Select {
                condition,
                accept,
                reject,
                ..
            } => {
                self.uniform(arguments, state, *condition)
                    && self.uniform(arguments, state, *accept)
                    && self.uniform(arguments, state, *reject)
            }
            Instruction::Math {
                arguments: inputs, ..
            } => inputs
                .iter()
                .all(|value| self.uniform(arguments, state, *value)),
            Instruction::MatrixLength { .. } => true,
            Instruction::WorkGroupUniformLoad { .. } => {
                state.values.get(&value).copied().unwrap_or(true)
            }
            Instruction::Call {
                function: callee,
                arguments: inputs,
                ..
            } => {
                let uniforms = inputs
                    .iter()
                    .map(|value| self.uniform(arguments, state, *value))
                    .collect::<Vec<_>>();
                self.callee(*callee, &uniforms).result_uniform
            }
            _ => false,
        }
    }

    fn root(&self, value: ValueId) -> Option<Root> {
        match self.declarations.get(&value).copied()? {
            Instruction::Address {
                address: Address::Local(index),
                ..
            } => Some(Root::Local(*index)),
            Instruction::Address {
                address: Address::Global(index),
                ..
            } => Some(Root::Global(*index)),
            Instruction::Access { base, .. } | Instruction::AccessIndex { base, .. } => {
                self.root(*base)
            }
            _ => None,
        }
    }

    fn block(&mut self, arguments: &[bool], block: &[Instruction], state: &mut State) {
        for instruction in block {
            match instruction {
                Instruction::Store { pointer, value } => {
                    if let Some(Root::Local(index)) = self.root(*pointer) {
                        state.locals[index as usize] = state.control_uniform
                            && self.uniform(arguments, state, *pointer)
                            && self.uniform(arguments, state, *value);
                    }
                }
                Instruction::Call {
                    function: callee,
                    arguments: inputs,
                    result,
                } => {
                    let uniforms = inputs
                        .iter()
                        .map(|value| self.uniform(arguments, state, *value))
                        .collect::<Vec<_>>();
                    let effect = self.callee(*callee, &uniforms);
                    if effect.barrier {
                        assert!(
                            state.control_uniform,
                            "a workgroup barrier is reached by different invocations through a Rust device call"
                        );
                    }
                    state.barrier |= effect.barrier;
                    if let Some(result) = result {
                        let uniform = state.control_uniform && effect.result_uniform;
                        state.values.insert(*result, uniform);
                    }
                }
                Instruction::Atomic { .. } => {}
                Instruction::WorkGroupUniformLoad { result, .. } => {
                    assert!(
                        state.control_uniform,
                        "a workgroup uniform load is reached by different invocations"
                    );
                    state.barrier = true;
                    state.values.insert(*result, true);
                }
                Instruction::Barrier(_) => {
                    assert!(
                        state.control_uniform,
                        "a workgroup barrier is reached by different invocations"
                    );
                    state.barrier = true;
                }
                Instruction::If {
                    condition,
                    accept,
                    reject,
                } => {
                    let uniform = self.uniform(arguments, state, *condition);
                    let mut accepted = state.clone();
                    accepted.control_uniform &= uniform;
                    self.block(arguments, accept, &mut accepted);
                    let mut rejected = state.clone();
                    rejected.control_uniform &= uniform;
                    self.block(arguments, reject, &mut rejected);
                    state.merge(&accepted);
                    state.merge(&rejected);
                    if !uniform
                        && (accepted.returned_nonuniform
                            || rejected.returned_nonuniform
                            || accepted.loop_exit_nonuniform
                            || rejected.loop_exit_nonuniform)
                    {
                        state.control_uniform = false;
                    }
                }
                Instruction::Switch {
                    selector,
                    cases,
                    default,
                } => {
                    let uniform = self.uniform(arguments, state, *selector);
                    let mut merged = state.clone();
                    for body in cases
                        .iter()
                        .map(|(_, body)| body)
                        .chain(std::iter::once(default))
                    {
                        let mut branch = state.clone();
                        branch.control_uniform &= uniform;
                        self.block(arguments, body, &mut branch);
                        merged.merge(&branch);
                        if !uniform && (branch.returned_nonuniform || branch.loop_exit_nonuniform) {
                            merged.control_uniform = false;
                        }
                    }
                    *state = merged;
                }
                Instruction::Loop { body, continuing } => {
                    let before = state.clone();
                    let mut head = before.clone();
                    let mut barriers = false;
                    let mut divergent_exit = false;
                    for pass in 0..=before.locals.len() {
                        let mut iteration = head.clone();
                        iteration.barrier = false;
                        iteration.loop_exit_nonuniform = false;
                        iteration.returned_nonuniform = false;
                        self.block(arguments, body, &mut iteration);
                        self.block(arguments, continuing, &mut iteration);
                        barriers |= iteration.barrier;
                        divergent_exit |= iteration.loop_exit_nonuniform;
                        assert!(
                            !barriers || !divergent_exit,
                            "a workgroup loop cannot leave some invocations behind at a barrier"
                        );
                        let next = before
                            .locals
                            .iter()
                            .zip(&iteration.locals)
                            .map(|(initial, current)| *initial && *current)
                            .collect::<Vec<_>>();
                        if next == head.locals {
                            state.merge(&iteration);
                            state.control_uniform =
                                before.control_uniform && !iteration.returned_nonuniform;
                            state.loop_exit_nonuniform = before.loop_exit_nonuniform;
                            state.barrier |= barriers;
                            break;
                        }
                        assert!(
                            pass < before.locals.len(),
                            "device loop uniformity did not converge"
                        );
                        head.locals = next;
                        head.control_uniform &= !iteration.returned_nonuniform;
                    }
                }
                Instruction::Return { value } => {
                    state.return_uniform &= state.control_uniform
                        && value.is_none_or(|value| self.uniform(arguments, state, value));
                    state.returned_nonuniform |= !state.control_uniform;
                }
                Instruction::Break | Instruction::Continue => {
                    state.loop_exit_nonuniform |= !state.control_uniform;
                }
                Instruction::Block(body) => self.block(arguments, body, state),
                other => {
                    if let Some(result) = other.result() {
                        let uniform = self.uniform(arguments, state, result);
                        state.values.insert(result, uniform);
                    }
                }
            }
        }
    }
}
