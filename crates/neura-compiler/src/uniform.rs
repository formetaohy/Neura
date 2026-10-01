use naga::{
    AddressSpace, Block, Expression, Function, GlobalVariable, Handle, LocalVariable, Module,
    Statement, StorageAccess,
};
use std::collections::HashMap;

#[derive(Clone, Copy)]
struct Effect {
    result_uniform: bool,
    barrier: bool,
}

#[derive(Clone)]
struct State {
    locals: Vec<bool>,
    values: Vec<bool>,
    control_uniform: bool,
    returned_nonuniform: bool,
    loop_exit_nonuniform: bool,
    return_uniform: bool,
    barrier: bool,
}

impl State {
    fn new(function: &Function) -> Self {
        Self {
            locals: vec![false; function.local_variables.len()],
            values: vec![false; function.expressions.len()],
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
        for (left, right) in self.values.iter_mut().zip(&other.values) {
            *left &= *right;
        }
        self.returned_nonuniform |= other.returned_nonuniform;
        self.loop_exit_nonuniform |= other.loop_exit_nonuniform;
        self.return_uniform &= other.return_uniform;
        self.barrier |= other.barrier;
    }
}

pub(crate) fn verify(module: &Module, entry: usize) {
    let mut verifier = Verifier {
        module,
        effects: HashMap::new(),
    };
    let function = &module.entry_points[entry].function;
    let arguments = function
        .arguments
        .iter()
        .map(|argument| {
            matches!(
                argument.binding,
                Some(naga::Binding::BuiltIn(
                    naga::BuiltIn::WorkGroupId
                        | naga::BuiltIn::WorkGroupSize
                        | naga::BuiltIn::NumWorkGroups
                ))
            )
        })
        .collect::<Vec<_>>();
    verifier.function(function, &arguments);
}

struct Verifier<'a> {
    module: &'a Module,
    effects: HashMap<(Handle<Function>, Vec<bool>), Effect>,
}

impl Verifier<'_> {
    fn callee(&mut self, handle: Handle<Function>, arguments: &[bool]) -> Effect {
        let key = (handle, arguments.to_vec());
        if let Some(effect) = self.effects.get(&key) {
            return *effect;
        }
        let function = &self.module.functions[handle];
        let effect = self.function(function, arguments);
        self.effects.insert(key, effect);
        effect
    }

    fn function(&mut self, function: &Function, arguments: &[bool]) -> Effect {
        assert_eq!(function.arguments.len(), arguments.len());
        let mut state = State::new(function);
        self.block(function, arguments, &function.body, &mut state);
        Effect {
            result_uniform: state.return_uniform,
            barrier: state.barrier,
        }
    }

    fn value(
        &self,
        function: &Function,
        arguments: &[bool],
        state: &State,
        handle: Handle<Expression>,
    ) -> bool {
        match &function.expressions[handle] {
            Expression::Literal(_) | Expression::Constant(_) | Expression::ZeroValue(_) => true,
            Expression::FunctionArgument(index) => arguments[*index as usize],
            Expression::LocalVariable(_) | Expression::GlobalVariable(_) => true,
            _ => state.values[handle.index()],
        }
    }

    fn root(&self, function: &Function, handle: Handle<Expression>) -> Option<Root> {
        match &function.expressions[handle] {
            Expression::LocalVariable(handle) => Some(Root::Local(*handle)),
            Expression::GlobalVariable(handle) => Some(Root::Global(*handle)),
            Expression::Access { base, .. } | Expression::AccessIndex { base, .. } => {
                self.root(function, *base)
            }
            _ => None,
        }
    }

    fn expression(
        &self,
        function: &Function,
        arguments: &[bool],
        state: &State,
        handle: Handle<Expression>,
    ) -> bool {
        let uniform = |handle| self.value(function, arguments, state, handle);
        match &function.expressions[handle] {
            Expression::Literal(_) | Expression::Constant(_) | Expression::ZeroValue(_) => true,
            Expression::FunctionArgument(index) => arguments[*index as usize],
            Expression::LocalVariable(_) | Expression::GlobalVariable(_) => true,
            Expression::Load { pointer } => {
                if !uniform(*pointer) {
                    return false;
                }
                match self.root(function, *pointer) {
                    Some(Root::Local(handle)) => state.locals[handle.index()],
                    Some(Root::Global(handle)) => {
                        match self.module.global_variables[handle].space {
                            AddressSpace::Storage { access } => {
                                !access.contains(StorageAccess::STORE)
                            }
                            AddressSpace::Uniform | AddressSpace::Immediate => true,
                            _ => false,
                        }
                    }
                    None => false,
                }
            }
            Expression::Access { base, index } => uniform(*base) && uniform(*index),
            Expression::AccessIndex { base, .. } => uniform(*base),
            Expression::Compose { components, .. } => components.iter().all(|part| uniform(*part)),
            Expression::Unary { expr, .. }
            | Expression::As { expr, .. }
            | Expression::ArrayLength(expr) => uniform(*expr),
            Expression::Binary { left, right, .. } => uniform(*left) && uniform(*right),
            Expression::Select {
                condition,
                accept,
                reject,
            } => uniform(*condition) && uniform(*accept) && uniform(*reject),
            Expression::Math {
                arg,
                arg1,
                arg2,
                arg3,
                ..
            } => {
                uniform(*arg)
                    && [arg1, arg2, arg3]
                        .into_iter()
                        .flatten()
                        .all(|expr| uniform(*expr))
            }
            Expression::Splat { value, .. } => uniform(*value),
            Expression::Swizzle { vector, .. } => uniform(*vector),
            _ => false,
        }
    }

    fn block(&mut self, function: &Function, arguments: &[bool], block: &Block, state: &mut State) {
        for statement in block {
            match statement {
                Statement::Emit(range) => {
                    for handle in range.clone() {
                        state.values[handle.index()] =
                            self.expression(function, arguments, state, handle);
                    }
                }
                Statement::Store { pointer, value } => {
                    if let Some(Root::Local(handle)) = self.root(function, *pointer) {
                        state.locals[handle.index()] = state.control_uniform
                            && self.value(function, arguments, state, *pointer)
                            && self.value(function, arguments, state, *value);
                    }
                }
                Statement::Call {
                    function: callee,
                    arguments: call_args,
                    result,
                } => {
                    let inputs = call_args
                        .iter()
                        .map(|arg| self.value(function, arguments, state, *arg))
                        .collect::<Vec<_>>();
                    let effect = self.callee(*callee, &inputs);
                    if effect.barrier {
                        assert!(
                            state.control_uniform,
                            "a workgroup barrier is reached by different invocations through a Rust device call"
                        );
                    }
                    state.barrier |= effect.barrier;
                    if let Some(result) = result {
                        state.values[result.index()] =
                            state.control_uniform && effect.result_uniform;
                    }
                }
                Statement::ControlBarrier(_) | Statement::MemoryBarrier(_) => {
                    assert!(
                        state.control_uniform,
                        "a workgroup barrier is reached by different invocations"
                    );
                    state.barrier = true;
                }
                Statement::If {
                    condition,
                    accept,
                    reject,
                } => {
                    let uniform = self.value(function, arguments, state, *condition);
                    let mut accepted = state.clone();
                    accepted.control_uniform &= uniform;
                    self.block(function, arguments, accept, &mut accepted);
                    let mut rejected = state.clone();
                    rejected.control_uniform &= uniform;
                    self.block(function, arguments, reject, &mut rejected);
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
                Statement::Switch { selector, cases } => {
                    let uniform = self.value(function, arguments, state, *selector);
                    let mut merged = state.clone();
                    for case in cases {
                        assert!(
                            !case.fall_through,
                            "Rust device matches have no fallthrough"
                        );
                        let mut branch = state.clone();
                        branch.control_uniform &= uniform;
                        self.block(function, arguments, &case.body, &mut branch);
                        merged.merge(&branch);
                        if !uniform && (branch.returned_nonuniform || branch.loop_exit_nonuniform) {
                            merged.control_uniform = false;
                        }
                    }
                    *state = merged;
                }
                Statement::Loop {
                    body,
                    continuing,
                    break_if,
                } => {
                    let before = state.clone();
                    let mut head = before.clone();
                    let mut barriers = false;
                    let mut divergent_exit = false;
                    for pass in 0..=before.locals.len() {
                        let mut iteration = head.clone();
                        iteration.barrier = false;
                        iteration.loop_exit_nonuniform = false;
                        iteration.returned_nonuniform = false;
                        self.block(function, arguments, body, &mut iteration);
                        self.block(function, arguments, continuing, &mut iteration);
                        if let Some(condition) = break_if {
                            assert!(
                                self.value(function, arguments, &iteration, *condition),
                                "a workgroup loop must use a uniform break condition"
                            );
                        }
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
                Statement::Return { value } => {
                    state.return_uniform &= state.control_uniform
                        && value.is_none_or(|expr| self.value(function, arguments, state, expr));
                    state.returned_nonuniform |= !state.control_uniform;
                }
                Statement::Break | Statement::Continue => {
                    state.loop_exit_nonuniform |= !state.control_uniform;
                }
                Statement::Block(block) => self.block(function, arguments, block, state),
                other => panic!("unsupported device synchronization effect: {other:?}"),
            }
        }
    }
}

enum Root {
    Local(Handle<LocalVariable>),
    Global(Handle<GlobalVariable>),
}
