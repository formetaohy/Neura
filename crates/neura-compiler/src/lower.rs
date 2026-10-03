mod statement;
mod value;

use crate::{Compiler, DeviceInstruction, ir};
use neura_shader_ir::{Argument, Block, BuiltIn, Function, Local, Space, TypeId, ValueId};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone, Copy)]
pub(super) struct Typed {
    pub(super) value: ValueId,
    pub(super) ty: TypeId,
}

#[derive(Clone)]
pub(super) enum Symbol {
    Value(Typed),
    Local(Typed),
    Array(Arc<[Typed]>),
    Constant(u32),
}

pub(super) struct FunctionLower<'a> {
    pub(super) compiler: &'a mut Compiler,
    pub(super) function: Function,
    pub(super) scopes: Vec<HashMap<String, Symbol>>,
    pub(super) blocks: Vec<Block>,
    values: Vec<ValueId>,
}

impl<'a> FunctionLower<'a> {
    pub(super) fn new(compiler: &'a mut Compiler, source: &ir::Function, entry: bool) -> Self {
        let mut arguments = Vec::new();
        let mut values = Vec::new();
        let mut scope = HashMap::new();
        for argument in &source.arguments {
            let ty = compiler.rust_type(&argument.ty);
            let builtin = entry.then(|| match argument.name.as_str() {
                "lid" => BuiltIn::LocalInvocationIndex,
                "group" => BuiltIn::WorkGroupId,
                "global" => BuiltIn::GlobalInvocationId,
                other => panic!("entry argument {other} is not a device builtin"),
            });
            let value = compiler.define(ty);
            assert!(
                scope
                    .insert(argument.name.clone(), Symbol::Value(Typed { value, ty }))
                    .is_none(),
                "a device argument name is unique"
            );
            values.push(value);
            arguments.push(Argument {
                name: argument.name.clone(),
                ty,
                builtin,
            });
        }
        let result = source.result.as_ref().map(|ty| compiler.rust_type(ty));
        Self {
            compiler,
            function: Function {
                name: source.name.clone(),
                arguments,
                result,
                locals: Vec::new(),
                body: Vec::new(),
            },
            scopes: vec![scope],
            blocks: vec![Vec::new()],
            values,
        }
    }

    pub(super) fn lower(mut self, body: &[ir::Statement]) -> Function {
        for (index, value) in self.values.clone().into_iter().enumerate() {
            self.push(DeviceInstruction::Argument {
                index: index as u32,
                result: value,
            });
        }
        self.statements(body);
        self.function.body = self.blocks.pop().expect("a function has a body");
        self.function
    }

    pub(super) fn push(&mut self, instruction: DeviceInstruction) {
        self.blocks
            .last_mut()
            .expect("a device instruction belongs to a block")
            .push(instruction);
    }

    pub(super) fn emit(
        &mut self,
        ty: TypeId,
        build: impl FnOnce(ValueId) -> DeviceInstruction,
    ) -> Typed {
        let value = self.compiler.define(ty);
        self.push(build(value));
        Typed { value, ty }
    }

    pub(super) fn declare_local(&mut self, name: &str, ty: TypeId) -> Typed {
        let index = self.function.locals.len() as u32;
        self.function.locals.push(Local {
            name: name.to_owned(),
            ty,
        });
        let pointer = self.compiler.pointer(Space::Function, ty);
        let local = self.emit(pointer, |result| DeviceInstruction::Address {
            address: neura_shader_ir::Address::Local(index),
            result,
        });
        assert!(
            self.scopes
                .last_mut()
                .expect("a local is declared in a block")
                .insert(name.to_owned(), Symbol::Local(local))
                .is_none(),
            "the Rust device local {name} is declared twice in a block"
        );
        local
    }

    pub(super) fn block(&mut self, body: impl FnOnce(&mut Self)) -> Block {
        self.scopes.push(HashMap::new());
        self.blocks.push(Vec::new());
        body(self);
        self.scopes.pop();
        self.blocks.pop().expect("a nested device block exists")
    }

    fn symbol(&self, name: &str) -> Option<Symbol> {
        self.scopes
            .iter()
            .rev()
            .find_map(|scope| scope.get(name).cloned())
    }

    pub(super) fn lookup(&self, name: &str) -> Option<Symbol> {
        self.symbol(name)
            .or_else(|| self.compiler.constant_value(name).map(Symbol::Constant))
    }

    pub(super) fn rust_type(&mut self, ty: &ir::Type) -> TypeId {
        self.compiler.rust_type(ty)
    }
}
