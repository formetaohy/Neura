mod statement;
mod value;

use crate::{Compiler, ir};
use naga::{
    BuiltIn, Expression, Function, FunctionArgument, FunctionResult, Handle, Statement, Type,
};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone, Copy)]
struct Typed {
    expr: Handle<Expression>,
    ty: Handle<Type>,
}

#[derive(Clone)]
enum Symbol {
    Value(Typed),
    Local(Typed),
    Array(Arc<[Typed]>),
    Constant(u32),
}

pub(super) struct FunctionLower<'a> {
    compiler: &'a mut Compiler,
    function: Function,
    scopes: Vec<HashMap<String, Symbol>>,
    blocks: Vec<naga::Block>,
}

impl<'a> FunctionLower<'a> {
    pub(super) fn new(compiler: &'a mut Compiler, source: &ir::Function, entry: bool) -> Self {
        let result = source.result.as_ref().map(|ty| FunctionResult {
            ty: compiler.rust_type(ty),
            binding: None,
        });
        let mut function = Function {
            name: Some(source.name.clone()),
            result,
            ..Default::default()
        };
        let mut arguments = HashMap::new();
        for argument in &source.arguments {
            let name = argument.name.clone();
            let ty = compiler.rust_type(&argument.ty);
            let binding = if entry {
                Some(naga::Binding::BuiltIn(match name.as_str() {
                    "lid" => BuiltIn::LocalInvocationIndex,
                    "group" => BuiltIn::WorkGroupId,
                    "global" => BuiltIn::GlobalInvocationId,
                    other => panic!("entry argument {other} is not a device builtin"),
                }))
            } else {
                None
            };
            let index = function.arguments.len() as u32;
            function.arguments.push(FunctionArgument {
                name: Some(name.clone()),
                ty,
                binding,
            });
            let expr = function
                .expressions
                .append(Expression::FunctionArgument(index), naga::Span::UNDEFINED);
            assert!(
                arguments
                    .insert(name, Symbol::Value(Typed { expr, ty }))
                    .is_none(),
                "a device argument name is unique"
            );
        }
        Self {
            compiler,
            function,
            scopes: vec![arguments],
            blocks: vec![naga::Block::new()],
        }
    }

    pub(super) fn lower(mut self, body: &[ir::Statement]) -> Function {
        self.statements(body);
        self.function.body = self.blocks.pop().expect("a function has a body");
        self.function
    }

    fn push(&mut self, statement: Statement) {
        self.blocks
            .last_mut()
            .expect("a device expression belongs to a block")
            .push(statement, naga::Span::UNDEFINED);
    }

    fn emit(&mut self, expression: Expression, ty: Handle<Type>) -> Typed {
        let immediate = matches!(
            expression,
            Expression::Literal(_)
                | Expression::Constant(_)
                | Expression::ZeroValue(_)
                | Expression::FunctionArgument(_)
                | Expression::LocalVariable(_)
                | Expression::GlobalVariable(_)
                | Expression::CallResult(_)
        );
        let start = self.function.expressions.len();
        let expr = self
            .function
            .expressions
            .append(expression, naga::Span::UNDEFINED);
        if !immediate {
            let range = self.function.expressions.range_from(start);
            self.push(Statement::Emit(range));
        }
        Typed { expr, ty }
    }

    fn block(&mut self, body: impl FnOnce(&mut Self)) -> naga::Block {
        self.scopes.push(HashMap::new());
        self.blocks.push(naga::Block::new());
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

    fn lookup(&self, name: &str) -> Option<Symbol> {
        self.symbol(name).or_else(|| {
            self.compiler
                .constants
                .get(name)
                .copied()
                .map(Symbol::Constant)
        })
    }
}

impl Compiler {
    fn rust_type(&mut self, ty: &ir::Type) -> Handle<Type> {
        match ty {
            ir::Type::Named(name) => self.ty(name),
            ir::Type::Array { element, length } => {
                let element = self.rust_type(element);
                let count = self.evaluate(length);
                self.array(element, Some(count))
            }
        }
    }

    fn evaluate(&self, expr: &ir::Expression) -> u32 {
        use ir::{BinaryOperator as Op, Expression as E, IntegerType};
        match expr {
            E::Integer { value, ty } if !matches!(ty, IntegerType::Signed) => *value,
            E::Name(name) => *self
                .constants
                .get(name)
                .unwrap_or_else(|| panic!("unknown device constant {name}")),
            E::Binary { op, left, right } => {
                let left = self.evaluate(left);
                let right = self.evaluate(right);
                match op {
                    Op::Add => left.checked_add(right).expect("device constant overflows"),
                    Op::Subtract => left.checked_sub(right).expect("device constant underflows"),
                    Op::Multiply => left.checked_mul(right).expect("device constant overflows"),
                    Op::Divide => left
                        .checked_div(right)
                        .expect("device constant divides by zero"),
                    _ => panic!("unsupported device constant expression"),
                }
            }
            other => panic!("a device constant must be known at compilation: {other:?}"),
        }
    }
}
