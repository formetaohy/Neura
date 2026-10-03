use std::collections::HashMap;

#[derive(Clone, Debug)]
pub enum Type {
    Named(String),
    Array {
        element: Box<Type>,
        length: Box<Expression>,
    },
}

#[derive(Clone, Copy, Debug)]
pub enum IntegerType {
    Inferred,
    Signed,
    Unsigned,
}

#[derive(Clone, Copy, Debug)]
pub enum UnaryOperator {
    Negate,
    Not,
}

#[derive(Clone, Copy, Debug)]
pub enum BinaryOperator {
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    BitAnd,
    BitOr,
    BitXor,
    LogicalAnd,
    LogicalOr,
    ShiftLeft,
    ShiftRight,
}

#[derive(Clone, Debug)]
pub enum Expression {
    Integer {
        value: u32,
        ty: IntegerType,
    },
    Float(f32),
    Bool(bool),
    Name(String),
    Field {
        base: Box<Self>,
        name: String,
    },
    Index {
        base: Box<Self>,
        index: Box<Self>,
    },
    Unary {
        op: UnaryOperator,
        value: Box<Self>,
    },
    Binary {
        op: BinaryOperator,
        left: Box<Self>,
        right: Box<Self>,
    },
    Call {
        name: String,
        arguments: Vec<Self>,
    },
    Repeat {
        value: Box<Self>,
        length: Box<Self>,
    },
    Cast {
        value: Box<Self>,
        ty: Type,
    },
    Reference(Box<Self>),
}

impl Expression {
    pub fn name(name: impl Into<String>) -> Self {
        Self::Name(name.into())
    }

    pub fn u32(value: u32) -> Self {
        Self::Integer {
            value,
            ty: IntegerType::Unsigned,
        }
    }

    pub fn field(base: Self, name: impl Into<String>) -> Self {
        Self::Field {
            base: Box::new(base),
            name: name.into(),
        }
    }

    pub fn call(name: impl Into<String>, arguments: Vec<Self>) -> Self {
        Self::Call {
            name: name.into(),
            arguments,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Argument {
    pub name: String,
    pub ty: Type,
}

#[derive(Clone, Debug)]
pub struct Function {
    pub name: String,
    pub arguments: Vec<Argument>,
    pub result: Option<Type>,
    pub body: Vec<Statement>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Pattern {
    Constant(String),
    Integer(u32),
    Default,
}

#[derive(Clone, Debug)]
pub struct Arm {
    pub pattern: Pattern,
    pub body: Vec<Statement>,
}

#[derive(Clone, Debug)]
pub enum Statement {
    Let {
        name: String,
        mutable: bool,
        value: Expression,
    },
    Assign {
        place: Expression,
        value: Expression,
        operator: Option<BinaryOperator>,
    },
    If {
        condition: Expression,
        accept: Vec<Self>,
        reject: Vec<Self>,
    },
    Match {
        selector: Expression,
        arms: Vec<Arm>,
    },
    For {
        name: String,
        start: Expression,
        end: Expression,
        step: Expression,
        unroll: bool,
        body: Vec<Self>,
    },
    While {
        condition: Expression,
        body: Vec<Self>,
    },
    Loop(Vec<Self>),
    Return(Option<Expression>),
    Break,
    Continue,
    Block(Vec<Self>),
    Expression(Expression),
}

pub fn match_count(statements: &[Statement]) -> usize {
    statements
        .iter()
        .map(|statement| match statement {
            Statement::Match { arms, .. } => {
                1 + arms.iter().map(|arm| match_count(&arm.body)).sum::<usize>()
            }
            Statement::If { accept, reject, .. } => match_count(accept) + match_count(reject),
            Statement::For { body, .. }
            | Statement::While { body, .. }
            | Statement::Loop(body)
            | Statement::Block(body) => match_count(body),
            _ => 0,
        })
        .sum()
}

pub fn match_cases(statements: &mut [Statement]) -> Option<&mut Vec<Arm>> {
    for statement in statements {
        match statement {
            Statement::Match { arms, .. } => return Some(arms),
            Statement::If { accept, reject, .. } => {
                if let Some(cases) = match_cases(accept) {
                    return Some(cases);
                }
                if let Some(cases) = match_cases(reject) {
                    return Some(cases);
                }
            }
            Statement::For { body, .. }
            | Statement::While { body, .. }
            | Statement::Loop(body)
            | Statement::Block(body) => {
                if let Some(cases) = match_cases(body) {
                    return Some(cases);
                }
            }
            _ => {}
        }
    }
    None
}

impl Expression {
    pub fn specialize(&mut self, constants: &HashMap<String, u32>, suffix: &str) {
        match self {
            Self::Name(name) => {
                if let Some(value) = constants.get(name) {
                    *self = Self::u32(*value);
                }
            }
            Self::Field { base, .. } => base.specialize(constants, suffix),
            Self::Index { base, index } => {
                base.specialize(constants, suffix);
                index.specialize(constants, suffix);
            }
            Self::Unary { value, .. } | Self::Cast { value, .. } | Self::Reference(value) => {
                value.specialize(constants, suffix)
            }
            Self::Binary { left, right, .. } => {
                left.specialize(constants, suffix);
                right.specialize(constants, suffix);
            }
            Self::Call { name, arguments } => {
                if let Some(base) = name.strip_prefix("template_") {
                    *name = format!("{base}_{suffix}");
                }
                for argument in arguments {
                    argument.specialize(constants, suffix);
                }
            }
            Self::Repeat { value, length } => {
                value.specialize(constants, suffix);
                length.specialize(constants, suffix);
            }
            Self::Integer { .. } | Self::Float(_) | Self::Bool(_) => {}
        }
    }
}

impl Statement {
    pub fn specialize(&mut self, constants: &HashMap<String, u32>, suffix: &str) {
        match self {
            Self::Let { value, .. } | Self::Expression(value) => {
                value.specialize(constants, suffix)
            }
            Self::Assign { place, value, .. } => {
                place.specialize(constants, suffix);
                value.specialize(constants, suffix);
            }
            Self::If {
                condition,
                accept,
                reject,
            } => {
                condition.specialize(constants, suffix);
                for statement in accept.iter_mut().chain(reject) {
                    statement.specialize(constants, suffix);
                }
            }
            Self::Match { selector, arms } => {
                selector.specialize(constants, suffix);
                for arm in arms {
                    for statement in &mut arm.body {
                        statement.specialize(constants, suffix);
                    }
                }
            }
            Self::For {
                start,
                end,
                step,
                body,
                ..
            } => {
                for expression in [start, end, step] {
                    expression.specialize(constants, suffix);
                }
                for statement in body {
                    statement.specialize(constants, suffix);
                }
            }
            Self::While { condition, body } => {
                condition.specialize(constants, suffix);
                for statement in body {
                    statement.specialize(constants, suffix);
                }
            }
            Self::Loop(body) | Self::Block(body) => {
                for statement in body {
                    statement.specialize(constants, suffix);
                }
            }
            Self::Return(value) => {
                if let Some(value) = value {
                    value.specialize(constants, suffix);
                }
            }
            Self::Break | Self::Continue => {}
        }
    }
}
