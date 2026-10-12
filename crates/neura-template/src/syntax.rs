use crate::value::Value;

pub(crate) enum Statement {
    Body(String),
    Print(Expression),
    If {
        arms: Vec<(Expression, Vec<Statement>)>,
        otherwise: Vec<Statement>,
    },
    For {
        name: String,
        source: Expression,
        body: Vec<Statement>,
    },
    Set {
        name: String,
        attribute: Option<String>,
        value: Expression,
    },
    Macro {
        name: String,
        parameters: Vec<(String, Option<Expression>)>,
        body: Vec<Statement>,
    },
}

pub(crate) enum Expression {
    Value(Value),
    Name(String),
    Attribute(Box<Expression>, String),
    Index(Box<Expression>, Box<Expression>),
    Slice {
        base: Box<Expression>,
        start: Option<Box<Expression>>,
        stop: Option<Box<Expression>>,
        step: Option<Box<Expression>>,
    },
    Call {
        callee: Box<Expression>,
        positional: Vec<Expression>,
        keywords: Vec<(String, Expression)>,
    },
    Filter {
        input: Box<Expression>,
        name: String,
        positional: Vec<Expression>,
    },
    Test {
        input: Box<Expression>,
        name: String,
        negated: bool,
        positional: Vec<Expression>,
    },
    Conditional {
        accept: Box<Expression>,
        condition: Box<Expression>,
        reject: Box<Expression>,
    },
    Negate(Box<Expression>),
    Not(Box<Expression>),
    And(Box<Expression>, Box<Expression>),
    Or(Box<Expression>, Box<Expression>),
    Compare {
        compare: Compare,
        left: Box<Expression>,
        right: Box<Expression>,
    },
    Add(Box<Expression>, Box<Expression>),
    Subtract(Box<Expression>, Box<Expression>),
    Concat(Box<Expression>, Box<Expression>),
}

#[derive(Clone, Copy)]
pub(crate) enum Compare {
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    In,
    NotIn,
}
