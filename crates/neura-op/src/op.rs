use neura_abi::Kind;
use neura_ir as ir;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Family {
    Binary,
    Unary,
}

impl Family {
    pub const fn operands(self) -> u32 {
        match self {
            Self::Binary => 2,
            Self::Unary => 1,
        }
    }

    pub const fn kind(self) -> Kind {
        match self {
            Self::Binary => Kind::Binary,
            Self::Unary => Kind::Unary,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Role {
    Operand,
    Other,
    Result,
}

impl Role {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Operand => "x",
            Self::Other => "o",
            Self::Result => "y",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Partial {
    Direct,
    Formula {
        roles: &'static [Role],
        build: fn() -> ir::Expression,
    },
}

impl Partial {
    pub fn formula(self) -> Option<ir::Expression> {
        match self {
            Self::Direct => None,
            Self::Formula { build, .. } => Some(build()),
        }
    }

    pub const fn roles(self) -> &'static [Role] {
        match self {
            Self::Direct => &[],
            Self::Formula { roles, .. } => roles,
        }
    }
}

pub const NONE: u32 = u32::MAX;

macro_rules! role {
    (x) => {
        Role::Operand
    };
    (o) => {
        Role::Other
    };
    (y) => {
        Role::Result
    };
}

macro_rules! partial {
    ((direct)) => {
        Partial::Direct
    };
    (([$($role:ident),*] => $expression:expr)) => {
        Partial::Formula {
            roles: &[$(role!($role)),*],
            build: || neura_macro::expression!($expression),
        }
    };
}

macro_rules! ops {
    (@declare $code:expr;) => {};
    (@declare $code:expr; $family:ident $name:ident = $label:literal apply ($apply:expr) partials [$($partial:tt),*]; $($rest:tt)*) => {
        pub const $name: u32 = $code;
        ops!(@declare $code + 1u32; $($rest)*);
    };
    ($($family:ident $name:ident = $label:literal apply ($apply:expr) partials [$($partial:tt),*];)+) => {
        ops!(@declare 0u32; $($family $name = $label apply ($apply) partials [$($partial),*];)+);

        #[derive(Clone, Copy, Debug)]
        pub struct Op {
            pub code: u32,
            pub family: Family,
            pub name: &'static str,
            pub partials: &'static [Partial],
        }

        impl Op {
            pub fn partial(self, slot: u32) -> Partial {
                *self.partials.get(slot as usize).unwrap_or_else(|| {
                    panic!("op {} carries no operand {slot}", self.name)
                })
            }

            pub fn apply_expression(self) -> ir::Expression {
                match self.code {
                    $($name => neura_macro::expression!($apply),)+
                    code => panic!("op {code} is not a declared pointwise op"),
                }
            }
        }

        pub const OPS: &[Op] = &[$(Op {
            code: $name,
            family: Family::$family,
            name: $label,
            partials: &[$(partial!($partial)),*],
        }),+];

        pub const COUNT: u32 = OPS.len() as u32;
    };
}

ops! {
    Binary ADD = "add" apply (a + b) partials [(direct), (direct)];
    Binary MUL = "mul" apply (a * b) partials [([o] => g * o), ([o] => g * o)];
    Binary SUB = "sub" apply (a - b) partials [(direct), ([] => -g)];
    Binary DIV = "div" apply (a / b) partials [([o] => g / o), ([x, o] => -g * o / (x * x))];
    Binary MAXIMUM = "maximum" apply (max(a, b)) partials [([x, o] => select(0.0, g, x > o)), ([x, o] => select(0.0, g, x > o))];
    Binary MINIMUM = "minimum" apply (min(a, b)) partials [([x, o] => select(0.0, g, x < o)), ([x, o] => select(0.0, g, x < o))];
    Unary RELU = "relu" apply (max(a, 0.0)) partials [([y] => select(0.0, g, y > 0.0))];
    Unary SQRT = "sqrt" apply (sqrt(a)) partials [([y] => g * 0.5 / y)];
    Unary RECIP = "recip" apply (1.0 / a) partials [([y] => -g * y * y)];
    Unary EXP = "exp" apply (exp(a)) partials [([y] => g * y)];
    Unary LOG = "log" apply (log(a)) partials [([y] => g * exp(-y))];
    Unary TANH = "tanh" apply (tanh(a)) partials [([y] => g * (1.0 - y * y))];
    Unary SIGMOID = "sigmoid" apply (sigmoid(a)) partials [([y] => g * y * (1.0 - y))];
    Unary NEG = "neg" apply (-a) partials [([] => -g)];
    Unary ABS = "abs" apply (abs(a)) partials [([x] => select(-g, g, x > 0.0))];
    Unary IDENTITY = "identity" apply (a) partials [(direct)];
}

pub fn of(code: u32) -> &'static Op {
    OPS.get(code as usize)
        .filter(|op| op.code == code)
        .unwrap_or_else(|| panic!("op {code} is not a declared pointwise op"))
}

pub fn kind(code: u32) -> Kind {
    of(code).family.kind()
}

pub fn name(code: u32) -> &'static str {
    of(code).name
}
