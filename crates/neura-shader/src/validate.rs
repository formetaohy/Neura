use crate::instruction::{Address, AtomicOp, BinaryOp, Instruction, MathFun, UnaryOp};
use crate::module::{Function, Module, Space, element_name};
use crate::ty::{Scalar, Type, TypeId, ValueId};

mod uniformity;

use std::collections::{HashMap, HashSet};
use uniformity::Analysis;

pub struct Verifier<'m> {
    module: &'m Module,
}

impl<'m> Verifier<'m> {
    pub fn new(module: &'m Module) -> Self {
        Self { module }
    }

    pub fn run(&self) {
        self.structure();
        for function in 0..self.module.functions().len() as u32 {
            Checker::new(self.module, function).run();
        }
        Analysis::new(self.module).run();
    }

    fn structure(&self) {
        let entry = self.module.entry();
        assert!(entry.result.is_none(), "a device entry returns no value");
        assert!(
            entry
                .arguments
                .iter()
                .all(|argument| argument.builtin.is_some()),
            "every device entry argument is a builtin"
        );
        let mut slots = self
            .module
            .globals()
            .iter()
            .filter_map(|global| global.binding.map(|binding| (binding.binding, global)))
            .collect::<Vec<_>>();
        slots.sort_by_key(|(slot, _)| *slot);
        for (index, (slot, global)) in slots.iter().enumerate() {
            assert_eq!(
                *slot as usize, index,
                "storage slots are dense but {} occupies {slot}",
                global.name
            );
            assert_eq!(
                global.space,
                Space::Storage,
                "the device global {} is bound yet lives in {} memory",
                global.name,
                global.space.name()
            );
        }
        for global in self.module.globals() {
            assert!(
                global.binding.is_some() || global.space == Space::WorkGroup,
                "the device global {} is neither bound nor a workgroup variable",
                global.name
            );
            assert!(
                !global.coherent || global.access.writable(),
                "the read-only device global {} claims to be coherent",
                global.name
            );
            self.storable(global.ty, &format!("the device global {}", global.name));
            if global.space == Space::Storage {
                let direct = matches!(self.module.ty(global.ty), Type::Array { count: None, .. });
                assert!(
                    direct || !self.runtime_array_inside(global.ty),
                    "the device global {} holds a runtime sized array beyond its own span",
                    global.name
                );
            }
        }
        for ty in self.module.types() {
            let Type::Struct {
                name,
                members,
                span,
            } = ty
            else {
                continue;
            };
            let mut end = 0;
            for member in members {
                assert!(
                    self.module.ty(member.ty).scalar() != Some(Scalar::Bool),
                    "the device struct {name} holds a boolean member {}",
                    member.name
                );
                assert!(
                    member.offset >= end,
                    "the device struct {name} overlaps its member {}",
                    member.name
                );
                assert!(
                    member.offset + self.module.size(member.ty) <= *span,
                    "the device struct {name} member {} outruns its span",
                    member.name
                );
                end = member.offset + self.module.size(member.ty);
            }
        }
    }

    fn runtime_array_inside(&self, ty: TypeId) -> bool {
        match self.module.ty(ty) {
            Type::Array { element, count } => {
                count.is_none() || self.runtime_array_inside(*element)
            }
            Type::Struct { members, .. } => members
                .iter()
                .any(|member| self.runtime_array_inside(member.ty)),
            _ => false,
        }
    }

    fn storable(&self, ty: TypeId, subject: &str) {
        match self.module.ty(ty) {
            Type::Scalar(scalar) => assert!(
                scalar.storable(),
                "{subject} holds the boolean {}",
                scalar.name()
            ),
            Type::Vector { scalar, .. } => assert!(
                scalar.storable(),
                "{subject} holds the boolean {}",
                scalar.name()
            ),
            Type::Array { element, .. } => self.storable(*element, subject),
            Type::Struct { members, .. } => {
                for member in members {
                    self.storable(member.ty, subject);
                }
            }
            Type::Atomic(scalar) => assert!(
                scalar.integer(),
                "{subject} atomically holds the non-integer {}",
                scalar.name()
            ),
            Type::Pointer { .. } => panic!("{subject} holds a device pointer"),
            Type::CooperativeMatrix { .. } => {
                panic!("{subject} holds a cooperative matrix")
            }
        }
    }
}

struct Checker<'m> {
    module: &'m Module,
    function: &'m Function,
    defined: HashSet<ValueId>,
    declarations: HashMap<ValueId, &'m Instruction>,
    loops: usize,
}

impl<'m> Checker<'m> {
    fn new(module: &'m Module, index: u32) -> Self {
        let function = &module.functions()[index as usize];
        let mut declarations = HashMap::new();
        collect(&function.body, &mut declarations);
        Self {
            module,
            function,
            defined: HashSet::new(),
            declarations,
            loops: 0,
        }
    }

    fn run(mut self) {
        self.block(&self.function.body);
    }

    fn block(&mut self, block: &[Instruction]) {
        for instruction in block {
            self.instruction(instruction);
            if let Some(result) = instruction.result() {
                if let Some(expected) = self.expected(instruction) {
                    assert_eq!(
                        self.module.value_ty(result),
                        expected,
                        "the device instruction {instruction:?} defines a value of the wrong type"
                    );
                }
                assert!(
                    self.module.constant_of(result).is_none(),
                    "the device function {} carries the instruction {instruction:?} whose result {} is a constant",
                    self.function.name,
                    result.index()
                );
                assert!(
                    self.defined.insert(result),
                    "the device value {} is defined twice",
                    result.0
                );
            }
        }
    }

    fn instruction(&mut self, instruction: &Instruction) {
        for operand in instruction.operands() {
            assert!(
                self.defined.contains(&operand) || self.module.constant_of(operand).is_some(),
                "the device value {} is used before it is defined in {}",
                operand.0,
                self.function.name
            );
        }
        match instruction {
            Instruction::Argument { index, .. } => assert!(
                (*index as usize) < self.function.arguments.len(),
                "the device function {} has no argument {index}",
                self.function.name
            ),
            Instruction::Address { address, .. } => self.address(*address),
            Instruction::Access { base, index, .. } => {
                let (space, element) = self.pointee(*base, "an indexed access");
                self.index(*index);
                self.expect_pointer(instruction, space, self.module.element(element));
            }
            Instruction::AccessIndex { base, index, .. } => {
                self.access_index(*base, *index, instruction);
            }
            Instruction::Load { pointer, .. } => {
                let (_, base) = self.pointee(*pointer, "a load");
                assert!(
                    !matches!(self.module.ty(base), Type::Array { .. }),
                    "a device array is read as a whole instead of element by element"
                );
            }
            Instruction::Store { pointer, value } => {
                self.writable(*pointer, "a device store");
                let pointer_ty = self.ty(*pointer);
                let (_, base) = self.pointee(*pointer, "a store");
                assert_eq!(
                    self.ty(*value),
                    self.module.loaded_ty(pointer_ty),
                    "a device store writes {} where {} stands",
                    element_name(self.module.ty(self.ty(*value))),
                    element_name(self.module.ty(base))
                );
            }
            Instruction::Unary { op, value, .. } => {
                let scalar = self.scalar(*value, "a unary operation");
                match op {
                    UnaryOp::Negate => assert!(
                        scalar.floating() || scalar.signed(),
                        "a device negation applies to the unsigned {}",
                        scalar.name()
                    ),
                    UnaryOp::LogicalNot => assert_eq!(
                        scalar,
                        Scalar::Bool,
                        "a device logical negation applies to the {}",
                        scalar.name()
                    ),
                    UnaryOp::BitwiseNot => assert!(
                        scalar.integer(),
                        "a device bitwise negation applies to the {}",
                        scalar.name()
                    ),
                }
            }
            Instruction::Binary {
                op,
                left,
                right,
                result,
            } => {
                let outer = self.lane(*left);
                match op {
                    BinaryOp::LogicalAnd | BinaryOp::LogicalOr => assert_eq!(
                        outer,
                        Scalar::Bool,
                        "a device logical operation applies to the {}",
                        outer.name()
                    ),
                    BinaryOp::And
                    | BinaryOp::Or
                    | BinaryOp::Xor
                    | BinaryOp::Modulo
                    | BinaryOp::ShiftLeft
                    | BinaryOp::ShiftRight => assert!(
                        outer.integer(),
                        "a device bitwise operation applies to the {}",
                        outer.name()
                    ),
                    _ => {}
                }
                let left_ty = self.ty(*left);
                let right_ty = self.ty(*right);
                let result_ty = self.ty(*result);
                if *op == BinaryOp::Multiply
                    && (self.lanes(left_ty) == 1) != (self.lanes(right_ty) == 1)
                {
                    let vector = if self.lanes(left_ty) == 1 {
                        right_ty
                    } else {
                        left_ty
                    };
                    let scalar = if self.lanes(left_ty) == 1 {
                        left_ty
                    } else {
                        right_ty
                    };
                    assert_eq!(
                        self.lane_of(left_ty),
                        self.lane_of(right_ty),
                        "a device product scales {} by {}",
                        element_name(self.module.ty(vector)),
                        element_name(self.module.ty(scalar))
                    );
                    assert_eq!(
                        result_ty,
                        vector,
                        "a device product in {} of a {} and a {} yields {}",
                        self.function.name,
                        element_name(self.module.ty(vector)),
                        element_name(self.module.ty(scalar)),
                        element_name(self.module.ty(result_ty))
                    );
                    return;
                }
                if matches!(op, BinaryOp::ShiftLeft | BinaryOp::ShiftRight) {
                    self.integer(*right, "a shift");
                    assert_eq!(result_ty, left_ty, "a device shift changes its type");
                    return;
                }
                assert_eq!(
                    left_ty,
                    right_ty,
                    "a device binary operation mixes {} and {}",
                    element_name(self.module.ty(left_ty)),
                    element_name(self.module.ty(right_ty))
                );
                if op.comparison() {
                    assert_eq!(
                        self.lane(*result),
                        Scalar::Bool,
                        "a device comparison yields {}",
                        element_name(self.module.ty(result_ty))
                    );
                    assert_eq!(
                        self.lanes(left_ty),
                        self.lanes(result_ty),
                        "a device comparison changes its lane count"
                    );
                } else {
                    assert_eq!(
                        result_ty, left_ty,
                        "a device binary operation changes its type"
                    );
                }
            }
            Instruction::Select {
                condition,
                accept,
                reject,
                ..
            } => {
                assert_eq!(
                    self.scalar(*condition, "a device select"),
                    Scalar::Bool,
                    "a device select is conditioned on a number"
                );
                assert_eq!(
                    self.ty(*accept),
                    self.ty(*reject),
                    "a device select chooses between different types"
                );
            }
            Instruction::Convert { value, result } => {
                let source = self.ty(*value);
                let target = self.ty(*result);
                assert_eq!(
                    self.lanes(source),
                    self.lanes(target),
                    "a device conversion changes its lane count"
                );
                assert!(
                    self.number(source) && self.number(target),
                    "a device conversion mixes {} and {}",
                    element_name(self.module.ty(source)),
                    element_name(self.module.ty(target))
                );
            }
            Instruction::Bitcast { value, result } => assert_eq!(
                self.module.size(self.ty(*value)),
                self.module.size(self.ty(*result)),
                "a device bitcast reinterprets a different width"
            ),
            Instruction::Math {
                fun,
                arguments,
                result,
            } => {
                assert_eq!(
                    arguments.len(),
                    fun.arity(),
                    "a device {} takes {} arguments",
                    fun.name(),
                    fun.arity()
                );
                if *fun == MathFun::UnpackHalf2x16 {
                    assert_eq!(
                        self.ty(arguments[0]),
                        self.module.scalar_id(Scalar::U32),
                        "a device half unpacking reads {}",
                        element_name(self.module.ty(self.ty(arguments[0])))
                    );
                    assert!(
                        matches!(
                            self.module.ty(self.ty(*result)),
                            Type::Vector {
                                scalar: Scalar::F32,
                                length: 2
                            }
                        ),
                        "a device half unpacking yields {}",
                        element_name(self.module.ty(self.ty(*result)))
                    );
                } else {
                    for argument in arguments {
                        assert_eq!(
                            self.ty(*argument),
                            self.ty(arguments[0]),
                            "a device {} mixes its argument types",
                            fun.name()
                        );
                    }
                    assert_eq!(
                        self.ty(*result),
                        self.ty(arguments[0]),
                        "a device {} changes its type",
                        fun.name()
                    );
                    let scalar = self.scalar(*result, "a device math function");
                    assert!(
                        scalar.floating() || matches!(fun, MathFun::Min | MathFun::Max),
                        "a device {} applies to the {}",
                        fun.name(),
                        scalar.name()
                    );
                }
            }
            Instruction::Compose {
                constituents,
                result,
            } => {
                let Type::Vector { scalar, length } = self.module.ty(self.ty(*result)) else {
                    panic!(
                        "a device composition builds {}",
                        element_name(self.module.ty(self.ty(*result)))
                    );
                };
                assert_eq!(
                    constituents.len() as u32,
                    *length,
                    "a device composition of {length} lanes takes {length} values"
                );
                for constituent in constituents {
                    assert_eq!(
                        self.ty(*constituent),
                        self.module.scalar_id(*scalar),
                        "a device composition mixes its component types"
                    );
                }
            }
            Instruction::Call {
                function,
                arguments,
                result,
            } => {
                let callee = &self.module.functions()[*function as usize];
                assert_eq!(
                    arguments.len(),
                    callee.arguments.len(),
                    "the device call to {} takes {} arguments",
                    callee.name,
                    callee.arguments.len()
                );
                for (argument, parameter) in arguments.iter().zip(&callee.arguments) {
                    assert_eq!(
                        self.ty(*argument),
                        parameter.ty,
                        "the device call to {} passes {} where {} stands",
                        callee.name,
                        element_name(self.module.ty(self.ty(*argument))),
                        element_name(self.module.ty(parameter.ty))
                    );
                }
                assert_eq!(
                    result.is_some(),
                    callee.returns_value(),
                    "the device call to {} disagrees about its result",
                    callee.name
                );
            }
            Instruction::Atomic {
                op,
                pointer,
                value,
                result,
            } => {
                self.writable(*pointer, "a device atomic");
                let (_, base) = self.pointee(*pointer, "an atomic operation");
                let Type::Atomic(scalar) = self.module.ty(base) else {
                    panic!(
                        "a device {} operates on {} instead of an atomic",
                        op.name(),
                        element_name(self.module.ty(base))
                    );
                };
                assert!(
                    *op == AtomicOp::Exchange || scalar.integer(),
                    "a device {} operates on the {}",
                    op.name(),
                    scalar.name()
                );
                assert_eq!(
                    self.ty(*value),
                    self.module.scalar_id(*scalar),
                    "a device {} counts {}",
                    op.name(),
                    element_name(self.module.ty(self.ty(*value)))
                );
                assert_eq!(
                    self.ty(*result),
                    self.module.scalar_id(*scalar),
                    "a device {} yields a different type",
                    op.name()
                );
            }
            Instruction::WorkGroupUniformLoad { pointer, .. } => assert_eq!(
                self.pointee(*pointer, "a uniform load").0,
                Space::WorkGroup,
                "a device uniform load reads memory no invocation shares"
            ),
            Instruction::Barrier(_) => {}
            Instruction::MatrixFill { value, result } => {
                let scalar = self.matrix_scalar(*result, "a matrix fill");
                assert_eq!(
                    self.ty(*value),
                    self.module.scalar_id(scalar),
                    "a device matrix fill writes {}",
                    element_name(self.module.ty(self.ty(*value)))
                );
            }
            Instruction::MatrixLoad {
                pointer,
                stride,
                result,
                ..
            } => {
                let scalar = self.matrix_scalar(*result, "a matrix load");
                let (_, base) = self.pointee(*pointer, "a matrix load");
                let element = self.matrix_element(base, "a matrix load");
                assert_eq!(
                    element,
                    scalar,
                    "a device matrix load reads {} where {} stands",
                    element.name(),
                    scalar.name()
                );
                self.integer(*stride, "a matrix stride");
            }
            Instruction::MatrixStore {
                pointer,
                value,
                stride,
                ..
            } => {
                let scalar = self.matrix_scalar(*value, "a matrix store");
                self.writable(*pointer, "a device matrix store");
                let (_, base) = self.pointee(*pointer, "a matrix store");
                let element = self.matrix_element(base, "a matrix store");
                assert_eq!(
                    element,
                    scalar,
                    "a device matrix store writes {} where {} stands",
                    scalar.name(),
                    element.name()
                );
                self.integer(*stride, "a matrix stride");
            }
            Instruction::MatrixMulAdd {
                left,
                right,
                accumulate,
                result,
            } => {
                let (scalar, rows, columns) = self.matrix(*result, "a matrix multiply");
                let (left_scalar, left_rows, left_columns) =
                    self.matrix(*left, "a matrix multiply");
                let (right_scalar, right_rows, right_columns) =
                    self.matrix(*right, "a matrix multiply");
                assert_eq!(
                    self.ty(*accumulate),
                    self.ty(*result),
                    "a device matrix multiply accumulates into a different matrix"
                );
                assert_eq!(
                    left_rows, rows,
                    "a device matrix multiply disagrees about its rows"
                );
                assert_eq!(
                    left_columns, right_rows,
                    "a device matrix multiply walks a different depth"
                );
                assert_eq!(
                    right_columns, columns,
                    "a device matrix multiply disagrees about its columns"
                );
                assert_eq!(
                    left_scalar,
                    right_scalar,
                    "a device matrix multiply mixes {} and {} inputs",
                    left_scalar.name(),
                    right_scalar.name()
                );
                assert!(
                    scalar == left_scalar || scalar == Scalar::F32 && left_scalar == Scalar::F16,
                    "a device matrix multiply accumulates {} inputs into {}",
                    left_scalar.name(),
                    scalar.name()
                );
            }
            Instruction::MatrixLength { ty, result } => {
                self.matrix_ty(*ty, "a matrix length");
                assert_eq!(
                    self.ty(*result),
                    self.module.scalar_id(Scalar::U32),
                    "a device matrix length is not a word"
                );
            }
            Instruction::MatrixExtract {
                value,
                index,
                result,
            } => {
                let scalar = self.matrix_scalar(*value, "a matrix extract");
                self.integer(*index, "a matrix index");
                assert_eq!(
                    self.ty(*result),
                    self.module.scalar_id(scalar),
                    "a device matrix extract yields {}",
                    element_name(self.module.ty(self.ty(*result)))
                );
            }
            Instruction::MatrixInsert {
                value,
                index,
                component,
                result,
            } => {
                let scalar = self.matrix_scalar(*value, "a matrix insert");
                self.integer(*index, "a matrix index");
                assert_eq!(
                    self.ty(*component),
                    self.module.scalar_id(scalar),
                    "a device matrix insert takes {}",
                    element_name(self.module.ty(self.ty(*component)))
                );
                assert_eq!(
                    self.ty(*result),
                    self.ty(*value),
                    "a device matrix insert changes its matrix"
                );
            }
            Instruction::If {
                condition,
                accept,
                reject,
            } => {
                assert_eq!(
                    self.scalar(*condition, "a device condition"),
                    Scalar::Bool,
                    "a device branch tests a number"
                );
                self.block(accept);
                self.block(reject);
            }
            Instruction::Switch {
                selector,
                cases,
                default,
            } => {
                self.integer(*selector, "a device switch");
                let mut seen = HashSet::new();
                for (value, body) in cases {
                    assert!(
                        seen.insert(*value),
                        "a device switch repeats the case {value}"
                    );
                    self.block(body);
                }
                self.block(default);
            }
            Instruction::Loop { body, continuing } => {
                self.loops += 1;
                self.block(body);
                self.block(continuing);
                self.loops -= 1;
            }
            Instruction::Break | Instruction::Continue => assert!(
                self.loops > 0,
                "a device loop control stands outside a loop"
            ),
            Instruction::Return { value } => assert_eq!(
                value.map(|value| self.ty(value)),
                self.function.result,
                "the device function {} returns a different type",
                self.function.name
            ),
            Instruction::Block(body) => self.block(body),
        }
    }

    fn expected(&self, instruction: &Instruction) -> Option<TypeId> {
        match instruction {
            Instruction::Unary { value, .. } => Some(self.ty(*value)),
            Instruction::Binary { op, left, .. } => {
                if op.comparison() {
                    Some(self.module.scalar_id(Scalar::Bool))
                } else {
                    Some(self.ty(*left))
                }
            }
            Instruction::Select { accept, .. } => Some(self.ty(*accept)),
            Instruction::Math { fun, arguments, .. } => {
                if *fun == MathFun::UnpackHalf2x16 {
                    None
                } else {
                    Some(self.ty(arguments[0]))
                }
            }
            Instruction::Load { pointer, .. } => Some(self.module.loaded_ty(self.ty(*pointer))),
            Instruction::Atomic { pointer, .. } => Some(self.module.loaded_ty(self.ty(*pointer))),
            Instruction::WorkGroupUniformLoad { pointer, .. } => {
                Some(self.module.loaded_ty(self.ty(*pointer)))
            }
            Instruction::Convert { result, .. } | Instruction::Bitcast { result, .. } => {
                Some(self.module.value_ty(*result))
            }
            _ => None,
        }
    }

    fn access_index(&self, base: ValueId, index: u32, instruction: &Instruction) {
        match self.module.ty(self.ty(base)) {
            Type::Pointer { space, base } => {
                let element = match self.module.ty(*base) {
                    Type::Array { element, .. } => *element,
                    Type::Struct { members, .. } => {
                        members
                            .get(index as usize)
                            .unwrap_or_else(|| panic!("a device struct has no member {index}"))
                            .ty
                    }
                    Type::Vector { scalar, .. } => self.module.scalar_id(*scalar),
                    other => panic!("a device indexes {}", element_name(other)),
                };
                self.expect_pointer(instruction, *space, element);
            }
            Type::Vector { scalar, length } => {
                assert!(index < *length, "a device vector has no lane {index}");
                assert_eq!(
                    self.module
                        .value_ty(instruction.result().expect("an access has a result")),
                    self.module.scalar_id(*scalar),
                    "a device lane access yields a different type"
                );
            }
            Type::Struct { members, .. } => {
                let member = members
                    .get(index as usize)
                    .unwrap_or_else(|| panic!("a device struct has no member {index}"));
                assert_eq!(
                    self.module
                        .value_ty(instruction.result().expect("an access has a result")),
                    member.ty,
                    "a device member access yields a different type"
                );
            }
            other => panic!("a device indexes {}", element_name(other)),
        }
    }

    fn address(&self, address: Address) {
        match address {
            Address::Global(index) => assert!(
                (index as usize) < self.module.globals().len(),
                "a device instruction addresses an undeclared global"
            ),
            Address::Local(index) => assert!(
                (index as usize) < self.function.locals.len(),
                "a device instruction addresses an undeclared local"
            ),
        }
    }

    fn expect_pointer(&self, instruction: &Instruction, space: Space, base: TypeId) {
        let result = self
            .module
            .value_ty(instruction.result().expect("an access has a result"));
        assert_eq!(
            self.module.ty(result),
            &Type::Pointer { space, base },
            "a device access does not reach into its base"
        );
    }

    fn writable(&self, pointer: ValueId, subject: &str) {
        let Some(index) = self.root(pointer) else {
            return;
        };
        let global = self.module.global(index);
        assert!(
            global.access.writable(),
            "{subject} writes into the read-only device global {}",
            global.name
        );
    }

    fn root(&self, value: ValueId) -> Option<u32> {
        match self.declarations.get(&value).copied()? {
            Instruction::Address {
                address: Address::Global(index),
                ..
            } => Some(*index),
            Instruction::Access { base, .. } | Instruction::AccessIndex { base, .. } => {
                self.root(*base)
            }
            _ => None,
        }
    }

    fn matrix(&self, value: ValueId, subject: &str) -> (Scalar, u32, u32) {
        self.matrix_ty(self.ty(value), subject)
    }

    fn matrix_element(&self, ty: TypeId, subject: &str) -> Scalar {
        match self.module.ty(ty) {
            Type::Scalar(scalar) => *scalar,
            Type::Vector { scalar, .. } => *scalar,
            other => panic!("{subject} reads {}", element_name(other)),
        }
    }

    fn matrix_ty(&self, ty: TypeId, subject: &str) -> (Scalar, u32, u32) {
        match self.module.ty(ty) {
            Type::CooperativeMatrix {
                scalar,
                rows,
                columns,
                ..
            } => (*scalar, *rows, *columns),
            other => panic!(
                "{subject} operates on {} instead of a cooperative matrix",
                element_name(other)
            ),
        }
    }

    fn matrix_scalar(&self, value: ValueId, subject: &str) -> Scalar {
        self.matrix(value, subject).0
    }

    fn pointee(&self, pointer: ValueId, subject: &str) -> (Space, TypeId) {
        match self.module.ty(self.ty(pointer)) {
            Type::Pointer { space, base } => (*space, *base),
            other => panic!("{subject} dereferences {}", element_name(other)),
        }
    }

    fn ty(&self, value: ValueId) -> TypeId {
        self.module.value_ty(value)
    }

    fn scalar(&self, value: ValueId, subject: &str) -> Scalar {
        let ty = self.ty(value);
        self.module.ty(ty).scalar().unwrap_or_else(|| {
            panic!(
                "{subject} operates on {} instead of a scalar",
                element_name(self.module.ty(ty))
            )
        })
    }

    fn index(&self, value: ValueId) {
        let ty = self.ty(value);
        assert!(
            matches!(self.module.ty(ty), Type::Scalar(Scalar::U32 | Scalar::I32)),
            "a device index is {}",
            element_name(self.module.ty(ty))
        );
    }

    fn integer(&self, value: ValueId, subject: &str) {
        let scalar = self.lane(value);
        assert!(scalar.integer(), "{subject} counts the {}", scalar.name());
    }

    fn lane(&self, value: ValueId) -> Scalar {
        self.lane_of(self.ty(value))
    }

    fn lane_of(&self, ty: TypeId) -> Scalar {
        match self.module.ty(ty) {
            Type::Scalar(scalar) => *scalar,
            Type::Vector { scalar, .. } => *scalar,
            other => panic!("a device operand is {}", element_name(other)),
        }
    }

    fn number(&self, ty: TypeId) -> bool {
        matches!(
            self.module.ty(ty),
            Type::Scalar(Scalar::U32 | Scalar::I32 | Scalar::F32 | Scalar::F16)
                | Type::Vector { .. }
        )
    }

    fn lanes(&self, ty: TypeId) -> u32 {
        match self.module.ty(ty) {
            Type::Scalar(_) => 1,
            Type::Vector { length, .. } => *length,
            other => panic!("{} is not a device number", element_name(other)),
        }
    }
}

fn collect<'m>(block: &'m [Instruction], declarations: &mut HashMap<ValueId, &'m Instruction>) {
    for instruction in block {
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
