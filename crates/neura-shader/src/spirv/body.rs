use crate::spirv::op::*;
use crate::spirv::writer::Writer;
use crate::{
    Address, AtomicOp, Barrier, BinaryOp, Constant, Instruction as DeviceInstruction, MathFun,
    MatrixLayout, Scalar, Space, Type, UnaryOp, ValueId,
};
use std::collections::HashSet;

struct Block {
    label: u32,
    body: Vec<Instruction>,
    reachable: bool,
}

#[derive(Clone, Copy)]
struct Loop {
    merge: u32,
    continuing: u32,
}

pub(crate) struct FunctionWriter<'w, 'm> {
    writer: &'w mut Writer<'m>,
    function: u32,
    blocks: Vec<Block>,
    current: Option<Block>,
    loops: Vec<Loop>,
    reached: HashSet<u32>,
}

impl<'w, 'm> FunctionWriter<'w, 'm> {
    pub(crate) fn new(writer: &'w mut Writer<'m>, function: u32) -> Self {
        Self {
            writer,
            function,
            blocks: Vec::new(),
            current: None,
            loops: Vec::new(),
            reached: HashSet::new(),
        }
    }

    pub(crate) fn run(mut self) {
        let function = self.writer.module.functions()[self.function as usize].clone();
        let id = self.writer.function_ids[self.function as usize];
        let result = function.result.map(|ty| self.writer.ty(ty));
        let void = self.writer.void();
        let function_type = self.function_type(&function, result);
        self.writer.push(
            Instruction::result(FUNCTION, result.unwrap_or(void), id)
                .word(0u32)
                .operand(function_type),
        );
        for (position, argument) in function.arguments.iter().enumerate() {
            if argument.builtin.is_some() {
                continue;
            }
            let value = self.writer.arguments[self.function as usize][position];
            let ty = self.writer.value_types[&value];
            let id = self.writer.values[&value];
            self.writer
                .push(Instruction::result(FUNCTION_PARAMETER, ty, id));
        }
        let entry = self.label();
        let mut body = Vec::new();
        for index in 0..function.locals.len() {
            let (pointer, variable) = self.writer.locals[self.function as usize][index];
            body.push(Instruction::result(VARIABLE, pointer, variable).word(storage::FUNCTION));
        }
        for (position, argument) in function.arguments.iter().enumerate() {
            let Some(builtin) = argument.builtin else {
                continue;
            };
            let value = self.writer.arguments[self.function as usize][position];
            let ty = self.writer.value_types[&value];
            let id = self.writer.values[&value];
            let pointer = self.writer.builtin(builtin);
            body.push(Instruction::result(LOAD, ty, id).operand(pointer));
        }
        self.current = Some(Block {
            label: entry,
            body,
            reachable: true,
        });
        self.block(&function.body);
        self.finish();
    }

    fn function_type(&mut self, function: &crate::Function, result: Option<u32>) -> u32 {
        let parameters = function
            .arguments
            .iter()
            .filter(|argument| argument.builtin.is_none())
            .map(|argument| self.writer.ty(argument.ty))
            .collect::<Vec<_>>();
        self.writer.function_type(result, &parameters)
    }

    fn finish(&mut self) {
        if let Some(block) = self.current.as_ref() {
            if block.reachable {
                assert!(
                    self.writer.module.functions()[self.function as usize]
                        .result
                        .is_none(),
                    "the device function {} ends without returning a value",
                    self.writer.module.functions()[self.function as usize].name
                );
                self.terminate(Instruction::plain(RETURN));
            } else {
                self.terminate(Instruction::plain(UNREACHABLE));
            }
        }
        for block in std::mem::take(&mut self.blocks) {
            self.writer.push(Instruction::defined(LABEL, block.label));
            for instruction in block.body {
                self.writer.push(instruction);
            }
        }
        self.writer.push(Instruction::plain(FUNCTION_END));
    }

    fn label(&mut self) -> u32 {
        self.writer.id()
    }

    fn start(&mut self, label: u32) {
        assert!(self.current.is_none(), "a device block is opened twice");
        self.current = Some(Block {
            label,
            body: Vec::new(),
            reachable: self.reached.contains(&label),
        });
    }

    fn push(&mut self, instruction: Instruction) {
        if let Some(current) = self.current.as_mut() {
            current.body.push(instruction);
        }
    }

    fn terminate(&mut self, instruction: Instruction) {
        self.push(instruction);
        if let Some(block) = self.current.take() {
            self.blocks.push(block);
        }
    }

    fn branch(&mut self, label: u32) {
        self.reached.insert(label);
        self.terminate(Instruction::plain(BRANCH).operand(Operand::Id(label)));
    }

    fn reach(&mut self, label: u32) {
        if self.current.is_some() {
            self.branch(label);
        }
    }

    fn value(&self, value: ValueId) -> u32 {
        *self
            .writer
            .values
            .get(&value)
            .unwrap_or_else(|| panic!("the device value {} is not defined", value.index()))
    }

    fn is_pointer(&self, value: ValueId) -> bool {
        let ty = self.writer.module.value_ty(value);
        matches!(self.writer.module.ty(ty), Type::Pointer { .. })
    }

    fn wrapped(&self, value: ValueId) -> bool {
        match self.writer.declared.get(&value) {
            Some(DeviceInstruction::Address {
                address: Address::Global(index),
                ..
            }) => self.writer.globals[*index as usize].wrapped,
            _ => false,
        }
    }

    fn pointer_space(&self, value: ValueId) -> Space {
        let ty = self.writer.module.value_ty(value);
        self.writer.module.pointee(ty).0
    }

    fn memory_operands(&mut self, pointer: ValueId, visible: bool) -> Vec<Operand> {
        if !self.writer.vulkan_memory_model {
            return Vec::new();
        }
        let space = self.pointer_space(pointer);
        if space == Space::Function {
            return Vec::new();
        }
        let scope = if space == Space::WorkGroup {
            scope::WORKGROUP
        } else {
            scope::DEVICE
        };
        let mask = memory_access::NON_PRIVATE_POINTER
            | if visible {
                memory_access::MAKE_POINTER_VISIBLE
            } else {
                memory_access::MAKE_POINTER_AVAILABLE
            };
        let scope = self.writer.integer(Scalar::U32, scope);
        vec![Operand::Word(mask), Operand::Id(scope)]
    }

    fn scalar(&self, value: ValueId) -> Scalar {
        let ty = self.writer.module.value_ty(value);
        match self.writer.module.ty(ty) {
            Type::Scalar(scalar) => *scalar,
            Type::Vector { scalar, .. } => *scalar,
            other => panic!(
                "the device value {} is {}",
                value.index(),
                crate::element_name(other)
            ),
        }
    }

    fn lanes(&self, value: ValueId) -> u32 {
        match self.writer.module.ty(self.writer.module.value_ty(value)) {
            Type::Scalar(_) => 1,
            Type::Vector { length, .. } => *length,
            other => panic!(
                "the device value {} is {}",
                value.index(),
                crate::element_name(other)
            ),
        }
    }

    fn block(&mut self, body: &[DeviceInstruction]) {
        for instruction in body {
            self.instruction(instruction);
        }
    }

    fn instruction(&mut self, instruction: &DeviceInstruction) {
        match instruction {
            DeviceInstruction::Argument { .. } | DeviceInstruction::Address { .. } => {}
            DeviceInstruction::Access {
                base,
                index,
                result,
            } => {
                let ty = self.writer.value_types[result];
                let id = self.value(*result);
                if self.is_pointer(*base) {
                    let mut chain =
                        Instruction::result(ACCESS_CHAIN, ty, id).operand(self.value(*base));
                    if self.wrapped(*base) {
                        let zero = self.writer.integer(Scalar::U32, 0);
                        chain = chain.operand(zero);
                    }
                    let index = self.value(*index);
                    self.push(chain.operand(index));
                } else {
                    let index = self.value(*index);
                    self.push(
                        Instruction::result(VECTOR_EXTRACT_DYNAMIC, ty, id)
                            .operand(self.value(*base))
                            .operand(index),
                    );
                }
            }
            DeviceInstruction::AccessIndex {
                base,
                index,
                result,
            } => {
                let ty = self.writer.value_types[result];
                let id = self.value(*result);
                if self.is_pointer(*base) {
                    let mut chain =
                        Instruction::result(ACCESS_CHAIN, ty, id).operand(self.value(*base));
                    if self.wrapped(*base) {
                        let zero = self.writer.integer(Scalar::U32, 0);
                        chain = chain.operand(zero);
                    }
                    let index = self.writer.integer(Scalar::U32, *index);
                    self.push(chain.operand(index));
                } else {
                    self.push(
                        Instruction::result(COMPOSITE_EXTRACT, ty, id)
                            .operand(self.value(*base))
                            .word(*index),
                    );
                }
            }
            DeviceInstruction::Load { pointer, result } => {
                let ty = self.writer.value_types[result];
                let id = self.value(*result);
                let operands = self.memory_operands(*pointer, true);
                let load = Instruction::result(LOAD, ty, id)
                    .operand(self.value(*pointer))
                    .operands(operands);
                self.push(load);
            }
            DeviceInstruction::Store { pointer, value } => {
                let operands = self.memory_operands(*pointer, false);
                let store = Instruction::plain(STORE)
                    .operand(self.value(*pointer))
                    .operand(self.value(*value))
                    .operands(operands);
                self.push(store);
            }
            DeviceInstruction::Unary { op, value, result } => {
                let scalar = self.scalar(*value);
                let opcode = match op {
                    UnaryOp::Negate => {
                        if scalar.floating() {
                            F_NEGATE
                        } else {
                            S_NEGATE
                        }
                    }
                    UnaryOp::LogicalNot => LOGICAL_NOT,
                    UnaryOp::BitwiseNot => NOT,
                };
                let ty = self.writer.value_types[result];
                let id = self.value(*result);
                self.push(Instruction::result(opcode, ty, id).operand(self.value(*value)));
            }
            DeviceInstruction::Binary {
                op,
                left,
                right,
                result,
            } => {
                let ty = self.writer.value_types[result];
                let id = self.value(*result);
                if *op == BinaryOp::Multiply && self.lanes(*left) != self.lanes(*right) {
                    let scalar = if self.lanes(*left) == 1 {
                        *left
                    } else {
                        *right
                    };
                    let vector = if self.lanes(*left) == 1 {
                        *right
                    } else {
                        *left
                    };
                    let vector_ty = self.writer.value_types[&vector];
                    let scalar_value = self.value(scalar);
                    if self.scalar(vector).floating() {
                        self.push(
                            Instruction::result(VECTOR_TIMES_SCALAR, ty, id)
                                .operand(self.value(vector))
                                .operand(scalar_value),
                        );
                    } else {
                        let splat = self.writer.id();
                        let mut instruction =
                            Instruction::result(COMPOSITE_CONSTRUCT, vector_ty, splat);
                        for _ in 0..self.lanes(vector) {
                            instruction = instruction.operand(scalar_value);
                        }
                        self.push(instruction);
                        self.push(
                            Instruction::result(I_MUL, ty, id)
                                .operand(self.value(vector))
                                .operand(splat),
                        );
                    }
                    return;
                }
                let scalar = self.scalar(*left);
                let opcode = binary_opcode(*op, scalar);
                self.push(
                    Instruction::result(opcode, ty, id)
                        .operand(self.value(*left))
                        .operand(self.value(*right)),
                );
            }
            DeviceInstruction::Select {
                condition,
                accept,
                reject,
                result,
            } => {
                assert_eq!(
                    self.lanes(*condition),
                    self.lanes(*accept),
                    "a device select between {} lanes is conditioned on {} lanes",
                    self.lanes(*accept),
                    self.lanes(*condition),
                );
                let ty = self.writer.value_types[result];
                let id = self.value(*result);
                self.push(
                    Instruction::result(SELECT, ty, id)
                        .operand(self.value(*condition))
                        .operand(self.value(*accept))
                        .operand(self.value(*reject)),
                );
            }
            DeviceInstruction::Convert { value, result } => {
                let source = self.scalar(*value);
                let target = self.scalar(*result);
                let ty = self.writer.value_types[result];
                let id = self.value(*result);
                match (source, target) {
                    _ if source == target => {
                        let present = self.value(*value);
                        self.writer.values.insert(*result, present);
                    }
                    (Scalar::F32 | Scalar::F16, Scalar::F32 | Scalar::F16) => self
                        .push(Instruction::result(F_CONVERT, ty, id).operand(self.value(*value))),
                    (Scalar::U32, Scalar::F32 | Scalar::F16) => self.push(
                        Instruction::result(CONVERT_U_TO_F, ty, id).operand(self.value(*value)),
                    ),
                    (Scalar::I32, Scalar::F32 | Scalar::F16) => self.push(
                        Instruction::result(CONVERT_S_TO_F, ty, id).operand(self.value(*value)),
                    ),
                    (Scalar::F32 | Scalar::F16, Scalar::U32) => self.push(
                        Instruction::result(CONVERT_F_TO_U, ty, id).operand(self.value(*value)),
                    ),
                    (Scalar::F32 | Scalar::F16, Scalar::I32) => self.push(
                        Instruction::result(CONVERT_F_TO_S, ty, id).operand(self.value(*value)),
                    ),
                    (Scalar::U32 | Scalar::I32, Scalar::U32 | Scalar::I32) => {
                        if source.bytes() != target.bytes() {
                            let opcode = if target.signed() {
                                S_CONVERT
                            } else {
                                U_CONVERT
                            };
                            self.push(
                                Instruction::result(opcode, ty, id).operand(self.value(*value)),
                            );
                        } else {
                            self.push(
                                Instruction::result(BITCAST, ty, id).operand(self.value(*value)),
                            );
                        }
                    }
                    _ => panic!("a device conversion from {} is undefined", source.name()),
                }
            }
            DeviceInstruction::Bitcast { value, result } => {
                let ty = self.writer.value_types[result];
                let id = self.value(*result);
                self.push(Instruction::result(BITCAST, ty, id).operand(self.value(*value)));
            }
            DeviceInstruction::Math {
                fun,
                arguments,
                result,
            } => {
                let ty = self.writer.value_types[result];
                let id = self.value(*result);
                if matches!(fun, MathFun::Min | MathFun::Max) {
                    let glsl = self.writer.glsl();
                    let scalar = self.scalar(arguments[0]);
                    let number = match (fun, scalar) {
                        (MathFun::Min, Scalar::F32 | Scalar::F16) => glsl::F_MIN,
                        (MathFun::Max, Scalar::F32 | Scalar::F16) => glsl::F_MAX,
                        (MathFun::Min, Scalar::U32) => glsl::U_MIN,
                        (MathFun::Max, Scalar::U32) => glsl::U_MAX,
                        (MathFun::Min, _) => glsl::S_MIN,
                        (MathFun::Max, _) => glsl::S_MAX,
                        _ => unreachable!(),
                    };
                    self.push(
                        Instruction::result(EXT_INST, ty, id)
                            .operand(glsl)
                            .operand(number)
                            .operand(self.value(arguments[0]))
                            .operand(self.value(arguments[1])),
                    );
                } else {
                    let number = match fun {
                        MathFun::Abs => glsl::ABS,
                        MathFun::Floor => glsl::FLOOR,
                        MathFun::Trunc => glsl::TRUNC,
                        MathFun::Sqrt => glsl::SQRT,
                        MathFun::Exp => glsl::EXP,
                        MathFun::Log => glsl::LOG,
                        MathFun::Sin => glsl::SIN,
                        MathFun::Cos => glsl::COS,
                        MathFun::Tanh => glsl::TANH,
                        MathFun::Pow => glsl::POW,
                        MathFun::UnpackHalf2x16 => glsl::UNPACK_HALF2X16,
                        MathFun::Min | MathFun::Max => unreachable!(),
                    };
                    let mut instruction = Instruction::result(EXT_INST, ty, id)
                        .operand(self.writer.glsl())
                        .operand(number);
                    for argument in arguments {
                        instruction = instruction.operand(self.value(*argument));
                    }
                    self.push(instruction);
                }
            }
            DeviceInstruction::Compose {
                constituents,
                result,
            } => {
                let ty = self.writer.value_types[result];
                let id = self.value(*result);
                let mut instruction = Instruction::result(COMPOSITE_CONSTRUCT, ty, id);
                for constituent in constituents {
                    instruction = instruction.operand(self.value(*constituent));
                }
                self.push(instruction);
            }
            DeviceInstruction::Call {
                function,
                arguments,
                result,
            } => {
                let callee = self.writer.function_ids[*function as usize];
                let (ty, id) = match result {
                    Some(result) => (self.writer.value_types[result], self.value(*result)),
                    None => {
                        let void = self.writer.void();
                        (void, self.writer.id())
                    }
                };
                let mut instruction = Instruction::result(FUNCTION_CALL, ty, id);
                instruction = instruction.operand(callee);
                for argument in arguments {
                    instruction = instruction.operand(self.value(*argument));
                }
                self.push(instruction);
            }
            DeviceInstruction::Atomic {
                op,
                pointer,
                value,
                result,
            } => {
                let space = self.pointer_space(*pointer);
                let scalar = self.scalar(*value);
                let opcode = atomic_opcode(*op, scalar);
                let scope = if space == Space::WorkGroup {
                    scope::WORKGROUP
                } else {
                    scope::DEVICE
                };
                let scope = self.writer.integer(Scalar::U32, scope);
                let relaxed = self.writer.integer(Scalar::U32, semantics::RELAXED);
                let ty = self.writer.value_types[result];
                let id = self.value(*result);
                self.push(
                    Instruction::result(opcode, ty, id)
                        .operand(self.value(*pointer))
                        .operand(scope)
                        .operand(relaxed)
                        .operand(self.value(*value)),
                );
            }
            DeviceInstruction::WorkGroupUniformLoad { pointer, result } => {
                let workgroup = self.writer.integer(Scalar::U32, scope::WORKGROUP);
                let ordering = self.writer.integer(
                    Scalar::U32,
                    semantics::ACQUIRE_RELEASE | semantics::WORKGROUP_MEMORY,
                );
                self.push(
                    Instruction::plain(CONTROL_BARRIER)
                        .operand(workgroup)
                        .operand(workgroup)
                        .operand(ordering),
                );
                let ty = self.writer.value_types[result];
                let id = self.value(*result);
                let operands = self.memory_operands(*pointer, true);
                self.push(
                    Instruction::result(LOAD, ty, id)
                        .operand(self.value(*pointer))
                        .operands(operands),
                );
            }
            DeviceInstruction::Barrier(barrier) => {
                let execution = self.writer.integer(Scalar::U32, scope::WORKGROUP);
                let (memory, ordering) = match barrier {
                    Barrier::WorkGroup => (
                        scope::WORKGROUP,
                        semantics::ACQUIRE_RELEASE | semantics::WORKGROUP_MEMORY,
                    ),
                    Barrier::Storage => (
                        scope::DEVICE,
                        semantics::ACQUIRE_RELEASE | semantics::UNIFORM_MEMORY,
                    ),
                };
                let memory = self.writer.integer(Scalar::U32, memory);
                let ordering = self.writer.integer(Scalar::U32, ordering);
                self.push(
                    Instruction::plain(CONTROL_BARRIER)
                        .operand(execution)
                        .operand(memory)
                        .operand(ordering),
                );
            }
            DeviceInstruction::MatrixFill { value, result } => {
                let ty = self.writer.value_types[result];
                let id = self.value(*result);
                self.push(
                    Instruction::result(COMPOSITE_CONSTRUCT, ty, id).operand(self.value(*value)),
                );
            }
            DeviceInstruction::MatrixLoad {
                pointer,
                stride,
                layout,
                result,
            } => {
                let ty = self.writer.value_types[result];
                let id = self.value(*result);
                let layout = self.writer.integer(Scalar::U32, layout_code(*layout));
                let operands = self.memory_operands(*pointer, true);
                self.push(
                    Instruction::result(COOPERATIVE_MATRIX_LOAD, ty, id)
                        .operand(self.value(*pointer))
                        .operand(layout)
                        .operand(self.value(*stride))
                        .operands(operands),
                );
            }
            DeviceInstruction::MatrixStore {
                pointer,
                value,
                stride,
                layout,
            } => {
                let layout = self.writer.integer(Scalar::U32, layout_code(*layout));
                let operands = self.memory_operands(*pointer, false);
                self.push(
                    Instruction::plain(COOPERATIVE_MATRIX_STORE)
                        .operand(self.value(*pointer))
                        .operand(self.value(*value))
                        .operand(layout)
                        .operand(self.value(*stride))
                        .operands(operands),
                );
            }
            DeviceInstruction::MatrixMulAdd {
                left,
                right,
                accumulate,
                result,
            } => {
                let ty = self.writer.value_types[result];
                let id = self.value(*result);
                self.push(
                    Instruction::result(COOPERATIVE_MATRIX_MUL_ADD, ty, id)
                        .operand(self.value(*left))
                        .operand(self.value(*right))
                        .operand(self.value(*accumulate)),
                );
            }
            DeviceInstruction::MatrixLength { ty, result } => {
                let matrix = self.writer.ty(*ty);
                let ty = self.writer.value_types[result];
                let id = self.value(*result);
                self.push(Instruction::result(COOPERATIVE_MATRIX_LENGTH, ty, id).operand(matrix));
            }
            DeviceInstruction::MatrixExtract {
                value,
                index,
                result,
            } => {
                let index = self.literal_index(*index);
                let ty = self.writer.value_types[result];
                let id = self.value(*result);
                self.push(
                    Instruction::result(COMPOSITE_EXTRACT, ty, id)
                        .operand(self.value(*value))
                        .word(index),
                );
            }
            DeviceInstruction::MatrixInsert {
                value,
                index,
                component,
                result,
            } => {
                let index = self.literal_index(*index);
                let ty = self.writer.value_types[result];
                let id = self.value(*result);
                self.push(
                    Instruction::result(COMPOSITE_INSERT, ty, id)
                        .operand(self.value(*component))
                        .operand(self.value(*value))
                        .word(index),
                );
            }
            DeviceInstruction::If {
                condition,
                accept,
                reject,
            } => {
                let accepted = self.label();
                let rejected = self.label();
                let merge = self.label();
                self.push(
                    Instruction::plain(SELECTION_MERGE)
                        .operand(Operand::Id(merge))
                        .word(0u32),
                );
                self.terminate(
                    Instruction::plain(BRANCH_CONDITIONAL)
                        .operand(self.value(*condition))
                        .operand(accepted)
                        .operand(rejected),
                );
                self.start(accepted);
                self.block(accept);
                self.reach(merge);
                self.start(rejected);
                self.block(reject);
                self.reach(merge);
                self.start(merge);
            }
            DeviceInstruction::Switch {
                selector,
                cases,
                default,
            } => {
                let merge = self.label();
                let labels = cases.iter().map(|_| self.label()).collect::<Vec<_>>();
                let fallback = self.label();
                let mut instruction = Instruction::plain(SWITCH)
                    .operand(self.value(*selector))
                    .operand(fallback);
                for ((value, _), label) in cases.iter().zip(&labels) {
                    instruction = instruction.word(*value).operand(*label);
                }
                self.push(
                    Instruction::plain(SELECTION_MERGE)
                        .operand(Operand::Id(merge))
                        .word(0u32),
                );
                self.terminate(instruction);
                for ((_, body), label) in cases.iter().zip(&labels) {
                    self.start(*label);
                    self.block(body);
                    self.reach(merge);
                }
                self.start(fallback);
                self.block(default);
                self.reach(merge);
                self.start(merge);
            }
            DeviceInstruction::Loop { body, continuing } => {
                let header = self.label();
                let block = self.label();
                let step = self.label();
                let merge = self.label();
                self.branch(header);
                self.start(header);
                self.push(
                    Instruction::plain(LOOP_MERGE)
                        .operand(Operand::Id(merge))
                        .operand(step)
                        .word(0u32),
                );
                self.branch(block);
                self.loops.push(Loop {
                    merge,
                    continuing: step,
                });
                self.start(block);
                self.block(body);
                self.reach(step);
                self.start(step);
                self.block(continuing);
                self.reach(header);
                self.loops.pop();
                self.start(merge);
            }
            DeviceInstruction::Break => {
                let target = self
                    .loops
                    .last()
                    .expect("a break stands inside a loop")
                    .merge;
                self.branch(target);
            }
            DeviceInstruction::Continue => {
                let target = self
                    .loops
                    .last()
                    .expect("a continue stands inside a loop")
                    .continuing;
                self.branch(target);
            }
            DeviceInstruction::Return { value } => match value {
                Some(value) => {
                    let instruction = Instruction::plain(RETURN_VALUE).operand(self.value(*value));
                    self.terminate(instruction);
                }
                None => self.terminate(Instruction::plain(RETURN)),
            },
            DeviceInstruction::Block(body) => self.block(body),
        }
    }

    fn literal_index(&self, index: ValueId) -> u32 {
        match self.writer.module.constant_of(index) {
            Some(Constant::U32(value)) => value,
            other => panic!("a device matrix index is not a word: {other:?}"),
        }
    }
}

fn layout_code(layout: MatrixLayout) -> u32 {
    match layout {
        MatrixLayout::RowMajor => 0,
        MatrixLayout::ColumnMajor => 1,
    }
}

fn atomic_opcode(op: AtomicOp, scalar: Scalar) -> u16 {
    match op {
        AtomicOp::Add => ATOMIC_I_ADD,
        AtomicOp::Subtract => ATOMIC_I_SUB,
        AtomicOp::Exchange => ATOMIC_EXCHANGE,
        AtomicOp::Min => {
            if scalar.signed() {
                ATOMIC_S_MIN
            } else {
                ATOMIC_U_MIN
            }
        }
        AtomicOp::Max => {
            if scalar.signed() {
                ATOMIC_S_MAX
            } else {
                ATOMIC_U_MAX
            }
        }
        AtomicOp::And => ATOMIC_AND,
        AtomicOp::Or => ATOMIC_OR,
        AtomicOp::Xor => ATOMIC_XOR,
    }
}

fn binary_opcode(op: BinaryOp, scalar: Scalar) -> u16 {
    match (op, scalar) {
        (BinaryOp::Add, Scalar::F32 | Scalar::F16) => F_ADD,
        (BinaryOp::Add, _) => I_ADD,
        (BinaryOp::Subtract, Scalar::F32 | Scalar::F16) => F_SUB,
        (BinaryOp::Subtract, _) => I_SUB,
        (BinaryOp::Multiply, Scalar::F32 | Scalar::F16) => F_MUL,
        (BinaryOp::Multiply, _) => I_MUL,
        (BinaryOp::Divide, Scalar::F32 | Scalar::F16) => F_DIV,
        (BinaryOp::Divide, Scalar::U32) => U_DIV,
        (BinaryOp::Divide, _) => S_DIV,
        (BinaryOp::Modulo, Scalar::F32 | Scalar::F16) => F_MOD,
        (BinaryOp::Modulo, Scalar::U32) => U_MOD,
        (BinaryOp::Modulo, _) => S_MOD,
        (BinaryOp::Equal, Scalar::F32 | Scalar::F16) => F_ORD_EQUAL,
        (BinaryOp::Equal, Scalar::Bool) => LOGICAL_EQUAL,
        (BinaryOp::Equal, _) => I_EQUAL,
        (BinaryOp::NotEqual, Scalar::F32 | Scalar::F16) => F_ORD_NOT_EQUAL,
        (BinaryOp::NotEqual, Scalar::Bool) => LOGICAL_NOT_EQUAL,
        (BinaryOp::NotEqual, _) => I_NOT_EQUAL,
        (BinaryOp::Less, Scalar::F32 | Scalar::F16) => F_ORD_LESS_THAN,
        (BinaryOp::Less, Scalar::U32) => U_LESS_THAN,
        (BinaryOp::Less, _) => S_LESS_THAN,
        (BinaryOp::LessEqual, Scalar::F32 | Scalar::F16) => F_ORD_LESS_THAN_EQUAL,
        (BinaryOp::LessEqual, Scalar::U32) => U_LESS_THAN_EQUAL,
        (BinaryOp::LessEqual, _) => S_LESS_THAN_EQUAL,
        (BinaryOp::Greater, Scalar::F32 | Scalar::F16) => F_ORD_GREATER_THAN,
        (BinaryOp::Greater, Scalar::U32) => U_GREATER_THAN,
        (BinaryOp::Greater, _) => S_GREATER_THAN,
        (BinaryOp::GreaterEqual, Scalar::F32 | Scalar::F16) => F_ORD_GREATER_THAN_EQUAL,
        (BinaryOp::GreaterEqual, Scalar::U32) => U_GREATER_THAN_EQUAL,
        (BinaryOp::GreaterEqual, _) => S_GREATER_THAN_EQUAL,
        (BinaryOp::And, _) => BITWISE_AND,
        (BinaryOp::Or, _) => BITWISE_OR,
        (BinaryOp::Xor, _) => BITWISE_XOR,
        (BinaryOp::LogicalAnd, _) => LOGICAL_AND,
        (BinaryOp::LogicalOr, _) => LOGICAL_OR,
        (BinaryOp::ShiftLeft, _) => SHIFT_LEFT_LOGICAL,
        (BinaryOp::ShiftRight, Scalar::I32) => SHIFT_RIGHT_ARITHMETIC,
        (BinaryOp::ShiftRight, _) => SHIFT_RIGHT_LOGICAL,
    }
}
