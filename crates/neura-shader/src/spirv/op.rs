pub const MAGIC: u32 = 0x0723_0203;
pub const EXTENSION: u16 = 10;

pub const NAME: u16 = 5;
pub const EXT_INST_IMPORT: u16 = 11;
pub const EXT_INST: u16 = 12;
pub const MEMORY_MODEL: u16 = 14;
pub const ENTRY_POINT: u16 = 15;
pub const EXECUTION_MODE: u16 = 16;
pub const CAPABILITY: u16 = 17;
pub const TYPE_VOID: u16 = 19;
pub const TYPE_BOOL: u16 = 20;
pub const TYPE_INT: u16 = 21;
pub const TYPE_FLOAT: u16 = 22;
pub const TYPE_VECTOR: u16 = 23;
pub const TYPE_ARRAY: u16 = 28;
pub const TYPE_RUNTIME_ARRAY: u16 = 29;
pub const TYPE_STRUCT: u16 = 30;
pub const TYPE_POINTER: u16 = 32;
pub const TYPE_FUNCTION: u16 = 33;
pub const CONSTANT_TRUE: u16 = 41;
pub const CONSTANT_FALSE: u16 = 42;
pub const CONSTANT: u16 = 43;
pub const CONSTANT_COMPOSITE: u16 = 44;
pub const CONSTANT_NULL: u16 = 46;
pub const FUNCTION: u16 = 54;
pub const FUNCTION_PARAMETER: u16 = 55;
pub const FUNCTION_END: u16 = 56;
pub const FUNCTION_CALL: u16 = 57;
pub const VARIABLE: u16 = 59;
pub const LOAD: u16 = 61;
pub const STORE: u16 = 62;
pub const ACCESS_CHAIN: u16 = 65;
pub const DECORATE: u16 = 71;
pub const MEMBER_DECORATE: u16 = 72;
pub const COMPOSITE_CONSTRUCT: u16 = 80;
pub const COMPOSITE_EXTRACT: u16 = 81;
pub const VECTOR_EXTRACT_DYNAMIC: u16 = 77;
pub const COMPOSITE_INSERT: u16 = 82;
pub const CONVERT_F_TO_U: u16 = 109;
pub const CONVERT_F_TO_S: u16 = 110;
pub const CONVERT_S_TO_F: u16 = 111;
pub const CONVERT_U_TO_F: u16 = 112;
pub const U_CONVERT: u16 = 113;
pub const S_CONVERT: u16 = 114;
pub const F_CONVERT: u16 = 115;
pub const BITCAST: u16 = 124;
pub const S_NEGATE: u16 = 126;
pub const F_NEGATE: u16 = 127;
pub const I_ADD: u16 = 128;
pub const F_ADD: u16 = 129;
pub const I_SUB: u16 = 130;
pub const F_SUB: u16 = 131;
pub const I_MUL: u16 = 132;
pub const F_MUL: u16 = 133;
pub const U_DIV: u16 = 134;
pub const S_DIV: u16 = 135;
pub const F_DIV: u16 = 136;
pub const U_MOD: u16 = 137;
pub const LOGICAL_EQUAL: u16 = 164;
pub const LOGICAL_NOT_EQUAL: u16 = 165;
pub const LOGICAL_OR: u16 = 166;
pub const LOGICAL_AND: u16 = 167;
pub const LOGICAL_NOT: u16 = 168;
pub const SELECT: u16 = 169;
pub const VECTOR_TIMES_SCALAR: u16 = 142;
pub const I_EQUAL: u16 = 170;
pub const I_NOT_EQUAL: u16 = 171;
pub const U_GREATER_THAN: u16 = 172;
pub const S_GREATER_THAN: u16 = 173;
pub const U_GREATER_THAN_EQUAL: u16 = 174;
pub const S_GREATER_THAN_EQUAL: u16 = 175;
pub const U_LESS_THAN: u16 = 176;
pub const S_LESS_THAN: u16 = 177;
pub const U_LESS_THAN_EQUAL: u16 = 178;
pub const S_LESS_THAN_EQUAL: u16 = 179;
pub const F_ORD_EQUAL: u16 = 180;
pub const F_ORD_NOT_EQUAL: u16 = 182;
pub const F_ORD_LESS_THAN: u16 = 184;
pub const F_ORD_GREATER_THAN: u16 = 186;
pub const F_ORD_LESS_THAN_EQUAL: u16 = 188;
pub const F_ORD_GREATER_THAN_EQUAL: u16 = 190;
pub const SHIFT_RIGHT_LOGICAL: u16 = 194;
pub const SHIFT_RIGHT_ARITHMETIC: u16 = 195;
pub const SHIFT_LEFT_LOGICAL: u16 = 196;
pub const BITWISE_OR: u16 = 197;
pub const BITWISE_XOR: u16 = 198;
pub const BITWISE_AND: u16 = 199;
pub const NOT: u16 = 200;
pub const CONTROL_BARRIER: u16 = 224;
pub const ATOMIC_EXCHANGE: u16 = 229;
pub const ATOMIC_I_ADD: u16 = 234;
pub const ATOMIC_I_SUB: u16 = 235;
pub const ATOMIC_S_MIN: u16 = 236;
pub const ATOMIC_U_MIN: u16 = 237;
pub const ATOMIC_S_MAX: u16 = 238;
pub const ATOMIC_U_MAX: u16 = 239;
pub const ATOMIC_AND: u16 = 240;
pub const ATOMIC_OR: u16 = 241;
pub const ATOMIC_XOR: u16 = 242;
pub const LOOP_MERGE: u16 = 246;
pub const SELECTION_MERGE: u16 = 247;
pub const LABEL: u16 = 248;
pub const BRANCH: u16 = 249;
pub const BRANCH_CONDITIONAL: u16 = 250;
pub const SWITCH: u16 = 251;
pub const RETURN: u16 = 253;
pub const UNREACHABLE: u16 = 255;
pub const RETURN_VALUE: u16 = 254;
pub const COOPERATIVE_MATRIX_TYPE: u16 = 4456;
pub const COOPERATIVE_MATRIX_LOAD: u16 = 4457;
pub const COOPERATIVE_MATRIX_STORE: u16 = 4458;
pub const COOPERATIVE_MATRIX_MUL_ADD: u16 = 4459;
pub const COOPERATIVE_MATRIX_LENGTH: u16 = 4460;

pub mod capability {
    pub const SHADER: u32 = 1;
    pub const FLOAT16: u32 = 9;
    pub const VULKAN_MEMORY_MODEL: u32 = 5345;
    pub const VULKAN_MEMORY_MODEL_DEVICE_SCOPE: u32 = 5346;
    pub const COOPERATIVE_MATRIX_KHR: u32 = 6022;
}

pub mod scope {
    pub const DEVICE: u32 = 1;
    pub const WORKGROUP: u32 = 2;
    pub const SUBGROUP: u32 = 3;
}

pub mod semantics {
    pub const RELAXED: u32 = 0;
    pub const ACQUIRE_RELEASE: u32 = 8;
    pub const UNIFORM_MEMORY: u32 = 64;
    pub const WORKGROUP_MEMORY: u32 = 256;
}

pub mod memory_access {
    pub const MAKE_POINTER_AVAILABLE: u32 = 8;
    pub const MAKE_POINTER_VISIBLE: u32 = 16;
    pub const NON_PRIVATE_POINTER: u32 = 32;
}

pub mod decoration {
    pub const BLOCK: u32 = 2;
    pub const ARRAY_STRIDE: u32 = 6;
    pub const BUILT_IN: u32 = 11;
    pub const COHERENT: u32 = 23;
    pub const NON_WRITABLE: u32 = 24;
    pub const BINDING: u32 = 33;
    pub const DESCRIPTOR_SET: u32 = 34;
    pub const OFFSET: u32 = 35;
}

pub mod built_in {
    pub const WORKGROUP_ID: u32 = 26;
    pub const GLOBAL_INVOCATION_ID: u32 = 28;
    pub const LOCAL_INVOCATION_INDEX: u32 = 29;
}

pub mod storage {
    pub const INPUT: u32 = 1;
    pub const WORKGROUP: u32 = 4;
    pub const FUNCTION: u32 = 7;
    pub const STORAGE_BUFFER: u32 = 12;
}

pub const EXECUTION_MODEL_GL_COMPUTE: u32 = 5;
pub const EXECUTION_MODE_LOCAL_SIZE: u32 = 17;
pub const ADDRESSING_MODEL_LOGICAL: u32 = 0;
pub const MEMORY_MODEL_GLSL450: u32 = 1;
pub const MEMORY_MODEL_VULKAN: u32 = 3;
pub const VERSION_1_3: u32 = 0x0001_0300;
pub const VERSION_1_4: u32 = 0x0001_0400;
pub const VERSION_1_6: u32 = 0x0001_0600;

pub mod glsl {
    pub const ABS: u32 = 4;
    pub const FLOOR: u32 = 8;
    pub const TRUNC: u32 = 3;
    pub const SIN: u32 = 13;
    pub const COS: u32 = 14;
    pub const TANH: u32 = 21;
    pub const POW: u32 = 26;
    pub const EXP: u32 = 27;
    pub const LOG: u32 = 28;
    pub const SQRT: u32 = 31;
    pub const F_MIN: u32 = 37;
    pub const U_MIN: u32 = 38;
    pub const S_MIN: u32 = 39;
    pub const F_MAX: u32 = 40;
    pub const U_MAX: u32 = 41;
    pub const S_MAX: u32 = 42;
    pub const UNPACK_HALF2X16: u32 = 62;
}

#[derive(Debug)]
pub enum Operand {
    Id(u32),
    Word(u32),
    Literal(String),
}

impl From<u32> for Operand {
    fn from(value: u32) -> Self {
        Self::Id(value)
    }
}

impl Operand {
    fn encode(&self, words: &mut Vec<u32>) {
        match self {
            Self::Id(id) => words.push(*id),
            Self::Word(word) => words.push(*word),
            Self::Literal(text) => {
                let mut bytes = text.as_bytes().to_vec();
                bytes.push(0);
                while !bytes.len().is_multiple_of(4) {
                    bytes.push(0);
                }
                for chunk in bytes.chunks(4) {
                    let mut word = [0u8; 4];
                    word.copy_from_slice(chunk);
                    words.push(u32::from_le_bytes(word));
                }
            }
        }
    }
}

impl Instruction {
    pub fn defines(&self) -> Option<u32> {
        let index = self.defines?;
        match self.operands.get(index) {
            Some(Operand::Id(id)) => Some(*id),
            other => panic!("a device instruction defines {other:?}"),
        }
    }
}

impl Operand {
    pub fn id(&self) -> u32 {
        match self {
            Self::Id(id) => *id,
            other => panic!("a device instruction expected an id, found {other:?}"),
        }
    }
}

#[derive(Debug)]
pub struct Instruction {
    pub opcode: u16,
    pub operands: Vec<Operand>,
    defines: Option<usize>,
}

impl Instruction {
    pub fn plain(opcode: u16) -> Self {
        Self {
            opcode,
            operands: Vec::new(),
            defines: None,
        }
    }

    pub fn defined(opcode: u16, id: u32) -> Self {
        let mut instruction = Self::plain(opcode).operand(id);
        instruction.defines = Some(0);
        instruction
    }

    pub fn result(opcode: u16, ty: u32, id: u32) -> Self {
        let mut instruction = Self::plain(opcode).operand(ty).operand(id);
        instruction.defines = Some(1);
        instruction
    }

    pub fn operand(mut self, operand: impl Into<Operand>) -> Self {
        self.operands.push(operand.into());
        self
    }

    pub fn word(self, value: u32) -> Self {
        self.operand(Operand::Word(value))
    }

    pub fn operands<I: IntoIterator<Item = Operand>>(mut self, operands: I) -> Self {
        self.operands.extend(operands);
        self
    }

    pub fn encode(&self, words: &mut Vec<u32>) {
        let mut operands = Vec::new();
        for operand in &self.operands {
            operand.encode(&mut operands);
        }
        let length = u32::try_from(operands.len() + 1).expect("a device instruction fits");
        words.push((length << 16) | u32::from(self.opcode));
        words.extend_from_slice(&operands);
    }
}
