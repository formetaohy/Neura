#[derive(Clone, PartialEq, Debug)]
pub enum Value {
    Unsigned8(u8),
    Signed8(i8),
    Unsigned16(u16),
    Signed16(i16),
    Unsigned32(u32),
    Signed32(i32),
    Single(f32),
    Flag(bool),
    Text(String),
    Array(Vec<Value>),
    Unsigned64(u64),
    Signed64(i64),
    Double(f64),
}

impl Value {
    pub const fn code(&self) -> u32 {
        match self {
            Self::Unsigned8(_) => 0,
            Self::Signed8(_) => 1,
            Self::Unsigned16(_) => 2,
            Self::Signed16(_) => 3,
            Self::Unsigned32(_) => 4,
            Self::Signed32(_) => 5,
            Self::Single(_) => 6,
            Self::Flag(_) => 7,
            Self::Text(_) => 8,
            Self::Array(_) => 9,
            Self::Unsigned64(_) => 10,
            Self::Signed64(_) => 11,
            Self::Double(_) => 12,
        }
    }

    pub const fn integer(&self) -> Option<u64> {
        match self {
            Self::Unsigned8(value) => Some(*value as u64),
            Self::Signed8(value) => Some(*value as u64),
            Self::Unsigned16(value) => Some(*value as u64),
            Self::Signed16(value) => Some(*value as u64),
            Self::Unsigned32(value) => Some(*value as u64),
            Self::Signed32(value) => Some(*value as u64),
            Self::Unsigned64(value) => Some(*value),
            Self::Signed64(value) => Some(*value as u64),
            Self::Single(_) | Self::Double(_) | Self::Flag(_) | Self::Text(_) | Self::Array(_) => {
                None
            }
        }
    }

    pub const fn number(&self) -> Option<f64> {
        match self {
            Self::Single(value) => Some(*value as f64),
            Self::Double(value) => Some(*value),
            Self::Unsigned8(_)
            | Self::Signed8(_)
            | Self::Unsigned16(_)
            | Self::Signed16(_)
            | Self::Unsigned32(_)
            | Self::Signed32(_)
            | Self::Unsigned64(_)
            | Self::Signed64(_)
            | Self::Flag(_)
            | Self::Text(_)
            | Self::Array(_) => None,
        }
    }

    pub const fn flag(&self) -> Option<bool> {
        match self {
            Self::Flag(value) => Some(*value),
            _ => None,
        }
    }

    pub fn text(&self) -> Option<&str> {
        match self {
            Self::Text(value) => Some(value),
            _ => None,
        }
    }

    pub fn array(&self) -> Option<&[Value]> {
        match self {
            Self::Array(values) => Some(values),
            _ => None,
        }
    }

    pub const fn name(&self) -> &'static str {
        match self {
            Self::Unsigned8(_) => "an unsigned byte",
            Self::Signed8(_) => "a signed byte",
            Self::Unsigned16(_) => "an unsigned half word",
            Self::Signed16(_) => "a signed half word",
            Self::Unsigned32(_) => "an unsigned word",
            Self::Signed32(_) => "a signed word",
            Self::Single(_) => "a single precision number",
            Self::Flag(_) => "a flag",
            Self::Text(_) => "a string",
            Self::Array(_) => "an array",
            Self::Unsigned64(_) => "an unsigned long",
            Self::Signed64(_) => "a signed long",
            Self::Double(_) => "a double precision number",
        }
    }
}
