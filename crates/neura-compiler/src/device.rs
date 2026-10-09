use crate::resource::{AtomicU32, Fvec2, Uvec4};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Reach {
    Kernel,
    Tile,
}

macro_rules! vocabulary {
    ($($variant:ident $name:literal $reach:ident $arity:literal $yields:literal;)+) => {
        #[derive(Clone, Copy, PartialEq, Eq, Debug)]
        pub enum Intrinsic {
            $($variant),+
        }

        impl Intrinsic {
            pub const ALL: &'static [Self] = &[$(Self::$variant),+];

            pub const fn name(self) -> &'static str {
                match self {
                    $(Self::$variant => $name),+
                }
            }

            pub const fn reach(self) -> Reach {
                match self {
                    $(Self::$variant => Reach::$reach),+
                }
            }

            pub const fn arity(self) -> u32 {
                match self {
                    $(Self::$variant => $arity),+
                }
            }

            pub const fn yields(self) -> bool {
                match self {
                    $(Self::$variant => $yields),+
                }
            }

            pub fn of(name: &str) -> Option<Self> {
                match name {
                    $($name => Some(Self::$variant),)+
                    _ => None,
                }
            }
        }
    };
}

vocabulary! {
    Select "select" Kernel 3 true;
    Max "max" Kernel 2 true;
    Min "min" Kernel 2 true;
    Abs "abs" Kernel 1 true;
    Sqrt "sqrt" Kernel 1 true;
    Exp "exp" Kernel 1 true;
    Log "log" Kernel 1 true;
    Tanh "tanh" Kernel 1 true;
    Trunc "trunc" Kernel 1 true;
    Sin "sin" Kernel 1 true;
    Cos "cos" Kernel 1 true;
    Pow "pow" Kernel 2 true;
    Floor "floor" Kernel 1 true;
    Uvec4 "uvec4" Kernel 4 true;
    BitcastU32 "bitcast_u32" Kernel 1 true;
    BitcastF32 "bitcast_f32" Kernel 1 true;
    UnpackHalf2x16 "unpack2x16float" Kernel 1 true;
    AtomicAdd "atomic_add" Kernel 2 true;
    AtomicSub "atomic_sub" Kernel 2 true;
    AtomicStore "atomic_store" Kernel 2 false;
    WorkgroupBarrier "workgroup_barrier" Kernel 0 false;
    StorageBarrier "storage_barrier" Kernel 0 false;
    F32 "f32" Tile 1 true;
    U32 "u32" Tile 1 true;
    I32 "i32" Tile 1 true;
    F16 "f16" Tile 1 true;
    ScalarArray "scalar_array" Tile 2 true;
    WorkgroupUniformLoad "workgroup_uniform_load" Tile 1 true;
    CoopmatAccumulator "coopmat_accumulator" Tile 1 true;
    CoopmatLoadRow "coopmat_load_row" Tile 2 true;
    CoopmatLoadBRow "coopmat_load_b_row" Tile 2 true;
    CoopmatLoadColumn "coopmat_load_column" Tile 2 true;
    CoopmatMuladd "coopmat_muladd" Tile 3 true;
    CoopmatStore "coopmat_store" Tile 3 false;
}

macro_rules! loop_forms {
    ($($variant:ident $name:literal $unroll:literal;)+) => {
        #[derive(Clone, Copy, PartialEq, Eq, Debug)]
        pub enum LoopForm {
            $($variant),+
        }

        impl LoopForm {
            pub const ALL: &'static [Self] = &[$(Self::$variant),+];

            pub const fn name(self) -> &'static str {
                match self {
                    $(Self::$variant => $name),+
                }
            }

            pub const fn unroll(self) -> bool {
                match self {
                    $(Self::$variant => $unroll),+
                }
            }

            pub fn of(name: &str) -> Option<Self> {
                match name {
                    $($name => Some(Self::$variant),)+
                    _ => None,
                }
            }
        }
    };
}

loop_forms! {
    Stride "stride" false;
    Unroll "unroll" true;
}

mod private {
    pub trait Scalar {}
    impl Scalar for u32 {}
    impl Scalar for i32 {}
    impl Scalar for f32 {}
}

pub trait Number: private::Scalar + Copy {}
impl Number for u32 {}
impl Number for i32 {}
impl Number for f32 {}

pub fn select<T: Copy>(_reject: T, _accept: T, _condition: bool) -> T {
    unreachable!("a device intrinsic exists only on the device")
}

pub fn max<T: Number>(_left: T, _right: T) -> T {
    unreachable!("a device intrinsic exists only on the device")
}

pub fn min<T: Number>(_left: T, _right: T) -> T {
    unreachable!("a device intrinsic exists only on the device")
}

pub fn abs(_value: f32) -> f32 {
    unreachable!("a device intrinsic exists only on the device")
}

pub fn sqrt(_value: f32) -> f32 {
    unreachable!("a device intrinsic exists only on the device")
}

pub fn exp(_value: f32) -> f32 {
    unreachable!("a device intrinsic exists only on the device")
}

pub fn log(_value: f32) -> f32 {
    unreachable!("a device intrinsic exists only on the device")
}

pub fn tanh(_value: f32) -> f32 {
    unreachable!("a device intrinsic exists only on the device")
}

pub fn trunc(_value: f32) -> f32 {
    unreachable!("a device intrinsic exists only on the device")
}

pub fn sin(_value: f32) -> f32 {
    unreachable!("a device intrinsic exists only on the device")
}

pub fn cos(_value: f32) -> f32 {
    unreachable!("a device intrinsic exists only on the device")
}

pub fn pow(_base: f32, _exponent: f32) -> f32 {
    unreachable!("a device intrinsic exists only on the device")
}

pub fn floor(_value: f32) -> f32 {
    unreachable!("a device intrinsic exists only on the device")
}

pub fn uvec4(_x: u32, _y: u32, _z: u32, _w: u32) -> Uvec4 {
    unreachable!("a device intrinsic exists only on the device")
}

pub fn bitcast_u32(_value: f32) -> u32 {
    unreachable!("a device intrinsic exists only on the device")
}

pub fn bitcast_f32(_value: u32) -> f32 {
    unreachable!("a device intrinsic exists only on the device")
}

pub fn unpack2x16float(_value: u32) -> Fvec2 {
    unreachable!("a device intrinsic exists only on the device")
}

pub fn atomic_add(_pointer: &AtomicU32, _value: u32) -> u32 {
    unreachable!("a device intrinsic exists only on the device")
}

pub fn atomic_sub(_pointer: &AtomicU32, _value: u32) -> u32 {
    unreachable!("a device intrinsic exists only on the device")
}

pub fn atomic_store(_pointer: &AtomicU32, _value: u32) {
    unreachable!("a device intrinsic exists only on the device")
}

pub fn workgroup_barrier() {
    unreachable!("a device intrinsic exists only on the device")
}

pub fn stride(_start: u32, _end: u32, _step: u32) -> core::ops::Range<u32> {
    unreachable!("a device loop exists only on the device")
}

pub fn unroll(_start: u32, _end: u32, _step: u32) -> core::ops::Range<u32> {
    unreachable!("a device loop exists only on the device")
}

pub fn storage_barrier() {
    unreachable!("a device intrinsic exists only on the device")
}
