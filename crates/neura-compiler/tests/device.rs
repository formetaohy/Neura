use neura_compiler::device::{Intrinsic, Reach};
use neura_compiler::{AtomicU32, BindingSpec, Compiler, ComputeProgram, Read, ReadWrite, kernel};
use neura_shader::{AtomicOp, Backend, Barrier, Instruction, MathFun};

#[kernel(workgroup_size = 8)]
fn kernel_vocabulary(
    lid: u32,
    input: Read<f32>,
    indices: Read<u32>,
    out: ReadWrite<f32>,
    counters: ReadWrite<AtomicU32>,
) {
    let value = input[lid];
    let chosen = select(0.0f32, value, value > 0.0f32);
    let smallest = min(value, chosen);
    let largest = max(smallest, abs(value));
    let root = sqrt(abs(largest));
    let growth = exp(root) + log(root + 1.0f32) + tanh(root);
    let wave = sin(root) + cos(root) + pow(root, 2.0f32) + growth;
    let rounded = trunc(wave) + floor(wave);
    let bits = bitcast_u32(rounded);
    let halves = unpack2x16float(bits);
    let lanes = uvec4(lid, bits, indices[lid], bitcast_u32(halves.x + halves.y));
    atomic_add(&counters[0], lanes.x);
    atomic_sub(&counters[1], lanes.y);
    atomic_store(&counters[2], lanes.z);
    workgroup_barrier();
    storage_barrier();
    out[lid] = bitcast_f32(lanes.w) + (lanes.x as f32);
}

#[neura_compiler::module]
mod tile_vocabulary {
    #[neura_compiler::kernel]
    fn main(lid: u32) {
        let mut registers = scalar_array(f16(0.0), 2u32);
        registers[0u32] = f16(u32(f32(i32(lid))));
        registers[1u32] = registers[0u32];
        claim[0u32] = u32(registers[1u32]);
        output[lid] = workgroup_uniform_load(&claim[0u32]);
        let left = coopmat_load_row(&halves[0u32], 16u32);
        let right = coopmat_load_b_row(&halves[0u32], 16u32);
        let mut accumulator = coopmat_accumulator(0.0);
        accumulator = coopmat_muladd(left, right, accumulator);
        coopmat_store(&panels[0u32], 16u32, accumulator);
    }
}

fn tile_program() -> ComputeProgram {
    let mut compiler = Compiler::empty();
    compiler.constant("COOPMAT_ROWS", 16);
    compiler.constant("COOPMAT_COLUMNS", 16);
    compiler.constant("COOPMAT_DEPTH", 16);
    compiler.storage_array("output", "u32", BindingSpec::writable_storage(0));
    compiler.workgroup("claim", "u32", 4);
    compiler.workgroup_bytes("halves", "f16", 512);
    compiler.workgroup("panels", "f32", 256);
    tile_vocabulary::define(&mut compiler);
    compiler.finish("tile vocabulary", "main", 64)
}

fn produces(intrinsic: Intrinsic) -> fn(&Instruction) -> bool {
    match intrinsic {
        Intrinsic::Select => |instruction| matches!(instruction, Instruction::Select { .. }),
        Intrinsic::Max => |instruction| {
            matches!(
                instruction,
                Instruction::Math {
                    fun: MathFun::Max,
                    ..
                }
            )
        },
        Intrinsic::Min => |instruction| {
            matches!(
                instruction,
                Instruction::Math {
                    fun: MathFun::Min,
                    ..
                }
            )
        },
        Intrinsic::Abs => |instruction| {
            matches!(
                instruction,
                Instruction::Math {
                    fun: MathFun::Abs,
                    ..
                }
            )
        },
        Intrinsic::Sqrt => |instruction| {
            matches!(
                instruction,
                Instruction::Math {
                    fun: MathFun::Sqrt,
                    ..
                }
            )
        },
        Intrinsic::Exp => |instruction| {
            matches!(
                instruction,
                Instruction::Math {
                    fun: MathFun::Exp,
                    ..
                }
            )
        },
        Intrinsic::Log => |instruction| {
            matches!(
                instruction,
                Instruction::Math {
                    fun: MathFun::Log,
                    ..
                }
            )
        },
        Intrinsic::Tanh => |instruction| {
            matches!(
                instruction,
                Instruction::Math {
                    fun: MathFun::Tanh,
                    ..
                }
            )
        },
        Intrinsic::Trunc => |instruction| {
            matches!(
                instruction,
                Instruction::Math {
                    fun: MathFun::Trunc,
                    ..
                }
            )
        },
        Intrinsic::Sin => |instruction| {
            matches!(
                instruction,
                Instruction::Math {
                    fun: MathFun::Sin,
                    ..
                }
            )
        },
        Intrinsic::Cos => |instruction| {
            matches!(
                instruction,
                Instruction::Math {
                    fun: MathFun::Cos,
                    ..
                }
            )
        },
        Intrinsic::Pow => |instruction| {
            matches!(
                instruction,
                Instruction::Math {
                    fun: MathFun::Pow,
                    ..
                }
            )
        },
        Intrinsic::Floor => |instruction| {
            matches!(
                instruction,
                Instruction::Math {
                    fun: MathFun::Floor,
                    ..
                }
            )
        },
        Intrinsic::Uvec4 => |instruction| matches!(instruction, Instruction::Compose { .. }),
        Intrinsic::BitcastU32 | Intrinsic::BitcastF32 => {
            |instruction| matches!(instruction, Instruction::Bitcast { .. })
        }
        Intrinsic::UnpackHalf2x16 => |instruction| {
            matches!(
                instruction,
                Instruction::Math {
                    fun: MathFun::UnpackHalf2x16,
                    ..
                }
            )
        },
        Intrinsic::AtomicAdd => |instruction| {
            matches!(
                instruction,
                Instruction::Atomic {
                    op: AtomicOp::Add,
                    ..
                }
            )
        },
        Intrinsic::AtomicSub => |instruction| {
            matches!(
                instruction,
                Instruction::Atomic {
                    op: AtomicOp::Subtract,
                    ..
                }
            )
        },
        Intrinsic::AtomicStore => |instruction| matches!(instruction, Instruction::Store { .. }),
        Intrinsic::WorkgroupBarrier => {
            |instruction| matches!(instruction, Instruction::Barrier(Barrier::WorkGroup))
        }
        Intrinsic::StorageBarrier => {
            |instruction| matches!(instruction, Instruction::Barrier(Barrier::Storage))
        }
        Intrinsic::F32 | Intrinsic::U32 | Intrinsic::I32 | Intrinsic::F16 => {
            |instruction| matches!(instruction, Instruction::Convert { .. })
        }
        Intrinsic::ScalarArray => |instruction| matches!(instruction, Instruction::Store { .. }),
        Intrinsic::WorkgroupUniformLoad => {
            |instruction| matches!(instruction, Instruction::WorkGroupUniformLoad { .. })
        }
        Intrinsic::CoopmatAccumulator => {
            |instruction| matches!(instruction, Instruction::MatrixFill { .. })
        }
        Intrinsic::CoopmatLoadRow | Intrinsic::CoopmatLoadBRow | Intrinsic::CoopmatLoadColumn => {
            |instruction| matches!(instruction, Instruction::MatrixLoad { .. })
        }
        Intrinsic::CoopmatMuladd => {
            |instruction| matches!(instruction, Instruction::MatrixMulAdd { .. })
        }
        Intrinsic::CoopmatStore => {
            |instruction| matches!(instruction, Instruction::MatrixStore { .. })
        }
    }
}

fn holds(body: &[Instruction], predicate: fn(&Instruction) -> bool) -> bool {
    body.iter().any(|instruction| {
        predicate(instruction)
            || match instruction {
                Instruction::If { accept, reject, .. } => {
                    holds(accept, predicate) || holds(reject, predicate)
                }
                Instruction::Switch { cases, default, .. } => {
                    cases.iter().any(|(_, body)| holds(body, predicate))
                        || holds(default, predicate)
                }
                Instruction::Loop { body, continuing } => {
                    holds(body, predicate) || holds(continuing, predicate)
                }
                Instruction::Block(body) => holds(body, predicate),
                _ => false,
            }
    })
}

#[test]
fn every_intrinsic_the_device_speaks_is_declared_once() {
    for intrinsic in Intrinsic::ALL.iter().copied() {
        assert_eq!(
            Intrinsic::of(intrinsic.name()),
            Some(intrinsic),
            "the device intrinsic {} is found by the name it declares",
            intrinsic.name(),
        );
        assert!(!intrinsic.name().is_empty());
    }
    assert_eq!(
        Intrinsic::of("a_name_no_intrinsic_carries"),
        None,
        "a name no intrinsic declares is a Rust device function",
    );
}

#[test]
fn a_kernel_calls_every_intrinsic_its_reach_declares() {
    let program = kernel_vocabulary();
    for intrinsic in Intrinsic::ALL
        .iter()
        .copied()
        .filter(|intrinsic| intrinsic.reach() == Reach::Kernel)
    {
        assert!(
            holds(&program.module().entry().body, produces(intrinsic)),
            "a kernel calls {} and the device program holds the instruction it lowers to",
            intrinsic.name(),
        );
    }
    for backend in [Backend::Vulkan, Backend::Dx12, Backend::Metal] {
        let _ = program.translate(backend);
    }
}

#[test]
fn the_tile_language_carries_the_intrinsics_a_kernel_cannot_declare() {
    let program = tile_program();
    for intrinsic in Intrinsic::ALL
        .iter()
        .copied()
        .filter(|intrinsic| intrinsic.reach() == Reach::Tile)
    {
        assert!(
            holds(&program.module().entry().body, produces(intrinsic)),
            "the tile language calls {} and the device program holds the instruction it lowers to",
            intrinsic.name(),
        );
    }
    for backend in [Backend::Vulkan, Backend::Metal] {
        let _ = program.translate(backend);
    }
}

#[neura_compiler::module]
mod few_arguments {
    #[neura_compiler::kernel]
    fn main(lid: u32) {
        output[lid] = min(f32(1.0));
    }
}

#[test]
#[should_panic(expected = "the device intrinsic min takes 2 arguments")]
fn an_intrinsic_the_table_counts_refuses_its_arguments() {
    let mut compiler = Compiler::empty();
    compiler.storage_array("output", "f32", BindingSpec::writable_storage(0));
    few_arguments::define(&mut compiler);
    compiler.finish("few arguments", "main", 64);
}
