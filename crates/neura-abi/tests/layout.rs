use neura_abi::{
    BINARY_ADD, BINARY_MUL, BoundsRecord, CHAIN_ADD, CHAIN_COUNT, CHAIN_MUL, CHAIN_RELU,
    CHAIN_SQRT, Geometry, KIND_COUNT, KIND_MATMUL, KIND_SOFTMAX, KIND_UNARY, MEDIUM, MatmulTile,
    NARROW, PROFILES, Profile, StepRecord, TaskRecord, UNARY_RECIP, UNARY_RELU, UNARY_SQRT,
    ValueRecord, WIDE, WORD_BYTES,
};
use std::mem::{align_of, offset_of, size_of};

fn refuses(action: impl FnOnce() + std::panic::UnwindSafe) -> bool {
    std::panic::catch_unwind(action).is_err()
}

#[test]
fn records_follow_the_shader_layout() {
    assert_eq!(size_of::<ValueRecord>(), 48);
    assert_eq!(offset_of!(ValueRecord, base), 0);
    assert_eq!(offset_of!(ValueRecord, dims), 16);
    assert_eq!(offset_of!(ValueRecord, strides), 32);
    assert_eq!(size_of::<TaskRecord>(), 52);
    assert_eq!(offset_of!(TaskRecord, geometry), 8);
    assert_eq!(offset_of!(TaskRecord, count), 16);
    assert_eq!(offset_of!(TaskRecord, a), 28);
    assert_eq!(offset_of!(TaskRecord, param), 40);
    assert_eq!(offset_of!(TaskRecord, chain), 44);
    assert_eq!(offset_of!(TaskRecord, steps), 48);
    assert_eq!(size_of::<StepRecord>(), 8);
    assert_eq!(offset_of!(StepRecord, op), 0);
    assert_eq!(offset_of!(StepRecord, operand), 4);
    assert_eq!(size_of::<BoundsRecord>(), 12);
    assert_eq!(offset_of!(BoundsRecord, first_task), 0);
    assert_eq!(offset_of!(BoundsRecord, task_count), 4);
    assert_eq!(offset_of!(BoundsRecord, wave), 8);
    assert_eq!(align_of::<ValueRecord>(), 4);
    assert_eq!(align_of::<TaskRecord>(), 4);
    assert_eq!(align_of::<StepRecord>(), 4);
}

#[test]
fn every_task_kind_is_named() {
    assert_eq!(KIND_COUNT, 10);
    for kind in 0..KIND_COUNT {
        assert!(!neura_abi::kind_name(kind).is_empty());
    }
    assert_eq!(neura_abi::kind_name(KIND_MATMUL), "matmul");
    assert_eq!(neura_abi::kind_name(KIND_SOFTMAX), "softmax");
    assert_eq!(neura_abi::kind_name(neura_abi::KIND_SUM_TO), "sum_to");
}

#[test]
fn every_op_code_is_named() {
    for op in neura_abi::BINARY_OPS {
        let _ = neura_abi::binary_name(*op);
    }
    for op in neura_abi::UNARY_OPS {
        let _ = neura_abi::unary_name(*op);
    }
    for op in neura_abi::CHAIN_OPS {
        let _ = neura_abi::chain_name(*op);
    }
    assert_eq!(neura_abi::binary_name(BINARY_ADD), "add");
    assert_eq!(neura_abi::binary_name(BINARY_MUL), "mul");
    assert_eq!(neura_abi::unary_name(UNARY_RELU), "relu");
    assert_eq!(neura_abi::unary_name(UNARY_SQRT), "sqrt");
    assert_eq!(neura_abi::unary_name(UNARY_RECIP), "recip");
    assert_eq!(neura_abi::chain_name(CHAIN_ADD), "add");
    assert_eq!(neura_abi::chain_name(CHAIN_MUL), "mul");
    assert_eq!(neura_abi::chain_name(CHAIN_RELU), "relu");
    assert_eq!(neura_abi::chain_name(CHAIN_SQRT), "sqrt");
    assert_eq!(neura_abi::chain_name(neura_abi::CHAIN_RECIP), "recip");
}

#[test]
fn every_elementwise_op_reaches_a_chain_op() {
    let mut reached = Vec::new();
    for op in neura_abi::BINARY_OPS {
        reached.push(neura_abi::chain_op(neura_abi::KIND_BINARY, *op));
    }
    for op in neura_abi::UNARY_OPS {
        reached.push(neura_abi::chain_op(KIND_UNARY, *op));
    }
    reached.sort_unstable();
    reached.dedup();
    let mut declared = neura_abi::CHAIN_OPS.to_vec();
    declared.sort_unstable();
    assert_eq!(
        reached, declared,
        "every elementwise op must land on a chain op the device can run",
    );
    assert_eq!(declared.len() as u32, CHAIN_COUNT);
}

#[test]
fn a_record_declares_what_the_device_reads() {
    let mut task: TaskRecord = bytemuck::Zeroable::zeroed();
    task.kind = KIND_MATMUL;
    task.geometry = 2;
    task.first = 3;
    task.count = 5;
    task.a = 2;
    task.b = 3;
    task.c = 4;
    task.out = 1;
    task.param = 0.5;
    task.chain = 7;
    task.steps = 2;
    let bytes = bytemuck::bytes_of(&task);
    assert_eq!(bytes.len(), 52);
    assert_eq!(
        u32::from_ne_bytes(bytes[0..4].try_into().unwrap()),
        KIND_MATMUL
    );
    assert_eq!(u32::from_ne_bytes(bytes[8..12].try_into().unwrap()), 2);
    assert_eq!(u32::from_ne_bytes(bytes[16..20].try_into().unwrap()), 5);
    assert_eq!(f32::from_ne_bytes(bytes[40..44].try_into().unwrap()), 0.5);
    assert_eq!(u32::from_ne_bytes(bytes[44..48].try_into().unwrap()), 7);
    assert_eq!(u32::from_ne_bytes(bytes[48..52].try_into().unwrap()), 2);
}

#[test]
fn a_step_declares_what_the_device_applies() {
    let step = StepRecord {
        op: CHAIN_RELU,
        operand: 9,
    };
    let bytes = bytemuck::bytes_of(&step);
    assert_eq!(bytes.len(), 8);
    assert_eq!(
        u32::from_ne_bytes(bytes[0..4].try_into().unwrap()),
        CHAIN_RELU
    );
    assert_eq!(u32::from_ne_bytes(bytes[4..8].try_into().unwrap()), 9);
}

#[test]
fn a_profile_offers_the_tiles_one_workgroup_carries() {
    for profile in PROFILES {
        for tile in profile.ladder() {
            assert_eq!(profile.workgroup(), tile.threads());
            assert_eq!(
                tile.registers() * profile.workgroup(),
                tile.rows() * tile.columns(),
            );
        }
        assert!(
            profile
                .ladder()
                .windows(2)
                .all(|pair| pair[0].tile_work() < pair[1].tile_work()),
            "a profile offers its tiles from the smallest to the widest",
        );
        assert_eq!(
            profile.shared_bytes(),
            profile
                .ladder()
                .iter()
                .map(|tile| tile.shared_bytes())
                .max()
                .unwrap()
                + u64::from(profile.workgroup()) * WORD_BYTES,
        );
        assert!(profile.fits(u32::MAX, profile.shared_bytes()));
        assert!(!profile.fits(profile.workgroup() - 1, profile.shared_bytes()));
        assert!(!profile.fits(u32::MAX, profile.shared_bytes() - 1));
    }
    assert!(
        NARROW.shared_bytes() < MEDIUM.shared_bytes(),
        "a wider workgroup stages more",
    );
    assert!(
        MEDIUM.shared_bytes() < WIDE.shared_bytes(),
        "a wider workgroup stages more",
    );
    assert!(
        WIDE.shared_bytes() > 16 * 1024,
        "the widest profile must ask the device for more than the baseline pool",
    );
    assert!(
        MEDIUM.shared_bytes() <= 16 * 1024,
        "the middle profile must run on the baseline pool",
    );
    assert!(
        PROFILES
            .windows(2)
            .all(|pair| pair[0].workgroup() < pair[1].workgroup()),
        "the profiles are ordered by the workgroup they hand the device",
    );
}

#[test]
fn a_profile_refuses_a_ladder_one_workgroup_cannot_carry() {
    assert!(refuses(|| {
        let _ = Profile::of(&[]);
    }));
    assert!(refuses(|| {
        static MIXED: &[MatmulTile] = &[
            MatmulTile::new(16, 16, 16, 8, 8),
            MatmulTile::new(32, 32, 16, 16, 16),
        ];
        let _ = Profile::of(MIXED);
    }));
}

#[test]
fn a_geometry_without_a_tile_declares_only_its_workgroup() {
    let geometry = Geometry::of(WIDE, &[]);
    assert!(geometry.tiles().is_empty());
    assert_eq!(geometry.workgroup(), WIDE.workgroup());
    assert_eq!(
        geometry.declarations(),
        format!("const WORKGROUP_SIZE: u32 = {}u;\n", WIDE.workgroup()),
    );
}

#[test]
fn a_geometry_declares_the_tiles_its_program_carries() {
    for profile in PROFILES {
        for tiles in [
            &profile.ladder()[..1],
            &profile.ladder()[..2],
            profile.ladder(),
        ] {
            let geometry = Geometry::of(*profile, tiles);
            let declarations = geometry.declarations();
            assert_eq!(geometry.workgroup(), profile.workgroup());
            assert_eq!(geometry.tiles(), tiles);
            assert!(declarations.contains(&format!(
                "const WORKGROUP_SIZE: u32 = {}u;",
                profile.workgroup()
            )));
            assert!(declarations.contains(&format!(
                "const MATMUL_LEFT_STAGE: u32 = {}u;",
                2 * tiles
                    .iter()
                    .map(|tile| tile.rows() * tile.depth())
                    .max()
                    .unwrap(),
            )));
            assert!(declarations.contains(&format!(
                "const MATMUL_RIGHT_STAGE: u32 = {}u;",
                2 * tiles
                    .iter()
                    .map(|tile| tile.depth() * tile.columns())
                    .max()
                    .unwrap(),
            )));
            for (index, tile) in tiles.iter().enumerate() {
                assert_eq!(geometry.geometry(*tile), index as u32);
                assert_eq!(geometry.tile(index as u32), *tile);
                for (suffix, value) in [
                    ("ROWS", tile.rows()),
                    ("COLUMNS", tile.columns()),
                    ("DEPTH", tile.depth()),
                    ("THREAD_ROWS", tile.thread_rows()),
                    ("THREAD_COLUMNS", tile.thread_columns()),
                    ("REGISTER_ROWS", tile.register_rows()),
                    ("REGISTER_COLUMNS", tile.register_columns()),
                ] {
                    let declaration = format!("const MATMUL_{suffix}_{index}: u32 = {value}u;");
                    assert!(
                        declarations.contains(&declaration),
                        "a program carrying {tiles:?} misses {declaration}",
                    );
                }
            }
            assert!(refuses(|| {
                let _ = geometry.tile(geometry.tiles().len() as u32);
            }));
            assert!(refuses(|| {
                let _ = geometry.geometry(MatmulTile::new(24, 24, 16, 8, 8));
            }));
        }
        assert!(refuses(|| {
            let _ = Geometry::of(*profile, &[MatmulTile::new(24, 24, 16, 8, 8)]);
        }));
    }
    assert!(refuses(|| {
        static WIDE_TILE: &[MatmulTile] = &[MatmulTile::new(64, 64, 16, 16, 16)];
        let _ = Geometry::of(NARROW, WIDE_TILE);
    }));
}

#[test]
fn a_matmul_tile_refuses_a_geometry_its_workgroup_cannot_carry() {
    assert!(refuses(|| {
        let _ = MatmulTile::new(16, 16, 16, 8, 0);
    }));
    assert!(refuses(|| {
        let _ = MatmulTile::new(16, 16, 16, 5, 8);
    }));
    assert!(refuses(|| {
        let _ = MatmulTile::new(16, 24, 16, 8, 16);
    }));
    assert!(refuses(|| {
        let _ = MatmulTile::new(0, 16, 16, 8, 8);
    }));
    assert!(refuses(|| {
        let _ = MatmulTile::new(16, 16, 0, 8, 8);
    }));
}
