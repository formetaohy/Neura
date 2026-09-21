use neura_abi::Kind;
use neura_abi::op::{self, OPS, Role};
use neura_abi::{
    BoundsRecord, Geometry, MEDIUM, MatmulTile, NARROW, PROFILES, Profile, StepRecord, TaskRecord,
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
    assert_eq!(offset_of!(TaskRecord, op), 4);
    assert_eq!(offset_of!(TaskRecord, geometry), 8);
    assert_eq!(offset_of!(TaskRecord, count), 16);
    assert_eq!(offset_of!(TaskRecord, a), 28);
    assert_eq!(offset_of!(TaskRecord, param), 40);
    assert_eq!(offset_of!(TaskRecord, chain), 44);
    assert_eq!(offset_of!(TaskRecord, steps), 48);
    assert_eq!(size_of::<StepRecord>(), 12);
    assert_eq!(offset_of!(StepRecord, op), 0);
    assert_eq!(offset_of!(StepRecord, operand), 4);
    assert_eq!(offset_of!(StepRecord, swapped), 8);
    assert_eq!(size_of::<BoundsRecord>(), 12);
    assert_eq!(offset_of!(BoundsRecord, first_task), 0);
    assert_eq!(offset_of!(BoundsRecord, task_count), 4);
    assert_eq!(offset_of!(BoundsRecord, wave), 8);
    assert_eq!(align_of::<ValueRecord>(), 4);
    assert_eq!(align_of::<TaskRecord>(), 4);
    assert_eq!(align_of::<StepRecord>(), 4);
}

#[test]
fn every_task_kind_is_declared_once() {
    assert_eq!(Kind::COUNT as usize, Kind::ALL.len());
    for (code, kind) in Kind::ALL.iter().enumerate() {
        assert_eq!(
            kind.code(),
            code as u32,
            "the {} kind leaves a hole",
            kind.name()
        );
        assert_eq!(Kind::of(kind.code()), *kind);
        assert!(!kind.name().is_empty());
        assert!(!kind.constant().is_empty());
    }
    let mut names = Kind::ALL.iter().map(|kind| kind.name()).collect::<Vec<_>>();
    let mut constants = Kind::ALL
        .iter()
        .map(|kind| kind.constant())
        .collect::<Vec<_>>();
    names.sort_unstable();
    names.dedup();
    constants.sort_unstable();
    constants.dedup();
    assert_eq!(names.len(), Kind::ALL.len(), "two kinds share a name");
    assert_eq!(
        constants.len(),
        Kind::ALL.len(),
        "two kinds share a device constant"
    );
    assert_eq!(Kind::Matmul.name(), "matmul");
    assert_eq!(Kind::LogSoftmax.name(), "log_softmax");
    assert!(Kind::Partial.pointwise());
    assert!(!Kind::Matmul.pointwise());
    assert!(Kind::Binary.chainable());
    assert!(!Kind::Partial.chainable());
    assert!(refuses(|| {
        let _ = Kind::of(Kind::COUNT);
    }));
}

#[test]
fn every_pointwise_op_is_declared_once() {
    assert_eq!(op::COUNT as usize, OPS.len());
    for (code, entry) in OPS.iter().enumerate() {
        assert_eq!(
            entry.code, code as u32,
            "the {} op leaves a hole",
            entry.name
        );
        assert_eq!(op::of(entry.code), entry);
        assert_eq!(op::name(entry.code), entry.name);
        assert_eq!(op::kind(entry.code), entry.family.kind());
        assert!(!entry.apply.is_empty());
        assert_eq!(
            entry.partials.len() as u32,
            entry.family.operands(),
            "the {} op declares a partial for every operand it reads",
            entry.name,
        );
        for slot in 0..entry.family.operands() {
            let _ = entry.partial(slot);
        }
        assert!(refuses(|| {
            let _ = entry.partial(entry.family.operands());
        }));
    }
    let mut names = OPS.iter().map(|entry| entry.name).collect::<Vec<_>>();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), OPS.len(), "two ops share a name");
    assert_eq!(op::name(op::ADD), "add");
    assert_eq!(op::name(op::RELU), "relu");
    assert_eq!(op::name(op::SIGMOID), "sigmoid");
    assert!(refuses(|| {
        let _ = op::of(op::COUNT);
    }));
    assert!(refuses(|| {
        let _ = op::of(op::NONE);
    }));
}

#[test]
fn every_partial_reads_exactly_the_roles_it_names() {
    for op in OPS {
        for slot in 0..op.family.operands() {
            let partial = op.partial(slot);
            let roles = partial.roles();
            assert!(
                !(roles.contains(&Role::Operand) && roles.contains(&Role::Result)),
                "the {} partial {slot} differentiates an operand and its own result at once",
                op.name,
            );
            assert!(
                !roles.contains(&Role::Result) || roles.len() == 1,
                "the {} partial {slot} reads its own result next to an operand",
                op.name,
            );
            if op.family == op::Family::Unary {
                assert!(
                    !roles.contains(&Role::Other),
                    "the {} partial {slot} reads a second operand its op never has",
                    op.name,
                );
            }
            let Some(formula) = partial.formula() else {
                assert!(
                    roles.is_empty(),
                    "the {} partial {slot} reads a role it never asks to be handed",
                    op.name,
                );
                continue;
            };
            assert!(
                mentions(formula, "g"),
                "the {} partial {slot} ignores the gradient it descends from",
                op.name,
            );
            for role in roles {
                assert!(
                    mentions(formula, role.name()),
                    "the {} partial {slot} asks for {} it never reads",
                    op.name,
                    role.name(),
                );
            }
        }
    }
}

fn mentions(formula: &str, name: &str) -> bool {
    formula
        .split(|character: char| !character.is_ascii_alphanumeric())
        .any(|word| word == name)
}

#[test]
fn a_record_declares_what_the_device_reads() {
    let mut task: TaskRecord = bytemuck::Zeroable::zeroed();
    task.kind = Kind::Matmul.code();
    task.op = op::MUL;
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
        Kind::Matmul.code()
    );
    assert_eq!(u32::from_ne_bytes(bytes[4..8].try_into().unwrap()), op::MUL);
    assert_eq!(u32::from_ne_bytes(bytes[8..12].try_into().unwrap()), 2);
    assert_eq!(u32::from_ne_bytes(bytes[16..20].try_into().unwrap()), 5);
    assert_eq!(f32::from_ne_bytes(bytes[40..44].try_into().unwrap()), 0.5);
    assert_eq!(u32::from_ne_bytes(bytes[44..48].try_into().unwrap()), 7);
    assert_eq!(u32::from_ne_bytes(bytes[48..52].try_into().unwrap()), 2);
}

#[test]
fn a_step_declares_what_the_device_applies() {
    let step = StepRecord {
        op: op::RELU,
        operand: 9,
        swapped: 1,
    };
    let bytes = bytemuck::bytes_of(&step);
    assert_eq!(bytes.len(), 12);
    assert_eq!(
        u32::from_ne_bytes(bytes[0..4].try_into().unwrap()),
        op::RELU
    );
    assert_eq!(u32::from_ne_bytes(bytes[4..8].try_into().unwrap()), 9);
    assert_eq!(u32::from_ne_bytes(bytes[8..12].try_into().unwrap()), 1);
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
        let staged = profile
            .ladder()
            .iter()
            .map(|tile| tile.shared_bytes())
            .max()
            .unwrap();
        assert!(
            profile.shared_bytes() >= staged + u64::from(profile.workgroup()) * WORD_BYTES,
            "a profile carries the widest tile it stages beside the scratch its reductions declare",
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
