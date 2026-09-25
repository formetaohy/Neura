use neura_abi::WORD_BYTES;
use neura_profile::{Geometry, MEDIUM, MatmulTile, NARROW, PROFILES, Profile, WIDE};

fn refuses(action: impl FnOnce() + std::panic::UnwindSafe) -> bool {
    std::panic::catch_unwind(action).is_err()
}

#[test]
fn a_profile_offers_the_tiles_one_workgroup_carries() {
    for profile in PROFILES {
        for (index, tile) in profile.tiles().iter().enumerate() {
            assert_eq!(profile.workgroup(), tile.threads());
            assert_eq!(
                tile.registers() * profile.workgroup(),
                tile.rows() * tile.columns(),
            );
            assert!(
                !profile.tiles()[..index].contains(tile),
                "a profile offers {tile:?} twice",
            );
        }
        assert!(
            profile
                .tiles()
                .windows(2)
                .all(|pair| pair[0].tile_work() <= pair[1].tile_work()),
            "a profile offers its tiles from the smallest to the widest",
        );
        let staged = profile
            .tiles()
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
fn a_profile_refuses_a_pool_one_workgroup_cannot_carry() {
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
fn a_geometry_declares_every_tile_its_profile_offers() {
    for profile in PROFILES {
        let geometry = Geometry::of(*profile);
        let (left, right) = geometry.stage_lengths();
        assert_eq!(geometry.workgroup(), profile.workgroup());
        assert_eq!(geometry.tiles(), profile.tiles());
        assert_eq!(
            left,
            2 * profile
                .tiles()
                .iter()
                .map(|tile| tile.left_stage())
                .max()
                .unwrap() as u32
        );
        assert_eq!(
            right,
            2 * profile
                .tiles()
                .iter()
                .map(|tile| tile.right_stage())
                .max()
                .unwrap() as u32
        );
        for (index, tile) in profile.tiles().iter().enumerate() {
            assert_eq!(geometry.geometry(*tile), index as u32);
            assert_eq!(geometry.tile(index as u32), *tile);
        }
        assert!(refuses(|| {
            let _ = geometry.tile(profile.tiles().len() as u32);
        }));
        assert!(refuses(|| {
            let _ = geometry.geometry(MatmulTile::new(24, 24, 16, 8, 8));
        }));
    }
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
