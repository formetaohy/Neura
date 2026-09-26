use neura_abi::WORD_BYTES;
use neura_profile::{Budget, Geometry, MatmulTile, Profile};
use std::panic::AssertUnwindSafe;

fn wide_device() -> Budget {
    Budget::of(1024, 48 << 10)
}

fn mid_device() -> Budget {
    Budget::of(512, 32 << 10)
}

fn refuses(action: impl FnOnce()) -> bool {
    std::panic::catch_unwind(AssertUnwindSafe(action)).is_err()
}

#[test]
fn a_device_budget_fits_every_profile_it_derives() {
    for budget in [Budget::BASELINE, mid_device(), wide_device()] {
        let profiles = Profile::derive(budget);
        assert!(!profiles.is_empty(), "a device derives no profile");
        for profile in &profiles {
            assert!(
                profile.fits(budget.threads(), budget.shared_bytes()),
                "{profile:?} outruns the {budget:?} it was derived from",
            );
            assert!(
                profile.workgroup() <= budget.threads(),
                "a derived profile hands the device more threads than it schedules",
            );
            assert!(
                profile
                    .tiles()
                    .iter()
                    .all(|tile| tile.threads() == profile.workgroup()),
                "a tile is carried by another workgroup than its profile hands the device",
            );
            assert!(
                profile
                    .tiles()
                    .windows(2)
                    .all(|pair| pair[0].tile_work() <= pair[1].tile_work()),
                "a profile offers its tiles from the smallest to the widest",
            );
            assert!(
                profile.tiles().windows(2).all(|pair| pair[0] != pair[1]),
                "a profile offers one tile twice",
            );
            assert!(
                profile.tiles().iter().any(|tile| tile.registers() >= 4),
                "a profile carries no tile a thread accumulates in registers",
            );
            assert!(
                !profile.tiles().is_empty() && profile.tiles().len() <= 32,
                "a profile carries no bounded set of tiles",
            );
        }
        assert!(
            profiles
                .windows(2)
                .all(|pair| pair[0].workgroup() < pair[1].workgroup()),
            "the profiles are ordered by the workgroup they hand the device",
        );
    }
}

#[test]
fn a_wider_device_derives_the_wider_tiles_it_can_stage() {
    let narrow = Profile::derive(Budget::BASELINE);
    let mid = Profile::derive(mid_device());
    let wide = Profile::derive(wide_device());
    assert!(narrow.len() <= mid.len() && mid.len() <= wide.len());
    assert!(
        narrow.iter().all(|profile| profile.workgroup() <= 256),
        "the baseline derives a workgroup beyond the threads it schedules",
    );
    assert!(
        wide.iter().any(|profile| profile.workgroup() == 1024),
        "a device of a thousand threads derives no thousand thread workgroup",
    );
    let widest = |profiles: &[Profile]| {
        profiles
            .iter()
            .flat_map(|profile| profile.tiles())
            .map(|tile| tile.rows() * tile.columns())
            .max()
            .expect("a profile offers a tile")
    };
    assert!(
        widest(&narrow) < widest(&mid) && widest(&mid) < widest(&wide),
        "a wider device derives no wider tile",
    );
    let holds = |profiles: &[Profile], rows: u32, columns: u32| {
        profiles.iter().any(|profile| {
            profile
                .tiles()
                .iter()
                .any(|tile| (tile.rows(), tile.columns()) == (rows, columns))
        })
    };
    assert!(
        holds(&wide, 128, 128),
        "a device that stages a four kilobyte block derives no 128x128 tile",
    );
    assert!(
        !holds(&narrow, 128, 128),
        "the baseline derives a tile beyond the pool it stages from",
    );
    assert!(
        holds(&narrow, 16, 16) && holds(&wide, 16, 16),
        "a device derives no tile for the narrowest product",
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
    assert!(refuses(|| {
        let _ = Budget::of(16, 48 << 10);
    }));
    assert!(refuses(|| {
        let _ = Budget::of(1024, 8 << 10);
    }));
}

#[test]
fn a_geometry_declares_every_tile_its_profile_offers() {
    for profile in Profile::derive(wide_device()) {
        let geometry = Geometry::of(
            profile.workgroup(),
            profile.shared_bytes(),
            profile.tiles(),
            &[],
        );
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
        assert!(
            profile.shared_bytes() >= (u64::from(left) + u64::from(right)) * WORD_BYTES,
            "a profile stages more than the shared pool it asks the device for",
        );
        for (index, tile) in profile.tiles().iter().enumerate() {
            assert_eq!(geometry.geometry(*tile), index as u32);
            assert_eq!(geometry.tile(index as u32), *tile);
        }
        let count = profile.tiles().len() as u32;
        assert!(refuses(|| {
            let _ = geometry.tile(count);
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
