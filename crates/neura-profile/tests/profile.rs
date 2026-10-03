use neura_abi::{Kind, WORD_BYTES};
use neura_profile::{
    AttentionTile, Budget, CooperativeMatrix, Geometry, MatmulStrategy, MatmulTile, Profile,
};
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

fn walked(profile: Profile) -> Vec<(u32, MatmulTile)> {
    profile
        .tiles()
        .iter()
        .enumerate()
        .map(|(index, tile)| (index as u32, *tile))
        .collect()
}

#[test]
fn a_device_budget_fits_every_profile_it_derives() {
    for budget in [Budget::BASELINE, mid_device(), wide_device()] {
        let profiles = Profile::derive(budget, None);
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
fn a_wider_device_derives_the_tiles_its_registers_and_pool_carry() {
    let cooperative = CooperativeMatrix::new(32, 16, 16, 16);
    for budget in [Budget::BASELINE, mid_device(), wide_device()] {
        for profile in Profile::derive(budget, Some(cooperative)) {
            assert!(
                profile.shared_bytes() <= budget.shared_bytes(),
                "{profile:?} stages a pool beyond the one a device program hands its workgroups",
            );
            assert!(
                profile.staging_bytes() + neura_profile::CLAIM_BYTES <= 21 << 10,
                "{profile:?} derives a staged menu beyond the pool one candidate stages from",
            );
            for tile in profile.tiles() {
                assert!(
                    (tile.registers() + tile.operands()) * profile.workgroup() <= 65_536,
                    "{tile:?} asks a {}-thread workgroup for {} registers a thread",
                    profile.workgroup(),
                    tile.registers() + tile.operands(),
                );
            }
        }
        assert!(
            Profile::derive(budget, Some(cooperative))
                .iter()
                .any(|profile| profile
                    .tiles()
                    .iter()
                    .any(|tile| matches!(tile.strategy(), MatmulStrategy::Cooperative))),
            "a device that holds cooperative matrices derives no cooperative tile for any workgroup",
        );
    }
    let holds = |profiles: &[Profile], rows: u32, columns: u32| {
        profiles.iter().any(|profile| {
            profile
                .tiles()
                .iter()
                .any(|tile| (tile.rows(), tile.columns()) == (rows, columns))
        })
    };
    let wide = Profile::derive(wide_device(), None);
    assert!(
        holds(&wide, 128, 128),
        "a device that stages a four kilobyte block derives no 128x128 tile",
    );
    assert!(
        holds(&wide, 16, 16),
        "a device derives no tile for the narrowest product",
    );
}

#[test]
fn a_cooperative_tile_takes_only_the_pool_the_staged_panels_leave() {
    let cooperative = CooperativeMatrix::new(32, 16, 16, 16);
    for budget in [Budget::BASELINE, mid_device(), wide_device()] {
        let plain = Profile::derive(budget, None);
        let gathered = Profile::derive(budget, Some(cooperative));
        for (plain, gathered) in plain.iter().zip(&gathered) {
            assert_eq!(
                gathered.staging_bytes(),
                plain.staging_bytes(),
                "a cooperative tile narrows the staged panels a profile stages",
            );
            assert!(
                gathered.shared_bytes() <= budget.shared_bytes(),
                "{gathered:?} asks a device for more pool than it offers",
            );
        }
    }
}

#[test]
fn a_profile_refuses_a_pool_one_workgroup_cannot_carry() {
    assert!(refuses(|| {
        let _ = Profile::of(&[]);
    }));
    assert!(refuses(|| {
        static MIXED: &[MatmulTile] = &[
            MatmulTile::new(MatmulStrategy::Staged, 16, 16, 16, 8, 8),
            MatmulTile::new(MatmulStrategy::Staged, 32, 32, 16, 16, 16),
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
fn a_geometry_declares_only_the_tiles_its_plan_walks() {
    for profile in Profile::derive(wide_device(), None) {
        let all = walked(profile);
        let walked = &all[..2];
        let geometry = Geometry::of(profile.workgroup(), profile.shared_bytes(), walked, &[]);
        let (left, right) = (
            walked.iter().map(|(_, tile)| tile.left_stage()).max(),
            walked.iter().map(|(_, tile)| tile.right_stage()).max(),
        );
        assert_eq!(geometry.workgroup(), profile.workgroup());
        assert_eq!(geometry.walked(), walked);
        assert_eq!(
            geometry.staging_bytes(),
            2 * (left.unwrap_or(0) + right.unwrap_or(0)) * WORD_BYTES,
            "a geometry stages the panels of the tiles its plan walks",
        );
        assert!(
            geometry.staging_bytes() < profile.staging_bytes(),
            "a plan that walks the narrow tiles of {profile:?} stages the panels of the wide ones",
        );
        assert_eq!(
            geometry.scratch_bytes(&[Kind::Matmul]),
            geometry.staging_bytes(),
            "a pool reserves exactly the panels a product stages",
        );
        assert!(
            geometry.workgroup_bytes(&[Kind::Matmul]) <= profile.shared_bytes(),
            "a geometry of two walked tiles outruns the profile that offers them",
        );
        let one = Geometry::of(
            profile.workgroup(),
            profile.shared_bytes(),
            &walked[..1],
            &[],
        );
        assert!(
            one.staging_bytes() <= geometry.staging_bytes(),
            "a plan that walks one tile stages more than a plan that walks two",
        );
    }
}

#[test]
fn every_tile_of_a_profile_is_affordable_on_its_own() {
    let cooperative = CooperativeMatrix::new(32, 16, 16, 16);
    for budget in [Budget::BASELINE, mid_device(), wide_device()] {
        for profile in Profile::derive(budget, Some(cooperative)) {
            for (index, tile) in profile.tiles().iter().enumerate() {
                let geometry = Geometry::of(
                    profile.workgroup(),
                    profile.shared_bytes(),
                    &[(index as u32, *tile)],
                    &[],
                );
                assert!(
                    geometry.workgroup_bytes(&[Kind::Matmul]) <= profile.shared_bytes(),
                    "a plan that walks {tile:?} alone outruns the profile that offers it",
                );
            }
        }
    }
}

#[test]
fn a_pool_is_as_wide_as_the_widest_body_that_stages_from_it() {
    for profile in Profile::derive(wide_device(), None) {
        let geometry = Geometry::of(
            profile.workgroup(),
            profile.shared_bytes(),
            &walked(profile)[..],
            &[AttentionTile::new(4, 8)],
        );
        let bodies = [Kind::Matmul, Kind::Attention, Kind::SumAxis, Kind::Argmax];
        let demands = bodies.map(|kind| geometry.scratch_bytes(&[kind]));
        assert!(
            demands.iter().all(|bytes| *bytes > 0),
            "a body that stages workgroup scratch declares none",
        );
        assert_eq!(
            geometry.scratch_bytes(&bodies),
            demands.iter().copied().max().expect("a body stages"),
            "one pool stages the widest body instead of partitioning itself among them",
        );
        assert!(
            geometry.scratch_bytes(&bodies) < demands.iter().sum(),
            "a pool that holds every body at once is no pool",
        );
        assert!(
            geometry.scratch_bytes(&bodies) + neura_profile::CLAIM_BYTES <= profile.shared_bytes(),
            "a pool outruns the profile that offers it",
        );
    }
}

#[test]
fn a_matmul_tile_refuses_a_geometry_its_workgroup_cannot_carry() {
    assert!(refuses(|| {
        let _ = MatmulTile::new(MatmulStrategy::Staged, 16, 16, 16, 8, 0);
    }));
    assert!(refuses(|| {
        let _ = MatmulTile::new(MatmulStrategy::Staged, 16, 16, 16, 5, 8);
    }));
    assert!(refuses(|| {
        let _ = MatmulTile::new(MatmulStrategy::Staged, 16, 24, 16, 8, 16);
    }));
    assert!(refuses(|| {
        let _ = MatmulTile::new(MatmulStrategy::Staged, 0, 16, 16, 8, 8);
    }));
    assert!(refuses(|| {
        let _ = MatmulTile::new(MatmulStrategy::Staged, 16, 16, 0, 8, 8);
    }));
}

#[test]
fn a_streamed_tile_carries_no_operand_through_shared_memory() {
    let tile = MatmulTile::new(MatmulStrategy::Streamed, 1, 256, 8, 1, 256);
    assert_eq!(tile.strategy(), MatmulStrategy::Streamed);
    assert_eq!(tile.left_stage(), 0);
    assert_eq!(tile.right_stage(), 0);
    assert_eq!(tile.shared_bytes(), 0);
    assert_eq!(tile.registers(), 1);
    assert_eq!(tile.threads(), 256);
}

#[test]
fn a_profile_offers_a_streamed_tile_for_the_narrowest_product() {
    for profile in Profile::derive(wide_device(), None) {
        let streamed = profile
            .tiles()
            .iter()
            .filter(|tile| tile.strategy() == MatmulStrategy::Streamed)
            .collect::<Vec<_>>();
        assert!(
            streamed.iter().any(|tile| tile.rows() == 1),
            "{profile:?} offers no product a single row rides",
        );
        assert!(
            streamed.iter().all(|tile| tile.shared_bytes() == 0),
            "a streamed product stages an operand beside the registers it accumulates in",
        );
        assert!(
            streamed
                .iter()
                .all(|tile| tile.columns().is_multiple_of(tile.thread_columns())),
            "a streamed product hands its threads a column band its grid does not divide",
        );
        assert!(
            profile
                .tiles()
                .iter()
                .any(|tile| tile.strategy() == MatmulStrategy::Staged && tile.shared_bytes() > 0),
            "{profile:?} offers no product that stages its operands",
        );
        let left = profile
            .tiles()
            .iter()
            .map(|tile| tile.left_stage())
            .max()
            .expect("a profile offers a tile");
        let right = profile
            .tiles()
            .iter()
            .map(|tile| tile.right_stage())
            .max()
            .expect("a profile offers a tile");
        assert_eq!(
            profile.staging_bytes(),
            2 * (left + right) * WORD_BYTES,
            "a profile counts the staging of the tiles it does not carry",
        );
    }
}
