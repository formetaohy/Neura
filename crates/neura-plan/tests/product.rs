use neura_abi::Element;
use neura_plan::{Contiguous, Product, Weight};
use neura_profile::{FoldedTile, MatmulStrategy, MatmulTile, Profile};

#[test]
fn a_plain_product_of_two_rows_stages_and_a_packed_one_folds() {
    let staged = MatmulTile::new(MatmulStrategy::Staged, 8, 32, 32, 8, 32);
    let folded = MatmulTile::folded(FoldedTile::new(1, 8, 8, 32));
    let fused = MatmulTile::folded(FoldedTile::staged(2, 8, 4, 32));
    let profile = Profile::within(&[staged, folded, fused], 40);
    let single = Product::of(
        1,
        1,
        1024,
        32,
        Weight::of(Element::Single, Contiguous::Reduction),
    );
    let plain = Product::of(
        1,
        2,
        1024,
        32,
        Weight::of(Element::Single, Contiguous::Reduction),
    );
    let packed = Product::of(
        1,
        2,
        1024,
        32,
        Weight::of(Element::Q8_0, Contiguous::Reduction),
    );
    assert_eq!(single.planned(profile), folded);
    assert_eq!(plain.planned(profile), staged);
    assert_eq!(packed.planned(profile), fused);
}
