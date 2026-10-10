pub fn random(count: u32, seed: u32) -> Vec<f32> {
    let mut entropy = seed | 1;
    neura_graph::Init::Uniform {
        low: -1.0,
        high: 1.0,
    }
    .samples(count, &mut entropy)
}
