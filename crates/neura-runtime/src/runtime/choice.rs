use super::{Program, Runtime, Weights};
use neura_abi::{Digest, Element};
use neura_gpu::AdapterInfo;
use neura_graph::Graph;
use neura_plan::{Contiguous, Product, Weight};
use neura_profile::{MatmulStrategy, MatmulTile, Profile};

const FORMAT: u32 = 2;
const HEAD_WORDS: usize = 4;
const PRODUCT_WORDS: usize = 7;
const TILE_WORDS: usize = 8;
const REDUCTION: u32 = 0;
const OUTPUT: u32 = 1;

pub(crate) struct Choice {
    profile: Profile,
    chosen: Vec<(Product, MatmulTile)>,
}

impl Choice {
    pub(crate) fn of(profile: Profile, chosen: &[(Product, MatmulTile)]) -> Self {
        Self {
            profile,
            chosen: chosen.to_vec(),
        }
    }

    pub(crate) fn profile(&self) -> Profile {
        self.profile
    }

    pub(crate) fn chosen(&self) -> &[(Product, MatmulTile)] {
        &self.chosen
    }

    pub(crate) fn encode(&self, profiles: &[Profile]) -> Vec<u8> {
        let menu = menu(self.profile, profiles)
            .unwrap_or_else(|| panic!("a remembered profile is one the device offers"));
        let mut bytes = Vec::new();
        push(
            &mut bytes,
            &[
                FORMAT,
                self.profile.workgroup(),
                self.profile.registers(),
                self.chosen.len() as u32,
            ],
        );
        bytes.extend_from_slice(&self.profile.shared_bytes().to_le_bytes());
        for (product, tile) in &self.chosen {
            let walked = menu
                .iter()
                .position(|held| *held == *tile)
                .unwrap_or_else(|| {
                    panic!("a remembered tile is one the profile that walks it offers")
                });
            push(
                &mut bytes,
                &[
                    product.planes(),
                    product.rows(),
                    product.columns(),
                    product.depth(),
                    product.weight().element().code(),
                    contiguous(product.weight().contiguous()),
                    walked as u32,
                ],
            );
            push(&mut bytes, &identity(*tile));
        }
        bytes
    }

    pub(crate) fn decode(bytes: &[u8], profiles: &[Profile]) -> Option<Self> {
        let mut held = Words::of(bytes);
        let head = held.take::<HEAD_WORDS>()?;
        if head[0] != FORMAT {
            return None;
        }
        let shared_bytes = held.long()?;
        let profile = profiles.iter().copied().find(|profile| {
            profile.workgroup() == head[1]
                && profile.registers() == head[2]
                && profile.shared_bytes() == shared_bytes
        })?;
        let mut chosen = Vec::with_capacity((head[3] as usize).min(profile.tiles().len()));
        for _ in 0..head[3] {
            let product = held.take::<PRODUCT_WORDS>()?;
            let element = Element::ALL.get(product[4] as usize).copied()?;
            let contiguous = match product[5] {
                REDUCTION => Contiguous::Reduction,
                OUTPUT => Contiguous::Output,
                _ => return None,
            };
            if product[..4].contains(&0) {
                return None;
            }
            let walked = *profile.tiles().get(product[6] as usize)?;
            if held.take::<TILE_WORDS>()? != identity(walked) {
                return None;
            }
            chosen.push((
                Product::of(
                    product[0],
                    product[1],
                    product[2],
                    product[3],
                    Weight::of(element, contiguous),
                ),
                walked,
            ));
        }
        held.done().then_some(Self { profile, chosen })
    }
}

pub(crate) fn key(device: &AdapterInfo, fingerprint: u64, profiles: &[Profile]) -> String {
    let mut digest = Digest::new();
    digest.word(FORMAT);
    digest.text(&device.name);
    digest.text(&device.id.to_string());
    digest.word(backend(device));
    digest.length(profiles.len());
    for profile in profiles {
        digest.word(profile.workgroup());
        digest.word(profile.registers());
        digest.length(profile.tiles().len());
        for tile in profile.tiles() {
            for word in identity(*tile) {
                digest.word(word);
            }
        }
    }
    format!("plan/{FORMAT}-{fingerprint:016x}-{:016x}", digest.finish())
}

impl Runtime {
    pub fn tuned(&self, graph: &Graph, weights: &Weights) -> Program {
        match self.remembered(graph) {
            Some(choice) => self.compile_chosen(graph, weights, choice.profile(), choice.chosen()),
            None => self.tune(graph, weights),
        }
    }

    pub(crate) fn remembered(&self, graph: &Graph) -> Option<Choice> {
        let profiles = self.profiles();
        let key = key(self.context.adapter_info(), graph.fingerprint(), &profiles);
        let bytes = self.context.device().artifact_cache().load(&key)?;
        Choice::decode(&bytes, &profiles)
    }

    pub(crate) fn remember(&self, graph: &Graph, choice: &Choice) {
        let profiles = self.profiles();
        let key = key(self.context.adapter_info(), graph.fingerprint(), &profiles);
        self.context
            .device()
            .artifact_cache()
            .store(&key, &choice.encode(&profiles));
    }
}

fn menu(profile: Profile, profiles: &[Profile]) -> Option<&[MatmulTile]> {
    profiles
        .iter()
        .find(|held| **held == profile)
        .map(|held| held.tiles())
}

fn identity(tile: MatmulTile) -> [u32; TILE_WORDS] {
    [
        strategy(tile.strategy()),
        tile.rows(),
        tile.columns(),
        tile.depth(),
        tile.threads(),
        tile.thread_rows(),
        tile.thread_columns(),
        tile.stages_a_run() as u32,
    ]
}

fn push(bytes: &mut Vec<u8>, words: &[u32]) {
    for word in words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
}

struct Words<'a> {
    bytes: &'a [u8],
}

impl<'a> Words<'a> {
    fn of(bytes: &'a [u8]) -> Self {
        Self { bytes }
    }

    fn take<const COUNT: usize>(&mut self) -> Option<[u32; COUNT]> {
        let (words, rest) = self.bytes.split_at_checked(COUNT * 4)?;
        self.bytes = rest;
        let mut held = [0u32; COUNT];
        for (slot, word) in held.iter_mut().zip(words.as_chunks::<4>().0) {
            *slot = u32::from_le_bytes(*word);
        }
        Some(held)
    }

    fn long(&mut self) -> Option<u64> {
        let (low, high) = (self.take::<1>()?[0], self.take::<1>()?[0]);
        Some(u64::from(low) | (u64::from(high) << 32))
    }

    fn done(&self) -> bool {
        self.bytes.is_empty()
    }
}

fn contiguous(contiguous: Contiguous) -> u32 {
    match contiguous {
        Contiguous::Reduction => REDUCTION,
        Contiguous::Output => OUTPUT,
    }
}

fn strategy(strategy: MatmulStrategy) -> u32 {
    match strategy {
        MatmulStrategy::Staged => 0,
        MatmulStrategy::Streamed => 1,
        MatmulStrategy::Cooperative => 2,
        MatmulStrategy::Folded => 3,
    }
}

fn backend(device: &AdapterInfo) -> u32 {
    match device.backend {
        neura_gpu::Backend::Vulkan => 0,
        neura_gpu::Backend::Dx12 => 1,
        neura_gpu::Backend::Metal => 2,
    }
}
