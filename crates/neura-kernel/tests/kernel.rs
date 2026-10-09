use neura_abi::{Element, Kind, RECORDS, WORD_BYTES};
use neura_pointwise::OPS;
use neura_profile::{
    AttentionTile, Budget, CLAIM_BYTES, CooperativeMatrix, Geometry, MatmulStrategy, MatmulTile,
    Profile,
};
use neura_shader::{
    Backend, BindingKind, ComputeProgram, Instruction, ShaderTranslation, Space, Type,
};

const DEVICE: Budget = Budget::of(1024, 48 << 10);

fn profiles() -> Vec<Profile> {
    Profile::derive(DEVICE, None)
}

fn walked(profile: Profile) -> Vec<(u32, MatmulTile)> {
    profile
        .tiles()
        .iter()
        .enumerate()
        .map(|(index, tile)| (index as u32, *tile))
        .collect()
}
use neura_kernel::{
    BINDINGS_OF_AUTHORED, BINDINGS_OF_PAGES, BINDINGS_WITHOUT_HEAP, Banks, Kernel, binding_count,
    bindings,
};
use std::collections::{BTreeSet, HashSet};
use std::sync::OnceLock;

fn all(profile: usize) -> &'static Kernel {
    static PROGRAMS: OnceLock<Vec<Kernel>> = OnceLock::new();
    let programs = PROGRAMS.get_or_init(|| {
        profiles()
            .iter()
            .map(|profile| {
                Kernel::assemble(
                    Kind::ALL,
                    Element::ALL,
                    Geometry::of(
                        profile.workgroup(),
                        profile.shared_bytes(),
                        &walked(*profile),
                        &[],
                    ),
                    false,
                    Banks::SINGLE,
                    false,
                )
            })
            .collect()
    });
    &programs[profile]
}

fn selected(profile: Profile, kinds: &[Kind], elements: &[Element]) -> Kernel {
    Kernel::assemble(
        kinds,
        elements,
        Geometry::of(
            profile.workgroup(),
            profile.shared_bytes(),
            &walked(profile),
            &[],
        ),
        false,
        Banks::SINGLE,
        false,
    )
}

fn functions(program: &ComputeProgram) -> BTreeSet<&str> {
    program
        .module()
        .functions()
        .iter()
        .map(|function| function.name.as_str())
        .collect()
}

fn switch_cases(program: &ComputeProgram, name: &str) -> (Vec<u32>, bool) {
    let function = program
        .module()
        .functions()
        .iter()
        .find(|function| function.name == name)
        .unwrap_or_else(|| panic!("the Rust device function {name} was compiled"));
    function
        .body
        .iter()
        .find_map(|instruction| match instruction {
            Instruction::Switch { cases, default, .. } => Some((
                cases.iter().map(|(value, _)| *value).collect::<Vec<_>>(),
                !default.is_empty(),
            )),
            _ => None,
        })
        .unwrap_or_else(|| panic!("the Rust device function {name} dispatches"))
}

#[test]
fn a_paged_program_binds_the_page_table_it_translates_weights_through() {
    let profile = profiles()[0];
    let geometry = Geometry::of(
        profile.workgroup(),
        profile.shared_bytes(),
        &walked(profile),
        &[],
    );
    let plain = Kernel::assemble(
        &[Kind::Matmul],
        &[Element::Single],
        geometry.clone(),
        false,
        Banks::SINGLE,
        false,
    );
    let paged = Kernel::assemble(
        &[Kind::Matmul],
        &[Element::Single],
        geometry,
        false,
        Banks::SINGLE,
        true,
    );
    assert!(!plain.paged());
    assert!(paged.paged());
    assert_eq!(paged.bindings().len(), plain.bindings().len() + 1);
    let table = neura_kernel::pages(Banks::SINGLE) as usize;
    assert_eq!(paged.bindings()[table].name, "pages");
    assert_eq!(paged.bindings()[table].kind, BindingKind::ReadOnlyStorage);
    for (binding, reflected) in bindings(false, Banks::SINGLE, true)
        .iter()
        .zip(paged.bindings())
    {
        assert_eq!(binding.name, reflected.name);
        assert_eq!(binding.binding, reflected.binding);
        assert_eq!(binding.kind, reflected.kind);
    }
    let paged_source = paged.program();
    let ShaderTranslation::Msl { source, .. } = paged_source.translate(Backend::Metal) else {
        panic!("Metal requires MSL");
    };
    assert!(
        source.contains("d_pages["),
        "the MSL of a paged program reads its page table",
    );
    let ShaderTranslation::Hlsl { source, .. } = paged_source.translate(Backend::Dx12) else {
        panic!("D3D12 requires HLSL");
    };
    assert!(
        source.contains("d_pages["),
        "the HLSL of a paged program reads its page table",
    );
    let plain_source = plain.program();
    let ShaderTranslation::Msl { source, .. } = plain_source.translate(Backend::Metal) else {
        panic!("Metal requires MSL");
    };
    assert!(
        !source.contains("d_pages["),
        "a program of a resident weight store pays no page table",
    );
}

#[test]
fn rust_source_compiles_into_three_native_shader_formats() {
    {
        let program = all(0).program();
        let ShaderTranslation::Spirv(words) = program.translate(Backend::Vulkan) else {
            panic!("Vulkan requires SPIR-V");
        };
        assert_eq!(words[0], 0x0723_0203);
        assert_eq!(words[1], 0x0001_0300, "a device program asks SPIR-V 1.3");
        let ShaderTranslation::Hlsl { source, entry } = program.translate(Backend::Dx12) else {
            panic!("D3D12 requires HLSL");
        };
        assert!(!entry.is_empty());
        assert!(source.contains("register(u2)"));
        assert!(source.contains("register(u4)"));
        assert!(source.contains("register(t5)"));
        let bare = bare_float_literals(&source);
        assert!(
            bare.is_empty(),
            "an HLSL program leaves these float literals to the compiler's default type: {bare:?}"
        );
        let ShaderTranslation::Msl {
            source,
            entry,
            size_bindings,
        } = program.translate(Backend::Metal)
        else {
            panic!("Metal requires MSL");
        };
        assert!(!entry.is_empty());
        assert!(size_bindings.is_empty());
        assert!(source.contains("[[buffer(2)]]"));
        assert!(source.contains("[[buffer(7)]]"));
        let bare = bare_float_literals(&source);
        assert!(
            bare.is_empty(),
            "MSL reads a bare decimal as a 64-bit double, and this program leaves these float literals bare: {bare:?}"
        );
    }
}

#[test]
fn every_task_compiles_for_every_native_backend() {
    {
        let program = selected(profiles()[0], Kind::ALL, Element::ALL).program();
        assert!(matches!(
            program.translate(Backend::Vulkan),
            ShaderTranslation::Spirv(_)
        ));
        assert!(matches!(
            program.translate(Backend::Dx12),
            ShaderTranslation::Hlsl { .. }
        ));
        assert!(matches!(
            program.translate(Backend::Metal),
            ShaderTranslation::Msl { .. }
        ));
    }
}

fn peek_switch(program: &ComputeProgram, name: &str) -> (Vec<u32>, bool) {
    let function = program
        .module()
        .functions()
        .iter()
        .find(|function| function.name == name)
        .unwrap_or_else(|| panic!("a device program declares {name}"));
    function
        .body
        .iter()
        .find_map(|instruction| match instruction {
            Instruction::Switch { cases, default, .. } => Some((
                cases.iter().map(|(value, _)| *value).collect::<Vec<_>>(),
                !default.is_empty(),
            )),
            _ => None,
        })
        .unwrap_or((Vec::new(), false))
}

#[test]
fn a_heap_of_one_bank_splits_no_address() {
    let program = selected(profiles()[0], &[Kind::Matmul], &[Element::Single]).program();
    assert_eq!(peek_switch(&program, "peek").0, Vec::<u32>::new());
    assert_eq!(peek_switch(&program, "poke").0, Vec::<u32>::new());
}

#[test]
fn a_heap_of_many_banks_binds_and_addresses_every_bank() {
    let banks = Banks::of(3, 11);
    let profile = profiles()[0];
    let kernel = Kernel::assemble(
        &[Kind::Matmul],
        &[Element::Single],
        Geometry::of(
            profile.workgroup(),
            profile.shared_bytes(),
            &walked(profile),
            &[],
        ),
        false,
        banks,
        false,
    );
    let expected = bindings(false, banks, false);
    assert_eq!(expected.len(), 2 + banks.count() as usize + 5);
    assert_eq!(kernel.bindings().len(), expected.len());
    for (binding, reflected) in expected.iter().zip(kernel.bindings()) {
        assert_eq!(binding.name, reflected.name);
        assert_eq!(binding.binding, reflected.binding);
        assert_eq!(binding.kind, reflected.kind);
    }
    assert_eq!(kernel.bindings()[2].name, "heap0");
    assert_eq!(kernel.bindings()[4].name, "heap2");
    assert_eq!(kernel.bindings()[5].name, "refusal");
    let program = kernel.program();
    assert_eq!(peek_switch(&program, "peek"), (vec![0, 1, 2], true));
    assert_eq!(peek_switch(&program, "poke"), (vec![0, 1, 2], true));
    let ShaderTranslation::Msl { source, .. } = program.translate(Backend::Metal) else {
        panic!("Metal requires MSL");
    };
    for bank in 0..3 {
        assert!(
            source.contains(&format!("[[buffer({})]]", 2 + bank)),
            "the MSL of a device program binds no bank {bank}",
        );
    }
}

#[test]
fn every_program_binds_the_storage_buffers_its_features_need() {
    for banks in [Banks::SINGLE, Banks::of(2, 10), Banks::of(5, 12)] {
        for authored in [false, true] {
            for paged in [false, true] {
                let declared = bindings(authored, banks, paged);
                let count = binding_count(banks, authored, paged);
                assert_eq!(
                    count,
                    BINDINGS_WITHOUT_HEAP
                        + banks.count()
                        + u32::from(paged) * BINDINGS_OF_PAGES
                        + u32::from(authored) * BINDINGS_OF_AUTHORED,
                    "the program of {banks:?} banks, {authored} authored walks and {paged} page tables declares another storage budget",
                );
                assert_eq!(
                    declared.len() as u32,
                    count,
                    "the program of {banks:?} banks, {authored} authored walks and {paged} page tables binds {} storage buffers",
                    declared.len(),
                );
                assert_eq!(
                    declared.last().expect("a program binds buffers").binding,
                    count - 1,
                    "the storage bindings of a program are dense",
                );
                assert_eq!(
                    declared
                        .iter()
                        .filter(|binding| binding.name == "pages")
                        .count(),
                    usize::from(paged),
                );
                assert_eq!(
                    declared
                        .iter()
                        .filter(|binding| matches!(
                            binding.name.as_str(),
                            "extents" | "measures" | "patches" | "patch_list"
                        ))
                        .count(),
                    usize::from(authored) * BINDINGS_OF_AUTHORED as usize,
                );
            }
        }
    }
}

#[test]
fn every_profile_compiles_the_rust_abi_and_bindings() {
    {
        for (index, profile) in profiles().iter().enumerate() {
            let kernel = all(index);
            let program = kernel.program();
            assert_eq!(kernel.workgroup_size(), profile.workgroup());
            assert_eq!(kernel.geometry().walked(), &walked(*profile)[..]);
            assert_eq!(
                kernel.bindings().len(),
                bindings(false, Banks::SINGLE, false).len()
            );
            assert_eq!(
                program.bindings().len(),
                bindings(false, Banks::SINGLE, false).len()
            );
            for (binding, reflected) in bindings(false, Banks::SINGLE, false)
                .iter()
                .zip(kernel.bindings())
            {
                assert_eq!(binding.name, reflected.name);
                assert_eq!(binding.binding, reflected.binding);
                assert_eq!(binding.kind, reflected.kind);
            }
            assert_eq!(program.reflected()[2].kind, BindingKind::ReadWriteStorage);
            for layout in RECORDS {
                let ty = program
                    .module()
                    .types()
                    .iter()
                    .find(|ty| matches!(ty, Type::Struct { name, .. } if name == layout.name))
                    .expect("all records originate in Rust");
                let Type::Struct { members, span, .. } = ty else {
                    panic!("a host record is a device struct");
                };
                assert_eq!(*span, layout.size);
                for (member, host) in members.iter().zip(layout.fields) {
                    assert_eq!(member.name, host.name);
                    assert_eq!(member.offset, host.offset);
                }
            }
        }
    }
}

#[test]
fn specialization_includes_only_reachable_rust_functions() {
    let kernel = selected(profiles()[0], &[Kind::Fill], &[Element::Single]);
    let program = kernel.program();
    let reachable = functions(&program);
    assert!(reachable.contains("run_fill"));
    for absent in [
        "run_matmul",
        "run_conv2d",
        "run_softmax",
        "run_scatter",
        "workgroup_choice",
        "fetch_half",
    ] {
        assert!(
            !reachable.contains(absent),
            "a fill kernel compiles unused function {absent}"
        );
    }
    let scratch = program
        .module()
        .globals()
        .iter()
        .filter(|global| global.space == Space::WorkGroup)
        .map(|global| global.name.clone())
        .collect::<Vec<_>>();
    assert_eq!(
        scratch,
        vec!["claim".to_owned()],
        "a fill kernel declares no workgroup scratch beside the scheduler's own claim",
    );
    let all = all(0).program();
    assert!(program.spirv().len() < all.spirv().len());
    assert_eq!(kernel.kinds(), &[Kind::Fill]);
}

#[test]
fn the_rust_dispatcher_only_accepts_its_selected_task_kinds() {
    let program = selected(
        profiles()[0],
        &[Kind::Fill, Kind::Binary],
        &[Element::Single],
    )
    .program();
    let (codes, has_default) = switch_cases(&program, "run_task");
    assert_eq!(codes.len(), 2);
    assert!(has_default, "a task dispatcher covers its default");
    assert_eq!(codes[0], Kind::Binary.code());
    assert_eq!(codes[1], Kind::Fill.code());
}

#[test]
fn every_declared_kind_reaches_the_device_body_it_names() {
    let program = selected(profiles()[0], Kind::ALL, Element::ALL).program();
    let reachable = functions(&program);
    for kind in Kind::ALL {
        assert!(
            reachable.contains(kind.entry()),
            "the {} kind names a device body {} that no module installs",
            kind.name(),
            kind.entry(),
        );
    }
    let (dispatched, _) = switch_cases(&program, "run_task");
    let dispatched = dispatched.into_iter().collect::<HashSet<_>>();
    assert_eq!(dispatched.len(), Kind::COUNT as usize);
    for kind in Kind::ALL {
        assert!(
            dispatched.contains(&kind.code()),
            "the task dispatcher carries no arm for the {} kind",
            kind.name(),
        );
    }
}

#[test]
fn the_rust_operation_dispatcher_contains_every_declared_code() {
    let program = selected(
        profiles()[0],
        &[Kind::Binary, Kind::Partial],
        &[Element::Single],
    )
    .program();
    let (defined, _) = switch_cases(&program, "op_apply");
    let defined = defined.into_iter().collect::<HashSet<_>>();
    assert_eq!(defined.len(), OPS.len());
    for op in OPS {
        assert!(defined.contains(&op.code));
    }
}

#[test]
fn every_tensor_element_compiles_only_the_loads_it_reads() {
    let single = selected(profiles()[0], &[Kind::Unary], &[Element::Single]).program();
    let half = selected(
        profiles()[0],
        &[Kind::Unary],
        &[Element::Single, Element::Half],
    )
    .program();
    assert_ne!(single, half);
    assert!(!functions(&single).contains("fetch_half"));
    assert!(functions(&half).contains("fetch_half"));
    assert!(!functions(&half).contains("fetch_bfloat16"));
    let quantized = selected(
        profiles()[0],
        &[Kind::Unary],
        &[Element::Single, Element::Int8],
    )
    .program();
    let reachable = functions(&quantized);
    assert!(reachable.contains("fetch_int8"));
    assert!(!reachable.contains("fetch_half"));
    assert!(!reachable.contains("fetch_bfloat16"));
}

#[test]
fn a_convert_carries_only_the_elements_it_packs() {
    let half = selected(
        profiles()[0],
        &[Kind::Convert],
        &[Element::Single, Element::Half],
    )
    .program();
    let reachable = functions(&half);
    assert!(reachable.contains("run_convert"));
    assert!(reachable.contains("run_convert_half"));
    assert!(!reachable.contains("run_convert_bfloat16"));
    let both = selected(
        profiles()[0],
        &[Kind::Convert],
        &[Element::Half, Element::Bfloat16],
    )
    .program();
    let reachable = functions(&both);
    assert!(reachable.contains("run_convert_half"));
    assert!(reachable.contains("run_convert_bfloat16"));
    assert!(!reachable.contains("run_convert_int8"));
    let quantized = selected(
        profiles()[0],
        &[Kind::Convert],
        &[Element::Single, Element::Int8],
    )
    .program();
    let reachable = functions(&quantized);
    assert!(reachable.contains("run_convert_int8"));
    assert!(!reachable.contains("run_convert_half"));
}

#[test]
fn matrix_specialization_contains_every_tile_in_the_profile() {
    for (index, profile) in profiles().iter().enumerate() {
        let kernel = all(index);
        let program = kernel.program();
        let names = functions(&program);
        for (tile, shape) in profile.tiles().iter().enumerate() {
            assert!(names.contains(format!("run_matmul_{tile}").as_str()));
            if shape.strategy() == neura_profile::MatmulStrategy::Staged {
                assert!(names.contains(format!("matmul_load_{tile}").as_str()));
            } else {
                assert!(
                    !names.contains(format!("matmul_load_{tile}").as_str()),
                    "a streamed product stages nothing through the workgroup it never loads",
                );
            }
        }
    }
}

fn workgroup_bytes(program: &ComputeProgram) -> u64 {
    program
        .module()
        .globals()
        .iter()
        .filter(|global| global.space == Space::WorkGroup)
        .map(|global| u64::from(program.module().size(global.ty)))
        .sum()
}

fn workgroup_global_bytes(program: &ComputeProgram, name: &str) -> u64 {
    program
        .module()
        .globals()
        .iter()
        .filter(|global| global.space == Space::WorkGroup && global.name == name)
        .map(|global| u64::from(program.module().size(global.ty)))
        .sum()
}

#[test]
fn a_cooperative_device_program_declares_the_half_panels_its_tiles_stage() {
    let shape = CooperativeMatrix::new(32, 16, 16, 16);
    let staged = Profile::derive(Budget::of(1024, 48 << 10), Some(shape))
        .into_iter()
        .filter(|profile| {
            profile
                .tiles()
                .iter()
                .any(|tile| matches!(tile.strategy(), MatmulStrategy::Cooperative))
        })
        .collect::<Vec<_>>();
    assert!(
        !staged.is_empty(),
        "a device of 16 by 16 fragments derives no profile a cooperative tile fits",
    );
    for profile in staged {
        let geometry = Geometry::of(
            profile.workgroup(),
            profile.shared_bytes(),
            &walked(profile),
            &[],
        );
        let kernel = Kernel::assemble(
            Kind::ALL,
            Element::ALL,
            geometry.clone(),
            false,
            Banks::SINGLE,
            false,
        );
        let program = kernel.program();
        assert_eq!(
            workgroup_bytes(&program),
            geometry.workgroup_bytes(Kind::ALL),
            "{profile:?} declares workgroup memory its geometry does not account for",
        );
        let panels = geometry
            .walked()
            .iter()
            .filter(|(_, tile)| matches!(tile.strategy(), MatmulStrategy::Cooperative))
            .map(|(_, tile)| tile.half_panels())
            .max()
            .expect("a cooperative tile stages its panels on the half grid");
        assert_eq!(
            geometry.half_panels(),
            panels,
            "a geometry sizes its half grid by another tile than the widest one it carries",
        );
        assert_eq!(
            workgroup_global_bytes(&program, "scratch_half"),
            2 * panels,
            "a device program stages {panels} half panels in workgroup memory it declares for another count",
        );
    }
}

#[test]
fn a_device_program_carries_only_the_tiles_its_plan_walks() {
    let profile = *profiles().last().expect("a profile");
    assert!(profile.tiles().len() > 2, "a profile carries a menu");
    let menu = walked(profile);
    let walked = &menu[..2];
    let geometry = Geometry::of(profile.workgroup(), profile.shared_bytes(), walked, &[]);
    let kernel = Kernel::assemble(
        Kind::ALL,
        Element::ALL,
        geometry.clone(),
        false,
        Banks::SINGLE,
        false,
    );
    let program = kernel.program();
    let names = functions(&program);
    for (index, _) in walked {
        assert!(names.contains(format!("run_matmul_{index}").as_str()));
    }
    for (index, _) in &menu[2..] {
        assert!(
            !names.contains(format!("run_matmul_{index}").as_str()),
            "a device program compiles the body of tile {index} its plan never walks",
        );
    }
    let (cases, _) = switch_cases(&program, "run_matmul");
    assert_eq!(
        cases,
        walked.iter().map(|(index, _)| *index).collect::<Vec<_>>()
    );
    let whole = all(profiles().len() - 1).program();
    assert!(
        program.spirv().len() < whole.spirv().len(),
        "a plan that walks two tiles of {profile:?} compiles the module of the whole menu",
    );
    assert!(
        workgroup_bytes(&program) < workgroup_bytes(&whole),
        "a plan that walks two tiles of {profile:?} declares the pool of the whole menu",
    );
    assert_eq!(
        workgroup_bytes(&program),
        geometry.workgroup_bytes(Kind::ALL),
        "a device program declares the pool of the tiles its plan walks",
    );
}

#[test]
fn workgroup_allocation_fits_the_advertised_profile() {
    for (index, profile) in profiles().iter().enumerate() {
        let program = all(index).program();
        let used = workgroup_bytes(&program);
        assert!(used > 0);
        assert!(
            used <= profile.shared_bytes(),
            "{profile:?} reserves {used} workgroup bytes"
        );
    }
}

#[test]
fn a_device_program_shares_one_scratch_pool_between_its_bodies() {
    let profile = *profiles().last().expect("a profile");
    let attention = [AttentionTile::new(4, 8)];
    let geometry = Geometry::of(
        profile.workgroup(),
        profile.shared_bytes(),
        &walked(profile),
        &attention,
    );
    let kernel = Kernel::assemble(
        Kind::ALL,
        Element::ALL,
        geometry.clone(),
        false,
        Banks::SINGLE,
        false,
    );
    let used = workgroup_bytes(&kernel.program());
    assert_eq!(
        used,
        geometry.scratch_bytes(Kind::ALL) + CLAIM_BYTES,
        "a device program declares the pool its widest body stages beside the word its workgroups claim with",
    );
    let partitioned = geometry.staging_bytes()
        + 2 * u64::from(attention[0].stage_words()) * WORD_BYTES
        + 2 * u64::from(profile.workgroup()) * WORD_BYTES;
    assert!(
        used < partitioned,
        "a device program lets every body stage from one pool instead of partitioning {partitioned} bytes among them",
    );
}

#[test]
fn the_same_rust_specialization_produces_the_same_device_program() {
    let first = selected(profiles()[0], &[Kind::Fill], &[Element::Single]).program();
    let second = selected(profiles()[0], &[Kind::Fill], &[Element::Single]).program();
    assert_eq!(first, second);
    assert_eq!(first.spirv(), second.spirv());
}

#[test]
fn attention_specialization_contains_every_tile_of_its_geometry() {
    let profile = *profiles().last().expect("a profile");
    let attention = [AttentionTile::new(4, 8), AttentionTile::new(2, 16)];
    let kernel = Kernel::assemble(
        &[
            Kind::Attention,
            Kind::AttentionQueryGrad,
            Kind::AttentionKeyGrad,
            Kind::AttentionValueGrad,
        ],
        &[Element::Single],
        Geometry::of(profile.workgroup(), profile.shared_bytes(), &[], &attention),
        false,
        Banks::SINGLE,
        false,
    );
    let program = kernel.program();
    let names = functions(&program);
    for (index, tile) in attention.iter().enumerate() {
        for name in [
            format!("run_attention_{index}"),
            format!("run_attention_query_grad_{index}"),
            format!("run_attention_key_grad_{index}"),
            format!("run_attention_value_grad_{index}"),
            format!("stage_attention_{index}"),
        ] {
            assert!(names.contains(name.as_str()), "{name} is missing");
        }
        assert_eq!(
            kernel.geometry().attention_tile(index as u32),
            *tile,
            "a device program carries the tile its geometry names",
        );
    }
    assert_eq!(
        kernel.geometry().scratch_bytes(&[Kind::Attention]),
        2 * u64::from(
            attention
                .iter()
                .map(|tile| tile.stage_words())
                .max()
                .expect("an attention stage"),
        ) * WORD_BYTES,
        "a device program stages the widest attention tile it carries",
    );
}

fn bare_float_literals(text: &str) -> Vec<String> {
    let bytes = text.as_bytes();
    let word = |byte: u8| byte.is_ascii_alphanumeric() || byte == b'_';
    let mut bare = Vec::new();
    let mut at = 0;
    while at < bytes.len() {
        let starts = bytes[at].is_ascii_digit() && (at == 0 || !word(bytes[at - 1]))
            || bytes[at] == b'-'
                && at + 1 < bytes.len()
                && bytes[at + 1].is_ascii_digit()
                && (at == 0 || !word(bytes[at - 1]));
        if !starts {
            at += 1;
            continue;
        }
        let mut cursor = at + usize::from(bytes[at] == b'-');
        while cursor < bytes.len() && bytes[cursor].is_ascii_digit() {
            cursor += 1;
        }
        let mut fractional = false;
        if cursor < bytes.len() && bytes[cursor] == b'.' {
            fractional = true;
            cursor += 1;
            while cursor < bytes.len() && bytes[cursor].is_ascii_digit() {
                cursor += 1;
            }
        }
        if cursor < bytes.len() && matches!(bytes[cursor], b'e' | b'E') {
            let mut exponent = cursor + 1;
            if exponent < bytes.len() && matches!(bytes[exponent], b'+' | b'-') {
                exponent += 1;
            }
            if exponent < bytes.len() && bytes[exponent].is_ascii_digit() {
                fractional = true;
                cursor = exponent;
                while cursor < bytes.len() && bytes[cursor].is_ascii_digit() {
                    cursor += 1;
                }
            }
        }
        if !fractional {
            at = cursor;
            continue;
        }
        if !matches!(bytes.get(cursor), Some(b'f' | b'F')) {
            bare.push(text[at..cursor].to_owned());
        }
        at = cursor + 1;
    }
    bare
}
