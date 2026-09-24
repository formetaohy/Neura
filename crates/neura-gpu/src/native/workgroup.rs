use spirv::{Decoration, Op, StorageClass};
use std::collections::{HashMap, HashSet};

fn instructions(words: &[u32]) -> impl Iterator<Item = &[u32]> {
    let mut remaining = &words[5..];
    std::iter::from_fn(move || {
        if remaining.is_empty() {
            return None;
        }
        let count = (remaining[0] >> 16) as usize;
        assert!(
            count > 0 && count <= remaining.len(),
            "SPIR-V contains a malformed instruction"
        );
        let (instruction, rest) = remaining.split_at(count);
        remaining = rest;
        Some(instruction)
    })
}

fn members(id: u32, types: &HashMap<u32, Vec<u32>>, found: &mut HashSet<u32>) {
    if !found.insert(id) {
        return;
    }
    let Some(ty) = types.get(&id) else {
        return;
    };
    match ty[0] & 0xffff {
        value if value == Op::TypeArray as u32 || value == Op::TypeRuntimeArray as u32 => {
            members(ty[2], types, found);
        }
        value if value == Op::TypeStruct as u32 => {
            for member in &ty[2..] {
                members(*member, types, found);
            }
        }
        _ => {}
    }
}

pub(crate) fn isolate(words: Vec<u32>) -> Vec<u32> {
    assert!(
        words.len() >= 5 && words[0] == 0x0723_0203,
        "a compute shader is SPIR-V"
    );
    let types = instructions(&words)
        .filter_map(|instruction| {
            let opcode = instruction[0] & 0xffff;
            if opcode == Op::TypeArray as u32
                || opcode == Op::TypeRuntimeArray as u32
                || opcode == Op::TypeStruct as u32
            {
                Some((instruction[1], instruction.to_vec()))
            } else {
                None
            }
        })
        .collect::<HashMap<_, _>>();
    let mut workgroup = HashSet::new();
    let mut external = HashSet::new();
    for instruction in instructions(&words) {
        if instruction[0] & 0xffff == Op::TypePointer as u32 {
            let found = if instruction[2] == StorageClass::Workgroup as u32 {
                &mut workgroup
            } else {
                &mut external
            };
            members(instruction[3], &types, found);
        }
    }
    let mut output = words[..5].to_vec();
    for instruction in instructions(&words) {
        let opcode = instruction[0] & 0xffff;
        let explicit = if opcode == Op::Decorate as u32 {
            instruction[2] == Decoration::ArrayStride as u32
                || instruction[2] == Decoration::Block as u32
                || instruction[2] == Decoration::BufferBlock as u32
        } else if opcode == Op::MemberDecorate as u32 {
            matches!(instruction[3],
                value if value == Decoration::Offset as u32
                    || value == Decoration::MatrixStride as u32
                    || value == Decoration::RowMajor as u32
                    || value == Decoration::ColMajor as u32
            )
        } else {
            false
        };
        if explicit && workgroup.contains(&instruction[1]) {
            assert!(
                !external.contains(&instruction[1]),
                "a Vulkan workgroup cannot share a decorated storage type",
            );
        } else {
            output.extend_from_slice(instruction);
        }
    }
    output
}
