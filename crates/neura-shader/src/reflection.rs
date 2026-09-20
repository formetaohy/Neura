use naga::front::wgsl;
use naga::{AddressSpace, StorageAccess};
use neura_gpu::BindingKind;
use std::fmt::Write as _;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShaderBinding {
    pub group: u32,
    pub binding: u32,
    pub name: String,
    pub kind: BindingKind,
}

pub fn reflect(source: &str) -> Vec<ShaderBinding> {
    let module = wgsl::parse_str(source).unwrap_or_else(|error| {
        panic!(
            "the assembled device program is not valid wgsl:\n{}",
            error.emit_to_string(source),
        )
    });
    let mut bindings = module
        .global_variables
        .iter()
        .filter_map(|(_, variable)| {
            variable.binding.map(|binding| ShaderBinding {
                group: binding.group,
                binding: binding.binding,
                name: variable
                    .name
                    .clone()
                    .unwrap_or_else(|| panic!("every device binding needs a name")),
                kind: kind_of(&variable.space),
            })
        })
        .collect::<Vec<_>>();
    bindings.sort_by_key(|binding| (binding.group, binding.binding));
    bindings
}

fn kind_of(space: &AddressSpace) -> BindingKind {
    let AddressSpace::Storage { access } = *space else {
        panic!("a device binding must be a storage buffer, not {space:?}");
    };
    if access.contains(StorageAccess::STORE) {
        BindingKind::ReadWriteStorage
    } else {
        BindingKind::ReadOnlyStorage
    }
}

pub fn describe(bindings: &[ShaderBinding]) -> String {
    let mut out = String::new();
    for binding in bindings {
        writeln!(
            out,
            "@group({}) @binding({}) {} {:?}",
            binding.group, binding.binding, binding.name, binding.kind,
        )
        .unwrap();
    }
    out
}
