use std::path::PathBuf;
use wgpu::Dx12Compiler;

const LIBRARY: &str = "dxcompiler.dll";

pub(crate) fn modern() -> Option<Dx12Compiler> {
    let library = locate()?;
    Some(Dx12Compiler::DynamicDxc {
        dxc_path: library.to_string_lossy().into_owned(),
    })
}

fn locate() -> Option<PathBuf> {
    candidates().into_iter().find(|path| path.is_file())
}

fn candidates() -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Ok(executable) = std::env::current_exe()
        && let Some(directory) = executable.parent()
    {
        candidates.push(directory.join(LIBRARY));
    }
    if let Some(kits) = std::env::var_os("ProgramFiles(x86)") {
        let bin = PathBuf::from(kits)
            .join("Windows Kits")
            .join("10")
            .join("bin");
        let mut sdk = std::fs::read_dir(&bin)
            .into_iter()
            .flatten()
            .flatten()
            .map(|entry| entry.path().join(architecture()).join(LIBRARY))
            .collect::<Vec<_>>();
        sdk.sort();
        sdk.reverse();
        candidates.extend(sdk);
    }
    candidates
}

const fn architecture() -> &'static str {
    if cfg!(target_arch = "aarch64") {
        "arm64"
    } else {
        "x64"
    }
}
