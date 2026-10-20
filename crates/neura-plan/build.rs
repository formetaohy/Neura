use std::env;
use std::fs;
use std::path::{Component, Path, PathBuf};

const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const PRIME: u64 = 0x0000_0100_0000_01b3;

fn main() {
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("a manifest"));
    let mut files = Vec::new();
    collect(&manifest.join("src"), &mut files);
    files.push(manifest.join("Cargo.toml"));
    files.push(manifest.join("build.rs"));
    let mut sources = files
        .into_iter()
        .map(|file| {
            println!("cargo:rerun-if-changed={}", file.display());
            let name = relative(&manifest, &file);
            (name, file)
        })
        .collect::<Vec<_>>();
    sources.sort_by(|left, right| left.0.cmp(&right.0));
    let mut digest = OFFSET;
    for (name, file) in sources {
        digest = feed(digest, name.as_bytes());
        digest = feed(digest, &[0]);
        let source =
            fs::read(&file).unwrap_or_else(|error| panic!("reading {}: {error}", file.display()));
        digest = feed_source(digest, &source);
        digest = feed(digest, &[0]);
    }
    println!("cargo:rerun-if-changed=build.rs");
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("an output directory"));
    fs::write(
        out.join("plan_build.rs"),
        format!("pub const PLAN_BUILD: u64 = 0x{digest:016x};\n"),
    )
    .expect("a generated identity file");
}

fn collect(directory: &Path, files: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(directory)
        .unwrap_or_else(|error| panic!("reading {}: {error}", directory.display()))
    {
        let entry = entry.expect("a planner source entry");
        let path = entry.path();
        if path.is_dir() {
            collect(&path, files);
        } else if path.extension().is_some_and(|held| held == "rs") {
            files.push(path);
        }
    }
}

fn relative(manifest: &Path, file: &Path) -> String {
    file.strip_prefix(manifest)
        .expect("a planner source lies under its manifest")
        .components()
        .filter_map(|component| match component {
            Component::Normal(name) => Some(name.to_string_lossy().into_owned()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("/")
}

fn feed_source(mut digest: u64, bytes: &[u8]) -> u64 {
    let mut at = 0;
    while at < bytes.len() {
        if bytes[at] == b'\r' && bytes.get(at + 1) == Some(&b'\n') {
            at += 1;
            continue;
        }
        digest = feed(digest, &bytes[at..at + 1]);
        at += 1;
    }
    digest
}

fn feed(mut digest: u64, bytes: &[u8]) -> u64 {
    for byte in bytes {
        digest ^= u64::from(*byte);
        digest = digest.wrapping_mul(PRIME);
    }
    digest
}
