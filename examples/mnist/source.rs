use flate2::read::GzDecoder;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;

const MIRROR: &str = "https://ossci-datasets.s3.amazonaws.com/mnist";
const IMAGES_MAGIC: u32 = 2_051;
const LABELS_MAGIC: u32 = 2_049;

#[derive(Clone, Copy)]
pub(crate) enum Split {
    Train,
    Test,
}

impl Split {
    fn name(self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Test => "t10k",
        }
    }

    fn digits(self) -> u32 {
        match self {
            Self::Train => 60_000,
            Self::Test => 10_000,
        }
    }
}

pub(crate) fn directory() -> PathBuf {
    std::env::var_os("NEURA_MNIST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("target")
                .join("mnist")
        })
}

pub(crate) fn images(directory: &Path, split: Split) -> (Vec<u8>, u32, u32) {
    let bytes = read(directory, &format!("{}-images-idx3-ubyte.gz", split.name()));
    let (dims, pixels) = parse(&bytes, IMAGES_MAGIC, 3);
    assert_eq!(
        [dims[0], dims[1], dims[2]],
        [split.digits(), 28, 28],
        "the image archive of the {:?} split holds {:?} digits where MNIST holds {} of 28 by 28",
        split.name(),
        dims,
        split.digits(),
    );
    (pixels, dims[1], dims[2])
}

pub(crate) fn labels(directory: &Path, split: Split) -> Vec<u8> {
    let bytes = read(directory, &format!("{}-labels-idx1-ubyte.gz", split.name()));
    let (dims, digits) = parse(&bytes, LABELS_MAGIC, 1);
    assert_eq!(
        dims,
        [split.digits()],
        "the label archive of the {:?} split holds {:?} labels where MNIST holds {}",
        split.name(),
        dims,
        split.digits(),
    );
    digits
}

fn read(directory: &Path, name: &str) -> Vec<u8> {
    let path = directory.join(name);
    if !path.exists() {
        fetch(directory, name);
    }
    let compressed = fs::read(&path).unwrap_or_else(|error| {
        panic!(
            "the MNIST archive {} of {} is unreadable: {error}",
            path.display(),
            directory.display(),
        )
    });
    let mut inflate = GzDecoder::new(compressed.as_slice());
    let mut bytes = Vec::new();
    inflate.read_to_end(&mut bytes).unwrap_or_else(|error| {
        panic!(
            "the MNIST archive {} of {} is not a gzip stream: {error}",
            path.display(),
            directory.display(),
        )
    });
    bytes
}

fn fetch(directory: &Path, name: &str) {
    let url = format!("{MIRROR}/{name}");
    let destination = directory.join(name);
    fs::create_dir_all(directory).unwrap_or_else(|error| {
        panic!(
            "the MNIST cache directory {} is not writable: {error}",
            directory.display(),
        )
    });
    println!("fetching {url} into {}", destination.display());
    for (tool, arguments) in attempts(&url, &destination) {
        if run(tool, &arguments) && destination.metadata().is_ok_and(|file| file.len() > 0) {
            return;
        }
        let _ = fs::remove_file(&destination);
    }
    panic!(
        "no curl, wget, or PowerShell fetched {url}: download the four archives of MNIST into {} yourself, or point NEURA_MNIST_DIR at a directory that holds them",
        directory.display(),
    );
}

fn attempts(url: &str, destination: &Path) -> Vec<(&'static str, Vec<String>)> {
    let path = destination.to_string_lossy().to_string();
    let request = format!("Invoke-WebRequest -Uri '{url}' -OutFile '{path}'");
    vec![
        (
            "curl",
            vec![
                "-fsSL".into(),
                "--retry".into(),
                "2".into(),
                "-o".into(),
                path.clone(),
                url.into(),
            ],
        ),
        ("wget", vec!["-q".into(), "-O".into(), path, url.into()]),
        (
            "pwsh",
            vec!["-NoProfile".into(), "-Command".into(), request.clone()],
        ),
        (
            "powershell",
            vec!["-NoProfile".into(), "-Command".into(), request],
        ),
    ]
}

fn run(tool: &str, arguments: &[String]) -> bool {
    Command::new(tool)
        .args(arguments)
        .status()
        .is_ok_and(|status| status.success())
}

fn parse(bytes: &[u8], magic: u32, rank: usize) -> (Vec<u32>, Vec<u8>) {
    let mut cursor = 0usize;
    let declared = take(bytes, &mut cursor);
    assert_eq!(
        declared, magic,
        "an IDX file of magic {magic} opens with the magic {declared}",
    );
    let dims = (0..rank)
        .map(|_| take(bytes, &mut cursor))
        .collect::<Vec<_>>();
    let elements = dims.iter().map(|dim| u64::from(*dim)).product::<u64>();
    let payload = &bytes[cursor..];
    assert_eq!(
        payload.len() as u64,
        elements,
        "an IDX header of {dims:?} holds {elements} numbers, and the file holds {}",
        payload.len(),
    );
    (dims, payload.to_vec())
}

fn take(bytes: &[u8], cursor: &mut usize) -> u32 {
    let word = bytes
        .get(*cursor..*cursor + 4)
        .unwrap_or_else(|| panic!("an IDX file ends after {} bytes", *cursor));
    *cursor += 4;
    u32::from_be_bytes(word.try_into().expect("four bytes of an IDX word"))
}
