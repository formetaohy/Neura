use std::ffi::OsStr;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

const PREFIX: &str = "neura-weights-";
const SUFFIX: &str = ".spill";
const ATTEMPTS: u32 = 32;

static NEXT: AtomicU64 = AtomicU64::new(0);

pub(crate) struct Spill {
    directory: PathBuf,
}

impl Spill {
    pub(crate) fn of(directory: &Path) -> Self {
        assert!(
            directory.is_dir(),
            "a weight spill keeps its pages in a directory that stands, and {} is no directory",
            directory.display(),
        );
        let spill = Self {
            directory: directory.to_path_buf(),
        };
        spill.sweep();
        spill
    }

    fn sweep(&self) {
        let entries = fs::read_dir(&self.directory).unwrap_or_else(|error| {
            panic!(
                "a weight spill could not scan the directory {}: {error}",
                self.directory.display(),
            )
        });
        for entry in entries {
            let Ok(entry) = entry else {
                continue;
            };
            if !leftover(&entry.file_name()) {
                continue;
            }
            let path = entry.path();
            let Ok(file) = File::open(&path) else {
                continue;
            };
            if file.try_lock().is_err() {
                continue;
            }
            let _ = fs::remove_file(&path);
        }
    }

    pub(crate) fn create(&self, bytes: u64) -> SpillFile {
        for _ in 0..ATTEMPTS {
            let serial = NEXT.fetch_add(1, Ordering::Relaxed);
            let name = format!("{PREFIX}{}-{serial}{SUFFIX}", std::process::id());
            let path = self.directory.join(name);
            let file = match OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .open(&path)
            {
                Ok(file) => file,
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!(
                    "a weight spill of {bytes} bytes could not be created at {}: {error}",
                    path.display(),
                ),
            };
            if file.try_lock().is_err() {
                continue;
            }
            file.set_len(bytes).unwrap_or_else(|error| {
                panic!(
                    "a weight spill of {bytes} bytes could not claim its bytes at {}: {error}",
                    path.display(),
                )
            });
            return SpillFile {
                file,
                path,
                read: 0,
                written: 0,
            };
        }
        panic!(
            "a weight spill of {bytes} bytes tried {ATTEMPTS} names in {} and every one belongs to a spill another run holds or to a leftover this run cannot remove",
            self.directory.display(),
        );
    }
}

fn leftover(name: &OsStr) -> bool {
    let Some(name) = name.to_str() else {
        return false;
    };
    let Some(rest) = name
        .strip_prefix(PREFIX)
        .and_then(|name| name.strip_suffix(SUFFIX))
    else {
        return false;
    };
    let Some((pid, serial)) = rest.split_once('-') else {
        return false;
    };
    pid.parse::<u64>().is_ok() && serial.parse::<u64>().is_ok()
}

pub(crate) struct SpillFile {
    file: File,
    path: PathBuf,
    read: u64,
    written: u64,
}

impl SpillFile {
    pub(crate) fn path(&self) -> PathBuf {
        self.path.clone()
    }

    pub(crate) fn read_bytes(&self) -> u64 {
        self.read
    }

    pub(crate) fn written_bytes(&self) -> u64 {
        self.written
    }

    pub(crate) fn read(&mut self, at: u64, bytes: &mut [u8]) {
        self.file
            .seek(SeekFrom::Start(at))
            .and_then(|_| self.file.read_exact(bytes))
            .unwrap_or_else(|error| {
                panic!(
                    "a weight spill at {} could not be read at {at}: {error}",
                    self.path.display(),
                )
            });
        self.read += bytes.len() as u64;
    }

    pub(crate) fn write(&mut self, at: u64, bytes: &[u8]) {
        self.file
            .seek(SeekFrom::Start(at))
            .and_then(|_| self.file.write_all(bytes))
            .unwrap_or_else(|error| {
                panic!(
                    "a weight spill at {} could not be written at {at}: {error}",
                    self.path.display(),
                )
            });
        self.written += bytes.len() as u64;
    }
}

impl Drop for SpillFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}
