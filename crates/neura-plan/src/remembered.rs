use crate::plan::Encoding;
use std::sync::Arc;

pub const DEFAULT_ENCODING_BYTES: u64 = 16 << 20;

pub(crate) struct Remembered {
    entries: Vec<Held>,
    bytes: u64,
    budget: u64,
    clock: u64,
}

struct Held {
    lengths: Vec<u32>,
    encoding: Arc<Encoding>,
    bytes: u64,
    used: u64,
}

impl Remembered {
    pub(crate) fn of(budget: u64) -> Self {
        assert!(
            budget > 0,
            "a plan that remembers no encoding walks every binding afresh",
        );
        Self {
            entries: Vec::new(),
            bytes: 0,
            budget,
            clock: 0,
        }
    }

    pub(crate) fn find(&mut self, lengths: &[u32]) -> Option<Arc<Encoding>> {
        let position = self
            .entries
            .iter()
            .position(|held| held.lengths == lengths)?;
        self.clock += 1;
        let held = &mut self.entries[position];
        held.used = self.clock;
        Some(held.encoding.clone())
    }

    pub(crate) fn insert(&mut self, lengths: &[u32], encoding: Arc<Encoding>) -> Arc<Encoding> {
        if let Some(held) = self.find(lengths) {
            return held;
        }
        let bytes = encoding.bytes() + lengths.len() as u64 * 4;
        if bytes > self.budget {
            return encoding;
        }
        while self.bytes + bytes > self.budget {
            let oldest = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, held)| held.used)
                .map(|(position, _)| position)
                .expect("a store over its budget holds an entry to evict");
            self.bytes -= self.entries.remove(oldest).bytes;
        }
        self.clock += 1;
        self.entries.push(Held {
            lengths: lengths.to_vec(),
            encoding: encoding.clone(),
            bytes,
            used: self.clock,
        });
        self.bytes += bytes;
        encoding
    }

    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    pub(crate) fn bytes(&self) -> u64 {
        self.bytes
    }
}
