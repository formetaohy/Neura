use crate::heap::{Allocation, Heap};
use crate::pool::{Pool, Recycled};
use neura_abi::{NO_PAGE, PAGE_WORDS, WORD_BYTES, pages_of};
use neura_gpu::{BufferUsages, GpuBuffer, GpuContext, Queue, Submission, SubmissionIndex};
use std::sync::{Arc, Mutex, PoisonError};

pub(crate) struct WeightStore {
    device: neura_gpu::Device,
    upload: std::sync::OnceLock<GpuBuffer>,
    pool: Arc<Pool>,
    store: Allocation,
    table: Option<GpuBuffer>,
    words: u64,
    pages: u32,
    slots: u32,
    mirror: Option<Mutex<Mirror>>,
}

struct Mirror {
    bytes: Vec<u8>,
    page_of: Vec<u32>,
    slot_of: Vec<u32>,
    dirty: Vec<bool>,
    used: Vec<u64>,
    clock: u64,
    chunks: Vec<Option<Recycled>>,
    spare: Vec<usize>,
    writebacks: Vec<Writeback>,
    captured: u64,
    transfers: u64,
}

struct Writeback {
    page: u32,
    chunk: usize,
    submission: Option<SubmissionIndex>,
}

#[derive(Clone, Copy)]
enum Origin {
    Chunk(usize),
    Slot(u32),
}

pub(crate) const fn page_bytes() -> u64 {
    PAGE_WORDS * WORD_BYTES
}

const WRITEBACK_CHUNKS: usize = 64;
const CAPTURE_PAGES: usize = 64;
const UPLOAD_PAGES: usize = 64;
const WRITEBACK_USAGE: BufferUsages = BufferUsages::COPY_SRC.union(BufferUsages::COPY_DST);
const READBACK_USAGE: BufferUsages = BufferUsages::COPY_DST.union(BufferUsages::MAP_READ);

impl Mirror {
    fn new(pages: u32, slots: u32) -> Self {
        Self {
            bytes: vec![0; pages as usize * page_bytes() as usize],
            page_of: vec![NO_PAGE; slots as usize],
            slot_of: vec![NO_PAGE; pages as usize],
            dirty: vec![false; slots as usize],
            used: vec![0; slots as usize],
            clock: 0,
            chunks: Vec::new(),
            spare: Vec::new(),
            writebacks: Vec::new(),
            captured: 0,
            transfers: 0,
        }
    }

    fn resident(&self, page: u32) -> Option<u32> {
        match self.slot_of[page as usize] {
            NO_PAGE => None,
            slot => Some(slot),
        }
    }

    fn touch(&mut self, slot: u32) {
        self.clock += 1;
        self.used[slot as usize] = self.clock;
    }

    fn vacuum(&mut self) {
        for slot in 0..self.page_of.len() {
            let held = self.page_of[slot];
            if held != NO_PAGE {
                self.slot_of[held as usize] = NO_PAGE;
            }
            self.page_of[slot] = NO_PAGE;
            self.dirty[slot] = false;
        }
    }

    fn table(&self) -> Vec<u32> {
        self.slot_of.clone()
    }
}

pub(crate) fn paged_for(words: u64, resident: Option<u64>) -> bool {
    let slots = match resident {
        None => return false,
        Some(bytes) => {
            assert!(
                bytes > 0 && bytes.is_multiple_of(WORD_BYTES),
                "a resident weight budget of {bytes} bytes holds no whole word",
            );
            u64::from(u32::try_from(bytes / page_bytes()).unwrap_or(u32::MAX))
        }
    };
    slots < pages_of(words)
}

impl WeightStore {
    pub(crate) fn new(
        context: &GpuContext,
        pool: &Arc<Pool>,
        heap: &Arc<Heap>,
        words: u64,
        resident: Option<u64>,
    ) -> Arc<Self> {
        let pages = u32::try_from(pages_of(words)).unwrap_or_else(|_| {
            panic!("a weight store of {words} words spans more pages than a device addresses")
        });
        let wanted = match resident {
            None => pages,
            Some(bytes) => {
                assert!(
                    bytes > 0 && bytes.is_multiple_of(WORD_BYTES),
                    "a resident weight budget of {bytes} bytes holds no whole word",
                );
                u32::try_from(bytes / page_bytes()).unwrap_or(u32::MAX)
            }
        };
        let slots = wanted.min(pages);
        let paged = slots < pages;
        let store = heap.allocate(if paged {
            u64::from(slots) * PAGE_WORDS
        } else {
            words
        });
        let table = paged.then(|| {
            GpuBuffer::new(
                context.device(),
                "neura weight pages",
                u64::from(pages).max(1) * WORD_BYTES,
                BufferUsages::STORAGE | BufferUsages::COPY_DST,
            )
        });
        let mirror = paged.then(|| Mutex::new(Mirror::new(pages, slots)));
        let store = Arc::new(Self {
            device: context.device().clone(),
            upload: std::sync::OnceLock::new(),
            pool: pool.clone(),
            store,
            table,
            words,
            pages,
            slots,
            mirror,
        });
        if let Some(table) = &store.table {
            let empty = vec![NO_PAGE; pages as usize];
            table.write_at(context.queue(), 0, bytemuck::cast_slice(&empty));
        }
        store
    }

    pub(crate) fn words(&self) -> u64 {
        self.words
    }

    pub(crate) fn bytes(&self) -> u64 {
        self.words * WORD_BYTES
    }

    pub(crate) fn pages(&self) -> u32 {
        self.pages
    }

    pub(crate) fn resident_pages(&self) -> u32 {
        self.slots
    }

    pub(crate) fn resident_bytes(&self) -> u64 {
        self.store.bytes()
    }

    pub(crate) fn paged(&self) -> bool {
        self.mirror.is_some()
    }

    pub(crate) fn buffer(&self) -> &GpuBuffer {
        self.store.buffer()
    }

    pub(crate) fn offset(&self) -> u64 {
        self.store.offset()
    }

    pub(crate) fn table(&self) -> Option<&GpuBuffer> {
        self.table.as_ref()
    }

    pub(crate) fn lives_on(&self, heap: &Arc<Heap>) -> bool {
        self.store.lives_on(heap)
    }

    pub(crate) fn readback_pages(&self) -> u64 {
        self.mirror.as_ref().map_or(0, |mirror| {
            mirror
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .captured
        })
    }

    pub(crate) fn readback_transfers(&self) -> u64 {
        self.mirror.as_ref().map_or(0, |mirror| {
            mirror
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .transfers
        })
    }

    fn bounds(&self, word: u64, bytes: u64) {
        assert!(
            bytes.is_multiple_of(WORD_BYTES)
                && word
                    .checked_add(bytes / WORD_BYTES)
                    .is_some_and(|end| end <= self.words),
            "a weight range of {bytes} bytes at word {word} outruns the {} words of the store",
            self.words,
        );
    }

    pub(crate) fn put(&self, queue: &Queue, word: u64, bytes: &[u8]) {
        self.bounds(word, bytes.len() as u64);
        let Some(mirror) = &self.mirror else {
            self.store
                .buffer()
                .write_at(queue, self.store.offset() + word * WORD_BYTES, bytes);
            return;
        };
        let mut mirror = mirror.lock().unwrap_or_else(PoisonError::into_inner);
        let base = word * WORD_BYTES;
        let from = (base / page_bytes()) as u32;
        let to =
            u32::try_from((base + bytes.len() as u64).div_ceil(page_bytes())).unwrap_or(u32::MAX);
        self.materialize_writebacks(&mut mirror, queue, |page| page >= from && page < to);
        let mut at = 0u64;
        while at < bytes.len() as u64 {
            let byte = base + at;
            let page = (byte / page_bytes()) as u32;
            let within = byte % page_bytes();
            let chunk = (bytes.len() as u64 - at).min(page_bytes() - within);
            match mirror.resident(page) {
                Some(slot) => {
                    self.store.buffer().write_at(
                        queue,
                        self.store.offset() + u64::from(slot) * page_bytes() + within,
                        &bytes[at as usize..(at + chunk) as usize],
                    );
                    mirror.dirty[slot as usize] = true;
                    mirror.touch(slot);
                }
                None => {
                    let start = page as usize * page_bytes() as usize + within as usize;
                    mirror.bytes[start..start + chunk as usize]
                        .copy_from_slice(&bytes[at as usize..(at + chunk) as usize]);
                }
            }
            at += chunk;
        }
    }

    pub(crate) fn get(&self, queue: &Queue, word: u64, bytes: u64) -> Vec<u8> {
        self.bounds(word, bytes);
        if bytes == 0 {
            return Vec::new();
        }
        let first = word * WORD_BYTES;
        let Some(mirror) = &self.mirror else {
            return self.read_device(queue, self.store.offset() + first, bytes);
        };
        let mut mirror = mirror.lock().unwrap_or_else(PoisonError::into_inner);
        self.materialize_range(&mut mirror, queue, first, first + bytes);
        self.flush_range(&mut mirror, queue, first, first + bytes);
        mirror.bytes[first as usize..(first + bytes) as usize].to_vec()
    }

    pub(crate) fn ensure(&self, queue: &Queue, pages: &[u32]) -> Vec<SubmissionIndex> {
        let Some(mirror) = &self.mirror else {
            return Vec::new();
        };
        assert!(
            pages.len() <= self.slots as usize,
            "the resident weights of {} pages cannot hold the {} pages one step of this plan walks; raise the resident weight budget, or bind a graph whose waves walk no more pages than it holds",
            self.slots,
            pages.len(),
        );
        let mut mirror = mirror.lock().unwrap_or_else(PoisonError::into_inner);
        let mut transfer: Option<Submission> = None;
        let mut submitted: Vec<SubmissionIndex> = Vec::new();
        let mut uploads: Vec<(u32, u32)> = Vec::new();
        let mut changed = false;
        for page in pages {
            if let Some(slot) = mirror.resident(*page) {
                mirror.touch(slot);
                continue;
            }
            changed = true;
            let held = mirror
                .writebacks
                .iter()
                .position(|writeback| writeback.page == *page)
                .map(|position| mirror.writebacks.remove(position));
            let slot = self.vacate(
                &mut mirror,
                queue,
                &mut transfer,
                &mut submitted,
                pages,
                *page,
            );
            match held {
                Some(writeback) => {
                    let chunk = mirror.chunks[writeback.chunk]
                        .as_ref()
                        .expect("a held page keeps the chunk its bytes were written back to");
                    let submission = transfer.get_or_insert_with(|| {
                        Submission::new(queue.device(), "neura weight writeback")
                    });
                    submission.copy(
                        chunk.buffer(),
                        0,
                        self.store.buffer(),
                        self.store.offset() + u64::from(slot) * page_bytes(),
                        page_bytes(),
                    );
                    mirror.chunks[writeback.chunk] = None;
                    mirror.spare.push(writeback.chunk);
                    mirror.dirty[slot as usize] = true;
                }
                None => {
                    uploads.push((*page, slot));
                    mirror.dirty[slot as usize] = false;
                }
            }
            mirror.page_of[slot as usize] = *page;
            mirror.slot_of[*page as usize] = slot;
            mirror.touch(slot);
        }
        let mut batches = uploads.chunks(self.batch_pages());
        loop {
            let batch = batches.next();
            if batch.is_none() && transfer.is_none() {
                break;
            }
            let carried = transfer.is_some();
            let mut submission = transfer
                .take()
                .unwrap_or_else(|| Submission::new(queue.device(), "neura weight upload"));
            if let Some(batch) = batch {
                let mut bytes = Vec::with_capacity(batch.len() * page_bytes() as usize);
                for (page, _) in batch {
                    let start = *page as usize * page_bytes() as usize;
                    bytes.extend_from_slice(&mirror.bytes[start..start + page_bytes() as usize]);
                }
                self.upload_buffer().write_at(queue, 0, &bytes);
                for (at, (_, slot)) in batch.iter().enumerate() {
                    submission.copy(
                        self.upload_buffer(),
                        at as u64 * page_bytes(),
                        self.store.buffer(),
                        self.store.offset() + u64::from(*slot) * page_bytes(),
                        page_bytes(),
                    );
                }
            }
            let index = submission.submit(queue);
            if carried {
                for writeback in &mut mirror.writebacks {
                    writeback.submission.get_or_insert(index);
                }
            }
            submitted.push(index);
        }
        if changed {
            self.reload(queue, &mut mirror);
        }
        submitted
    }

    fn batch_pages(&self) -> usize {
        (self.slots as usize).min(UPLOAD_PAGES)
    }

    fn upload_buffer(&self) -> &GpuBuffer {
        self.upload.get_or_init(|| {
            GpuBuffer::new(
                &self.device,
                "neura weight upload",
                self.batch_pages() as u64 * page_bytes(),
                WRITEBACK_USAGE,
            )
        })
    }

    fn vacate(
        &self,
        mirror: &mut Mirror,
        queue: &Queue,
        transfer: &mut Option<Submission>,
        submitted: &mut Vec<SubmissionIndex>,
        pages: &[u32],
        page: u32,
    ) -> u32 {
        let free = (0..self.slots).find(|slot| mirror.page_of[*slot as usize] == NO_PAGE);
        let victim = free
            .or_else(|| {
                (0..self.slots)
                    .filter(|slot| {
                        let kept = mirror.page_of[*slot as usize];
                        kept != NO_PAGE && kept != page && pages.binary_search(&kept).is_err()
                    })
                    .min_by_key(|slot| mirror.used[*slot as usize])
            })
            .unwrap_or_else(|| {
                panic!(
                    "the resident weights of {} pages cannot hold the {} pages one step of this plan walks; raise the resident weight budget, or bind a graph whose waves walk no more pages than it holds",
                    self.slots,
                    pages.len(),
                )
            });
        let kept = mirror.page_of[victim as usize];
        if kept != NO_PAGE {
            if mirror.dirty[victim as usize] {
                self.write_back(mirror, queue, transfer, submitted, victim, kept);
            }
            mirror.slot_of[kept as usize] = NO_PAGE;
        }
        mirror.page_of[victim as usize] = NO_PAGE;
        mirror.dirty[victim as usize] = false;
        victim
    }

    fn write_back(
        &self,
        mirror: &mut Mirror,
        queue: &Queue,
        transfer: &mut Option<Submission>,
        submitted: &mut Vec<SubmissionIndex>,
        victim: u32,
        page: u32,
    ) {
        let index = match self.spare_chunk(mirror) {
            Some(index) => index,
            None => {
                self.flush_transfer(mirror, queue, transfer, submitted);
                submitted.extend(self.materialize_writebacks(mirror, queue, |_| true));
                self.spare_chunk(mirror)
                    .expect("a captured batch releases the chunks it held")
            }
        };
        let chunk = Recycled::claim(
            &self.pool,
            "neura weight page",
            page_bytes(),
            WRITEBACK_USAGE,
        );
        let submission = transfer
            .get_or_insert_with(|| Submission::new(queue.device(), "neura weight writeback"));
        submission.copy(
            self.store.buffer(),
            self.store.offset() + u64::from(victim) * page_bytes(),
            chunk.buffer(),
            0,
            page_bytes(),
        );
        mirror.chunks[index] = Some(chunk);
        mirror.writebacks.push(Writeback {
            page,
            chunk: index,
            submission: None,
        });
    }

    fn spare_chunk(&self, mirror: &mut Mirror) -> Option<usize> {
        if let Some(index) = mirror.spare.pop() {
            return Some(index);
        }
        if mirror.chunks.len() < WRITEBACK_CHUNKS {
            mirror.chunks.push(None);
            return Some(mirror.chunks.len() - 1);
        }
        None
    }

    fn flush_transfer(
        &self,
        mirror: &mut Mirror,
        queue: &Queue,
        transfer: &mut Option<Submission>,
        submitted: &mut Vec<SubmissionIndex>,
    ) {
        if let Some(submission) = transfer.take() {
            let index = submission.submit(queue);
            for writeback in &mut mirror.writebacks {
                writeback.submission.get_or_insert(index);
            }
            submitted.push(index);
        }
    }

    fn materialize_writebacks(
        &self,
        mirror: &mut Mirror,
        queue: &Queue,
        select: impl Fn(u32) -> bool,
    ) -> Vec<SubmissionIndex> {
        let mut sources = Vec::new();
        let mut kept = Vec::new();
        for writeback in mirror.writebacks.drain(..) {
            if select(writeback.page) {
                assert!(
                    writeback.submission.is_some(),
                    "a page the host reads back is a page the queue already carries",
                );
                sources.push((writeback.page, Origin::Chunk(writeback.chunk)));
            } else {
                kept.push(writeback);
            }
        }
        mirror.writebacks = kept;
        let chunks = sources
            .iter()
            .map(|(_, origin)| match origin {
                Origin::Chunk(chunk) => *chunk,
                Origin::Slot(_) => unreachable!("a writeback holds a chunk"),
            })
            .collect::<Vec<usize>>();
        let submitted = self.capture(mirror, queue, &sources);
        for chunk in chunks {
            mirror.chunks[chunk] = None;
            mirror.spare.push(chunk);
        }
        submitted
    }

    fn capture(
        &self,
        mirror: &mut Mirror,
        queue: &Queue,
        sources: &[(u32, Origin)],
    ) -> Vec<SubmissionIndex> {
        let mut submitted = Vec::new();
        for batch in sources.chunks(CAPTURE_PAGES) {
            let bytes = batch.len() as u64 * page_bytes();
            let staging =
                Recycled::claim(&self.pool, "neura weight readback", bytes, READBACK_USAGE);
            let mut submission = Submission::new(queue.device(), "neura weight readback");
            for (at, (_, origin)) in batch.iter().enumerate() {
                let (buffer, offset) = match origin {
                    Origin::Chunk(chunk) => (
                        mirror.chunks[*chunk]
                            .as_ref()
                            .expect("a writeback holds the chunk it was handed")
                            .buffer(),
                        0,
                    ),
                    Origin::Slot(slot) => (
                        self.store.buffer(),
                        self.store.offset() + u64::from(*slot) * page_bytes(),
                    ),
                };
                submission.copy(
                    buffer,
                    offset,
                    staging.buffer(),
                    at as u64 * page_bytes(),
                    page_bytes(),
                );
            }
            let index = submission.submit(queue);
            let read = staging.buffer().read(queue, index, bytes);
            for (at, (page, _)) in batch.iter().enumerate() {
                let start = *page as usize * page_bytes() as usize;
                mirror.bytes[start..start + page_bytes() as usize].copy_from_slice(
                    &read[at * page_bytes() as usize..(at + 1) * page_bytes() as usize],
                );
            }
            mirror.captured += batch.len() as u64;
            mirror.transfers += 1;
            submitted.push(index);
        }
        submitted
    }

    pub(crate) fn mark_dirty(&self, pages: &[u32]) {
        let Some(mirror) = &self.mirror else {
            return;
        };
        let mut mirror = mirror.lock().unwrap_or_else(PoisonError::into_inner);
        for page in pages {
            if let Some(slot) = mirror.resident(*page) {
                mirror.dirty[slot as usize] = true;
            }
        }
    }

    pub(crate) fn pour(&self, queue: &Queue, bytes: &[u8]) {
        self.bounds(0, bytes.len() as u64);
        let Some(mirror) = &self.mirror else {
            self.store
                .buffer()
                .write_at(queue, self.store.offset(), bytes);
            return;
        };
        let mut mirror = mirror.lock().unwrap_or_else(PoisonError::into_inner);
        self.materialize_all(&mut mirror, queue);
        mirror.vacuum();
        let copy = bytes.len().min(mirror.bytes.len());
        mirror.bytes[..copy].copy_from_slice(&bytes[..copy]);
        mirror.bytes[copy..].fill(0);
        self.reload(queue, &mut mirror);
    }

    pub(crate) fn flush(&self, queue: &Queue) {
        let Some(mirror) = &self.mirror else {
            return;
        };
        let mut mirror = mirror.lock().unwrap_or_else(PoisonError::into_inner);
        self.materialize_all(&mut mirror, queue);
        self.flush_range(&mut mirror, queue, 0, self.words * WORD_BYTES);
    }

    fn materialize_range(&self, mirror: &mut Mirror, queue: &Queue, first: u64, end: u64) {
        let from = (first / page_bytes()) as u32;
        let to =
            u32::try_from(end.div_ceil(page_bytes())).expect("a weight range fits the page table");
        self.materialize_writebacks(mirror, queue, |page| page >= from && page < to);
    }

    fn materialize_all(&self, mirror: &mut Mirror, queue: &Queue) {
        self.materialize_writebacks(mirror, queue, |_| true);
    }

    fn flush_range(&self, mirror: &mut Mirror, queue: &Queue, first: u64, end: u64) {
        let mut sources = Vec::new();
        for slot in 0..self.slots {
            let kept = mirror.page_of[slot as usize];
            if kept == NO_PAGE || !mirror.dirty[slot as usize] {
                continue;
            }
            let start = u64::from(kept) * page_bytes();
            if start >= end || start + page_bytes() <= first {
                continue;
            }
            sources.push((kept, Origin::Slot(slot)));
            mirror.dirty[slot as usize] = false;
        }
        self.capture(mirror, queue, &sources);
    }

    pub(crate) fn checkpoint(&self, queue: &Queue) -> Vec<u8> {
        if self.words == 0 {
            return Vec::new();
        }
        let Some(mirror) = &self.mirror else {
            return self.read_device(queue, self.store.offset(), self.words * WORD_BYTES);
        };
        self.flush(queue);
        let mirror = mirror.lock().unwrap_or_else(PoisonError::into_inner);
        mirror.bytes[..(self.words * WORD_BYTES) as usize].to_vec()
    }

    fn reload(&self, queue: &Queue, mirror: &mut Mirror) {
        let table = self
            .table
            .as_ref()
            .expect("a paged weight store carries the pages it streams");
        table.write_at(queue, 0, bytemuck::cast_slice(&mirror.table()));
    }

    fn read_device(&self, queue: &Queue, offset: u64, bytes: u64) -> Vec<u8> {
        assert!(bytes > 0, "a device read of no bytes has no answer");
        let staging = Recycled::claim(
            &self.pool,
            "neura weights",
            bytes,
            BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        );
        let mut submission = Submission::new(queue.device(), "neura weights");
        submission.copy(self.store.buffer(), offset, staging.buffer(), 0, bytes);
        let submission = submission.submit(queue);
        staging.buffer().read(queue, submission, bytes)
    }
}
