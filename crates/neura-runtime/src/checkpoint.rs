use neura_abi::{Element, WORD_BYTES};
use neura_tape::Region;

const MAGIC: [u8; 4] = *b"NRCP";
const VERSION: u32 = 5;
const HEADER_BYTES: usize = 32;
const ENTRY_BYTES: usize = 20;

#[derive(Clone, Copy)]
struct Entry {
    word: u64,
    elements: u32,
    element: Element,
    scale: f32,
}

pub struct Checkpoint {
    bytes: Vec<u8>,
    entries: Vec<Entry>,
    state: Vec<Entry>,
    weights_words: u64,
    state_words: u64,
    payload: usize,
}

impl Checkpoint {
    pub(crate) fn of(weights: &Region, state: &Region, payload: Vec<u8>) -> Self {
        let parameters = entries(weights);
        let training = entries(state);
        let weights_words = weights.words();
        let state_words = state.words().max(weights_words);
        assert_eq!(
            payload.len() as u64,
            state_words * WORD_BYTES,
            "a checkpoint of {} bytes carries the {state_words} words its store holds",
            payload.len(),
        );
        let payload_at = HEADER_BYTES + (parameters.len() + training.len()) * ENTRY_BYTES;
        let mut bytes = Vec::with_capacity(payload_at + payload.len());
        bytes.extend_from_slice(&MAGIC);
        bytes.extend_from_slice(&VERSION.to_le_bytes());
        bytes.extend_from_slice(&(parameters.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&(training.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&weights_words.to_le_bytes());
        bytes.extend_from_slice(&state_words.to_le_bytes());
        for entry in parameters.iter().chain(&training) {
            bytes.extend_from_slice(&entry.word.to_le_bytes());
            bytes.extend_from_slice(&entry.elements.to_le_bytes());
            bytes.extend_from_slice(&entry.element.code().to_le_bytes());
            bytes.extend_from_slice(&entry.scale.to_le_bytes());
        }
        bytes.extend_from_slice(&payload);
        Self {
            bytes,
            entries: parameters,
            state: training,
            weights_words,
            state_words,
            payload: payload_at,
        }
    }

    pub fn decode(bytes: &[u8]) -> Self {
        assert!(
            bytes.len() >= HEADER_BYTES,
            "a checkpoint holds at least {HEADER_BYTES} bytes of header, not {}",
            bytes.len(),
        );
        assert_eq!(
            &bytes[..4],
            &MAGIC,
            "the bytes a checkpoint decodes from open with {} where this framework signs {}",
            String::from_utf8_lossy(&bytes[..4]),
            String::from_utf8_lossy(&MAGIC),
        );
        let version = u32::from_le_bytes(bytes[4..8].try_into().expect("a version word"));
        assert_eq!(
            version, VERSION,
            "a checkpoint of format version {version} predates or postdates format version {VERSION}",
        );
        let tensors =
            u32::from_le_bytes(bytes[8..12].try_into().expect("a tensor count word")) as usize;
        let state =
            u32::from_le_bytes(bytes[12..16].try_into().expect("a state count word")) as usize;
        let weights_words =
            u64::from_le_bytes(bytes[16..24].try_into().expect("a weight word count"));
        let state_words = u64::from_le_bytes(bytes[24..32].try_into().expect("a store word count"));
        let payload = HEADER_BYTES + (tensors + state) * ENTRY_BYTES;
        assert!(
            bytes.len() >= payload,
            "a checkpoint of {tensors} tensors beside {state} of training state needs {payload} bytes of header, not {}",
            bytes.len(),
        );
        let records = bytes[HEADER_BYTES..payload]
            .as_chunks::<ENTRY_BYTES>()
            .0
            .iter()
            .map(|entry| Entry {
                word: u64::from_le_bytes(entry[..8].try_into().expect("a tensor address")),
                elements: u32::from_le_bytes(entry[8..12].try_into().expect("an element count")),
                element: Element::of(u32::from_le_bytes(
                    entry[12..16].try_into().expect("an element code"),
                )),
                scale: f32::from_le_bytes(entry[16..].try_into().expect("a quantum scale")),
            })
            .collect::<Vec<Entry>>();
        assert!(
            weights_words <= state_words,
            "a checkpoint of {weights_words} words of parameters holds {state_words} words of store",
        );
        assert_eq!(
            (bytes.len() - payload) as u64,
            state_words * WORD_BYTES,
            "a checkpoint of {state_words} words packs {} bytes of store",
            bytes.len() - payload,
        );
        Self {
            bytes: bytes.to_vec(),
            entries: records[..tensors].to_vec(),
            state: records[tensors..].to_vec(),
            weights_words,
            state_words,
            payload,
        }
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn tensors(&self) -> usize {
        self.entries.len()
    }

    pub fn elements(&self, tensor: usize) -> u32 {
        self.entry(tensor).elements
    }

    pub fn element(&self, tensor: usize) -> Element {
        self.entry(tensor).element
    }

    pub fn scale(&self, tensor: usize) -> f32 {
        self.entry(tensor).scale
    }

    pub fn state_tensors(&self) -> usize {
        self.state.len()
    }

    pub(crate) fn payload(&self, words: u64) -> &[u8] {
        let bytes = words * WORD_BYTES;
        assert!(
            bytes <= (self.bytes.len() - self.payload) as u64,
            "a store of {words} words asks for {bytes} bytes where this checkpoint holds {}",
            self.bytes.len() - self.payload,
        );
        &self.bytes[self.payload..self.payload + bytes as usize]
    }

    pub(crate) fn matches(&self, weights: &Region, state: &Region) {
        assert_eq!(
            self.entries.len(),
            weights.tensors(),
            "a checkpoint of {} tensors pours into a region of {} tensors; a store loads only a checkpoint of the very same model parameters",
            self.entries.len(),
            weights.tensors(),
        );
        assert_section(&self.entries, weights, "parameter");
        assert_eq!(
            self.weights_words,
            weights.words(),
            "a checkpoint of {} words of parameters pours into a region of {} words",
            self.weights_words,
            weights.words(),
        );
        if state.tensors() == 0 {
            return;
        }
        assert_eq!(
            self.state.len(),
            state.tensors(),
            "a checkpoint of {} tensors of training state pours into a region of {} tensors; an optimizer resumes only from the state it wrote",
            self.state.len(),
            state.tensors(),
        );
        assert_section(&self.state, state, "training state");
        assert_eq!(
            self.state_words,
            state.words(),
            "a checkpoint of {self:?} carries {} words of store where the region holds {}",
            self.state_words,
            state.words(),
        );
    }

    fn entry(&self, tensor: usize) -> Entry {
        *self
            .entries
            .get(tensor)
            .unwrap_or_else(|| panic!("this checkpoint holds {} tensors", self.entries.len()))
    }
}

impl std::fmt::Debug for Checkpoint {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        out.debug_struct("Checkpoint")
            .field("tensors", &self.entries.len())
            .field("state", &self.state.len())
            .field("weights_words", &self.weights_words)
            .field("state_words", &self.state_words)
            .finish()
    }
}

fn entries(region: &Region) -> Vec<Entry> {
    region
        .entries()
        .iter()
        .map(|entry| Entry {
            word: entry.word,
            elements: u32::try_from(entry.elements).expect("a tensor fits a u32 element count"),
            element: entry.element,
            scale: entry.scale,
        })
        .collect()
}

fn assert_section(records: &[Entry], region: &Region, section: &str) {
    for (index, (record, entry)) in records.iter().zip(region.entries()).enumerate() {
        assert!(
            record.elements == u32::try_from(entry.elements).expect("a tensor fits a u32")
                && record.element == entry.element,
            "tensor {index} of the {section} holds {} {} elements where the region holds {} {}",
            record.elements,
            record.element.name(),
            entry.elements,
            entry.element.name(),
        );
        assert!(
            record.scale == entry.scale,
            "tensor {index} of the {section} quantizes by {} where the region quantizes by {}",
            record.scale,
            entry.scale,
        );
        assert_eq!(
            record.word, entry.word,
            "tensor {index} of the {section} lies at word {} where the region holds it at word {}",
            record.word, entry.word,
        );
    }
}
