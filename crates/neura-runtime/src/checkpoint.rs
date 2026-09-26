use neura_abi::{Element, WORD_BYTES};
use neura_program::Region;

const MAGIC: [u8; 4] = *b"NRCP";
const VERSION: u32 = 2;
const HEADER_BYTES: usize = 20;
const ENTRY_BYTES: usize = 8;

pub struct Checkpoint {
    bytes: Vec<u8>,
    entries: Vec<(u32, Element)>,
    payload: usize,
}

impl Checkpoint {
    pub(crate) fn of(region: &Region, payload: Vec<u8>) -> Self {
        assert_eq!(
            payload.len() as u64,
            region.bytes(),
            "a checkpoint of {} bytes carries the {} bytes its region holds",
            payload.len(),
            region.bytes(),
        );
        let entries = region
            .entries()
            .iter()
            .map(|entry| {
                (
                    u32::try_from(entry.elements).expect("a tensor fits a u32 element count"),
                    entry.element,
                )
            })
            .collect::<Vec<_>>();
        let payload_at = HEADER_BYTES + entries.len() * ENTRY_BYTES;
        let mut bytes = Vec::with_capacity(payload_at + region.bytes() as usize);
        bytes.extend_from_slice(&MAGIC);
        bytes.extend_from_slice(&VERSION.to_le_bytes());
        bytes.extend_from_slice(&(entries.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&region.words().to_le_bytes());
        for (elements, element) in &entries {
            bytes.extend_from_slice(&elements.to_le_bytes());
            bytes.extend_from_slice(&element.code().to_le_bytes());
        }
        bytes.extend_from_slice(&payload);
        Self {
            bytes,
            entries,
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
        let words = u64::from_le_bytes(bytes[12..20].try_into().expect("a word count"));
        let payload = HEADER_BYTES + tensors * ENTRY_BYTES;
        assert!(
            bytes.len() >= payload,
            "a checkpoint of {tensors} tensors needs {payload} bytes of header, not {}",
            bytes.len(),
        );
        let entries = bytes[HEADER_BYTES..payload]
            .as_chunks::<ENTRY_BYTES>()
            .0
            .iter()
            .map(|entry| {
                (
                    u32::from_le_bytes(entry[..4].try_into().expect("an element count")),
                    Element::of(u32::from_le_bytes(
                        entry[4..].try_into().expect("an element code"),
                    )),
                )
            })
            .collect::<Vec<_>>();
        let packed = entries
            .iter()
            .map(|(elements, element)| element.words(u64::from(*elements)))
            .sum::<u64>();
        assert!(
            packed <= words,
            "a checkpoint of {words} words carries {packed} words of tensors",
        );
        assert_eq!(
            (bytes.len() - payload) as u64,
            words * WORD_BYTES,
            "a checkpoint of {words} words packs {} bytes of store",
            bytes.len() - payload,
        );
        Self {
            bytes: bytes.to_vec(),
            entries,
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
        self.entries
            .get(tensor)
            .unwrap_or_else(|| panic!("this checkpoint holds {} tensors", self.entries.len()))
            .0
    }

    pub fn element(&self, tensor: usize) -> Element {
        self.entries
            .get(tensor)
            .unwrap_or_else(|| panic!("this checkpoint holds {} tensors", self.entries.len()))
            .1
    }

    pub(crate) fn payload(&self) -> &[u8] {
        &self.bytes[self.payload..]
    }

    pub(crate) fn matches(&self, region: &Region) {
        assert_eq!(
            self.entries.len(),
            region.tensors(),
            "a checkpoint of {} tensors pours into a region of {} tensors; a store loads only a checkpoint of the very same parameters in the very same order",
            self.entries.len(),
            region.tensors(),
        );
        for (index, ((checkpoint, element), entry)) in
            self.entries.iter().zip(region.entries()).enumerate()
        {
            assert!(
                u64::from(*checkpoint) == entry.elements && *element == entry.element,
                "tensor {index} of the checkpoint holds {checkpoint} {} elements where the region holds {} {}",
                element.name(),
                entry.elements,
                entry.element.name(),
            );
        }
        assert_eq!(
            self.payload().len() as u64,
            region.bytes(),
            "a checkpoint of {} bytes pours into a region of {} bytes",
            self.payload().len(),
            region.bytes(),
        );
    }
}
