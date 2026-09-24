use neura_abi::WORD_BYTES;
use neura_precision::Precision;
use neura_program::Region;

const MAGIC: [u8; 4] = *b"NRCP";
const VERSION: u32 = 1;
const HEADER_BYTES: usize = 24;
const ENTRY_BYTES: usize = 4;

pub struct Checkpoint {
    bytes: Vec<u8>,
    precision: Precision,
    entries: Vec<u32>,
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
            .map(|(_, elements)| {
                u32::try_from(*elements).expect("a tensor fits a u32 element count")
            })
            .collect::<Vec<_>>();
        let payload_at = HEADER_BYTES + entries.len() * ENTRY_BYTES;
        let mut bytes = Vec::with_capacity(payload_at + region.bytes() as usize);
        bytes.extend_from_slice(&MAGIC);
        bytes.extend_from_slice(&VERSION.to_le_bytes());
        bytes.extend_from_slice(&region.precision().code().to_le_bytes());
        bytes.extend_from_slice(&(entries.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&region.words().to_le_bytes());
        for elements in &entries {
            bytes.extend_from_slice(&elements.to_le_bytes());
        }
        bytes.extend_from_slice(&payload);
        Self {
            bytes,
            precision: region.precision(),
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
        let code = u32::from_le_bytes(bytes[8..12].try_into().expect("a precision word"));
        let precision = match code {
            0 => Precision::Single,
            1 => Precision::Half,
            _ => {
                panic!("a checkpoint carries the precision code {code}, and no device stores by it")
            }
        };
        let tensors =
            u32::from_le_bytes(bytes[12..16].try_into().expect("a tensor count word")) as usize;
        let words = u64::from_le_bytes(bytes[16..24].try_into().expect("a word count"));
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
            .map(|word| u32::from_le_bytes(*word))
            .collect::<Vec<_>>();
        let elements = entries
            .iter()
            .map(|elements| u64::from(*elements))
            .sum::<u64>();
        assert!(
            elements <= precision.elements_per_word() * words,
            "a checkpoint of {words} words spans {elements} elements where its store packs at most {}",
            precision.elements_per_word() * words,
        );
        assert_eq!(
            (bytes.len() - payload) as u64,
            words * WORD_BYTES,
            "a checkpoint of {words} words packs {} bytes of store",
            bytes.len() - payload,
        );
        Self {
            bytes: bytes.to_vec(),
            precision,
            entries,
            payload,
        }
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn precision(&self) -> Precision {
        self.precision
    }

    pub fn tensors(&self) -> usize {
        self.entries.len()
    }

    pub(crate) fn payload(&self) -> &[u8] {
        &self.bytes[self.payload..]
    }

    pub(crate) fn matches(&self, region: &Region) {
        assert_eq!(
            self.precision,
            region.precision(),
            "a checkpoint of a {:?} store pours into a {:?} region",
            self.precision,
            region.precision(),
        );
        assert_eq!(
            self.entries.len(),
            region.tensors(),
            "a checkpoint of {} tensors pours into a region of {} tensors; a store loads only a checkpoint of the very same parameters in the very same order",
            self.entries.len(),
            region.tensors(),
        );
        for (index, (checkpoint, (_, elements))) in
            self.entries.iter().zip(region.entries()).enumerate()
        {
            assert_eq!(
                u64::from(*checkpoint),
                *elements,
                "tensor {index} of the checkpoint holds {} elements where the region holds {elements}",
                checkpoint,
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
