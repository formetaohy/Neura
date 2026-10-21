use std::collections::VecDeque;

const MAGIC: &[u8; 4] = b"NBS1";

#[derive(Clone, Debug)]
pub struct BatchSnapshot {
    pub(crate) batch: u32,
    pub(crate) context: u32,
    pub(crate) window: u32,
    pub(crate) sampled: bool,
    pub(crate) prefix: Option<Vec<u32>>,
    pub(crate) slots: Vec<Option<SlotSnapshot>>,
    pub(crate) seeds: Vec<u32>,
    pub(crate) temperatures: Vec<f32>,
    pub(crate) keeps: Vec<f32>,
    pub(crate) cumulatives: Vec<f32>,
    pub(crate) state_names: Vec<String>,
    pub(crate) states: Vec<Vec<f32>>,
}

#[derive(Clone, Debug)]
pub(crate) struct SlotSnapshot {
    pub(crate) cursor: u32,
    pub(crate) phase: PhaseSnapshot,
}

#[derive(Clone, Debug)]
pub(crate) enum PhaseSnapshot {
    Prompting(VecDeque<u32>),
    Generating(u32),
    Suspended,
}

impl BatchSnapshot {
    pub fn batch(&self) -> u32 {
        self.batch
    }

    pub fn context(&self) -> u32 {
        self.context
    }

    pub fn window(&self) -> u32 {
        self.window
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        words(&mut bytes, [self.batch, self.context, self.window]);
        word(&mut bytes, u32::from(self.sampled));
        match &self.prefix {
            Some(tokens) => {
                word(&mut bytes, 1);
                tokens_out(&mut bytes, tokens);
            }
            None => word(&mut bytes, 0),
        }
        word(
            &mut bytes,
            u32::try_from(self.slots.len()).expect("a batch holds fewer planes than a word counts"),
        );
        for slot in &self.slots {
            match slot {
                None => word(&mut bytes, 0),
                Some(slot) => {
                    let tag = match &slot.phase {
                        PhaseSnapshot::Prompting(_) => 1,
                        PhaseSnapshot::Generating(_) => 2,
                        PhaseSnapshot::Suspended => 3,
                    };
                    word(&mut bytes, tag);
                    word(&mut bytes, slot.cursor);
                    match &slot.phase {
                        PhaseSnapshot::Prompting(tokens) => {
                            let tokens = tokens.iter().copied().collect::<Vec<u32>>();
                            tokens_out(&mut bytes, &tokens);
                        }
                        PhaseSnapshot::Generating(token) => word(&mut bytes, *token),
                        PhaseSnapshot::Suspended => {}
                    }
                }
            }
        }
        word(
            &mut bytes,
            u32::try_from(self.seeds.len()).expect("a batch holds fewer planes than a word counts"),
        );
        for seed in &self.seeds {
            word(&mut bytes, *seed);
        }
        for values in [&self.temperatures, &self.keeps, &self.cumulatives] {
            word(
                &mut bytes,
                u32::try_from(values.len()).expect("a batch holds fewer planes than a word counts"),
            );
            for value in values {
                bytes.extend_from_slice(&value.to_bits().to_le_bytes());
            }
        }
        word(
            &mut bytes,
            u32::try_from(self.state_names.len())
                .expect("a batch names fewer states than a word counts"),
        );
        for name in &self.state_names {
            let name = name.as_bytes();
            word(
                &mut bytes,
                u32::try_from(name.len()).expect("a state name fits a word"),
            );
            bytes.extend_from_slice(name);
        }
        word(
            &mut bytes,
            u32::try_from(self.states.len())
                .expect("a batch holds fewer states than a word counts"),
        );
        for values in &self.states {
            word(
                &mut bytes,
                u32::try_from(values.len()).expect("a batch holds fewer planes than a word counts"),
            );
            for value in values {
                bytes.extend_from_slice(&value.to_bits().to_le_bytes());
            }
        }
        bytes
    }

    pub fn from_bytes(bytes: &[u8]) -> Self {
        let mut input = Reader { bytes, at: 0 };
        assert_eq!(
            input.take(4),
            MAGIC,
            "a batch snapshot opens with the four bytes {MAGIC:?}",
        );
        let batch = input.word();
        let context = input.word();
        let window = input.word();
        let sampled = match input.word() {
            0 => false,
            1 => true,
            other => panic!("a batch snapshot samples by {other}"),
        };
        let prefix = match input.word() {
            0 => None,
            _ => Some(input.tokens()),
        };
        let planes = input.count();
        assert_eq!(
            planes, batch,
            "a snapshot of {batch} planes names {planes} slots",
        );
        let mut slots = Vec::with_capacity(planes as usize);
        for _ in 0..planes {
            let tag = input.word();
            let slot = match tag {
                0 => None,
                _ => {
                    let cursor = input.word();
                    let phase = match tag {
                        1 => PhaseSnapshot::Prompting(input.tokens().into()),
                        2 => PhaseSnapshot::Generating(input.word()),
                        3 => PhaseSnapshot::Suspended,
                        _ => panic!("a slot snapshot holds phase {tag}"),
                    };
                    Some(SlotSnapshot { cursor, phase })
                }
            };
            slots.push(slot);
        }
        let seeds = input.words();
        let temperatures = input.floats();
        let keeps = input.floats();
        let cumulatives = input.floats();
        let names = input.count();
        let mut state_names = Vec::with_capacity(names as usize);
        for _ in 0..names {
            let length = input.word() as usize;
            state_names.push(
                std::str::from_utf8(input.take(length))
                    .expect("a state name holds UTF-8")
                    .to_owned(),
            );
        }
        let held = input.count();
        assert_eq!(
            held, names,
            "a snapshot names {names} states and holds {held} of them",
        );
        let mut states = Vec::with_capacity(held as usize);
        for _ in 0..held {
            states.push(input.floats());
        }
        input.finish();
        assert_eq!(
            seeds.len(),
            batch as usize,
            "a snapshot of {batch} planes carries {} seeds",
            seeds.len(),
        );
        for values in [&temperatures, &keeps, &cumulatives] {
            assert_eq!(
                values.len(),
                batch as usize,
                "a snapshot of {batch} planes carries {} values",
                values.len(),
            );
        }
        for values in &states {
            assert_eq!(
                values.len(),
                batch as usize,
                "a state of {batch} planes carries {} values",
                values.len(),
            );
        }
        Self {
            batch,
            context,
            window,
            sampled,
            prefix,
            slots,
            seeds,
            temperatures,
            keeps,
            cumulatives,
            state_names,
            states,
        }
    }
}

fn word(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn words(bytes: &mut Vec<u8>, values: impl IntoIterator<Item = u32>) {
    for value in values {
        word(bytes, value);
    }
}

fn tokens_out(bytes: &mut Vec<u8>, tokens: &[u32]) {
    word(
        bytes,
        u32::try_from(tokens.len()).expect("a prompt holds fewer tokens than a word counts"),
    );
    words(bytes, tokens.iter().copied());
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, bytes: usize) -> &'a [u8] {
        let end = self
            .at
            .checked_add(bytes)
            .expect("a snapshot offset fits an address");
        assert!(
            end <= self.bytes.len(),
            "a snapshot holds {} bytes where its reader asks for {end}",
            self.bytes.len(),
        );
        let held = &self.bytes[self.at..end];
        self.at = end;
        held
    }

    fn word(&mut self) -> u32 {
        u32::from_le_bytes(
            self.take(4)
                .try_into()
                .expect("a snapshot word spans four bytes"),
        )
    }

    fn count(&mut self) -> u32 {
        let count = self.word();
        assert!(
            (count as usize) <= self.bytes.len(),
            "a snapshot counts {count} values in {} bytes",
            self.bytes.len(),
        );
        count
    }

    fn words(&mut self) -> Vec<u32> {
        let count = self.count();
        (0..count).map(|_| self.word()).collect()
    }

    fn floats(&mut self) -> Vec<f32> {
        let count = self.count();
        (0..count).map(|_| f32::from_bits(self.word())).collect()
    }

    fn tokens(&mut self) -> Vec<u32> {
        self.words()
    }

    fn finish(&self) {
        assert_eq!(
            self.at,
            self.bytes.len(),
            "a snapshot holds {} bytes where its reader stops at {}",
            self.bytes.len(),
            self.at,
        );
    }
}
