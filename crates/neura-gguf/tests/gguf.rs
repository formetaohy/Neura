use neura_abi::Element;
use neura_gguf::{Gguf, Value};
use neura_source::Source;

#[path = "support/mod.rs"]
mod support;

use support::{Meta, build};

fn refuses(action: impl FnOnce()) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(action)).is_err()
}

fn kind(element: Element) -> u32 {
    match element {
        Element::Single => 0,
        Element::Half => 1,
        Element::Q8_0 => 8,
        Element::Q3K => 11,
        Element::Q4K => 12,
        Element::Q5K => 13,
        Element::Q6K => 14,
        Element::Bfloat16 => 30,
        other => panic!("{} storage carries no ggml type", other.name()),
    }
}

#[test]
fn a_gguf_hands_back_the_metadata_it_holds() {
    let fixture = build(
        32,
        &[
            ("general.alignment", Meta::Unsigned32(32)),
            ("example.architecture", Meta::Text("nothing in particular")),
            ("example.count", Meta::Unsigned64(9)),
            ("example.ratio", Meta::Single(0.125)),
            ("example.precise", Meta::Double(0.5)),
            ("example.flag", Meta::Flag(true)),
            ("example.byte", Meta::Unsigned8(200)),
            ("example.signed", Meta::Signed8(-5)),
            (
                "example.tags",
                Meta::Array(&[Meta::Text("one"), Meta::Text("two")]),
            ),
            (
                "example.counts",
                Meta::Array(&[Meta::Unsigned32(3), Meta::Unsigned32(4)]),
            ),
        ],
        &[],
    );
    let gguf = Gguf::decode(&fixture.bytes);
    assert_eq!(gguf.tensors(), 0);
    assert_eq!(
        gguf.value("example.architecture").and_then(Value::text),
        Some("nothing in particular"),
    );
    assert_eq!(
        gguf.value("example.count").and_then(Value::integer),
        Some(9),
    );
    assert_eq!(
        gguf.value("example.ratio").and_then(Value::number),
        Some(0.125),
    );
    assert_eq!(
        gguf.value("example.precise").and_then(Value::number),
        Some(0.5),
    );
    assert_eq!(gguf.value("example.flag").and_then(Value::flag), Some(true));
    assert_eq!(
        gguf.value("example.byte").and_then(Value::integer),
        Some(200),
    );
    assert_eq!(
        gguf.value("example.signed").and_then(Value::integer),
        Some(-5i64 as u64),
    );
    let counts = gguf
        .value("example.counts")
        .and_then(Value::array)
        .expect("a gguf holds the array it was given");
    assert_eq!(counts[1].integer(), Some(4));
    let tags = gguf
        .value("example.tags")
        .and_then(Value::array)
        .expect("a gguf holds the array it was given");
    assert_eq!(tags.len(), 2);
    assert_eq!(tags[1].text(), Some("two"));
    assert!(gguf.value("example.absent").is_none());
}

#[test]
fn a_gguf_names_the_element_every_tensor_packs() {
    let held = [
        (Element::Single, 8u64),
        (Element::Half, 16),
        (Element::Q8_0, 32),
        (Element::Q3K, 256),
        (Element::Q4K, 256),
        (Element::Q5K, 256),
        (Element::Q6K, 256),
        (Element::Bfloat16, 4),
    ];
    let payloads = held
        .iter()
        .map(|(element, elements)| vec![0u8; element.payload_bytes(*elements) as usize])
        .collect::<Vec<Vec<u8>>>();
    let shapes = held
        .iter()
        .map(|(_, elements)| vec![*elements])
        .collect::<Vec<Vec<u64>>>();
    let tensors = held
        .iter()
        .enumerate()
        .map(|(at, (element, _))| {
            (
                element.name(),
                kind(*element),
                &shapes[at][..],
                &payloads[at][..],
            )
        })
        .collect::<Vec<_>>();
    let fixture = build(32, &[("general.alignment", Meta::Unsigned32(32))], &tensors);
    let gguf = Gguf::decode(&fixture.bytes);
    assert_eq!(gguf.tensors(), held.len());
    for (element, elements) in &held {
        let tensor = gguf
            .tensor(element.name())
            .unwrap_or_else(|| panic!("a gguf holds the tensor {}", element.name()));
        assert_eq!(tensor.element, *element);
        assert_eq!(tensor.elements, *elements);
        assert_eq!(tensor.shape[0], *elements);
        assert_eq!(tensor.rank, 1);
        assert_eq!(tensor.bytes, element.payload_bytes(*elements));
        assert_eq!(tensor.at, fixture.at[element.name()]);
        let placed = gguf
            .placed(element.name())
            .expect("a gguf places its tensors");
        assert_eq!(placed.element, *element);
        assert_eq!(placed.elements, *elements);
        assert_eq!(placed.scale, 1.0);
        assert_eq!(placed.payload_bytes, tensor.bytes);
        assert_eq!(placed.table_bytes, 0);
        let mut handed = Vec::new();
        gguf.payload(element.name(), &mut |chunk| handed.extend_from_slice(chunk));
        let start = (fixture.data + tensor.at) as usize;
        assert_eq!(
            handed,
            fixture.bytes[start..start + tensor.bytes as usize].to_vec(),
            "a gguf hands out the bytes of {} and nothing else",
            element.name(),
        );
    }
}

#[test]
fn a_gguf_refuses_a_tensor_this_framework_cannot_read() {
    let fixture = build(
        32,
        &[("general.alignment", Meta::Unsigned32(32))],
        &[("packed", 23, &[256], &[0u8; 136])],
    );
    assert!(
        refuses(move || {
            Gguf::decode(&fixture.bytes);
        }),
        "a gguf whose tensor packs IQ4_XS numbers is refused by name",
    );
}

#[test]
fn a_gguf_refuses_a_file_that_does_not_follow_its_layout() {
    let fixture = build(
        32,
        &[("general.alignment", Meta::Unsigned32(32))],
        &[("weights", 12, &[256], &[0u8; 144])],
    );
    assert!(
        refuses(|| {
            let mut bytes = fixture.bytes.clone();
            bytes[..4].copy_from_slice(b"GGFU");
            Gguf::decode(&bytes);
        }),
        "a file that does not open with GGUF holds no gguf",
    );
    assert!(
        refuses(|| {
            let mut bytes = fixture.bytes.clone();
            bytes[4..8].copy_from_slice(&9u32.to_le_bytes());
            Gguf::decode(&bytes);
        }),
        "a gguf of a version this reader does not follow is refused",
    );
    assert!(
        refuses(|| {
            let short = build(
                32,
                &[("general.alignment", Meta::Unsigned32(32))],
                &[("weights", 12, &[256], &[0u8; 144])],
            );
            Gguf::decode(&short.bytes[..8]);
        }),
        "a file that stops inside its own header holds no gguf",
    );
    assert!(
        refuses(|| {
            let broken = build(
                32,
                &[("general.alignment", Meta::Unsigned32(32))],
                &[("weights", 12, &[128], &[0u8; 144])],
            );
            Gguf::decode(&broken.bytes);
        }),
        "a gguf whose q4_k tensor holds half a block is refused",
    );
    assert!(
        refuses(|| {
            let cut = build(
                32,
                &[("general.alignment", Meta::Unsigned32(32))],
                &[("weights", 2, &[32], &[0u8; 18])],
            );
            Gguf::decode(&cut.bytes[..cut.data as usize]);
        }),
        "a gguf whose tensor data stops short of the tensor it holds is refused",
    );
    assert!(
        refuses(|| {
            let broken = build(
                8,
                &[("general.alignment", Meta::Unsigned32(32))],
                &[
                    ("first", 12, &[256], &[0u8; 144]),
                    ("second", 12, &[256], &[0u8; 144]),
                ],
            );
            Gguf::decode(&broken.bytes);
        }),
        "a gguf whose tensor begins off the alignment it declares is refused",
    );
}

#[test]
fn a_gguf_file_hands_a_store_the_same_bytes_as_the_whole_file() {
    let fixture = build(
        32,
        &[("general.alignment", Meta::Unsigned32(16))],
        &[("weights", 12, &[256], &[7u8; 144])],
    );
    let directory = std::env::temp_dir().join(format!("neura-gguf-{}", std::process::id()));
    std::fs::create_dir_all(&directory).expect("a directory for one gguf file");
    let path = directory.join("weights.gguf");
    std::fs::write(&path, &fixture.bytes).expect("a gguf file");
    let file = neura_gguf::GgufFile::open(&path);
    assert_eq!(file.tensors(), 1);
    assert_eq!(file.path(), path.as_path());
    let held = Gguf::decode(&fixture.bytes);
    assert_eq!(
        file.tensor("weights"),
        held.tensor("weights"),
        "a gguf file holds the tensor table the bytes of the file hold",
    );
    let mut streamed = Vec::new();
    file.payload("weights", &mut |chunk| streamed.extend_from_slice(chunk));
    let mut whole = Vec::new();
    held.payload("weights", &mut |chunk| whole.extend_from_slice(chunk));
    assert_eq!(streamed, whole);
    assert!(file.reads() > 0);
    std::fs::remove_dir_all(&directory).expect("the directory of one gguf file");
}
