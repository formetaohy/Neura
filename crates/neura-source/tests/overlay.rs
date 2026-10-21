use neura_abi::Element;
use neura_source::{Overlay, Placed, Source};

struct Tensors(&'static [(&'static str, u8)]);

impl Source for Tensors {
    fn placed(&self, name: &str) -> Option<Placed> {
        self.0
            .iter()
            .find(|(held, _)| *held == name)
            .map(|_| Placed {
                element: Element::Single,
                elements: 1,
                scale: 1.0,
                payload_bytes: 1,
                table_bytes: 0,
            })
    }

    fn payload(&self, name: &str, sink: &mut dyn FnMut(&[u8])) {
        let (_, byte) = self
            .0
            .iter()
            .find(|(held, _)| *held == name)
            .unwrap_or_else(|| panic!("this source holds no tensor named {name}"));
        sink(std::slice::from_ref(byte));
    }

    fn table(&self, name: &str, _sink: &mut dyn FnMut(&[u8])) {
        panic!("this source quantizes no tensor, and {name} asks for quanta");
    }
}

#[test]
fn an_overlay_reads_the_tensor_it_lays_over_a_base() {
    let base = Tensors(&[("shared", 1), ("base", 2)]);
    let over = Tensors(&[("shared", 3), ("adapter", 4)]);
    let held = Overlay::new(&base, &over);
    for (name, wanted) in [("shared", 3u8), ("base", 2), ("adapter", 4)] {
        assert_eq!(
            held.placed(name),
            Some(Placed {
                element: Element::Single,
                elements: 1,
                scale: 1.0,
                payload_bytes: 1,
                table_bytes: 0,
            }),
            "{name} is placed by the source that holds it",
        );
        let mut bytes = Vec::new();
        held.payload(name, &mut |chunk| bytes.extend_from_slice(chunk));
        assert_eq!(
            bytes,
            vec![wanted],
            "{name} reads from the source that holds it"
        );
    }
    assert!(held.placed("missing").is_none());
}
