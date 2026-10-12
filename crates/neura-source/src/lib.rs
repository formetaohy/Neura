use neura_abi::Element;

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Placed {
    pub element: Element,
    pub elements: u64,
    pub scale: f32,
    pub payload_bytes: u64,
    pub table_bytes: u64,
}

pub trait Source {
    fn placed(&self, name: &str) -> Option<Placed>;

    fn payload(&self, name: &str, sink: &mut dyn FnMut(&[u8]));

    fn table(&self, name: &str, sink: &mut dyn FnMut(&[u8]));
}
