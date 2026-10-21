use neura_abi::Element;

mod overlay;

pub use overlay::Overlay;

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

impl<T: Source + ?Sized> Source for &T {
    fn placed(&self, name: &str) -> Option<Placed> {
        (**self).placed(name)
    }

    fn payload(&self, name: &str, sink: &mut dyn FnMut(&[u8])) {
        (**self).payload(name, sink);
    }

    fn table(&self, name: &str, sink: &mut dyn FnMut(&[u8])) {
        (**self).table(name, sink);
    }
}
