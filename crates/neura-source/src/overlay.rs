use crate::{Placed, Source};

pub struct Overlay<Base, Over> {
    base: Base,
    over: Over,
}

impl<Base: Source, Over: Source> Overlay<Base, Over> {
    pub fn new(base: Base, over: Over) -> Self {
        Self { base, over }
    }
}

impl<Base: Source, Over: Source> Source for Overlay<Base, Over> {
    fn placed(&self, name: &str) -> Option<Placed> {
        self.over.placed(name).or_else(|| self.base.placed(name))
    }

    fn payload(&self, name: &str, sink: &mut dyn FnMut(&[u8])) {
        match self.over.placed(name) {
            Some(_) => self.over.payload(name, sink),
            None => self.base.payload(name, sink),
        }
    }

    fn table(&self, name: &str, sink: &mut dyn FnMut(&[u8])) {
        match self.over.placed(name) {
            Some(_) => self.over.table(name, sink),
            None => self.base.table(name, sink),
        }
    }
}
