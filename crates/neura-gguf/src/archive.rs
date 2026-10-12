use crate::{Gguf, GgufFile, Metadata, Tensor};

pub trait Archive {
    fn metadata(&self) -> Metadata<'_>;

    fn tensor(&self, name: &str) -> Option<Tensor>;
}

impl Archive for Gguf {
    fn metadata(&self) -> Metadata<'_> {
        Gguf::metadata(self)
    }

    fn tensor(&self, name: &str) -> Option<Tensor> {
        Gguf::tensor(self, name)
    }
}

impl Archive for GgufFile {
    fn metadata(&self) -> Metadata<'_> {
        GgufFile::metadata(self)
    }

    fn tensor(&self, name: &str) -> Option<Tensor> {
        GgufFile::tensor(self, name)
    }
}
