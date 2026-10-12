use crate::Value;
use std::collections::BTreeMap;

#[derive(Clone, Copy)]
pub struct Metadata<'a> {
    held: &'a BTreeMap<String, Value>,
}

impl<'a> Metadata<'a> {
    pub(crate) const fn of(held: &'a BTreeMap<String, Value>) -> Self {
        Self { held }
    }

    pub fn has(&self, key: &str) -> bool {
        self.held.contains_key(key)
    }

    pub fn value(&self, key: &str) -> &'a Value {
        self.held.get(key).unwrap_or_else(|| {
            panic!("this gguf holds no {key}");
        })
    }

    pub fn text(&self, key: &str) -> &'a str {
        self.value(key).text().unwrap_or_else(|| {
            panic!(
                "the {key} of this gguf is {} where a string stands",
                self.value(key).name(),
            );
        })
    }

    pub fn texts(&self, key: &str) -> Vec<&'a str> {
        self.array(key)
            .iter()
            .map(|value| {
                value.text().unwrap_or_else(|| {
                    panic!(
                        "the {key} of this gguf holds {} where a string stands",
                        value.name(),
                    );
                })
            })
            .collect()
    }

    pub fn flag(&self, key: &str) -> bool {
        self.value(key).flag().unwrap_or_else(|| {
            panic!(
                "the {key} of this gguf is {} where a flag stands",
                self.value(key).name(),
            );
        })
    }

    pub fn word(&self, key: &str) -> u32 {
        let held = self.integer(key);
        u32::try_from(held).unwrap_or_else(|_| {
            panic!("the {key} of this gguf holds {held} where a word holds what fits one");
        })
    }

    pub fn single(&self, key: &str) -> f32 {
        self.value(key).number().map_or_else(
            || {
                panic!(
                    "the {key} of this gguf is {} where a number stands",
                    self.value(key).name(),
                );
            },
            |held| held as f32,
        )
    }

    pub fn optional_word(&self, key: &str) -> Option<u32> {
        self.held.get(key).map(|_| self.word(key))
    }

    pub fn array(&self, key: &str) -> &'a [Value] {
        self.value(key).array().unwrap_or_else(|| {
            panic!(
                "the {key} of this gguf is {} where an array stands",
                self.value(key).name(),
            );
        })
    }

    fn integer(&self, key: &str) -> u64 {
        self.value(key).integer().unwrap_or_else(|| {
            panic!(
                "the {key} of this gguf is {} where a whole number stands",
                self.value(key).name(),
            );
        })
    }
}
