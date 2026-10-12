use std::cell::RefCell;
use std::rc::Rc;

#[derive(Clone, Debug)]
pub enum Value {
    Missing,
    Empty,
    Flag(bool),
    Integer(i64),
    Number(f64),
    Text(String),
    List(Vec<Value>),
    Map(Vec<(String, Value)>),
    Namespace(Rc<RefCell<Vec<(String, Value)>>>),
}

impl Value {
    pub fn truth(&self) -> bool {
        match self {
            Self::Missing | Self::Empty => false,
            Self::Flag(held) => *held,
            Self::Integer(held) => *held != 0,
            Self::Number(held) => *held != 0.0,
            Self::Text(held) => !held.is_empty(),
            Self::List(held) => !held.is_empty(),
            Self::Map(held) => !held.is_empty(),
            Self::Namespace(held) => !held.borrow().is_empty(),
        }
    }

    pub fn text(&self) -> String {
        match self {
            Self::Missing => String::new(),
            Self::Empty => "None".to_owned(),
            Self::Flag(held) => if *held { "True" } else { "False" }.to_owned(),
            Self::Integer(held) => held.to_string(),
            Self::Number(held) => {
                if held.is_finite() && held.fract() == 0.0 {
                    format!("{held:.1}")
                } else {
                    format!("{held}")
                }
            }
            Self::Text(held) => held.clone(),
            Self::List(_) | Self::Map(_) | Self::Namespace(_) => self.repr(),
        }
    }

    pub fn kind(&self) -> &'static str {
        match self {
            Self::Missing => "what the template leaves undefined",
            Self::Empty => "none",
            Self::Flag(_) => "a flag",
            Self::Integer(_) => "a whole number",
            Self::Number(_) => "a number",
            Self::Text(_) => "a string",
            Self::List(_) => "a list",
            Self::Map(_) => "a mapping",
            Self::Namespace(_) => "a namespace",
        }
    }

    pub(crate) fn repr(&self) -> String {
        match self {
            Self::Text(held) => format!("'{}'", held.replace('\\', "\\\\").replace('\'', "\\'")),
            Self::List(held) => {
                let items = held.iter().map(Self::repr).collect::<Vec<String>>();
                format!("[{}]", items.join(", "))
            }
            Self::Map(held) => {
                let items = held
                    .iter()
                    .map(|(key, value)| {
                        format!("{}: {}", Self::Text(key.clone()).repr(), value.repr())
                    })
                    .collect::<Vec<String>>();
                format!("{{{}}}", items.join(", "))
            }
            Self::Namespace(held) => Self::Map(held.borrow().clone()).repr(),
            other => other.text(),
        }
    }

    pub(crate) fn equals(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Missing, Self::Missing) | (Self::Empty, Self::Empty) => true,
            (Self::Missing, _) | (_, Self::Missing) | (Self::Empty, _) | (_, Self::Empty) => false,
            (Self::Flag(held), Self::Flag(other)) => held == other,
            (Self::Integer(held), Self::Integer(other)) => held == other,
            (Self::Integer(held), Self::Number(other)) => *held as f64 == *other,
            (Self::Number(held), Self::Integer(other)) => *held == *other as f64,
            (Self::Number(held), Self::Number(other)) => held == other,
            (Self::Text(held), Self::Text(other)) => held == other,
            (Self::List(held), Self::List(other)) => {
                held.len() == other.len()
                    && held
                        .iter()
                        .zip(other)
                        .all(|(held, other)| held.equals(other))
            }
            (Self::Map(held), Self::Map(other)) => {
                held.len() == other.len()
                    && held.iter().all(|(key, value)| {
                        other
                            .iter()
                            .find(|(other, _)| other == key)
                            .is_some_and(|(_, other)| value.equals(other))
                    })
            }
            (Self::Namespace(held), Self::Namespace(other)) => Rc::ptr_eq(held, other),
            _ => false,
        }
    }

    pub(crate) fn greater(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Integer(held), Self::Integer(other)) => held > other,
            (Self::Integer(held), Self::Number(other)) => *held as f64 > *other,
            (Self::Number(held), Self::Integer(other)) => *held > *other as f64,
            (Self::Number(held), Self::Number(other)) => held > other,
            (Self::Text(held), Self::Text(other)) => held > other,
            _ => panic!(
                "a template weighs {} against {} where no order stands",
                self.kind(),
                other.kind(),
            ),
        }
    }

    pub(crate) fn add(&self, other: &Self) -> Self {
        match (self, other) {
            (Self::Text(held), Self::Text(other)) => Self::Text(format!("{held}{other}")),
            (Self::Integer(held), Self::Integer(other)) => {
                Self::Integer(held.checked_add(*other).unwrap_or_else(|| {
                    panic!("a template adds {held} to {other} where no whole number holds the sum")
                }))
            }
            (Self::Integer(held), Self::Number(other)) => Self::Number(*held as f64 + other),
            (Self::Number(held), Self::Integer(other)) => Self::Number(held + *other as f64),
            (Self::Number(held), Self::Number(other)) => Self::Number(held + other),
            _ => panic!("a template adds {} to {}", other.kind(), self.kind(),),
        }
    }

    pub(crate) fn concat(&self, other: &Self) -> Self {
        Self::Text(format!("{}{}", self.text(), other.text()))
    }

    pub(crate) fn attribute(&self, name: &str) -> Self {
        match self {
            Self::Missing => Self::Missing,
            Self::Map(held) => held
                .iter()
                .find(|(key, _)| key == name)
                .map_or(Self::Missing, |(_, value)| value.clone()),
            Self::Namespace(held) => held
                .borrow()
                .iter()
                .find(|(key, _)| key == name)
                .map_or(Self::Missing, |(_, value)| value.clone()),
            _ => panic!("a template reads the attribute {name} of {}", self.kind()),
        }
    }

    pub(crate) fn assign(&self, name: &str, value: Self) {
        match self {
            Self::Namespace(held) => {
                let mut held = held.borrow_mut();
                match held.iter_mut().find(|(key, _)| key == name) {
                    Some((_, slot)) => *slot = value,
                    None => held.push((name.to_owned(), value)),
                }
            }
            _ => panic!("a template writes the attribute {name} of {}", self.kind()),
        }
    }

    pub(crate) fn subscript(&self, key: &Self) -> Self {
        match (self, key) {
            (Self::List(held), Self::Integer(at)) => {
                let at = within(*at, held.len());
                held[at].clone()
            }
            (Self::Map(held), Self::Text(key)) => held
                .iter()
                .find(|(name, _)| name == key)
                .map_or(Self::Missing, |(_, value)| value.clone()),
            (Self::Namespace(held), Self::Text(key)) => held
                .borrow()
                .iter()
                .find(|(name, _)| name == key)
                .map_or(Self::Missing, |(_, value)| value.clone()),
            (Self::Text(held), Self::Integer(at)) => {
                let held = held.chars().collect::<Vec<char>>();
                let at = within(*at, held.len());
                Self::Text(held[at].to_string())
            }
            _ => panic!("a template reads the {} of {}", key.kind(), self.kind(),),
        }
    }

    pub(crate) fn slice(&self, start: Option<i64>, stop: Option<i64>, step: Option<i64>) -> Self {
        let Self::List(held) = self else {
            panic!("a template slices {} where a list stands", self.kind());
        };
        let step = step.unwrap_or(1);
        assert!(step != 0, "a template slices a list by no step at all");
        let len = held.len() as i64;
        let bound = |at: i64| if at < 0 { at + len } else { at };
        let mut out = Vec::new();
        if step > 0 {
            let first = start.map_or(0, |at| bound(at).clamp(0, len));
            let last = stop.map_or(len, |at| bound(at).clamp(0, len));
            let mut at = first;
            while at < last {
                out.push(held[at as usize].clone());
                at += step;
            }
        } else {
            let first = start.map_or(len - 1, |at| bound(at).clamp(-1, len - 1));
            let last = stop.map_or(-len - 1, bound);
            let mut at = first;
            while at > last {
                if at >= 0 && at < len {
                    out.push(held[at as usize].clone());
                }
                at += step;
            }
        }
        Self::List(out)
    }

    pub(crate) fn contains(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Text(held), Self::Text(other)) => held.contains(other.as_str()),
            (Self::List(held), other) => held.iter().any(|item| item.equals(other)),
            (Self::Map(held), Self::Text(key)) => held.iter().any(|(name, _)| name == key),
            (Self::Namespace(held), Self::Text(key)) => {
                held.borrow().iter().any(|(name, _)| name == key)
            }
            _ => panic!(
                "a template asks whether {} holds {}",
                self.kind(),
                other.kind(),
            ),
        }
    }

    pub(crate) fn length(&self) -> usize {
        match self {
            Self::Missing => 0,
            Self::Text(held) => held.chars().count(),
            Self::List(held) => held.len(),
            Self::Map(held) => held.len(),
            Self::Namespace(held) => held.borrow().len(),
            _ => panic!("a template measures the length of {}", self.kind()),
        }
    }

    pub(crate) fn method(&self, name: &str, arguments: &[Self]) -> Self {
        let Self::Text(held) = self else {
            panic!("a template calls {name} on {}", self.kind());
        };
        let first = || {
            arguments
                .first()
                .unwrap_or_else(|| panic!("a template calls {name} with no argument"))
                .text()
        };
        match name {
            "split" => Self::List(
                held.split(first().as_str())
                    .map(|piece| Self::Text(piece.to_owned()))
                    .collect(),
            ),
            "strip" | "lstrip" | "rstrip" => {
                let held = if arguments.is_empty() {
                    match name {
                        "strip" => held.trim().to_owned(),
                        "lstrip" => held.trim_start().to_owned(),
                        _ => held.trim_end().to_owned(),
                    }
                } else {
                    let set = first().chars().collect::<Vec<char>>();
                    match name {
                        "strip" => held
                            .trim_matches(|character| set.contains(&character))
                            .to_owned(),
                        "lstrip" => held
                            .trim_start_matches(|character| set.contains(&character))
                            .to_owned(),
                        _ => held
                            .trim_end_matches(|character| set.contains(&character))
                            .to_owned(),
                    }
                };
                Self::Text(held)
            }
            "startswith" => Self::Flag(held.starts_with(first().as_str())),
            "endswith" => Self::Flag(held.ends_with(first().as_str())),
            _ => panic!(
                "a template calls {name} on a string, and this renderer reads no such method"
            ),
        }
    }

    pub(crate) fn json(&self) -> String {
        match self {
            Self::Missing | Self::Empty => "null".to_owned(),
            Self::Flag(held) => held.to_string(),
            Self::Integer(held) => held.to_string(),
            Self::Number(held) => format!("{held}"),
            Self::Text(held) => {
                let mut out = String::from("\"");
                for character in held.chars() {
                    let code = character as u32;
                    match character {
                        '"' => out.push_str("\\\""),
                        '\\' => out.push_str("\\\\"),
                        '\n' => out.push_str("\\n"),
                        '\r' => out.push_str("\\r"),
                        '\t' => out.push_str("\\t"),
                        '<' => out.push_str("\\u003c"),
                        '>' => out.push_str("\\u003e"),
                        '&' => out.push_str("\\u0026"),
                        '\'' => out.push_str("\\u0027"),
                        _ if !(0x20..=0x7e).contains(&code) => {
                            if code <= 0xffff {
                                out.push_str(&format!("\\u{code:04x}"));
                            } else {
                                let held = code - 0x10000;
                                let high = 0xd800 + (held >> 10);
                                let low = 0xdc00 + (held & 0x3ff);
                                out.push_str(&format!("\\u{high:04x}\\u{low:04x}"));
                            }
                        }
                        held => out.push(held),
                    }
                }
                out.push('"');
                out
            }
            Self::List(held) => {
                let items = held.iter().map(Self::json).collect::<Vec<String>>();
                format!("[{}]", items.join(", "))
            }
            Self::Map(held) => {
                let mut entries = held.clone();
                entries.sort_by(|(left, _), (right, _)| left.cmp(right));
                let items = entries
                    .iter()
                    .map(|(key, value)| {
                        format!("{}: {}", Self::Text(key.clone()).json(), value.json())
                    })
                    .collect::<Vec<String>>();
                format!("{{{}}}", items.join(", "))
            }
            Self::Namespace(held) => Self::Map(held.borrow().clone()).json(),
        }
    }
}

fn within(at: i64, len: usize) -> usize {
    let held = if at < 0 { at + len as i64 } else { at };
    assert!(
        (0..len as i64).contains(&held),
        "a template reads the element {at} of a list of {len} elements",
    );
    held as usize
}

impl From<&str> for Value {
    fn from(held: &str) -> Self {
        Self::Text(held.to_owned())
    }
}

impl From<String> for Value {
    fn from(held: String) -> Self {
        Self::Text(held)
    }
}

impl From<bool> for Value {
    fn from(held: bool) -> Self {
        Self::Flag(held)
    }
}

impl From<i64> for Value {
    fn from(held: i64) -> Self {
        Self::Integer(held)
    }
}

impl From<f64> for Value {
    fn from(held: f64) -> Self {
        Self::Number(held)
    }
}

impl From<Vec<Value>> for Value {
    fn from(held: Vec<Value>) -> Self {
        Self::List(held)
    }
}
