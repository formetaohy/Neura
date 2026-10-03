use super::JSON_DEPTH_CEILING;

pub(crate) enum Json {
    Object(Vec<(String, Json)>),
    Array(Vec<Json>),
    String(String),
    Number(f64),
    Bool,
    Null,
}

impl Json {
    pub(super) fn parse(text: &str) -> Self {
        let mut parser = Parser {
            bytes: text.as_bytes(),
            at: 0,
        };
        let value = parser.value(0);
        parser.whitespace();
        assert!(
            parser.at == parser.bytes.len(),
            "a container header holds {} bytes beyond its JSON",
            parser.bytes.len() - parser.at,
        );
        value
    }

    pub(super) fn kind(&self) -> &'static str {
        match self {
            Self::Object(_) => "an object",
            Self::Array(_) => "an array",
            Self::String(_) => "a string",
            Self::Number(_) => "a number",
            Self::Bool => "a boolean",
            Self::Null => "null",
        }
    }

    pub(super) fn into_integer(self, what: &str) -> u64 {
        let Json::Number(number) = self else {
            panic!("{what} holds {} where it holds a whole number", self.kind());
        };
        assert!(
            number.fract() == 0.0 && number >= 0.0 && number <= u64::MAX as f64,
            "{what} holds {number}, which is not a whole number",
        );
        number as u64
    }
}

struct Parser<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Parser<'_> {
    fn value(&mut self, depth: usize) -> Json {
        assert!(
            depth < JSON_DEPTH_CEILING,
            "a container header nests JSON beyond {JSON_DEPTH_CEILING} levels",
        );
        self.whitespace();
        match self.peek() {
            b'{' => self.object(depth),
            b'[' => self.array(depth),
            b'"' => Json::String(self.string()),
            b't' => {
                self.literal("true");
                Json::Bool
            }
            b'f' => {
                self.literal("false");
                Json::Bool
            }
            b'n' => {
                self.literal("null");
                Json::Null
            }
            b'-' | b'0'..=b'9' => self.number(),
            byte => panic!(
                "a container header holds {} where JSON holds a value",
                char::from(byte),
            ),
        }
    }

    fn object(&mut self, depth: usize) -> Json {
        self.take();
        let mut fields = Vec::new();
        self.whitespace();
        if self.peek() == b'}' {
            self.take();
            return Json::Object(fields);
        }
        loop {
            self.whitespace();
            let key = self.string();
            self.whitespace();
            assert!(
                self.take() == b':',
                "a container header holds a key where JSON holds a colon",
            );
            fields.push((key, self.value(depth + 1)));
            self.whitespace();
            match self.take() {
                b',' => continue,
                b'}' => return Json::Object(fields),
                other => panic!(
                    "a container header holds {} where JSON holds a comma or a brace",
                    char::from(other),
                ),
            }
        }
    }

    fn array(&mut self, depth: usize) -> Json {
        self.take();
        let mut items = Vec::new();
        self.whitespace();
        if self.peek() == b']' {
            self.take();
            return Json::Array(items);
        }
        loop {
            items.push(self.value(depth + 1));
            self.whitespace();
            match self.take() {
                b',' => continue,
                b']' => return Json::Array(items),
                other => panic!(
                    "a container header holds {} where JSON holds a comma or a bracket",
                    char::from(other),
                ),
            }
        }
    }

    fn string(&mut self) -> String {
        assert!(
            self.take() == b'"',
            "a container header holds text where JSON holds a string",
        );
        let mut text = Vec::new();
        loop {
            match self.take() {
                b'"' => {
                    return String::from_utf8(text).expect("a container header holds UTF-8");
                }
                b'\\' => {
                    let mut carried = [0u8; 4];
                    text.extend_from_slice(self.escape().encode_utf8(&mut carried).as_bytes());
                }
                byte if byte < 0x20 => {
                    panic!("a container header holds control byte {byte} inside a string",)
                }
                byte => text.push(byte),
            }
        }
    }

    fn escape(&mut self) -> char {
        match self.take() {
            b'"' => '"',
            b'\\' => '\\',
            b'/' => '/',
            b'b' => '\u{8}',
            b'f' => '\u{c}',
            b'n' => '\n',
            b'r' => '\r',
            b't' => '\t',
            b'u' => {
                let code = self.hex4();
                if (0xd800..0xdc00).contains(&code) {
                    assert!(
                        self.take() == b'\\' && self.take() == b'u',
                        "a container header holds a lone UTF-16 surrogate",
                    );
                    let low = self.hex4();
                    assert!(
                        (0xdc00..0xe000).contains(&low),
                        "a container header holds a high surrogate beside {low:x}",
                    );
                    return char::from_u32(0x1_0000 + ((code - 0xd800) << 10) + (low - 0xdc00))
                        .expect("a surrogate pair holds a character");
                }
                char::from_u32(code).expect("an escape holds a character")
            }
            other => panic!(
                "a container header holds an unknown escape \\{}",
                char::from(other),
            ),
        }
    }

    fn hex4(&mut self) -> u32 {
        let mut value = 0u32;
        for _ in 0..4 {
            let byte = self.take();
            let digit = char::from(byte).to_digit(16).unwrap_or_else(|| {
                panic!(
                    "a container header holds {} inside an escape",
                    char::from(byte)
                )
            });
            value = value * 16 + digit;
        }
        value
    }

    fn number(&mut self) -> Json {
        let start = self.at;
        if self.peek() == b'-' {
            self.take();
        }
        while self.peek().is_ascii_digit() {
            self.take();
        }
        if self.peek() == b'.' {
            self.take();
            while self.peek().is_ascii_digit() {
                self.take();
            }
        }
        if matches!(self.peek(), b'e' | b'E') {
            self.take();
            if matches!(self.peek(), b'+' | b'-') {
                self.take();
            }
            while self.peek().is_ascii_digit() {
                self.take();
            }
        }
        let text = std::str::from_utf8(&self.bytes[start..self.at])
            .expect("a container header holds UTF-8");
        let number = text.parse::<f64>().unwrap_or_else(|_| {
            panic!("a container header holds {text} where JSON holds a number")
        });
        Json::Number(number)
    }

    fn literal(&mut self, text: &str) {
        assert!(
            self.bytes[self.at..].starts_with(text.as_bytes()),
            "a container header holds {} where JSON holds {text}",
            String::from_utf8_lossy(&self.bytes[self.at..self.bytes.len().min(self.at + 5)]),
        );
        self.at += text.len();
    }

    fn whitespace(&mut self) {
        while matches!(self.peek(), b' ' | b'\n' | b'\r' | b'\t') {
            self.at += 1;
        }
    }

    fn peek(&self) -> u8 {
        self.bytes.get(self.at).copied().unwrap_or(0)
    }

    fn take(&mut self) -> u8 {
        let byte = self.bytes.get(self.at).copied().unwrap_or_else(|| {
            panic!("a container header ends inside its JSON");
        });
        self.at += 1;
        byte
    }
}
