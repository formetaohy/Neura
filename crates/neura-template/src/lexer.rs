#[derive(Clone, PartialEq, Debug)]
pub(crate) enum Token {
    Body(String),
    Print,
    Block,
    Close,
    Name(String),
    Text(String),
    Whole(i64),
    Decimal(f64),
    Symbol(String),
}

enum Tag {
    Print,
    Block,
    Comment,
}

impl Tag {
    fn closer(&self) -> &'static str {
        match self {
            Self::Print => "}}",
            Self::Block => "%}",
            Self::Comment => "#}",
        }
    }
}

pub(crate) fn lex(source: &str) -> Vec<Token> {
    let mut held = Vec::new();
    let mut body = String::new();
    let mut trim = false;
    let mut at = 0usize;
    while at < source.len() {
        let rest = &source[at..];
        let tag = if rest.starts_with("{{") {
            Some(Tag::Print)
        } else if rest.starts_with("{%") {
            Some(Tag::Block)
        } else if rest.starts_with("{#") {
            Some(Tag::Comment)
        } else {
            None
        };
        let Some(tag) = tag else {
            let character = rest
                .chars()
                .next()
                .expect("a byte of a template stands on a character");
            at += character.len_utf8();
            if trim {
                if character.is_whitespace() {
                    continue;
                }
                trim = false;
            }
            body.push(character);
            continue;
        };
        at += 2;
        let left = source[at..].starts_with('-');
        if left {
            at += 1;
        }
        let closer = tag.closer();
        let end = source[at..].find(closer).unwrap_or_else(|| {
            panic!("a template opens a tag at byte {at} and holds no {closer} that closes it");
        });
        let inner = &source[at..at + end];
        let right = inner.ends_with('-');
        let inner = inner.strip_suffix('-').unwrap_or(inner);
        at += end + closer.len();
        if left {
            while body.ends_with(char::is_whitespace) {
                body.pop();
            }
        }
        if !body.is_empty() {
            held.push(Token::Body(std::mem::take(&mut body)));
        }
        trim = right;
        match tag {
            Tag::Comment => {}
            Tag::Print => {
                held.push(Token::Print);
                held.extend(words(inner));
                held.push(Token::Close);
            }
            Tag::Block => {
                held.push(Token::Block);
                held.extend(words(inner));
                held.push(Token::Close);
            }
        }
    }
    if !body.is_empty() {
        held.push(Token::Body(body));
    }
    held
}

fn words(source: &str) -> Vec<Token> {
    let mut held = Vec::new();
    let mut at = 0usize;
    while at < source.len() {
        let character = source[at..]
            .chars()
            .next()
            .expect("a byte of a tag stands on a character");
        if character.is_whitespace() {
            at += character.len_utf8();
            continue;
        }
        if character == '\'' || character == '"' {
            let (text, next) = quoted(source, at, character);
            held.push(Token::Text(text));
            at = next;
            continue;
        }
        if character.is_ascii_digit() {
            let (token, next) = number(source, at);
            held.push(token);
            at = next;
            continue;
        }
        if character.is_alphabetic() || character == '_' {
            let start = at;
            while at < source.len() {
                let held = source[at..].chars().next().expect("a character");
                if !held.is_alphanumeric() && held != '_' {
                    break;
                }
                at += held.len_utf8();
            }
            held.push(Token::Name(source[start..at].to_owned()));
            continue;
        }
        let (symbol, next) = punctuation(source, at).unwrap_or_else(|| {
            panic!("a tag of a template holds the character {character} where this renderer reads no part of an expression");
        });
        held.push(Token::Symbol(symbol.to_owned()));
        at = next;
    }
    held
}

fn quoted(source: &str, at: usize, quote: char) -> (String, usize) {
    let mut held = String::new();
    let mut at = at + quote.len_utf8();
    while at < source.len() {
        let character = source[at..].chars().next().expect("a character");
        at += character.len_utf8();
        if character == quote {
            return (held, at);
        }
        if character != '\\' {
            held.push(character);
            continue;
        }
        let escaped = source[at..].chars().next().unwrap_or_else(|| {
            panic!("a template ends a string with a backslash that escapes nothing");
        });
        at += escaped.len_utf8();
        held.push(match escaped {
            'n' => '\n',
            't' => '\t',
            'r' => '\r',
            '0' => '\0',
            other => other,
        });
    }
    panic!("a template opens the string {held:?} and holds no {quote} that closes it");
}

fn number(source: &str, at: usize) -> (Token, usize) {
    let start = at;
    let mut cursor = at;
    let mut decimal = false;
    while cursor < source.len() {
        let character = source[cursor..].chars().next().expect("a character");
        if character.is_ascii_digit() {
            cursor += 1;
            continue;
        }
        if character == '.' && !decimal {
            decimal = true;
            cursor += 1;
            continue;
        }
        break;
    }
    let held = &source[start..cursor];
    if decimal {
        (
            Token::Decimal(held.parse().unwrap_or_else(|_| {
                panic!("a template holds {held} where a number stands");
            })),
            cursor,
        )
    } else {
        (
            Token::Whole(held.parse().unwrap_or_else(|_| {
                panic!("a template holds {held} where a whole number stands");
            })),
            cursor,
        )
    }
}

fn punctuation(source: &str, at: usize) -> Option<(&'static str, usize)> {
    for symbol in ["**", "//", "==", "!=", "<=", ">="] {
        if source[at..].starts_with(symbol) {
            return Some((symbol, at + symbol.len()));
        }
    }
    let character = source[at..].chars().next()?;
    let held = "+-*/%~=,.:|()[]<>"
        .contains(character)
        .then_some(character)?;
    Some((
        match held {
            '+' => "+",
            '-' => "-",
            '*' => "*",
            '/' => "/",
            '%' => "%",
            '~' => "~",
            '=' => "=",
            ',' => ",",
            '.' => ".",
            ':' => ":",
            '|' => "|",
            '(' => "(",
            ')' => ")",
            '[' => "[",
            ']' => "]",
            '<' => "<",
            '>' => ">",
            _ => unreachable!("the test above holds every symbol of this list"),
        },
        at + held.len_utf8(),
    ))
}
