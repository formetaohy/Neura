use crate::syntax::{Compare, Expression, Statement};
use crate::template::Message;
use crate::value::Value;
use std::cell::RefCell;
use std::rc::Rc;

struct Macro<'t> {
    name: &'t str,
    parameters: &'t [(String, Option<Expression>)],
    body: &'t [Statement],
}

pub(crate) struct Renderer<'t> {
    scopes: Vec<Vec<(String, Value)>>,
    macros: Vec<Macro<'t>>,
    out: String,
}

impl<'t> Renderer<'t> {
    pub(crate) fn render(
        body: &'t [Statement],
        messages: &[Message],
        values: &[(&str, Value)],
    ) -> String {
        let mut held = Self {
            scopes: vec![Vec::new()],
            macros: Vec::new(),
            out: String::new(),
        };
        let held_messages = messages
            .iter()
            .map(|message| {
                let mut entries = vec![
                    ("role".to_owned(), Value::Text(message.role().to_owned())),
                    ("content".to_owned(), message.content().clone()),
                ];
                entries.extend(message.held().iter().cloned());
                Value::Map(entries)
            })
            .collect::<Vec<Value>>();
        held.bind("messages", Value::List(held_messages));
        for (name, value) in values {
            held.bind(name, value.clone());
        }
        held.run(body);
        held.out
    }

    fn run(&mut self, body: &'t [Statement]) {
        for held in body {
            self.statement(held);
        }
    }

    fn statement(&mut self, held: &'t Statement) {
        match held {
            Statement::Body(text) => self.out.push_str(text),
            Statement::Print(expression) => {
                let value = self.expression(expression);
                self.out.push_str(&value.text());
            }
            Statement::If { arms, otherwise } => {
                for (condition, body) in arms {
                    if self.expression(condition).truth() {
                        self.run(body);
                        return;
                    }
                }
                self.run(otherwise);
            }
            Statement::For { name, source, body } => {
                let walked = self.expression(source);
                let items = match walked {
                    Value::List(items) => items,
                    Value::Map(entries) => entries
                        .into_iter()
                        .map(|(key, _)| Value::Text(key))
                        .collect::<Vec<Value>>(),
                    Value::Text(text) => text
                        .chars()
                        .map(|character| Value::Text(character.to_string()))
                        .collect::<Vec<Value>>(),
                    other => panic!("a template walks {} where a list stands", other.kind()),
                };
                let length = items.len();
                for at in 0..length {
                    let loop_value = loop_value(&items, at);
                    self.scopes.push(vec![
                        (name.clone(), items[at].clone()),
                        ("loop".to_owned(), loop_value),
                    ]);
                    self.run(body);
                    self.scopes.pop();
                }
            }
            Statement::Set {
                name,
                attribute,
                value,
            } => {
                let value = self.expression(value);
                match attribute {
                    Some(attribute) => self.lookup(name).assign(attribute, value),
                    None => self.bind(name, value),
                }
            }
            Statement::Macro {
                name,
                parameters,
                body,
            } => self.macros.push(Macro {
                name,
                parameters,
                body,
            }),
        }
    }

    fn expression(&mut self, held: &Expression) -> Value {
        match held {
            Expression::Value(value) => value.clone(),
            Expression::Name(name) => self.lookup(name),
            Expression::Attribute(base, name) => self.expression(base).attribute(name),
            Expression::Index(base, key) => {
                let base = self.expression(base);
                let key = self.expression(key);
                base.subscript(&key)
            }
            Expression::Slice {
                base,
                start,
                stop,
                step,
            } => {
                let base = self.expression(base);
                let start = start.as_ref().map(|held| self.whole(held));
                let stop = stop.as_ref().map(|held| self.whole(held));
                let step = step.as_ref().map(|held| self.whole(held));
                base.slice(start, stop, step)
            }
            Expression::Call {
                callee,
                positional,
                keywords,
            } => {
                let positional = positional
                    .iter()
                    .map(|held| self.expression(held))
                    .collect::<Vec<Value>>();
                let keywords = keywords
                    .iter()
                    .map(|(name, held)| (name.clone(), self.expression(held)))
                    .collect::<Vec<(String, Value)>>();
                self.call(callee, &positional, &keywords)
            }
            Expression::Filter {
                input,
                name,
                positional,
            } => {
                let input = self.expression(input);
                let positional = positional
                    .iter()
                    .map(|held| self.expression(held))
                    .collect::<Vec<Value>>();
                self.filter(input, name, &positional)
            }
            Expression::Test {
                input,
                name,
                negated,
                positional,
            } => {
                let input = self.expression(input);
                let positional = positional
                    .iter()
                    .map(|held| self.expression(held))
                    .collect::<Vec<Value>>();
                let held = holds(&input, name, &positional);
                Value::Flag(if *negated { !held } else { held })
            }
            Expression::Conditional {
                accept,
                condition,
                reject,
            } => {
                if self.expression(condition).truth() {
                    self.expression(accept)
                } else {
                    self.expression(reject)
                }
            }
            Expression::Negate(held) => match self.expression(held) {
                Value::Integer(held) => Value::Integer(-held),
                Value::Number(held) => Value::Number(-held),
                other => panic!("a template negates {}", other.kind()),
            },
            Expression::Not(held) => Value::Flag(!self.expression(held).truth()),
            Expression::And(left, right) => {
                let left = self.expression(left);
                if left.truth() {
                    self.expression(right)
                } else {
                    left
                }
            }
            Expression::Or(left, right) => {
                let left = self.expression(left);
                if left.truth() {
                    left
                } else {
                    self.expression(right)
                }
            }
            Expression::Compare {
                compare,
                left,
                right,
            } => {
                let left = self.expression(left);
                let right = self.expression(right);
                Value::Flag(self.compare(*compare, &left, &right))
            }
            Expression::Add(left, right) => {
                let left = self.expression(left);
                let right = self.expression(right);
                left.add(&right)
            }
            Expression::Subtract(left, right) => {
                let left = self.expression(left);
                let right = self.expression(right);
                self.subtract(left, right)
            }
            Expression::Concat(left, right) => {
                let left = self.expression(left);
                let right = self.expression(right);
                left.concat(&right)
            }
        }
    }

    fn call(
        &mut self,
        callee: &Expression,
        positional: &[Value],
        keywords: &[(String, Value)],
    ) -> Value {
        match callee {
            Expression::Attribute(base, name) => self.expression(base).method(name, positional),
            Expression::Name(name) => match name.as_str() {
                "namespace" => Value::Namespace(Rc::new(RefCell::new(keywords.to_vec()))),
                "raise_exception" => panic!(
                    "the template raises {}",
                    positional
                        .first()
                        .map_or_else(|| "an exception".to_owned(), Value::text,),
                ),
                _ => self.invoke(name, positional, keywords),
            },
            _ => panic!(
                "a template calls a value this renderer reads no callee of, and it calls macros and functions by name",
            ),
        }
    }

    fn invoke(&mut self, name: &str, positional: &[Value], keywords: &[(String, Value)]) -> Value {
        let Some((parameters, body)) = self
            .macros
            .iter()
            .rev()
            .find(|held| held.name == name)
            .map(|held| (held.parameters, held.body))
        else {
            panic!("a template calls {name}, and this renderer reads no such macro or function");
        };
        assert!(
            positional.len() <= parameters.len()
                && keywords.iter().all(|(keyword, _)| parameters
                    .iter()
                    .any(|(parameter, _)| parameter == keyword)),
            "the template calls the macro {name} with an argument it names no parameter for",
        );
        let mut scope = Vec::new();
        for (at, (parameter, default)) in parameters.iter().enumerate() {
            let value = positional
                .get(at)
                .cloned()
                .or_else(|| {
                    keywords
                        .iter()
                        .find(|(keyword, _)| keyword == parameter)
                        .map(|(_, value)| value.clone())
                })
                .or_else(|| default.as_ref().map(|held| self.expression(held)))
                .unwrap_or(Value::Missing);
            scope.push((parameter.clone(), value));
        }
        self.scopes.push(scope);
        let saved = std::mem::take(&mut self.out);
        self.run(body);
        let captured = std::mem::replace(&mut self.out, saved);
        self.scopes.pop();
        Value::Text(captured)
    }

    fn filter(&mut self, input: Value, name: &str, positional: &[Value]) -> Value {
        assert!(
            positional.is_empty(),
            "the template filters a value through the {name} filter, and this renderer reads no argument of it",
        );
        match name {
            "trim" => Value::Text(input.text().trim().to_owned()),
            "string" => Value::Text(input.text()),
            "safe" => input,
            "length" => Value::Integer(input.length() as i64),
            "tojson" => Value::Text(input.json()),
            other => panic!(
                "the template filters a value through {other}, and this renderer reads no such filter",
            ),
        }
    }

    fn compare(&self, compare: Compare, left: &Value, right: &Value) -> bool {
        match compare {
            Compare::Equal => left.equals(right),
            Compare::NotEqual => !left.equals(right),
            Compare::Less => right.greater(left),
            Compare::LessEqual => !left.greater(right),
            Compare::Greater => left.greater(right),
            Compare::GreaterEqual => !right.greater(left),
            Compare::In => right.contains(left),
            Compare::NotIn => !right.contains(left),
        }
    }

    fn subtract(&self, left: Value, right: Value) -> Value {
        match (left, right) {
            (Value::Integer(left), Value::Integer(right)) => Value::Integer(left - right),
            (Value::Integer(left), Value::Number(right)) => Value::Number(left as f64 - right),
            (Value::Number(left), Value::Integer(right)) => Value::Number(left - right as f64),
            (Value::Number(left), Value::Number(right)) => Value::Number(left - right),
            (left, right) => panic!("a template takes {} from {}", right.kind(), left.kind(),),
        }
    }

    fn whole(&mut self, held: &Expression) -> i64 {
        match self.expression(held) {
            Value::Integer(value) => value,
            other => panic!("a template slices a list by {}", other.kind()),
        }
    }

    fn bind(&mut self, name: &str, value: Value) {
        let scope = self
            .scopes
            .last_mut()
            .expect("a template renders within a scope");
        match scope.iter_mut().find(|(key, _)| key == name) {
            Some((_, slot)) => *slot = value,
            None => scope.push((name.to_owned(), value)),
        }
    }

    fn lookup(&self, name: &str) -> Value {
        for scope in self.scopes.iter().rev() {
            if let Some((_, value)) = scope.iter().find(|(key, _)| key == name) {
                return value.clone();
            }
        }
        Value::Missing
    }
}

fn loop_value(items: &[Value], at: usize) -> Value {
    let length = items.len();
    Value::Map(vec![
        ("index0".to_owned(), Value::Integer(at as i64)),
        ("index".to_owned(), Value::Integer(at as i64 + 1)),
        ("first".to_owned(), Value::Flag(at == 0)),
        ("last".to_owned(), Value::Flag(at + 1 == length)),
        ("length".to_owned(), Value::Integer(length as i64)),
        (
            "previtem".to_owned(),
            if at == 0 {
                Value::Missing
            } else {
                items[at - 1].clone()
            },
        ),
        (
            "nextitem".to_owned(),
            if at + 1 == length {
                Value::Missing
            } else {
                items[at + 1].clone()
            },
        ),
    ])
}

fn holds(value: &Value, name: &str, positional: &[Value]) -> bool {
    assert!(
        positional.is_empty(),
        "the template weighs a value against the {name} test, and this renderer reads no argument of it",
    );
    match name {
        "defined" => !matches!(value, Value::Missing),
        "undefined" => matches!(value, Value::Missing),
        "none" => matches!(value, Value::Empty),
        "string" => matches!(value, Value::Text(_)),
        "number" => matches!(value, Value::Integer(_) | Value::Number(_)),
        "integer" => matches!(value, Value::Integer(_)),
        "boolean" => matches!(value, Value::Flag(_)),
        "true" => matches!(value, Value::Flag(true)),
        "false" => matches!(value, Value::Flag(false)),
        "iterable" => matches!(value, Value::Text(_) | Value::List(_) | Value::Map(_)),
        "mapping" => matches!(value, Value::Map(_) | Value::Namespace(_)),
        "sequence" => matches!(value, Value::Text(_) | Value::List(_)),
        other => panic!(
            "the template weighs a value against the {other} test, and this renderer reads no such test",
        ),
    }
}
