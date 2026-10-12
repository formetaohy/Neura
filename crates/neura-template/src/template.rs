use crate::lexer::lex;
use crate::parser::Parser;
use crate::render::Renderer;
use crate::syntax::Statement;
use crate::value::Value;
use neura_gguf::Archive;

pub struct Template {
    body: Vec<Statement>,
}

impl Template {
    pub fn read(source: &impl Archive) -> Self {
        Self::parse(source.metadata().text("tokenizer.chat_template"))
    }

    pub fn parse(source: &str) -> Self {
        Self {
            body: Parser::of(lex(source)).program(),
        }
    }

    pub fn render(&self, messages: &[Message], values: &[(&str, Value)]) -> String {
        Renderer::render(&self.body, messages, values)
    }
}

pub struct Message {
    role: String,
    content: Value,
    held: Vec<(String, Value)>,
}

impl Message {
    pub fn new(role: impl Into<String>, content: impl Into<Value>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            held: Vec::new(),
        }
    }

    pub fn with(mut self, name: &str, value: impl Into<Value>) -> Self {
        assert!(
            name != "role" && name != "content",
            "a message walks under its role and its content, and {name} names one of them",
        );
        self.held.push((name.to_owned(), value.into()));
        self
    }

    pub fn role(&self) -> &str {
        &self.role
    }

    pub fn content(&self) -> &Value {
        &self.content
    }

    pub fn held(&self) -> &[(String, Value)] {
        &self.held
    }
}
