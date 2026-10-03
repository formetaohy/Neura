use std::fmt::Display;

pub struct Source {
    out: String,
    indent: usize,
}

impl Source {
    pub fn new() -> Self {
        Self {
            out: String::new(),
            indent: 0,
        }
    }

    pub fn line(&mut self, text: impl Display) {
        for _ in 0..self.indent {
            self.out.push_str("    ");
        }
        self.out.push_str(&text.to_string());
        self.out.push('\n');
    }

    pub fn raw(&mut self, text: &str) {
        self.out.push_str(text);
    }

    pub fn enter(&mut self) {
        self.indent += 1;
    }

    pub fn open(&mut self, text: impl Display) {
        self.line(format!("{text} {{"));
        self.indent += 1;
    }

    pub fn close(&mut self, text: impl Display) {
        self.indent -= 1;
        self.line(text);
    }

    pub fn finish(self) -> String {
        self.out
    }
}
