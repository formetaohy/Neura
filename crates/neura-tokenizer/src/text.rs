#[derive(Default)]
pub struct Text {
    held: Vec<u8>,
}

impl Text {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, piece: &[u8]) -> String {
        self.held.extend_from_slice(piece);
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.held) {
                Ok(text) => {
                    out.push_str(text);
                    self.held.clear();
                    break;
                }
                Err(error) => {
                    let valid = error.valid_up_to();
                    if valid > 0 {
                        out.push_str(
                            std::str::from_utf8(&self.held[..valid])
                                .expect("a valid prefix reads as text"),
                        );
                        self.held.drain(..valid);
                    }
                    match error.error_len() {
                        Some(width) => {
                            out.push_str(&String::from_utf8_lossy(&self.held[..width]));
                            self.held.drain(..width);
                        }
                        None => break,
                    }
                }
            }
        }
        out
    }

    pub fn flush(&mut self) -> String {
        let out = String::from_utf8_lossy(&self.held).into_owned();
        self.held.clear();
        out
    }
}
