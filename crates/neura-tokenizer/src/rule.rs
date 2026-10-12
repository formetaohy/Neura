pub trait Rule {
    fn pieces<'a>(&self, text: &'a str) -> Vec<&'a str>;
}
