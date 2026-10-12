use crate::lexer::Token;
use crate::syntax::{Compare, Expression, Statement};
use crate::value::Value;

pub(crate) struct Parser {
    tokens: Vec<Token>,
    at: usize,
}

impl Parser {
    pub(crate) fn of(tokens: Vec<Token>) -> Self {
        Self { tokens, at: 0 }
    }

    pub(crate) fn program(mut self) -> Vec<Statement> {
        let held = self.statements(&[]);
        if let Some(token) = self.peek() {
            panic!("a template holds {token:?} where its end stands");
        }
        held
    }

    fn statements(&mut self, ends: &[&str]) -> Vec<Statement> {
        let mut held = Vec::new();
        while let Some(token) = self.peek() {
            match token.clone() {
                Token::Body(text) => {
                    self.at += 1;
                    held.push(Statement::Body(text));
                }
                Token::Print => {
                    self.at += 1;
                    let held_expression = self.expression();
                    self.close();
                    held.push(Statement::Print(held_expression));
                }
                Token::Block => {
                    let name = self.tag_name().to_owned();
                    if ends.contains(&name.as_str()) {
                        return held;
                    }
                    held.push(self.statement(&name));
                }
                other => panic!("a template holds {other:?} where a body stands"),
            }
        }
        assert!(
            ends.is_empty(),
            "a template ends where its {} stands",
            ends.join(" or "),
        );
        held
    }

    fn statement(&mut self, name: &str) -> Statement {
        match name {
            "if" => self.if_statement(),
            "for" => self.for_statement(),
            "set" => self.set_statement(),
            "macro" => self.macro_statement(),
            other => {
                panic!("a template holds the {other} tag, and this renderer reads no such tag")
            }
        }
    }

    fn if_statement(&mut self) -> Statement {
        self.open("if");
        let mut arms = Vec::new();
        let condition = self.expression();
        self.close();
        let body = self.statements(&["elif", "else", "endif"]);
        arms.push((condition, body));
        loop {
            let name = self.tag_name().to_owned();
            match name.as_str() {
                "elif" => {
                    self.open("elif");
                    let condition = self.expression();
                    self.close();
                    let body = self.statements(&["elif", "else", "endif"]);
                    arms.push((condition, body));
                }
                "else" => {
                    self.open("else");
                    self.close();
                    let otherwise = self.statements(&["endif"]);
                    self.open("endif");
                    self.close();
                    return Statement::If { arms, otherwise };
                }
                "endif" => {
                    self.open("endif");
                    self.close();
                    return Statement::If {
                        arms,
                        otherwise: Vec::new(),
                    };
                }
                other => panic!("a template holds the {other} tag where an if ends"),
            }
        }
    }

    fn for_statement(&mut self) -> Statement {
        self.open("for");
        let name = self.name();
        self.expect_name("in");
        let source = self.expression();
        self.close();
        let body = self.statements(&["endfor"]);
        self.open("endfor");
        self.close();
        Statement::For { name, source, body }
    }

    fn set_statement(&mut self) -> Statement {
        self.open("set");
        let name = self.name();
        let attribute = if self.take_symbol(".") {
            Some(self.name())
        } else {
            None
        };
        self.expect_symbol("=");
        let value = self.expression();
        self.close();
        Statement::Set {
            name,
            attribute,
            value,
        }
    }

    fn macro_statement(&mut self) -> Statement {
        self.open("macro");
        let name = self.name();
        self.expect_symbol("(");
        let mut parameters = Vec::new();
        while !self.take_symbol(")") {
            let held = self.name();
            let default = if self.take_symbol("=") {
                Some(self.expression())
            } else {
                None
            };
            parameters.push((held, default));
            if !self.take_symbol(",") {
                self.expect_symbol(")");
                break;
            }
        }
        self.close();
        let body = self.statements(&["endmacro"]);
        self.open("endmacro");
        self.close();
        Statement::Macro {
            name,
            parameters,
            body,
        }
    }

    fn expression(&mut self) -> Expression {
        self.conditional()
    }

    fn conditional(&mut self) -> Expression {
        let accept = self.or();
        if !self.take_name("if") {
            return accept;
        }
        let condition = self.or();
        self.expect_name("else");
        let reject = self.conditional();
        Expression::Conditional {
            accept: Box::new(accept),
            condition: Box::new(condition),
            reject: Box::new(reject),
        }
    }

    fn or(&mut self) -> Expression {
        let mut held = self.and();
        while self.take_name("or") {
            held = Expression::Or(Box::new(held), Box::new(self.and()));
        }
        held
    }

    fn and(&mut self) -> Expression {
        let mut held = self.negation();
        while self.take_name("and") {
            held = Expression::And(Box::new(held), Box::new(self.negation()));
        }
        held
    }

    fn negation(&mut self) -> Expression {
        if self.take_name("not") {
            return Expression::Not(Box::new(self.negation()));
        }
        self.comparison()
    }

    fn comparison(&mut self) -> Expression {
        let mut held = self.addition();
        loop {
            let compare = if self.take_symbol("==") {
                Some(Compare::Equal)
            } else if self.take_symbol("!=") {
                Some(Compare::NotEqual)
            } else if self.take_symbol("<=") {
                Some(Compare::LessEqual)
            } else if self.take_symbol(">=") {
                Some(Compare::GreaterEqual)
            } else if self.take_symbol("<") {
                Some(Compare::Less)
            } else if self.take_symbol(">") {
                Some(Compare::Greater)
            } else if self.take_name("in") {
                Some(Compare::In)
            } else if self.at_name("not") && self.at_name_at(1, "in") {
                self.at += 2;
                Some(Compare::NotIn)
            } else if self.take_name("is") {
                let negated = self.take_name("not");
                let name = self.name();
                let positional = if self.take_symbol("(") {
                    self.arguments().0
                } else {
                    Vec::new()
                };
                held = Expression::Test {
                    input: Box::new(held),
                    name,
                    negated,
                    positional,
                };
                continue;
            } else {
                None
            };
            let Some(compare) = compare else { break };
            held = Expression::Compare {
                compare,
                left: Box::new(held),
                right: Box::new(self.addition()),
            };
        }
        held
    }

    fn addition(&mut self) -> Expression {
        let mut held = self.concat();
        loop {
            if self.take_symbol("+") {
                held = Expression::Add(Box::new(held), Box::new(self.concat()));
            } else if self.take_symbol("-") {
                held = Expression::Subtract(Box::new(held), Box::new(self.concat()));
            } else {
                break;
            }
        }
        held
    }

    fn concat(&mut self) -> Expression {
        let mut held = self.filtered();
        while self.take_symbol("~") {
            held = Expression::Concat(Box::new(held), Box::new(self.filtered()));
        }
        held
    }

    fn filtered(&mut self) -> Expression {
        let mut held = self.postfix();
        while self.take_symbol("|") {
            let name = self.name();
            let positional = if self.take_symbol("(") {
                self.arguments().0
            } else {
                Vec::new()
            };
            held = Expression::Filter {
                input: Box::new(held),
                name,
                positional,
            };
        }
        held
    }

    fn postfix(&mut self) -> Expression {
        let mut held = self.primary();
        loop {
            if self.take_symbol(".") {
                held = Expression::Attribute(Box::new(held), self.name());
            } else if self.take_symbol("[") {
                held = self.subscript(held);
            } else if self.take_symbol("(") {
                let (positional, keywords) = self.arguments();
                held = Expression::Call {
                    callee: Box::new(held),
                    positional,
                    keywords,
                };
            } else {
                break;
            }
        }
        held
    }

    fn subscript(&mut self, base: Expression) -> Expression {
        let start = if self.at_symbol(":") {
            None
        } else {
            Some(Box::new(self.expression()))
        };
        if !self.take_symbol(":") {
            self.expect_symbol("]");
            let Some(start) = start else {
                panic!("a template reads the element of a list no index names");
            };
            return Expression::Index(Box::new(base), start);
        }
        let stop = if self.at_symbol(":") || self.at_symbol("]") {
            None
        } else {
            Some(Box::new(self.expression()))
        };
        let step = if self.take_symbol(":") {
            Some(Box::new(self.expression()))
        } else {
            None
        };
        self.expect_symbol("]");
        Expression::Slice {
            base: Box::new(base),
            start,
            stop,
            step,
        }
    }

    fn arguments(&mut self) -> (Vec<Expression>, Vec<(String, Expression)>) {
        let mut positional = Vec::new();
        let mut keywords = Vec::new();
        if self.take_symbol(")") {
            return (positional, keywords);
        }
        loop {
            let keyword = matches!(self.peek(), Some(Token::Name(_))) && self.at_symbol_at(1, "=");
            if keyword {
                let name = self.name();
                self.expect_symbol("=");
                keywords.push((name, self.expression()));
            } else {
                positional.push(self.expression());
            }
            if self.take_symbol(",") {
                continue;
            }
            self.expect_symbol(")");
            return (positional, keywords);
        }
    }

    fn primary(&mut self) -> Expression {
        let Some(token) = self.peek().cloned() else {
            panic!("a template ends where an expression stands");
        };
        match token {
            Token::Name(name) => {
                self.at += 1;
                match name.as_str() {
                    "true" => Expression::Value(Value::Flag(true)),
                    "false" => Expression::Value(Value::Flag(false)),
                    "none" => Expression::Value(Value::Empty),
                    _ => Expression::Name(name),
                }
            }
            Token::Text(text) => {
                self.at += 1;
                Expression::Value(Value::Text(text))
            }
            Token::Whole(held) => {
                self.at += 1;
                Expression::Value(Value::Integer(held))
            }
            Token::Decimal(held) => {
                self.at += 1;
                Expression::Value(Value::Number(held))
            }
            Token::Symbol(symbol) if symbol == "-" => {
                self.at += 1;
                Expression::Negate(Box::new(self.negation()))
            }
            Token::Symbol(symbol) if symbol == "+" => {
                self.at += 1;
                self.negation()
            }
            Token::Symbol(symbol) if symbol == "(" => {
                self.at += 1;
                let held = self.expression();
                self.expect_symbol(")");
                held
            }
            other => panic!("a template holds {other:?} where an expression stands"),
        }
    }

    fn tag_name(&self) -> &str {
        match self.peek() {
            Some(Token::Block) => {}
            other => panic!("a template holds {other:?} where a tag stands"),
        }
        match self.tokens.get(self.at + 1) {
            Some(Token::Name(name)) => name,
            other => panic!("a template holds {other:?} where a tag names itself"),
        }
    }

    fn open(&mut self, name: &str) {
        let held = self.tag_name();
        assert_eq!(held, name, "a template closes the {name} tag with {held}",);
        self.at += 1;
        self.expect_name(name);
    }

    fn close(&mut self) {
        match self.peek() {
            Some(Token::Close) => self.at += 1,
            other => panic!("a template holds {other:?} where a tag ends"),
        }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.at)
    }

    fn name(&mut self) -> String {
        let Some(Token::Name(name)) = self.peek().cloned() else {
            panic!("a template holds {:?} where a name stands", self.peek());
        };
        self.at += 1;
        name
    }

    fn expect_name(&mut self, name: &str) {
        let held = self.name();
        assert_eq!(held, name, "a template holds {held} where {name} stands");
    }

    fn take_name(&mut self, name: &str) -> bool {
        if self.at_name(name) {
            self.at += 1;
            return true;
        }
        false
    }

    fn at_name(&self, name: &str) -> bool {
        matches!(self.peek(), Some(Token::Name(held)) if held == name)
    }

    fn at_name_at(&self, skip: usize, name: &str) -> bool {
        matches!(self.tokens.get(self.at + skip), Some(Token::Name(held)) if held == name)
    }

    fn at_symbol_at(&self, skip: usize, symbol: &str) -> bool {
        matches!(self.tokens.get(self.at + skip), Some(Token::Symbol(held)) if held == symbol)
    }

    fn expect_symbol(&mut self, symbol: &str) {
        let held = self.peek().cloned();
        let Some(Token::Symbol(symbol_held)) = held else {
            panic!("a template holds {held:?} where {symbol} stands");
        };
        assert_eq!(
            symbol_held, symbol,
            "a template holds {symbol_held} where {symbol} stands"
        );
        self.at += 1;
    }

    fn take_symbol(&mut self, symbol: &str) -> bool {
        if self.at_symbol(symbol) {
            self.at += 1;
            return true;
        }
        false
    }

    fn at_symbol(&self, symbol: &str) -> bool {
        matches!(self.peek(), Some(Token::Symbol(held)) if held == symbol)
    }
}
