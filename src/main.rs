use std::fs;

pub mod parser;
use parser::{Lexer, Parser, Tag};

fn main() {
    let source = fs::read_to_string("./src/main.kuu").expect("WUT?");

    // let mut parser = Parser::new(&source);
    // let ast = parser.parse_stmts();

    let mut lexer = Lexer::new(&source);
    println!("{:?}", lexer.next_token());
    println!("{:?}", lexer.next_token());
    println!("{:?}", lexer.expect(Tag::Assign));
    println!("{:?}", lexer.peek_token());
    println!("{:?}", lexer.next_token());
    println!("{:?}", lexer.next_token());
    println!("{:?}", lexer.next_token());
    // println!("{:?}", ast);
}
