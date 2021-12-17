use criterion::{black_box, criterion_group, criterion_main, Criterion};

#[path="../src/main.rs"]
mod kuu;

const SOURCE: &'static str = "

anna = |x, y, z| {
    a = x + y
    b = y + z
}

";

fn fibonacci(source: &str) -> Result<Vec<kuu::parser::Stmt>, kuu::parser::ParserError> {
    let mut parser = kuu::parser::Parser::new(source);
    parser.parse_stmts()
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("PARSER", |b| b.iter(|| fibonacci(black_box(SOURCE))));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);