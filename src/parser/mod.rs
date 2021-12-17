use std::borrow::Cow;
use std::convert::AsRef;
use std::mem;

const KEYWORDS: [(&'static str, Tag); 3] = [
    ("return", Tag::Return),
    ("if", Tag::If),
    ("while", Tag::While)
];

fn kw_or_ident(ident: &str) -> Tag {
    for (kw, tag) in KEYWORDS {
        if kw == ident {
            return tag;
        }
    }

    Tag::Identifier
}

#[derive(Debug)]
pub enum ParserError {
    Expected(Tag),

    MalformedFunction,
    MalformedExpression,

    UnknownByte(u8),
    UnknownToken,

    UnmatchedAssignment,

    A
}

#[derive(Debug, PartialEq)]
pub enum Tag {
    Return,
    If,
    While,

    Add,
    Sub,
    Mul,

    AddAssign,
    SubAssign,
    MulAssign,

    LT,
    GT,

    Assign,

    Open,
    Close,
    
    OpenSquare,
    CloseSquare,

    OpenCurly,
    CloseCurly,

    VerticalBar,

    Identifier,
    Float,
    Integer,

    Comma,
    Dot,

    EoF,
}

#[derive(Debug)]
pub struct Token {
    tag: Tag,

    start: usize,
    end: usize,
}

impl AsRef<Token> for Token {
    fn as_ref(&self) -> &Token {
        self
    }
}

#[derive(Debug)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Pow,

    LT,
}

#[derive(Debug)]
pub enum Assoc {
    Left,
    Right,
}

#[derive(Debug)]
pub enum Expr {
    Integer(i64),
    Float(f64),
    Variable(String),

    BinOp(Op, Box<Expr>, Box<Expr>),

    Function(Vec<String>, Vec<Stmt>),
    FunctionInline(Vec<String>, Box<Expr>),
}

#[derive(Debug)]
pub enum Stmt {
    Assign(Vec<Var>, Vec<Expr>),
    AssignOp(Op, Var, Expr),
    Return(Expr),
    If(Expr, Vec<Stmt>),
    While(Expr, Vec<Stmt>),
}

#[derive(Debug)]
pub enum Var {
    Simple(String),
    Lookup(Box<Var>, Box<Var>),
    Index(Box<Var>, Box<Expr>),
}

#[derive(Debug)]
pub struct Parser<'a> {
    source: Cow<'a, str>,
    index: usize,
    index_peek: usize,
    current_token: Option<Token>,

    depth_curly: usize,
}


#[derive(Debug)]
pub struct Lexer<'input> {
    source: Cow<'input, str>,
    offset: usize,
    peeked: Option<Token>,

    open_braces: usize,
    open_brackets: usize,
    open_parentheses: usize,
}

impl<'input> Lexer<'input> {
    pub fn new<T>(input: T) -> Self where T: Into<Cow<'input, str>> {
        Lexer {
            source: input.into(),
            offset: 0,
            peeked: None,

            open_braces: 0,
            open_brackets: 0,
            open_parentheses: 0,
        }
    }

    pub fn next_token(&mut self) -> Result<Token, ParserError> {
        match self.peeked.take() {
            Some(token) => return Ok(token),
            _ => (),
        }

        // Ignore whitespace characters
        let start = self.offset + self.source[self.offset..].bytes()
            .take_while(u8::is_ascii_whitespace)
            .count();
        
        let bytes = self.source.as_bytes();

        let (tag, len) = match bytes.get(start) {
            Some(&b'+') => match bytes.get(start + 1) {
                Some(&b'=') => (Tag::AddAssign, 2),
                _ => (Tag::Add, 1),
            },

            Some(&b'-') => match bytes.get(start + 1) {
                Some(&b'=') => (Tag::SubAssign, 2),
                _ => (Tag::Sub, 1),
            },

            Some(&b'=') => (Tag::Assign, 1),
            Some(&b'(') => (Tag::Open, 1),
            Some(&b')') => (Tag::Close, 1),
            Some(&b'{') => (Tag::OpenCurly, 1),
            Some(&b'}') => (Tag::CloseCurly, 1),
            Some(&b'|') => (Tag::VerticalBar, 1),
            Some(&b',') => (Tag::Comma, 1),
            Some(&b'.') => (Tag::Dot, 1),

            Some(&b) if b.is_ascii_alphabetic() => {
                let len = 1 + self.source[(start + 1)..].bytes()
                    .take_while(u8::is_ascii_alphanumeric)
                    .count();
                
                (kw_or_ident(&self.source[start..(start + len)]), len)
            },
            
            Some(&b) if b.is_ascii_digit() => {
                let len = 1 + self.source[(start + 1)..].bytes()
                    .take_while(u8::is_ascii_digit)
                    .count(); 
                
                (Tag::Integer, len)
            },

            Some(&b) => return Err(ParserError::UnknownByte(b)),

            None => (Tag::EoF, 0),
        };

        self.offset = start + len;

        Ok(Token { tag, start, end: start + len })
    }

    pub fn peek_token(&mut self) -> Result<&Token, ParserError> {
        if self.peeked.is_some() {
            Ok(self.peeked.as_ref().unwrap())
        } else {
            self.peeked = Some(self.next_token()?);
            Ok(self.peeked.as_ref().unwrap())
        }
    }

    pub fn expect(&mut self, tag: Tag) -> Result<Token, ParserError> {
        let token = self.next_token()?;

        if token.tag == tag {
            Ok(token)
        } else {
            assert!(self.peeked.is_none());
            self.peeked = Some(token);
            Err(ParserError::Expected(tag))
        }
    }
}


impl<'a> Parser<'a> {
    pub fn new<T>(source: T) -> Self where T: Into<Cow<'a, str>> {
        Parser {
            source: source.into(),
            
            index: 0,
            index_peek: 0,


            current_token: None,

            depth_curly: 0,
        }
    }

    fn read_token(&self, token: &Token) -> &str {
        unsafe { self.source.get_unchecked(token.start..token.end) }
    }

    pub fn peek(&mut self) -> Result<Token, ParserError> {
        let start = self.index + self.source[self.index..].bytes()
            .take_while(u8::is_ascii_whitespace)
            .count();

        let bytes = self.source.as_bytes();

        let (tag, len) = match bytes.get(start) {
            Some(&b'+') => match bytes.get(start + 1) {
                Some(&b'=') => (Tag::AddAssign, 2),
                _ => (Tag::Add, 1),
            },

            Some(&b'-') => match bytes.get(start + 1) {
                Some(&b'=') => (Tag::SubAssign, 2),
                _ => (Tag::Sub, 1),
            },

            Some(&b'(') => (Tag::Open, 1),
            Some(&b')') => (Tag::Close, 1),
            Some(&b'{') => (Tag::OpenCurly, 1),
            Some(&b'}') => (Tag::CloseCurly, 1),
            Some(&b'|') => (Tag::VerticalBar, 1),
            Some(&b',') => (Tag::Comma, 1),
            Some(&b'.') => (Tag::Dot, 1),

            Some(&b'=') => (Tag::Assign, 1),

            Some(&b) if b.is_ascii_alphabetic() => {
                let len = 1 + self.source[(start + 1)..].bytes()
                    .take_while(u8::is_ascii_alphanumeric)
                    .count();
                
                (kw_or_ident(&self.source[start..(start + len)]), len)
            },
            
            Some(&b) if b.is_ascii_digit() => {
                let len = 1 + self.source[(start + 1)..].bytes()
                    .take_while(u8::is_ascii_digit)
                    .count(); 
                
                (Tag::Integer, len)
            },

            Some(&b) => return Err(ParserError::UnknownByte(b)),

            None => (Tag::EoF, 0),
        };

        let end = start + len;
        let token = Token { tag, start, end };

        self.index_peek = end;

        Ok(token)
    }


    pub fn _peek(&mut self) -> Result<&Token, ParserError> {
        let start = self.index + self.source[self.index..].bytes()
            .take_while(u8::is_ascii_whitespace)
            .count();

        let bytes = self.source.as_bytes();

        let (tag, len) = match bytes.get(start) {
            Some(&b'+') => match bytes.get(start + 1) {
                Some(&b'=') => (Tag::AddAssign, 2),
                _ => (Tag::Add, 1),
            },

            Some(&b'-') => match bytes.get(start + 1) {
                Some(&b'=') => (Tag::SubAssign, 2),
                _ => (Tag::Sub, 1),
            },

            Some(&b'(') => (Tag::Open, 1),
            Some(&b')') => (Tag::Close, 1),
            Some(&b'{') => (Tag::OpenCurly, 1),
            Some(&b'}') => (Tag::CloseCurly, 1),
            Some(&b'|') => (Tag::VerticalBar, 1),
            Some(&b',') => (Tag::Comma, 1),
            Some(&b'.') => (Tag::Dot, 1),

            Some(&b'=') => (Tag::Assign, 1),

            Some(&b) if b.is_ascii_alphabetic() => {
                let len = 1 + self.source[(start + 1)..].bytes()
                    .take_while(u8::is_ascii_alphanumeric)
                    .count();
                
                (kw_or_ident(&self.source[start..(start + len)]), len)
            },
            
            Some(&b) if b.is_ascii_digit() => {
                let len = 1 + self.source[(start + 1)..].bytes()
                    .take_while(u8::is_ascii_digit)
                    .count(); 
                
                (Tag::Integer, len)
            },

            Some(&b) => return Err(ParserError::UnknownByte(b)),

            None => (Tag::EoF, 0),
        };

        

        let end = start + len;

        self.current_token = Some(Token { tag, start, end });

        Ok(self.current_token.as_ref().unwrap())
    }

    fn expect_(&mut self, tag: Tag) -> Result<Token, ParserError> {
        match mem::replace(&mut self.current_token, None) {
            Some(token) if token.tag == tag => Ok(token),
            Some(_) => Err(ParserError::Expected(tag)),
            None => match self.peek() {
                Ok(peek) if peek.tag == tag => Ok(peek),
                err => err,
            }
        }
    }

    pub fn expect(&mut self, tag: Tag) -> Result<Token, ParserError> {
        let token= self.peek()?;
        
        if token.tag == tag {
            self.index = token.end;
            Ok(token)
        } else {
            Err(ParserError::Expected(tag))
        }
    }

    pub fn consume(&mut self, token: &Token) {
        self.index = token.end;
    }

    /*

    Block is a list of statements surrounded by curly braces.

    */

    pub fn parse_block(&mut self) -> Result<Vec<Stmt>, ParserError> {
        self.expect(Tag::OpenCurly)?;
        self.depth_curly += 1;

        let body = self.parse_stmts()?;

        self.expect(Tag::CloseCurly)?;
        self.depth_curly -= 1;

        return Ok(body);
    }

    // funcbody ::= ‘(’ [parlist] ‘)’ block end
    // parlist ::= namelist [‘,’ ‘...’] | ‘...’
    // namelist ::= Name {‘,’ Name}

    pub fn parse_expr_func(&mut self) -> Result<Expr, ParserError> {
        self.expect(Tag::VerticalBar)?;

        let mut args = Vec::new();
        let mut peek = self.peek()?;
        
        match peek.tag {
            Tag::Identifier => {
                self.consume(&peek);
                args.push(self.read_token(&peek).into());

                loop {
                    peek = self.peek()?;
                    
                    if peek.tag == Tag::Comma {
                        self.consume(&peek);
                        let var = self.expect(Tag::Identifier)?;

                        args.push(self.read_token(&var).into());
                    } else {
                        break;
                    }
                }

                self.expect(Tag::VerticalBar)?;
            }

            Tag::VerticalBar => {
                self.consume(&peek);
            },

            _ => return Err(ParserError::MalformedFunction),
        }

        peek = self.peek()?;

        if peek.tag == Tag::OpenCurly {
            let body = self.parse_block()?;
            let func = Expr::Function(args, body);
            
            return Ok(func);
        } else {
            let expr = self.parse_expression()?;
            let func = Expr::FunctionInline(args, Box::new(expr));
            return Ok(func);
        }
    }

    /*
    
    compute_expr(min_prec):
        result = compute_atom()

        while cur token is a binary operator with precedence >= min_prec:
            prec, assoc = precedence and associativity of current token
            if assoc is left:
            next_min_prec = prec + 1
            else:
            next_min_prec = prec
            rhs = compute_expr(next_min_prec)
            result = compute operator(result, rhs)

        return result
        
    */

    pub fn parse_identifier(&mut self) -> Result<Expr, ParserError> {
        let token = self.expect(Tag::Identifier)?;
        Ok(Expr::Variable(self.read_token(&token).to_owned()))
    }

    pub fn parse_integer(&mut self) -> Result<Expr, ParserError> {
        let token = self.expect(Tag::Integer)?;
        Ok(Expr::Integer(self.read_token(&token).parse().unwrap()))
    }

    pub fn parse_atom(&mut self) -> Result<Expr, ParserError> {
        let token = self.peek()?;

        match token.tag {
            Tag::Integer     => self.parse_integer(),
            Tag::Identifier  => self.parse_identifier(),
            Tag::VerticalBar => self.parse_expr_func(),
            _ => return Err(ParserError::MalformedExpression)
        }
    }

    pub fn parse_expression_with_precedence(&mut self, min_prec: usize) -> Result<Expr, ParserError> {
        let mut result = self.parse_atom()?;


        

        /*
        
        compute_expr(min_prec):
            result = compute_atom()

            while cur token is a binary operator with precedence >= min_prec:
                prec, assoc = precedence and associativity of current token
                if assoc is left:
                next_min_prec = prec + 1
                else:
                next_min_prec = prec
                rhs = compute_expr(next_min_prec)
                result = compute operator(result, rhs)

            return result
        
        */

        loop {
            let token = self.peek()?;

            /*
            
                 or
                and
                <     >     <=    >=    ~=    ==
                |
                ~
                &
                <<    >>
                ..
                +     -
                *     /     //    %
                unary operators (not   #     -     ~)

            
            */

            let (op, prec, assoc) = match token.tag {
                Tag::Add => (Op::Add, 1, Assoc::Left),
                Tag::Sub => (Op::Sub, 1, Assoc::Left),
                Tag::Mul => (Op::Mul, 2, Assoc::Left),
                Tag::LT  => (Op::LT,  1, Assoc::Left),
                _ => break
            };

            if prec < min_prec {
                break;
            }

            self.consume(&token);

            let next_mic_prec = match assoc {
                Assoc::Left  => prec + 1,
                Assoc::Right => prec,
            };

            let rhs = self.parse_expression_with_precedence(next_mic_prec)?;
            result = Expr::BinOp(op, Box::new(result), Box::new(rhs));

        }
        
        Ok(result)
    }

    // var ::=  Name | prefixexp ‘[’ exp ‘]’ | prefixexp ‘.’ Name
    fn parse_var(&mut self) -> Result<Var, ParserError> {
        let ident = self.expect(Tag::Identifier)?;
        
        let mut var = Var::Simple(self.read_token(&ident).into());
        let mut peek = self.peek()?;

        if !matches!(peek.tag, Tag::Dot | Tag::OpenSquare) {
            return Ok(var);
        }

        loop {
            match peek.tag {

                Tag::Dot => {
                    self.consume(&peek);

                    let lookup = self.expect(Tag::Identifier)?;
                    let lookup_ident = self.read_token(&lookup);
                    let lookup_var = Var::Simple(lookup_ident.into());

                    println!("{:?}", self.peek()?);

                    var = Var::Lookup(Box::new(var), Box::new(lookup_var));
                },

                Tag::OpenSquare => {
                    self.consume(&peek);
                    
                    let lookup = self.parse_expression()?;
                    self.expect(Tag::CloseSquare)?;

                    let lookup_var = Var::Index(Box::new(var), Box::new(lookup));

                    var = lookup_var;
                },

                _ => break,
            }
            
            peek = self.peek()?;
        }

        Ok(var)
    }

    // varlist ::= var {‘,’ var}
    fn parse_varlist(&mut self) -> Result<Vec<Var>, ParserError> {
        let mut vars = Vec::new();
        
        let mut var = self.parse_var()?; 
        vars.push(var);

        while let Ok(_) = self.expect(Tag::Comma) {
            var = self.parse_var()?;
            vars.push(var);
        }

        Ok(vars)
    }

    // explist ::= exp {‘,’ exp}
    fn parse_explist(&mut self) -> Result<Vec<Expr>, ParserError> {
        let mut exprs = Vec::new();

        let mut expr = self.parse_expression()?;
        exprs.push(expr);

        while let Ok(_) = self.expect(Tag::Comma) {
            expr = self.parse_expression()?;
            exprs.push(expr);
        }

        Ok(exprs)
    }

    // stat ::= varlist ‘=’ explist
    // star ::= var '+=' exp
    pub fn parse_stmt_assignment(&mut self) -> Result<Stmt, ParserError> {
        let mut var = self.parse_var()?;
        let peek = self.peek()?;

        if peek.tag == Tag::AddAssign {
            self.consume(&peek);
            let expr = self.parse_expression()?;
            let stmt_assign_op = Stmt::AssignOp(Op::Add, var, expr);

            return Ok(stmt_assign_op);
        }
        
        let mut vars = Vec::new();
        vars.push(var);
        
        while let Ok(_) = self.expect(Tag::Comma) {
            var = self.parse_var()?;
            vars.push(var);
        }

        self.expect(Tag::Assign)?;
        let exprs = self.parse_explist()?;
        
        if vars.len() != exprs.len() {
            return Err(ParserError::UnmatchedAssignment);
        }

        let stmt_assign = Stmt::Assign(vars, exprs);

        Ok(stmt_assign)
    }

    pub fn parse_expression(&mut self) -> Result<Expr, ParserError> {
        self.parse_expression_with_precedence(0)
    }

    pub fn parse_return(&mut self) -> Result<Stmt, ParserError> {
        self.expect(Tag::Return)?;

        let value = self.parse_expression()?;

        Ok(Stmt::Return(value))
    }

    pub fn parse_if(&mut self) -> Result<Stmt, ParserError> {
        self.expect(Tag::If)?;

        let cond = self.parse_expression()?;
        let body = self.parse_block()?;

        Ok(Stmt::If(cond, body))
    }

    pub fn parse_while(&mut self) -> Result<Stmt, ParserError> {
        self.expect(Tag::While)?;

        let cond = self.parse_expression()?;
        let body = self.parse_block()?;

        Ok(Stmt::While(cond, body))
    }

    pub fn parse_stmts(&mut self) -> Result<Vec<Stmt>, ParserError> {
        let mut output = Vec::new();

        loop {
            let token = self.peek()?;

            if token.tag == Tag::EoF {
                break;
            }

            let stmt = match token.tag {
                Tag::Identifier => self.parse_stmt_assignment()?,

                Tag::If => self.parse_if()?,
                Tag::While => self.parse_while()?,

                Tag::Return => self.parse_return()?,

                Tag::CloseCurly if self.depth_curly > 0 => return Ok(output),

                _ => return Err(ParserError::UnknownToken)
            };

            output.push(stmt);
        }
        
        Ok(output)
    }
}

