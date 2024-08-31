use combine::{optional, token};
use combine::parser::char::{spaces, digit, char, letter, string};
use combine::{between, choice, many1, sep_by, ParseError, Parser};
use combine::parser::repeat::chainl1;
use combine::stream::Stream;
use combine::parser;

use crate::ast::{Expr, Prototype, Function};

fn parse_prototype<I>() -> impl Parser<I, Output = Prototype>
  where I: Stream<Token = char>,
        I::Error: ParseError<I::Token, I::Range, I::Position>,
{
  (
    many1(choice((letter(), digit(), char('_')))),
    between(
      char('('),
      char(')'),
      sep_by(many1(choice((letter(), digit(), char('_')))), char(' '))
    )
  )
    .map(|(id, args)| Prototype::new(id, args))
}
fn parse_extern<I>() -> impl Parser<I, Output = Prototype>
  where I: Stream<Token = char>,
        I::Error: ParseError<I::Token, I::Range, I::Position>,
{
  string("extern")
    .with(spaces())
    .with(parse_prototype())
}

/// Parses a function definition
pub fn parse_definition<I>() -> impl Parser<I, Output = Function>
  where I: Stream<Token = char>,
        I::Error: ParseError<I::Token, I::Range, I::Position>,
{
  println!("Parsing a function definition");
  (
    string("def"),
    spaces(),
    parse_prototype(),
    expression_parser()
  )
    .map(|(_, _, proto, body)| Function::new(proto, body))
}

fn parse_identifier_expr<I>() -> impl Parser<I, Output = Expr>
  where I: Stream<Token = char>,
        I::Error: ParseError<I::Token, I::Range, I::Position>,
{
  (
    many1(choice((letter(), digit(), char('_')))),
    optional(spaces()),
    optional(
      between(
        char('('),
        char(')'), sep_by(expr(), spaces())
      )
    )
  )
    .map(|(id, _, args)| {
      match args {
        Some(args) => Expr::Call {
          callee: id,
          args,
        },
        None => Expr::Variable(id),
      }
    })
}

fn parse_paren_expr<I>() -> impl Parser<I, Output = Expr>
where
    I: Stream<Token = char>,
    I::Error: ParseError<I::Token, I::Range, I::Position>,
{
  // between(char('('), char(')'), expr())
  between(char('('), char(')'), expr_parser())
}

fn integer_part<I>() -> impl Parser<I, Output = f64>
  where I: Stream<Token = char>,
        I::Error: ParseError<I::Token, I::Range, I::Position>,
{
  many1(digit()).map(|string: String| string.parse::<f64>().unwrap())
}

fn decimal_part<I>() -> impl Parser<I, Output = f64>
  where I: Stream<Token = char>,
        I::Error: ParseError<I::Token, I::Range, I::Position>,
{
  many1(digit()).map(|digits: String| {
    let mut decimal = "0.".to_owned();
    decimal.push_str(&digits);
    decimal.parse::<f64>().unwrap()
  })
}

fn decimal<I>() -> impl Parser<I, Output = f64>
  where I: Stream<Token = char>,
        I::Error: ParseError<I::Token, I::Range, I::Position>,
{
  optional(integer_part())
    .skip(char('.'))
    .and(optional(decimal_part()))
    .map(|(integer, decimal)| integer
    .unwrap_or(0.0) + decimal
    .unwrap_or(0.0))
}


pub fn parse_number_expr<I>() -> impl Parser<I, Output = Expr>
where
    I: Stream<Token = char>,
    I::Error: ParseError<I::Token, I::Range, I::Position>,
{
  decimal().map(Expr::Number)
}

fn expr_<'a, I>() -> impl Parser<I, Output = Expr>
  where I: Stream<Token = char>,
        I::Error: ParseError<I::Token, I::Range, I::Position>,
{

  let skip_spaces = || spaces().silent();

  choice((
    parse_number_expr(),
    parse_paren_expr(),
    parse_identifier_expr(),
  )).skip(skip_spaces())
}

parser!{
  pub(crate) fn expr[I]()(I) -> Expr
  where [I: Stream<Token = char>]
  {
    expr_()
  }
}

parser!{
  pub(crate) fn expr_parser[I]()(I) -> Expr
  where [I: Stream<Token = char>]
  {
    expression_parser()
  }
}


fn parse_primary<I>() -> impl Parser<I, Output = Expr>
where
    I: Stream<Token = char>,
    I::Error: ParseError<I::Token, I::Range, I::Position>,
{
  spaces()
    .with(
      choice((
        parse_number_expr(),
        parse_paren_expr(),
        parse_identifier_expr(),
      ))
    ).skip(spaces())
}

fn create_binop_node(c: char) -> impl Fn(Expr, Expr) -> Expr {
  move |l: Expr, r: Expr| Expr::BinOp {
      op: c,
      lhs: Box::new(l),
      rhs: Box::new(r),
  }
}

macro_rules! create_binop {
  ($c:expr) => {
    return create_binop_node($c)
  }
}

pub fn expression_parser<I>() -> impl Parser<I, Output = Expr>
where
    I: Stream<Token = char>,
    I::Error: ParseError<I::Token, I::Range, I::Position>,
{
    let factor = spaces()
      .with(parse_primary())
      .skip(spaces());

    let lt = token::<I>('<').map(|c| create_binop!(c));

    let div = token::<I>('/').map(|c| create_binop!(c));
    let mul = token::<I>('*').map(|c| create_binop!(c));

    // let term = chainl1(chainl1(chainl1(factor, lt), div), mul);

    let term = chainl1(chainl1(factor, div), mul);

    let sub = token::<I>('-').map(|c| create_binop!(c));
    let add = token::<I>('+').map(|c| create_binop!(c));

    let expr = chainl1(chainl1(chainl1(term, sub), add), lt);

    expr
}


#[cfg(test)]
mod tests {
  use combine::Parser;
  use crate::ast::*;
  use super::*;
  use Expr::*;

  #[test]
  fn test_lt_op_precedence() {
    let result = expression_parser().parse("3.0 < 4.0 * 2.0").unwrap().0;
    let expected = BinOp {
      op: '<',
      lhs: Box::new(Number(3.0)),
      rhs: Box::new(BinOp {
        op: '*',
        lhs: Box::new(Number(4.0)),
        rhs: Box::new(Number(2.0)),
      }),
    };

    assert_eq!(result, expected);
  }

  #[test]
  fn test_op_with_var() {
    let result = expression_parser().parse("3.0 + 4.0 * x").unwrap().0;
    let expected = BinOp {
      op: '+',
      lhs: Box::new(Number(3.0)),
      rhs: Box::new(BinOp {
        op: '*',
        lhs: Box::new(Number(4.0)),
        rhs: Box::new(Variable("x".to_string())),
      }),
    };

    assert_eq!(result, expected);
  }

  #[test]
  fn test_sub_and_add_op_precedence() {
    let result = expression_parser().parse("3.0 + 4.0 - 2.0").unwrap().0;
    let expected = BinOp {
      op: '+',
      lhs: Box::new(Number(3.0)),
      rhs: Box::new(BinOp {
        op: '-',
        lhs: Box::new(Number(4.0)),
        rhs: Box::new(Number(2.0)),
      }),
    };

    assert_eq!(result, expected);
  }

  #[test]
  fn test_div_and_mul_op_precedence() {
    let result = expression_parser().parse("3.0 * 4.0 / 2.0").unwrap().0;
    let expected = BinOp {
      op: '*',
      lhs: Box::new(Number(3.0)),
      rhs: Box::new(BinOp {
        op: '/',
        lhs: Box::new(Number(4.0)),
        rhs: Box::new(Number(2.0)),
      }),

    };

    assert_eq!(result, expected);
  }

  #[test]
  fn test_all_op_precedence() {
    let result = expression_parser().parse("3.0 + 4.0 * 2.0 / 2.0 - 1.0").unwrap().0;
    let expected = BinOp {
      op: '+',
      lhs: Box::new(Number(3.0)),
      rhs: Box::new(BinOp {
        op: '-',
        lhs: Box::new(BinOp {
          op: '*',
          lhs: Box::new(Number(4.0)),
          rhs: Box::new(BinOp {
            op: '/',
            lhs: Box::new(Number(2.0)),
            rhs: Box::new(Number(2.0)),
          }),
        }),
        rhs: Box::new(Number(1.0)),
      })
    };
    assert_eq!(result, expected);
  }

  #[test]
  fn test_op_with_paren() {
    let result = expression_parser().parse("(3.0 + 4.0) * 2.0").unwrap().0;
    let expected = BinOp {
      op: '*',
      lhs: Box::new(BinOp {
        op: '+',
        lhs: Box::new(Number(3.0)),
        rhs: Box::new(Number(4.0)),
      }),
      rhs: Box::new(Number(2.0)),
    };
    assert_eq!(result, expected);
  }

  #[test]
  fn test_simple_number() {
    let result = expression_parser().parse("32.1").unwrap().0;
    let expected = Number(32.1);
    assert_eq!(result, expected);
  }

  #[test]
  fn test_simple_op() {
    let result = expression_parser().parse("32.1 + 20.2").unwrap().0;
    let expected = BinOp {
      op: '+',
      lhs: Box::new(Number(32.1)),
      rhs: Box::new(Number(20.2)),
    };
    assert_eq!(result, expected);
  }

  #[test]
  fn test_parse_primary() {
    let result = parse_primary().parse(" 3.14  ").unwrap().0;
    assert_eq!(result, Expr::Number(3.14));
    let result = parse_primary().parse("44.2 ").unwrap().0;
    assert_eq!(result, Expr::Number(44.2));
  }

  #[test]
  fn test_number() {
    let result = parse_number_expr().parse("3.14").unwrap().0;
    assert_eq!(result, Expr::Number(3.14));
    let result = parse_number_expr().parse("30.0").unwrap().0;
    assert_eq!(result, Expr::Number(30.0));
  }

  #[test]
  fn test_paren_expr() {
    let result = parse_paren_expr().parse("(3.14)").unwrap().0;
    assert_eq!(result, Expr::Number(3.14));
  }

  #[test]
  fn test_parse_prototype() {
    let result = parse_prototype().parse("foo(x y z)").unwrap().0;
    let expected = Prototype::new("foo".to_string(), vec!["x".to_string(), "y".to_string(), "z".to_string()]);
    assert_eq!(result, expected);
  }

  #[test]
  fn test_parse_definition() {
    let result = parse_definition().parse("def foo(x y z) 3.14 + 0.2").unwrap().0;
    let expected = Function::new(
      Prototype::new("foo".to_string(), vec!["x".to_string(), "y".to_string(), "z".to_string()]),
      Expr::BinOp {
        op: '+',
        lhs: Box::new(Expr::Number(3.14)),
        rhs: Box::new(Expr::Number(0.2)),
      }
    );
    assert_eq!(result, expected);
    let result = parse_definition().parse("def foo() 3.14 + 0.2").unwrap().0;
    let expected = Function::new(
      Prototype::new("foo".to_string(), vec![]),
      Expr::BinOp { op: '+', lhs: Box::new(Expr::Number(3.14)), rhs: Box::new(Expr::Number(0.2)) },
    );
    assert_eq!(result, expected);
  }

  #[test]
  fn test_parse_definition_with_var() {

    use Expr::*;

    let result = parse_definition().parse("def foo(x y z) 3.14 + x * (y - z)").unwrap().0;
    let expected = Function::new(
      Prototype::new("foo".to_string(), vec!["x".to_string(), "y".to_string(), "z".to_string()]),
      BinOp {
        op: '+',
        lhs: Box::new(Number(3.14)),
        rhs: Box::new(BinOp {
          op: '*',
          lhs: Box::new(Variable("x".to_string())),
          rhs: Box::new(BinOp {
            op: '-',
            lhs: Box::new(Variable("y".to_string())),
            rhs: Box::new(Variable("z".to_string())),
          }),
        }),
      }
    );
    assert_eq!(result, expected);
  }

  #[test]
  fn test_parse_definition_with_call() {
    use Expr::*;
    let result = parse_definition().parse("def foo(x y) x + foo(y 4.0)").unwrap().0;
    let expected = Function::new(
      Prototype::new("foo".to_string(), vec!["x".to_string(), "y".to_string()]),
      BinOp {
        op: '+',
        lhs: Box::new(Variable("x".to_string())),
        rhs: Box::new(Call {
          callee: "foo".to_string(),
          args: vec![Variable("y".to_string()), Number(4.0)],
        }),
      }
    );
    assert_eq!(result, expected);
  }

  #[test]
  fn test_identifier() {
    let result = parse_identifier_expr().parse("foo").unwrap().0;
    assert_eq!(result, Expr::Variable("foo".to_string()));

    let result = parse_identifier_expr().parse("foo (bar 3.14)").unwrap().0;
    assert_eq!(result, Expr::Call {
      callee: "foo".to_string(),
      args: vec![Expr::Variable("bar".to_string()), Expr::Number(3.14)],
    });

    let result = parse_identifier_expr().parse("foo(bar 3.14)").unwrap().0;
    assert_eq!(result, Expr::Call {
      callee: "foo".to_string(),
      args: vec![Expr::Variable("bar".to_string()), Expr::Number(3.14)],
    });
  }
}