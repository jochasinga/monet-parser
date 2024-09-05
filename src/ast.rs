use std::fmt::{self, Display, Formatter};
use num_bigint::BigInt;
use num_traits::cast::ToPrimitive;

#[derive(Debug, PartialEq)]
pub struct Prototype {
  pub name: String,
  pub args: Vec<String>,
}

impl Prototype {
  pub fn new(name: String, args: Vec<String>) -> Self {
    Self { name, args }
  }
  pub fn name(&self) -> String {
    self.name.clone()
  }
}

impl Default for Prototype {
  fn default() -> Self {
    Self {
      name: "tmp".to_string(),
      args: Vec::new(),
    }
  }

}

#[derive(Debug, Default, PartialEq)]
pub struct Function {
  pub prototype: Prototype,
  pub body: Expr,
}

impl Function {
  pub fn new(prototype: Prototype, body: Expr) -> Self {
    Self { prototype, body }
  }
}

#[derive(Debug, PartialEq)]
pub enum Expr {
  Int(BigInt),
  Uint(BigInt),
  Number(f64),
  Variable(String),
  BinOp {
    op: char,
    lhs: Box<Expr>,
    rhs: Box<Expr>,
  },
  Call {
    callee: String,
    args: Vec<Expr>,
  },
  Nothing,
}

impl Display for Expr {
  fn fmt(&self, f: &mut Formatter) -> fmt::Result {
    match self {
      Expr::Int(n) => write!(f, "Expr::Int({})", n),
      Expr::Uint(n) => write!(f, "Expr::Uint({})", n),
      Expr::Number(n) => {
        let n = format!("{:.4}", n);
        write!(f, "Expr::Number({})", n)
      },
      Expr::Variable(v) => write!(f, "Expr::Variable({})", v),
      Expr::BinOp { op, lhs, rhs } => {
        write!(f, "Expr::BinOp {{ op: '{}', lhs: Box::new({}), rhs: Box::new({}) }}", op, lhs, rhs)
      },
      Expr::Call { callee, args } => {
        let args = args.iter().map(|arg| arg.to_string()).collect::<Vec<String>>().join(" ");
        write!(f, "{} {}", callee, args)
      },
      Expr::Nothing => write!(f, ""),
    }
  }
}

impl Expr {
  pub fn eval(&self) -> f64 {
    match self {
      Expr::Int(n) => n.to_f64().unwrap(),
      Expr::Uint(n) => n.to_f64().unwrap(),
      Expr::Number(n) => *n,
      Expr::Variable(_) => 0.0,
      Expr::BinOp { op, lhs, rhs } => {
        let l = lhs.eval();
        let r = rhs.eval();
        match op {
          '+' => l + r,
          '-' => l - r,
          '*' => l * r,
          '/' => l / r,
          _ => 0.0,
        }
      },
      Expr::Call { callee, args } => {
        let callee = callee.clone();
        let args = args.iter().map(|arg| arg.eval()).collect::<Vec<f64>>();
        match callee.as_str() {
          "add" => args[0] + args[1],
          "sub" => args[0] - args[1],
          "mul" => args[0] * args[1],
          "div" => args[0] / args[1],
          _ => 0.0,
        }
      },
      Expr::Nothing => 0.0,
    }
  }
}

// impl ToString for Expr {
//   fn to_string(&self) -> String {
//     match self {
//       Expr::Number(n) => n.to_string(),
//       Expr::Variable(v) => v.clone(),
//       Expr::BinOp { op, lhs, rhs } => format!("{} {} {}", lhs.to_string(), op, rhs.to_string()),
//       Expr::Call { callee, args } => {
//         let args = args.iter().map(|arg| arg.to_string()).collect::<Vec<String>>().join(" ");
//         format!("{} {}", callee, args)
//       },
//       Expr::Nothing => "".to_string(),
//     }
//   }
// }

impl Default for Expr {
  fn default() -> Self {
    Self::Nothing
  }
}