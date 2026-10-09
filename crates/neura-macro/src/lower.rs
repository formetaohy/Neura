use proc_macro2::TokenStream;
use quote::quote;
use syn::{Expr, FnArg, Lit, Pat, Stmt};

pub(crate) fn function(source: &syn::ItemFn) -> syn::Result<TokenStream> {
    let signature = &source.sig;
    if !signature.generics.params.is_empty()
        || signature.asyncness.is_some()
        || signature.unsafety.is_some()
        || signature.abi.is_some()
    {
        return Err(syn::Error::new_spanned(
            signature,
            "device functions use concrete safe Rust signatures",
        ));
    }
    let name = signature.ident.to_string();
    let mut arguments = Vec::new();
    for argument in &signature.inputs {
        let FnArg::Typed(argument) = argument else {
            return Err(syn::Error::new_spanned(
                argument,
                "device functions have no receiver",
            ));
        };
        let Pat::Ident(ident) = argument.pat.as_ref() else {
            return Err(syn::Error::new_spanned(
                argument,
                "device arguments have simple names",
            ));
        };
        let arg_name = ident.ident.to_string();
        let ty = ty(&argument.ty)?;
        arguments.push(quote!(neura_ast::Argument {
            name: #arg_name.into(), ty: #ty,
        }));
    }
    let result = match &signature.output {
        syn::ReturnType::Default => quote!(None),
        syn::ReturnType::Type(_, returned) => {
            let ty = ty(returned)?;
            quote!(Some(#ty))
        }
    };
    let returns = matches!(source.sig.output, syn::ReturnType::Type(_, _));
    let body = tail_statements(&source.block.stmts, returns)?;
    Ok(quote!(neura_ast::Function {
        name: #name.into(),
        arguments: vec![#(#arguments),*],
        result: #result,
        body: vec![#(#body),*],
    }))
}

fn ty(source: &syn::Type) -> syn::Result<TokenStream> {
    match source {
        syn::Type::Path(path) if path.path.get_ident().is_some() => {
            let name = path
                .path
                .get_ident()
                .expect("a simple type has an identifier")
                .to_string();
            let name = if name == "Uvec3" {
                "uvec3"
            } else {
                name.as_str()
            };
            Ok(quote!(neura_ast::Type::Named(#name.into())))
        }
        syn::Type::Array(array) => {
            let element = ty(&array.elem)?;
            let length = expression(&array.len)?;
            Ok(quote!(neura_ast::Type::Array {
                element: Box::new(#element), length: Box::new(#length),
            }))
        }
        _ => Err(syn::Error::new_spanned(source, "unsupported device type")),
    }
}

pub(crate) fn statements(source: &[Stmt]) -> syn::Result<Vec<TokenStream>> {
    source.iter().map(statement).collect()
}

fn tail_statements(source: &[Stmt], returns: bool) -> syn::Result<Vec<TokenStream>> {
    let last = source.len().checked_sub(1);
    source
        .iter()
        .enumerate()
        .map(|(index, statement)| {
            if returns
                && last == Some(index)
                && let Stmt::Expr(expr, None) = statement
            {
                return tail_statement(expr);
            }
            self::statement(statement)
        })
        .collect()
}

fn tail_statement(source: &Expr) -> syn::Result<TokenStream> {
    if matches!(
        source,
        Expr::Return(_) | Expr::Loop(_) | Expr::While(_) | Expr::Break(_) | Expr::Continue(_)
    ) {
        return executable(source);
    }
    tail_expression(source)
}

fn tail_expression(source: &Expr) -> syn::Result<TokenStream> {
    let path = quote!(neura_ast::Statement);
    match source {
        Expr::If(expr) => if_statement(expr, true),
        Expr::Match(selection) => match_statement(selection, true),
        Expr::Block(block) => {
            let body = tail_statements(&block.block.stmts, true)?;
            Ok(quote!(#path::Block(vec![#(#body),*])))
        }
        other => {
            let value = expression(other)?;
            Ok(quote!(#path::Return(Some(#value))))
        }
    }
}

fn block_statements(block: &syn::Block, tail: bool) -> syn::Result<Vec<TokenStream>> {
    if tail {
        tail_statements(&block.stmts, true)
    } else {
        statements(&block.stmts)
    }
}

fn branch(source: &Expr, tail: bool) -> syn::Result<Vec<TokenStream>> {
    if let Expr::Block(block) = source {
        return block_statements(&block.block, tail);
    }
    Ok(vec![if tail {
        tail_statement(source)?
    } else {
        executable(source)?
    }])
}

fn if_statement(expr: &syn::ExprIf, tail: bool) -> syn::Result<TokenStream> {
    let path = quote!(neura_ast::Statement);
    let condition = expression(&expr.cond)?;
    let accept = block_statements(&expr.then_branch, tail)?;
    let reject = expr
        .else_branch
        .as_ref()
        .map_or_else(|| Ok(Vec::new()), |(_, source)| branch(source, tail))?;
    Ok(quote!(#path::If {
        condition: #condition,
        accept: vec![#(#accept),*],
        reject: vec![#(#reject),*],
    }))
}

fn match_statement(selection: &syn::ExprMatch, tail: bool) -> syn::Result<TokenStream> {
    let path = quote!(neura_ast::Statement);
    let selector = expression(&selection.expr)?;
    let mut arms = Vec::new();
    for arm in &selection.arms {
        if arm.guard.is_some() {
            return Err(syn::Error::new_spanned(
                arm,
                "device matches do not have guards",
            ));
        }
        let pattern = match &arm.pat {
            Pat::Wild(_) => quote!(neura_ast::Pattern::Default),
            Pat::Lit(lit) => {
                let Lit::Int(number) = &lit.lit else {
                    return Err(syn::Error::new_spanned(
                        lit,
                        "device cases are integer constants",
                    ));
                };
                let (value, _) = integer(number)?;
                quote!(neura_ast::Pattern::Integer(#value))
            }
            Pat::Path(path) if path.path.segments.len() > 1 => {
                let name = path
                    .path
                    .segments
                    .iter()
                    .map(|part| part.ident.to_string())
                    .collect::<Vec<_>>()
                    .join("::");
                quote!(neura_ast::Pattern::Constant(#name.into()))
            }
            _ => {
                return Err(syn::Error::new_spanned(
                    &arm.pat,
                    "device cases require a qualified Rust constant",
                ));
            }
        };
        let body = branch(&arm.body, tail)?;
        arms.push(quote!(neura_ast::Arm {
            pattern: #pattern, body: vec![#(#body),*],
        }));
    }
    Ok(quote!(#path::Match { selector: #selector, arms: vec![#(#arms),*] }))
}

fn statement(source: &Stmt) -> syn::Result<TokenStream> {
    match source {
        Stmt::Local(local) => {
            let Pat::Ident(ident) = &local.pat else {
                return Err(syn::Error::new_spanned(
                    &local.pat,
                    "device locals use named bindings",
                ));
            };
            let Some(init) = &local.init else {
                return Err(syn::Error::new_spanned(
                    local,
                    "a device local needs a value",
                ));
            };
            if init.diverge.is_some() {
                return Err(syn::Error::new_spanned(
                    local,
                    "device locals do not use let-else",
                ));
            }
            let name = ident.ident.to_string();
            let mutable = ident.mutability.is_some();
            let value = expression(&init.expr)?;
            Ok(quote!(neura_ast::Statement::Let {
                name: #name.into(), mutable: #mutable, value: #value,
            }))
        }
        Stmt::Expr(source, _) => executable(source),
        other => Err(syn::Error::new_spanned(
            other,
            "device functions contain only executable Rust",
        )),
    }
}

fn executable(source: &Expr) -> syn::Result<TokenStream> {
    let path = quote!(neura_ast::Statement);
    match source {
        Expr::If(expr) => if_statement(expr, false),
        Expr::Match(selection) => match_statement(selection, false),
        Expr::ForLoop(loop_) => {
            let Pat::Ident(variable) = loop_.pat.as_ref() else {
                return Err(syn::Error::new_spanned(
                    &loop_.pat,
                    "device loop counters have simple names",
                ));
            };
            let iterator = expression(&loop_.expr)?;
            let name = variable.ident.to_string();
            let body = statements(&loop_.body.stmts)?;
            Ok(quote!(#path::For {
                name: #name.into(), iterator: #iterator, body: vec![#(#body),*],
            }))
        }
        Expr::While(loop_) => {
            let condition = expression(&loop_.cond)?;
            let body = statements(&loop_.body.stmts)?;
            Ok(quote!(#path::While { condition: #condition, body: vec![#(#body),*] }))
        }
        Expr::Loop(loop_) => {
            let body = statements(&loop_.body.stmts)?;
            Ok(quote!(#path::Loop(vec![#(#body),*])))
        }
        Expr::Return(return_) => {
            let value = match &return_.expr {
                Some(value) => {
                    let value = expression(value)?;
                    quote!(Some(#value))
                }
                None => quote!(None),
            };
            Ok(quote!(#path::Return(#value)))
        }
        Expr::Break(_) => Ok(quote!(#path::Break)),
        Expr::Continue(_) => Ok(quote!(#path::Continue)),
        Expr::Assign(assign) => {
            let place = expression(&assign.left)?;
            let value = expression(&assign.right)?;
            Ok(quote!(#path::Assign { place: #place, value: #value, operator: None }))
        }
        Expr::Binary(binary) => {
            if let Some(op) = assignment(&binary.op) {
                let op = quote!(Some(#op));
                let place = expression(&binary.left)?;
                let value = expression(&binary.right)?;
                Ok(quote!(#path::Assign { place: #place, value: #value, operator: #op }))
            } else {
                let value = expression(source)?;
                Ok(quote!(#path::Expression(#value)))
            }
        }
        Expr::Block(block) => {
            let body = statements(&block.block.stmts)?;
            Ok(quote!(#path::Block(vec![#(#body),*])))
        }
        Expr::Paren(inner) => executable(&inner.expr),
        _ => {
            let value = expression(source)?;
            Ok(quote!(#path::Expression(#value)))
        }
    }
}

fn assignment(op: &syn::BinOp) -> Option<TokenStream> {
    let name = match op {
        syn::BinOp::AddAssign(_) => "Add",
        syn::BinOp::SubAssign(_) => "Subtract",
        syn::BinOp::MulAssign(_) => "Multiply",
        syn::BinOp::DivAssign(_) => "Divide",
        syn::BinOp::RemAssign(_) => "Modulo",
        syn::BinOp::BitAndAssign(_) => "BitAnd",
        syn::BinOp::BitOrAssign(_) => "BitOr",
        syn::BinOp::BitXorAssign(_) => "BitXor",
        syn::BinOp::ShlAssign(_) => "ShiftLeft",
        syn::BinOp::ShrAssign(_) => "ShiftRight",
        _ => return None,
    };
    let name = syn::Ident::new(name, proc_macro2::Span::call_site());
    Some(quote!(neura_ast::BinaryOperator::#name))
}

fn binary(op: &syn::BinOp) -> Option<TokenStream> {
    let name = match op {
        syn::BinOp::Add(_) => "Add",
        syn::BinOp::Sub(_) => "Subtract",
        syn::BinOp::Mul(_) => "Multiply",
        syn::BinOp::Div(_) => "Divide",
        syn::BinOp::Rem(_) => "Modulo",
        syn::BinOp::Eq(_) => "Equal",
        syn::BinOp::Ne(_) => "NotEqual",
        syn::BinOp::Lt(_) => "Less",
        syn::BinOp::Le(_) => "LessEqual",
        syn::BinOp::Gt(_) => "Greater",
        syn::BinOp::Ge(_) => "GreaterEqual",
        syn::BinOp::BitAnd(_) => "BitAnd",
        syn::BinOp::BitOr(_) => "BitOr",
        syn::BinOp::BitXor(_) => "BitXor",
        syn::BinOp::And(_) => "LogicalAnd",
        syn::BinOp::Or(_) => "LogicalOr",
        syn::BinOp::Shl(_) => "ShiftLeft",
        syn::BinOp::Shr(_) => "ShiftRight",
        _ => return None,
    };
    let name = syn::Ident::new(name, proc_macro2::Span::call_site());
    Some(quote!(neura_ast::BinaryOperator::#name))
}

pub(crate) fn expression(source: &Expr) -> syn::Result<TokenStream> {
    let path = quote!(neura_ast::Expression);
    match source {
        Expr::Lit(lit) => match &lit.lit {
            Lit::Int(value) => {
                let (value, kind) = integer(value)?;
                Ok(quote!(#path::Integer { value: #value, ty: #kind }))
            }
            Lit::Float(value) => {
                let value = value.base10_parse::<f32>()?;
                Ok(quote!(#path::Float(#value)))
            }
            Lit::Bool(value) => {
                let value = value.value;
                Ok(quote!(#path::Bool(#value)))
            }
            _ => Err(syn::Error::new_spanned(
                lit,
                "device literals are u32, i32, f32 or bool",
            )),
        },
        Expr::Path(expr_path) => {
            if expr_path
                .path
                .segments
                .iter()
                .any(|part| !part.arguments.is_empty())
            {
                return Err(syn::Error::new_spanned(
                    expr_path,
                    "device paths do not have type arguments",
                ));
            }
            let name = expr_path
                .path
                .segments
                .iter()
                .map(|part| part.ident.to_string())
                .collect::<Vec<_>>()
                .join("::");
            Ok(quote!(#path::Name(#name.into())))
        }
        Expr::Field(field) => {
            let base = expression(&field.base)?;
            let syn::Member::Named(member) = &field.member else {
                return Err(syn::Error::new_spanned(field, "device fields have names"));
            };
            let name = member.to_string();
            Ok(quote!(#path::Field { base: Box::new(#base), name: #name.into() }))
        }
        Expr::Index(index) => {
            let base = expression(&index.expr)?;
            let index = expression(&index.index)?;
            Ok(quote!(#path::Index { base: Box::new(#base), index: Box::new(#index) }))
        }
        Expr::Unary(unary) => {
            let op = match unary.op {
                syn::UnOp::Neg(_) => quote!(neura_ast::UnaryOperator::Negate),
                syn::UnOp::Not(_) => quote!(neura_ast::UnaryOperator::Not),
                _ => {
                    return Err(syn::Error::new_spanned(
                        unary,
                        "device unary operations are - or !",
                    ));
                }
            };
            let value = expression(&unary.expr)?;
            Ok(quote!(#path::Unary { op: #op, value: Box::new(#value) }))
        }
        Expr::Binary(binary_expr) => {
            let Some(op) = binary(&binary_expr.op) else {
                return Err(syn::Error::new_spanned(
                    binary_expr,
                    "compound assignments are statements",
                ));
            };
            let left = expression(&binary_expr.left)?;
            let right = expression(&binary_expr.right)?;
            Ok(quote!(#path::Binary { op: #op, left: Box::new(#left), right: Box::new(#right) }))
        }
        Expr::Call(call) => {
            let name = call_name(&call.func)?;
            let args = call
                .args
                .iter()
                .map(expression)
                .collect::<syn::Result<Vec<_>>>()?;
            Ok(quote!(#path::Call { name: #name.into(), arguments: vec![#(#args),*] }))
        }
        Expr::Repeat(repeat) => {
            let value = expression(&repeat.expr)?;
            let length = expression(&repeat.len)?;
            Ok(quote!(#path::Repeat { value: Box::new(#value), length: Box::new(#length) }))
        }
        Expr::Cast(cast) => {
            let value = expression(&cast.expr)?;
            let ty = ty(&cast.ty)?;
            Ok(quote!(#path::Cast { value: Box::new(#value), ty: #ty }))
        }
        Expr::Reference(reference) => {
            let value = expression(&reference.expr)?;
            Ok(quote!(#path::Reference(Box::new(#value))))
        }
        Expr::Paren(paren) => expression(&paren.expr),
        Expr::Group(group) => expression(&group.expr),
        _ => Err(syn::Error::new_spanned(
            source,
            "unsupported Rust device expression",
        )),
    }
}

fn call_name(source: &Expr) -> syn::Result<String> {
    let Expr::Path(path) = source else {
        return Err(syn::Error::new_spanned(
            source,
            "device calls use direct names",
        ));
    };
    let Some(ident) = path.path.get_ident() else {
        return Err(syn::Error::new_spanned(
            source,
            "device calls use direct names",
        ));
    };
    Ok(ident.to_string())
}

fn integer(number: &syn::LitInt) -> syn::Result<(u32, TokenStream)> {
    let kind = match number.suffix() {
        "" | "usize" => quote!(neura_ast::IntegerType::Inferred),
        "u32" => quote!(neura_ast::IntegerType::Unsigned),
        "i32" => quote!(neura_ast::IntegerType::Signed),
        _ => {
            return Err(syn::Error::new_spanned(
                number,
                "device integers are u32 or i32",
            ));
        }
    };
    let value = number.to_string();
    let digits = value
        .strip_suffix(number.suffix())
        .expect("a Rust integer has a suffix")
        .replace('_', "");
    let parsed = if let Some(hex) = digits.strip_prefix("0x") {
        u32::from_str_radix(hex, 16)
    } else if let Some(binary) = digits.strip_prefix("0b") {
        u32::from_str_radix(binary, 2)
    } else if let Some(octal) = digits.strip_prefix("0o") {
        u32::from_str_radix(octal, 8)
    } else {
        digits.parse::<u32>()
    };
    let value =
        parsed.map_err(|_| syn::Error::new_spanned(number, "device integer exceeds u32"))?;
    Ok((value, kind))
}
