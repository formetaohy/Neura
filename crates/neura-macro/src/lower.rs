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
        arguments.push(quote!(neura_ir::Argument {
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
    let body = statements(&source.block.stmts)?;
    Ok(quote!(neura_ir::Function {
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
            Ok(quote!(neura_ir::Type::Named(#name.into())))
        }
        syn::Type::Array(array) => {
            let element = ty(&array.elem)?;
            let length = expression(&array.len)?;
            Ok(quote!(neura_ir::Type::Array {
                element: Box::new(#element), length: Box::new(#length),
            }))
        }
        _ => Err(syn::Error::new_spanned(source, "unsupported device type")),
    }
}

pub(crate) fn statements(source: &[Stmt]) -> syn::Result<Vec<TokenStream>> {
    source.iter().map(statement).collect()
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
            Ok(quote!(neura_ir::Statement::Let {
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

fn branch(source: &Expr) -> syn::Result<Vec<TokenStream>> {
    if let Expr::Block(block) = source {
        statements(&block.block.stmts)
    } else {
        Ok(vec![executable(source)?])
    }
}

fn executable(source: &Expr) -> syn::Result<TokenStream> {
    let path = quote!(neura_ir::Statement);
    match source {
        Expr::If(expr) => {
            let condition = expression(&expr.cond)?;
            let accept = statements(&expr.then_branch.stmts)?;
            let reject = expr
                .else_branch
                .as_ref()
                .map_or_else(|| Ok(Vec::new()), |(_, branch)| self::branch(branch))?;
            Ok(quote!(#path::If {
                condition: #condition,
                accept: vec![#(#accept),*],
                reject: vec![#(#reject),*],
            }))
        }
        Expr::Match(selection) => {
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
                    Pat::Wild(_) => quote!(neura_ir::Pattern::Default),
                    Pat::Lit(lit) => {
                        let Lit::Int(number) = &lit.lit else {
                            return Err(syn::Error::new_spanned(
                                lit,
                                "device cases are integer constants",
                            ));
                        };
                        let (value, _) = integer(number)?;
                        quote!(neura_ir::Pattern::Integer(#value))
                    }
                    Pat::Path(path) if path.path.segments.len() > 1 => {
                        let name = path
                            .path
                            .segments
                            .iter()
                            .map(|part| part.ident.to_string())
                            .collect::<Vec<_>>()
                            .join("::");
                        quote!(neura_ir::Pattern::Constant(#name.into()))
                    }
                    _ => {
                        return Err(syn::Error::new_spanned(
                            &arm.pat,
                            "device cases require a qualified Rust constant",
                        ));
                    }
                };
                let body = branch(&arm.body)?;
                arms.push(quote!(neura_ir::Arm {
                    pattern: #pattern, body: vec![#(#body),*],
                }));
            }
            Ok(quote!(#path::Match { selector: #selector, arms: vec![#(#arms),*] }))
        }
        Expr::ForLoop(loop_) => {
            let Pat::Ident(variable) = loop_.pat.as_ref() else {
                return Err(syn::Error::new_spanned(
                    &loop_.pat,
                    "device loop counters have simple names",
                ));
            };
            let Expr::Call(call) = loop_.expr.as_ref() else {
                return Err(syn::Error::new_spanned(
                    &loop_.expr,
                    "device loops use stride or unroll",
                ));
            };
            let name = call_name(&call.func)?;
            if call.args.len() != 3 || !matches!(name.as_str(), "stride" | "unroll") {
                return Err(syn::Error::new_spanned(
                    &loop_.expr,
                    "device loops use stride(start, end, step) or unroll(start, end, step)",
                ));
            }
            let start = expression(&call.args[0])?;
            let end = expression(&call.args[1])?;
            let step = expression(&call.args[2])?;
            let name = variable.ident.to_string();
            let unroll = call_name(&call.func)? == "unroll";
            let body = statements(&loop_.body.stmts)?;
            Ok(quote!(#path::For {
                name: #name.into(), start: #start, end: #end, step: #step,
                unroll: #unroll, body: vec![#(#body),*],
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
    Some(quote!(neura_ir::BinaryOperator::#name))
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
    Some(quote!(neura_ir::BinaryOperator::#name))
}

pub(crate) fn expression(source: &Expr) -> syn::Result<TokenStream> {
    let path = quote!(neura_ir::Expression);
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
                syn::UnOp::Neg(_) => quote!(neura_ir::UnaryOperator::Negate),
                syn::UnOp::Not(_) => quote!(neura_ir::UnaryOperator::Not),
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
        "" | "usize" => quote!(neura_ir::IntegerType::Inferred),
        "u32" => quote!(neura_ir::IntegerType::Unsigned),
        "i32" => quote!(neura_ir::IntegerType::Signed),
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
