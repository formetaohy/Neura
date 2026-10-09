mod lower;

use proc_macro::TokenStream;
use proc_macro2::TokenStream as Tokens;
use quote::quote;
use std::collections::BTreeSet;
use syn::parse::{Parse, ParseStream, Parser};
use syn::visit_mut::{self, VisitMut};
use syn::{FnArg, GenericArgument, Pat, PathArguments, Token};

struct KernelOptions {
    size: u32,
}

impl Parse for KernelOptions {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let name: syn::Ident = input.parse()?;
        if name != "workgroup_size" {
            return Err(syn::Error::new_spanned(
                name,
                "kernel attributes specify workgroup_size",
            ));
        }
        input.parse::<Token![=]>()?;
        let size: syn::LitInt = input.parse()?;
        let size = size.base10_parse::<u32>()?;
        if size == 0 || !input.is_empty() {
            return Err(input.error("a kernel has one positive workgroup size"));
        }
        Ok(Self { size })
    }
}

struct Shared {
    name: syn::Ident,
    element: syn::Ident,
    count: u32,
}

struct DeviceModule {
    name: syn::Ident,
    visibility: syn::Visibility,
    shared: Vec<Shared>,
    functions: Vec<syn::ItemFn>,
    entry: Option<syn::Ident>,
}

fn is_entry_marker(attr: &syn::Attribute) -> bool {
    let segments = &attr.path().segments;
    segments.last().is_some_and(|part| part.ident == "kernel")
        && (segments.len() == 1 || segments.len() == 2 && segments[0].ident == "neura_compiler")
}

fn shared_memory(item: syn::ItemMacro) -> syn::Result<Shared> {
    if !item.mac.path.is_ident("workgroup") {
        return Err(syn::Error::new_spanned(
            &item.mac.path,
            "a device module declares workgroup memory with workgroup!(NAME: [element; count])",
        ));
    }
    let declaration = item.mac.tokens.clone();
    let parse = |input: ParseStream<'_>| {
        let name: syn::Ident = input.parse()?;
        input.parse::<Token![:]>()?;
        let ty: syn::Type = input.parse()?;
        Ok((name, ty))
    };
    let (name, ty) = parse.parse2(declaration)?;
    let syn::Type::Array(array) = &ty else {
        return Err(syn::Error::new_spanned(
            &ty,
            "device workgroup memory is an array",
        ));
    };
    let syn::Type::Path(path) = array.elem.as_ref() else {
        return Err(syn::Error::new_spanned(
            array.elem.as_ref(),
            "device workgroup memory holds u32, i32 or f32",
        ));
    };
    let Some(element) = path.path.get_ident().cloned() else {
        return Err(syn::Error::new_spanned(
            array.elem.as_ref(),
            "device workgroup memory holds u32, i32 or f32",
        ));
    };
    if !matches!(element.to_string().as_str(), "u32" | "i32" | "f32") {
        return Err(syn::Error::new_spanned(
            element,
            "device workgroup memory holds u32, i32 or f32",
        ));
    }
    if name.to_string().chars().any(char::is_uppercase) {
        return Err(syn::Error::new_spanned(
            &name,
            "device workgroup memory uses a snake case name",
        ));
    }
    let syn::Expr::Lit(literal) = &array.len else {
        return Err(syn::Error::new_spanned(
            &array.len,
            "a device workgroup array counts a literal element count",
        ));
    };
    let syn::Lit::Int(count) = &literal.lit else {
        return Err(syn::Error::new_spanned(
            &literal.lit,
            "a device workgroup array counts a literal element count",
        ));
    };
    let count = count.base10_parse::<u32>()?;
    if count == 0 {
        return Err(syn::Error::new_spanned(
            &name,
            "a device workgroup array holds at least one element",
        ));
    }
    Ok(Shared {
        name,
        element,
        count,
    })
}

fn device_module(input: syn::ItemMod) -> syn::Result<DeviceModule> {
    let Some((_, items)) = input.content else {
        return Err(syn::Error::new_spanned(
            input,
            "a kernel module is defined inline",
        ));
    };
    if let Some(attribute) = input.attrs.first() {
        return Err(syn::Error::new_spanned(
            attribute,
            "a kernel module carries no attribute of its own",
        ));
    }
    let visibility = input.vis;
    let name = input.ident;
    let mut shared = Vec::new();
    let mut functions = Vec::new();
    let mut entry = None;
    for item in items {
        match item {
            syn::Item::Fn(mut function) => {
                let mut marked = false;
                for attr in &function.attrs {
                    if !is_entry_marker(attr) {
                        return Err(syn::Error::new_spanned(
                            attr,
                            "kernel module functions only use #[kernel]",
                        ));
                    }
                    if !matches!(attr.meta, syn::Meta::Path(_)) || marked {
                        return Err(syn::Error::new_spanned(
                            attr,
                            "a device module declares one #[kernel] entry",
                        ));
                    }
                    if !matches!(function.sig.output, syn::ReturnType::Default) {
                        return Err(syn::Error::new_spanned(
                            &function.sig.output,
                            "compute entries do not return values",
                        ));
                    }
                    marked = true;
                }
                function.attrs.clear();
                if marked {
                    if entry.is_some() {
                        return Err(syn::Error::new_spanned(
                            &function.sig.ident,
                            "a device module declares one #[kernel] entry",
                        ));
                    }
                    entry = Some(function.sig.ident.clone());
                }
                functions.push(function);
            }
            syn::Item::Macro(item) => shared.push(shared_memory(item)?),
            other => {
                return Err(syn::Error::new_spanned(
                    other,
                    "a kernel module contains Rust functions and workgroup arrays",
                ));
            }
        }
    }
    if functions.is_empty() {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "a kernel module needs functions",
        ));
    }
    Ok(DeviceModule {
        name,
        visibility,
        shared,
        functions,
        entry,
    })
}

fn shared_declarations(shared: &[Shared]) -> Vec<Tokens> {
    shared
        .iter()
        .map(|shared| {
            let name = shared.name.to_string();
            let element = shared.element.to_string();
            let count = shared.count;
            quote!(compiler.workgroup(#name, #element, #count);)
        })
        .collect()
}

fn shared_locals(source: &syn::ItemFn, shared: &[Shared]) -> Vec<syn::Stmt> {
    shared
        .iter()
        .filter_map(|shared| {
            let mut scan = Mentions {
                name: &shared.name,
                present: false,
                written: false,
            };
            let mut function = source.clone();
            scan.visit_item_fn_mut(&mut function);
            if !scan.present {
                return None;
            }
            let name = &shared.name;
            let element = &shared.element;
            let count = shared.count as usize;
            Some(if scan.written {
                syn::parse_quote!(let mut #name = ::neura_compiler::Workgroup::<#element, #count>::new();)
            } else {
                syn::parse_quote!(let #name = ::neura_compiler::Workgroup::<#element, #count>::new();)
            })
        })
        .collect()
}

struct Mentions<'a> {
    name: &'a syn::Ident,
    present: bool,
    written: bool,
}

impl Mentions<'_> {
    fn mark(&mut self, place: &mut syn::Expr) {
        let present = self.present;
        self.present = false;
        self.visit_expr_mut(place);
        if self.present {
            self.written = true;
        }
        self.present |= present;
    }
}

impl VisitMut for Mentions<'_> {
    fn visit_ident_mut(&mut self, ident: &mut syn::Ident) {
        if ident == self.name {
            self.present = true;
        }
    }

    fn visit_expr_assign_mut(&mut self, assignment: &mut syn::ExprAssign) {
        self.mark(&mut assignment.left);
        visit_mut::visit_expr_assign_mut(self, assignment);
    }

    fn visit_expr_binary_mut(&mut self, binary: &mut syn::ExprBinary) {
        if matches!(
            binary.op,
            syn::BinOp::AddAssign(_)
                | syn::BinOp::SubAssign(_)
                | syn::BinOp::MulAssign(_)
                | syn::BinOp::DivAssign(_)
                | syn::BinOp::RemAssign(_)
                | syn::BinOp::BitAndAssign(_)
                | syn::BinOp::BitOrAssign(_)
                | syn::BinOp::BitXorAssign(_)
                | syn::BinOp::ShlAssign(_)
                | syn::BinOp::ShrAssign(_)
        ) {
            self.mark(&mut binary.left);
        }
        visit_mut::visit_expr_binary_mut(self, binary);
    }
}

fn expand_definitions(module: DeviceModule) -> syn::Result<Tokens> {
    let DeviceModule {
        name,
        visibility,
        shared,
        functions,
        entry,
    } = module;
    let declarations = shared_declarations(&shared);
    let definitions = functions
        .iter()
        .map(lower::function)
        .collect::<syn::Result<Vec<_>>>()?;
    let uses_entry = entry.is_some().then(|| quote!(let _ = ENTRY;));
    let entry = entry.map(|entry| {
        quote!(
            pub const ENTRY: &str = stringify!(#entry);
        )
    });
    Ok(quote! {
        #visibility mod #name {
            #entry
            pub(crate) fn define(compiler: &mut ::neura_compiler::Compiler) {
                use ::neura_compiler::ast as neura_ast;
                #uses_entry
                #(#declarations)*
                #(compiler.function(#definitions);)*
            }
        }
    })
}

struct Bindings {
    builtins: syn::punctuated::Punctuated<FnArg, syn::token::Comma>,
    records: BTreeSet<String>,
    statements: Vec<Tokens>,
}

fn storage_path(ty: &syn::Type) -> Option<&syn::PathSegment> {
    let syn::Type::Path(path) = ty else {
        return None;
    };
    let segments = path.path.segments.iter().collect::<Vec<_>>();
    let [segment] = segments.as_slice() else {
        return None;
    };
    matches!(segment.ident.to_string().as_str(), "Read" | "ReadWrite").then_some(*segment)
}

fn storage_element(ty: &syn::Type) -> syn::Result<String> {
    let Some(segment) = storage_path(ty) else {
        return Err(syn::Error::new_spanned(
            ty,
            "storage arguments use Read<T> or ReadWrite<T>",
        ));
    };
    let PathArguments::AngleBracketed(generics) = &segment.arguments else {
        return Err(syn::Error::new_spanned(
            ty,
            "storage arguments specify an element type",
        ));
    };
    let arguments = generics.args.iter().collect::<Vec<_>>();
    let [GenericArgument::Type(element)] = arguments.as_slice() else {
        return Err(syn::Error::new_spanned(
            generics,
            "storage buffers have one scalar element type",
        ));
    };
    let syn::Type::Path(path) = element else {
        return Err(syn::Error::new_spanned(
            element,
            "storage buffers contain u32, f32, AtomicU32 or an ABI record",
        ));
    };
    let Some(ident) = path.path.get_ident() else {
        return Err(syn::Error::new_spanned(
            element,
            "storage buffers contain u32, f32, AtomicU32 or an ABI record",
        ));
    };
    let name = ident.to_string();
    if name == "AtomicU32" || name == "u32" || name == "f32" || name.ends_with("Record") {
        Ok(name)
    } else {
        Err(syn::Error::new_spanned(
            ident,
            "storage buffers contain u32, f32, AtomicU32 or an ABI record",
        ))
    }
}

fn bindings(entry: &mut syn::ItemFn) -> syn::Result<Bindings> {
    let mut builtins = syn::punctuated::Punctuated::new();
    let mut records = BTreeSet::new();
    let mut statements = Vec::new();
    let mut slot = 0u32;
    for input in &mut entry.sig.inputs {
        let FnArg::Typed(arg) = input else {
            return Err(syn::Error::new_spanned(input, "kernel arguments are named"));
        };
        let Pat::Ident(variable) = arg.pat.as_mut() else {
            return Err(syn::Error::new_spanned(
                arg,
                "kernel arguments have simple names",
            ));
        };
        let argument_name = variable.ident.to_string();
        if matches!(argument_name.as_str(), "lid" | "group" | "global") {
            let expected = if argument_name == "lid" {
                "u32"
            } else {
                "Uvec3"
            };
            let Some(actual) = (match arg.ty.as_ref() {
                syn::Type::Path(path) => path.path.get_ident(),
                _ => None,
            }) else {
                return Err(syn::Error::new_spanned(
                    &arg.ty,
                    format!("builtin {argument_name} has type {expected}"),
                ));
            };
            if actual != expected {
                return Err(syn::Error::new_spanned(
                    &arg.ty,
                    format!("builtin {argument_name} has type {expected}"),
                ));
            }
            builtins.push(input.clone());
            continue;
        }
        let access = storage_path(&arg.ty)
            .map(|segment| segment.ident.to_string())
            .ok_or_else(|| {
                syn::Error::new_spanned(&arg.ty, "storage arguments use Read<T> or ReadWrite<T>")
            })?;
        let element = storage_element(&arg.ty)?;
        let element = match element.strip_suffix("Record") {
            Some(record) => {
                records.insert(record.to_owned());
                record.to_owned()
            }
            None => element,
        };
        let spec = if access == "ReadWrite" {
            quote!(::neura_compiler::BindingSpec::writable_storage(#slot))
        } else {
            quote!(::neura_compiler::BindingSpec::storage(#slot))
        };
        statements.push(quote!(compiler.storage_array(#argument_name, #element, #spec);));
        slot += 1;
    }
    if slot == 0 {
        return Err(syn::Error::new_spanned(
            &entry.sig,
            "a compute kernel binds at least one storage buffer",
        ));
    }
    Ok(Bindings {
        builtins,
        records,
        statements,
    })
}

fn assert_scalar_arguments(function: &syn::ItemFn) -> syn::Result<()> {
    for input in &function.sig.inputs {
        let FnArg::Typed(arg) = input else {
            return Err(syn::Error::new_spanned(
                input,
                "device helpers have no receiver",
            ));
        };
        if storage_path(&arg.ty).is_some() {
            return Err(syn::Error::new_spanned(
                &arg.ty,
                "only the entry reads storage buffers, and device helpers take scalars",
            ));
        }
    }
    Ok(())
}

fn writable_arguments(function: &mut syn::ItemFn) {
    for input in &mut function.sig.inputs {
        let FnArg::Typed(arg) = input else {
            continue;
        };
        let Pat::Ident(variable) = arg.pat.as_mut() else {
            continue;
        };
        if storage_path(&arg.ty).is_some_and(|segment| segment.ident == "ReadWrite") {
            variable.mutability = Some(syn::token::Mut::default());
        }
    }
}

fn record_declarations(records: impl Iterator<Item = String>) -> Vec<Tokens> {
    records
        .map(|name| {
            quote! {
                compiler.record(*::neura_compiler::abi::RECORDS.iter()
                    .find(|record| record.name == #name)
                    .unwrap_or_else(|| panic!("kernel ABI record {} is unknown", #name)));
            }
        })
        .collect()
}

fn expand_kernel_function(
    options: KernelOptions,
    mut original: syn::ItemFn,
) -> syn::Result<Tokens> {
    if !matches!(original.sig.output, syn::ReturnType::Default) {
        return Err(syn::Error::new_spanned(
            &original.sig.output,
            "compute entries do not return values",
        ));
    }
    let name = original.sig.ident.clone();
    let visibility = original.vis.clone();
    let entry = bindings(&mut original)?;
    let mut body = original.clone();
    body.sig.inputs = entry.builtins;
    let definition = lower::function(&body)?;
    let records = record_declarations(entry.records.into_iter());
    let statements = entry.statements;
    let mut typecheck = original;
    writable_arguments(&mut typecheck);
    typecheck.vis = syn::Visibility::Inherited;
    let size = options.size;
    Ok(quote! {
        #visibility fn #name() -> ::neura_compiler::ComputeProgram {
            static PROGRAM: ::std::sync::OnceLock<::neura_compiler::ComputeProgram> =
                ::std::sync::OnceLock::new();
            PROGRAM.get_or_init(|| {
                use ::neura_compiler::ast as neura_ast;
                let mut compiler = ::neura_compiler::Compiler::empty();
                compiler.constant("WORKGROUP_SIZE", #size);
                #(#records)*
                #(#statements)*
                compiler.function(#definition);
                compiler.finish(stringify!(#name), stringify!(#name), #size)
            }).clone()
        }
        const _: () = {
            use ::neura_compiler::device::*;
            const WORKGROUP_SIZE: u32 = #size;
            #typecheck
            let _ = #name;
            let _ = WORKGROUP_SIZE;
        };
    })
}

fn expand_kernel_module(options: KernelOptions, module: syn::ItemMod) -> syn::Result<Tokens> {
    let module = device_module(module)?;
    let name = module.name.clone();
    let visibility = module.visibility.clone();
    let Some(entry) = module.entry.clone() else {
        return Err(syn::Error::new_spanned(
            &module.functions[0].sig.ident,
            "a standalone kernel module marks one #[kernel] entry",
        ));
    };
    let declarations = shared_declarations(&module.shared);
    let mut definitions = Vec::new();
    let mut records = Vec::new();
    let mut statements = Vec::new();
    for function in &module.functions {
        if function.sig.ident == entry {
            let mut entry_function = function.clone();
            let entry_bindings = bindings(&mut entry_function)?;
            let mut body = entry_function;
            body.sig.inputs = entry_bindings.builtins;
            definitions.push(lower::function(&body)?);
            records.extend(record_declarations(entry_bindings.records.into_iter()));
            statements.extend(entry_bindings.statements);
            continue;
        }
        assert_scalar_arguments(function)?;
        definitions.push(lower::function(function)?);
    }
    let checks = module.functions.iter().map(|function| {
        let locals = shared_locals(function, &module.shared);
        let mut check = function.clone();
        writable_arguments(&mut check);
        check.vis = syn::Visibility::Inherited;
        if !locals.is_empty() {
            check.block.stmts.splice(0..0, locals);
        }
        quote!(#check)
    });
    let names = module.functions.iter().map(|function| &function.sig.ident);
    let size = options.size;
    Ok(quote! {
        #visibility fn #name() -> ::neura_compiler::ComputeProgram {
            static PROGRAM: ::std::sync::OnceLock<::neura_compiler::ComputeProgram> =
                ::std::sync::OnceLock::new();
            PROGRAM.get_or_init(|| {
                use ::neura_compiler::ast as neura_ast;
                let mut compiler = ::neura_compiler::Compiler::empty();
                compiler.constant("WORKGROUP_SIZE", #size);
                #(#declarations)*
                #(#records)*
                #(#statements)*
                #(compiler.function(#definitions);)*
                compiler.finish(stringify!(#name), stringify!(#entry), #size)
            }).clone()
        }
        const _: () = {
            use ::neura_compiler::device::*;
            const WORKGROUP_SIZE: u32 = #size;
            #(#checks)*
            #(let _ = #names;)*
            let _ = WORKGROUP_SIZE;
        };
    })
}

#[proc_macro_attribute]
pub fn module(attr: TokenStream, item: TokenStream) -> TokenStream {
    let result = if attr.is_empty() {
        syn::parse::<syn::ItemMod>(item)
            .and_then(device_module)
            .and_then(expand_definitions)
    } else {
        Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "kernel modules have no attributes",
        ))
    };
    result.unwrap_or_else(syn::Error::into_compile_error).into()
}

#[proc_macro]
pub fn expression(input: TokenStream) -> TokenStream {
    let result =
        syn::parse::<syn::Expr>(input).and_then(|expression| lower::expression(&expression));
    result.unwrap_or_else(syn::Error::into_compile_error).into()
}

#[proc_macro_attribute]
pub fn kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    let result = syn::parse::<KernelOptions>(attr).and_then(|options| {
        syn::parse::<syn::Item>(item).and_then(|item| match item {
            syn::Item::Fn(function) => expand_kernel_function(options, function),
            syn::Item::Mod(module) => expand_kernel_module(options, module),
            other => Err(syn::Error::new_spanned(
                other,
                "a kernel is a function or a module of functions",
            )),
        })
    });
    result.unwrap_or_else(syn::Error::into_compile_error).into()
}
