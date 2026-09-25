mod lower;

use proc_macro::TokenStream;
use proc_macro2::TokenStream as Tokens;
use quote::quote;
use std::collections::BTreeSet;
use syn::parse::{Parse, ParseStream};
use syn::{FnArg, GenericArgument, Pat, PathArguments, Token};

struct Workgroup {
    size: u32,
}

impl Parse for Workgroup {
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

fn expand_module(input: Tokens) -> syn::Result<Tokens> {
    let module = syn::parse2::<syn::ItemMod>(input)?;
    let Some((_, items)) = module.content else {
        return Err(syn::Error::new_spanned(
            module,
            "a kernel module is defined inline",
        ));
    };
    let mut definitions = Vec::new();
    let mut entry = None;
    for item in items {
        let syn::Item::Fn(mut function) = item else {
            return Err(syn::Error::new_spanned(
                item,
                "a kernel module contains Rust functions",
            ));
        };
        for attr in &function.attrs {
            let segments = &attr.path().segments;
            let marker = segments.last().is_some_and(|part| part.ident == "kernel")
                && (segments.len() == 1
                    || segments.len() == 2 && segments[0].ident == "neura_compiler");
            if !marker {
                return Err(syn::Error::new_spanned(
                    attr,
                    "kernel module functions only use #[kernel]",
                ));
            }
            if !matches!(attr.meta, syn::Meta::Path(_)) || entry.is_some() {
                return Err(syn::Error::new_spanned(
                    attr,
                    "a module has one dynamically sized #[kernel] entry",
                ));
            }
            if !matches!(function.sig.output, syn::ReturnType::Default) {
                return Err(syn::Error::new_spanned(
                    &function.sig.output,
                    "compute entries do not return values",
                ));
            }
            entry = Some(function.sig.ident.clone());
        }
        function.attrs.clear();
        definitions.push(lower::function(&function)?);
    }
    if definitions.is_empty() {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "a kernel module needs functions",
        ));
    }
    let use_entry = entry.as_ref().map(|_| quote!(let _ = ENTRY;));
    let entry = entry.map(|name| {
        quote!(
            pub const ENTRY: &str = stringify!(#name);
        )
    });
    Ok(quote! {
        #entry
        pub(crate) fn define(compiler: &mut ::neura_compiler::Compiler) {
            use ::neura_compiler::ir as neura_ir;
            #use_entry
            #(compiler.function(#definitions);)*
        }
    })
}

#[proc_macro_attribute]
pub fn module(attr: TokenStream, item: TokenStream) -> TokenStream {
    let result = if attr.is_empty() {
        expand_module(item.into())
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

fn expand_kernel(options: Workgroup, mut original: syn::ItemFn) -> syn::Result<Tokens> {
    if !matches!(original.sig.output, syn::ReturnType::Default) {
        return Err(syn::Error::new_spanned(
            &original.sig.output,
            "compute entries do not return values",
        ));
    }
    let name = original.sig.ident.clone();
    let visibility = original.vis.clone();
    let mut builtin_args = syn::punctuated::Punctuated::new();
    let mut bindings = Vec::new();
    let mut records = BTreeSet::new();
    let mut slot = 0u32;
    for input in &mut original.sig.inputs {
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
            builtin_args.push(input.clone());
            continue;
        }
        let syn::Type::Path(path) = arg.ty.as_ref() else {
            return Err(syn::Error::new_spanned(
                &arg.ty,
                "storage arguments use Read<T> or ReadWrite<T>",
            ));
        };
        let Some(segment) = (path.path.segments.len() == 1).then(|| {
            path.path
                .segments
                .first()
                .expect("a storage argument has one type")
        }) else {
            return Err(syn::Error::new_spanned(
                &arg.ty,
                "storage arguments use Read<T> or ReadWrite<T>",
            ));
        };
        let spec = match segment.ident.to_string().as_str() {
            "Read" => quote!(::neura_compiler::BindingSpec::storage(#slot)),
            "DynamicRead" => quote!(::neura_compiler::BindingSpec::dynamic_storage(#slot)),
            "ReadWrite" => quote!(::neura_compiler::BindingSpec::writable_storage(#slot)),
            other => {
                return Err(syn::Error::new_spanned(
                    &arg.ty,
                    format!("unsupported storage access {other}"),
                ));
            }
        };
        let PathArguments::AngleBracketed(generics) = &segment.arguments else {
            return Err(syn::Error::new_spanned(
                &arg.ty,
                "storage arguments specify an element type",
            ));
        };
        if generics.args.len() != 1 {
            return Err(syn::Error::new_spanned(
                generics,
                "storage buffers have one scalar element type",
            ));
        }
        let Some(GenericArgument::Type(syn::Type::Path(ty))) = generics.args.first() else {
            return Err(syn::Error::new_spanned(
                generics,
                "storage buffers contain u32 or f32",
            ));
        };
        let Some(ty) = ty.path.get_ident() else {
            return Err(syn::Error::new_spanned(
                ty,
                "storage buffers contain u32 or f32",
            ));
        };
        let element = ty.to_string();
        let element = if matches!(element.as_str(), "u32" | "f32") {
            element
        } else if let Some(record) = element.strip_suffix("Record") {
            let record = record.to_owned();
            records.insert(record.clone());
            record
        } else {
            return Err(syn::Error::new_spanned(
                ty,
                "storage buffers contain u32, f32 or an ABI record",
            ));
        };
        if segment.ident == "ReadWrite" {
            variable.mutability = Some(syn::token::Mut::default());
        }
        bindings.push(quote!(compiler.storage_array(#argument_name, #element, #spec);));
        slot += 1;
    }
    if slot == 0 {
        return Err(syn::Error::new_spanned(
            &original.sig,
            "a compute kernel binds at least one storage buffer",
        ));
    }
    let mut body = original.clone();
    body.sig.inputs = builtin_args;
    let definition = lower::function(&body)?;
    let mut typecheck = original;
    let hidden = syn::Ident::new("check", name.span());
    typecheck.sig.ident = hidden.clone();
    typecheck.vis = syn::Visibility::Inherited;
    let size = options.size;
    let records = records.into_iter().map(|name| {
        quote! {
            compiler.record(*::neura_compiler::abi::RECORDS.iter()
                .find(|record| record.name == #name)
                .unwrap_or_else(|| panic!("kernel ABI record {} is unknown", #name)));
        }
    });
    Ok(quote! {
        #visibility fn #name() -> ::neura_compiler::ComputeProgram {
            static PROGRAM: ::std::sync::OnceLock<::neura_compiler::ComputeProgram> =
                ::std::sync::OnceLock::new();
            PROGRAM.get_or_init(|| {
                use ::neura_compiler::ir as neura_ir;
                let mut compiler = ::neura_compiler::Compiler::new();
                #(#records)*
                #(#bindings)*
                compiler.function(#definition);
                compiler.finish(stringify!(#name), stringify!(#name), #size)
            }).clone()
        }
        const _: () = {
            #typecheck
            let _ = #hidden;
        };
    })
}

#[proc_macro_attribute]
pub fn kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    let result = syn::parse::<Workgroup>(attr).and_then(|opts| {
        syn::parse::<syn::ItemFn>(item).and_then(|function| expand_kernel(opts, function))
    });
    result.unwrap_or_else(syn::Error::into_compile_error).into()
}
