fn main() {
    for backend in ["dx12", "metal", "vulkan"] {
        println!("cargo::rustc-check-cfg=cfg({backend}_backend)");
    }
    println!("cargo::rustc-check-cfg=cfg(multiple_backends)");

    let os = std::env::var("CARGO_CFG_TARGET_OS").expect("cargo names the target OS");
    let backends: &[&str] = match os.as_str() {
        "windows" => &["dx12", "vulkan"],
        "macos" | "ios" => &["metal"],
        "linux" | "android" => &["vulkan"],
        unsupported => panic!("no compute backend targets {unsupported}"),
    };
    for backend in backends {
        println!("cargo::rustc-cfg={backend}_backend");
    }
    if backends.len() > 1 {
        println!("cargo::rustc-cfg=multiple_backends");
    }
}
