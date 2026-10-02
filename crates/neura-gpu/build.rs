fn main() {
    for backend in ["dx12", "metal", "vulkan"] {
        println!("cargo::rustc-check-cfg=cfg({backend}_backend)");
        println!("cargo::rustc-check-cfg=cfg(platform_{backend})");
    }
    println!("cargo::rustc-check-cfg=cfg(multiple_backends)");

    let os = std::env::var("CARGO_CFG_TARGET_OS").expect("cargo names the target OS");
    let (compiled, platform): (&[&str], &[&str]) = match os.as_str() {
        "windows" => (&["dx12", "vulkan"], &["dx12", "vulkan"]),
        "macos" => (&["metal", "vulkan"], &["metal"]),
        "ios" => (&["metal"], &["metal"]),
        "linux" | "android" => (&["vulkan"], &["vulkan"]),
        unsupported => panic!("no compute backend targets {unsupported}"),
    };
    for backend in compiled {
        println!("cargo::rustc-cfg={backend}_backend");
    }
    for backend in platform {
        println!("cargo::rustc-cfg=platform_{backend}");
    }
    if compiled.len() > 1 {
        println!("cargo::rustc-cfg=multiple_backends");
    }
}
