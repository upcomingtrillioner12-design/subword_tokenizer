fn main() {
    cc::Build::new()
        .file("src/cpp/bpe.cpp")
        .cpp(true)
        .flag("-mmacosx-version-min=11.0")
        .compile("bpe");

    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-lib=c++");
    } else {
        println!("cargo:rustc-link-lib=stdc++");
    }
}
