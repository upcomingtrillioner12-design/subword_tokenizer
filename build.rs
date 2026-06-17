fn main() {
    cc::Build::new()
        .file("src/cpp/bpe.cpp")
        .cpp(true)  // Tell cc to compile as C++
        .compile("bpe");

    // Link the correct C++ standard library per platform
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-lib=c++");
    } else {
        println!("cargo:rustc-link-lib=stdc++");
    }
}
