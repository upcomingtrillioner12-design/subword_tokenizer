fn main() {
    cc::Build::new()
        .file("src/cpp/bpe.cpp")
        .cpp(true)
        .std("c++17")
        .compile("bpe");

    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rerun-if-changed=src/cpp/bpe.cpp");
}
