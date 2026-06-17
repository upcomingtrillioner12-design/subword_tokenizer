fn main() {
    cc::Build::new()
        .file("src/cpp/bpe.cpp")
        .cpp(true)  // Tell cc to compile as C++
        .compile("bpe");
    
    // Tell cargo to link against C++ standard library
    println!("cargo:rustc-link-lib=stdc++");
}
