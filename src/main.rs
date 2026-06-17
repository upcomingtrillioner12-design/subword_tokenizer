unsafe extern "C" {
    fn train_bpe(text: *const std::ffi::c_char, vocab_size: i32);
}

fn main() {
    let text = "This is sample text for BPE training. We will build subword tokenization.";
    let vocab_size = 1000;
    
    println!("$$$$ Starting BPE Tokenizer Training");
    println!("$$$ Corpus: {}", text);
    println!("$$ Vocab Size: {}", vocab_size);
    
    let c_text = std::ffi::CString::new(text).unwrap();
    unsafe {
        train_bpe(c_text.as_ptr(), vocab_size);
    }
    
    println!("### BPE Training Complete!");
}
