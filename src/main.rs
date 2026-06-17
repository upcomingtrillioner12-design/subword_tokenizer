unsafe extern "C" {
    fn train_bpe(text: *const std::ffi::c_char, vocab_size: i32);
}

fn main() {
    let text = "This is sample text for BPE training. We will build subword tokenization. \
                 Byte Pair Encoding is a compression technique used in natural language processing. \
                 It repeatedly merges the most frequent pair of bytes in a sequence until reaching \
                 a target vocabulary size. This approach is commonly used in transformer models like \
                 BERT and GPT to convert raw text into subword tokens that balance vocabulary size \
                 and coverage. The algorithm is simple yet powerful for building tokenizers.";
    let vocab_size = 350;
    
    println!("$$$$ Starting BPE Tokenizer Training");
    println!("$$$ Corpus: {} ... (truncated)", &text[..std::cmp::min(80, text.len())]);
    println!("$$ Vocab Size: {}", vocab_size);
    
    let c_text = std::ffi::CString::new(text).unwrap();
    unsafe {
        train_bpe(c_text.as_ptr(), vocab_size);
    }
    
    println!("### BPE Training Complete!");
}
