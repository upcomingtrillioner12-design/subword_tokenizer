use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::ffi::c_char;

/// BPE (Byte Pair Encoding) tokenizer model
/// Contains learned vocabulary and merge operations
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BPEModel {
    pub vocab: Vec<String>,
    pub merges: Vec<(String, String)>,
    pub vocab_size: i32,
}

impl BPEModel {
    /// Create a new BPEModel from components
    pub fn new(vocab: Vec<String>, merges: Vec<(String, String)>, vocab_size: i32) -> Self {
        BPEModel {
            vocab,
            merges,
            vocab_size,
        }
    }

    /// Apply learned merges to tokenize new text
    /// Returns a vector of tokens after applying all learned merge operations
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        // Initialize: split text into characters
        let mut tokens: Vec<String> = text.chars().map(|c| c.to_string()).collect();
        
        // Apply each learned merge in order
        for (first, second) in &self.merges {
            let merged = format!("{}{}", first, second);
            let mut new_tokens = Vec::new();
            
            let mut i = 0;
            while i < tokens.len() {
                if i + 1 < tokens.len() && tokens[i] == *first && tokens[i + 1] == *second {
                    new_tokens.push(merged.clone());
                    i += 2;
                } else {
                    new_tokens.push(tokens[i].clone());
                    i += 1;
                }
            }
            tokens = new_tokens;
        }
        
        tokens
    }

    /// Save model to JSON file
    pub fn save(&self, path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(&self)?;
        fs::write(path, json)?;
        Ok(())
    }

    /// Load model from JSON file
    pub fn load(path: &PathBuf) -> Result<Self, Box<dyn std::error::Error>> {
        let json = fs::read_to_string(path)?;
        let model = serde_json::from_str(&json)?;
        Ok(model)
    }
}

/// FFI declarations for C++ BPE training implementation
pub mod ffi {
    use std::ffi::c_char;

    unsafe extern "C" {
        pub fn train_bpe_ffi(text: *const c_char, vocab_size: i32);
        pub fn get_vocab_count() -> i32;
        pub fn get_merge_count() -> i32;
        pub fn get_vocab_item(index: i32) -> *const c_char;
        pub fn get_merge_pair(index: i32, first: *mut *const c_char, second: *mut *const c_char);
    }
}


/// Train a BPE model on corpus text
/// Returns a trained BPEModel with learned vocabulary and merge operations
pub fn train(corpus_text: &str, vocab_size: i32) -> Result<BPEModel, Box<dyn std::error::Error>> {
    let c_text = std::ffi::CString::new(corpus_text)?;
    
    // Call C++ training implementation
    unsafe {
        ffi::train_bpe_ffi(c_text.as_ptr(), vocab_size);
    }

    // Collect vocabulary from C++
    let vocab_count = unsafe { ffi::get_vocab_count() };
    let mut vocab = Vec::new();
    
    for i in 0..vocab_count {
        let vocab_ptr = unsafe { ffi::get_vocab_item(i) };
        if !vocab_ptr.is_null() {
            if let Ok(vocab_str) = unsafe { std::ffi::CStr::from_ptr(vocab_ptr).to_str() } {
                vocab.push(vocab_str.to_string());
            }
        }
    }

    // Collect merges from C++
    let merge_count = unsafe { ffi::get_merge_count() };
    let mut merges = Vec::new();
    
    for i in 0..merge_count {
        let mut first_ptr: *const c_char = std::ptr::null();
        let mut second_ptr: *const c_char = std::ptr::null();
        
        unsafe {
            ffi::get_merge_pair(i, &mut first_ptr, &mut second_ptr);
        }
        
        let first = if !first_ptr.is_null() {
            unsafe { std::ffi::CStr::from_ptr(first_ptr).to_str().unwrap_or("").to_string() }
        } else {
            String::new()
        };
        
        let second = if !second_ptr.is_null() {
            unsafe { std::ffi::CStr::from_ptr(second_ptr).to_str().unwrap_or("").to_string() }
        } else {
            String::new()
        };
        
        if !first.is_empty() && !second.is_empty() {
            merges.push((first, second));
        }
    }

    Ok(BPEModel::new(vocab, merges, vocab_size))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bpe_model_creation() {
        let vocab = vec!["a".to_string(), "b".to_string()];
        let merges = vec![];
        let model = BPEModel::new(vocab, merges, 256);
        assert_eq!(model.vocab_size, 256);
    }

    #[test]
    fn test_tokenize_empty_merges() {
        let vocab = vec!["h".to_string(), "e".to_string(), "l".to_string(), "o".to_string()];
        let model = BPEModel::new(vocab, vec![], 256);
        let tokens = model.tokenize("hello");
        assert_eq!(tokens, vec!["h", "e", "l", "l", "o"]);
    }

    #[test]
    fn test_tokenize_with_merges() {
        let vocab = vec!["h".to_string(), "e".to_string(), "l".to_string(), "o".to_string(), "he".to_string()];
        let merges = vec![("h".to_string(), "e".to_string())];
        let model = BPEModel::new(vocab, merges, 256);
        let tokens = model.tokenize("hello");
        assert_eq!(tokens, vec!["he", "l", "l", "o"]);
    }

    #[test]
    fn test_tokenize_deterministic() {
        let vocab = vec!["a".to_string(), "b".to_string(), "ab".to_string()];
        let merges = vec![("a".to_string(), "b".to_string())];
        let model = BPEModel::new(vocab, merges, 256);
        let tokens1 = model.tokenize("ab");
        let tokens2 = model.tokenize("ab");
        assert_eq!(tokens1, tokens2);
    }

    #[test]
    fn test_tokenize_multiple_merges() {
        let vocab = vec!["a".to_string(), "b".to_string(), "c".to_string(), "ab".to_string(), "abc".to_string()];
        let merges = vec![
            ("a".to_string(), "b".to_string()),
            ("ab".to_string(), "c".to_string()),
        ];
        let model = BPEModel::new(vocab, merges, 256);
        let tokens = model.tokenize("abc");
        assert_eq!(tokens, vec!["abc"]);
    }
}
