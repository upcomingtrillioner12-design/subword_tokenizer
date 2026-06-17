use clap::Parser;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

mod tests;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BPEModel {
    pub vocab: Vec<String>,
    pub merges: Vec<(String, String)>,
    pub vocab_size: i32,
}

impl BPEModel {
    /// Apply learned merges to tokenize new text
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
}

#[derive(Parser, Debug)]
#[command(name = "Subword Tokenizer")]
#[command(about = "A BPE tokenizer built in Rust + C++", long_about = None)]
struct Args {
    /// Path to input corpus file (if not provided, uses sample text)
    #[arg(short, long)]
    corpus: Option<PathBuf>,

    /// Target vocabulary size
    #[arg(short, long, default_value = "350")]
    vocab_size: i32,

    /// Path to save trained model as JSON
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Apply tokenization to a test string (inference mode)
    #[arg(short, long)]
    tokenize: Option<String>,

    /// Path to load trained model from JSON
    #[arg(short, long)]
    model: Option<PathBuf>,
}

unsafe extern "C" {
    fn train_bpe_ffi(text: *const std::ffi::c_char, vocab_size: i32);
    fn get_vocab_count() -> i32;
    fn get_merge_count() -> i32;
    fn get_vocab_item(index: i32) -> *const std::ffi::c_char;
    fn get_merge_pair(index: i32, first: *mut *const std::ffi::c_char, second: *mut *const std::ffi::c_char);
}

fn main() {
    let args = Args::parse();

    // Read corpus
    let corpus_text = if let Some(corpus_path) = args.corpus {
        match fs::read_to_string(&corpus_path) {
            Ok(content) => {
                println!("✓ Loaded corpus from: {:?}", corpus_path);
                content
            }
            Err(e) => {
                eprintln!("✗ Error reading corpus file: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        // Default sample text
        "This is sample text for BPE training. We will build subword tokenization. \
         Byte Pair Encoding is a compression technique used in natural language processing. \
         It repeatedly merges the most frequent pair of bytes in a sequence until reaching \
         a target vocabulary size. This approach is commonly used in transformer models like \
         BERT and GPT to convert raw text into subword tokens that balance vocabulary size \
         and coverage. The algorithm is simple yet powerful for building tokenizers."
            .to_string()
    };

    println!("\n╔════════════════════════════════════════════╗");
    println!("║      Subword Tokenizer (BPE)              ║");
    println!("╚════════════════════════════════════════════╝");

    println!("\n📊 Configuration:");
    println!("   Corpus size: {} chars", corpus_text.len());
    println!("   Target vocab size: {}", args.vocab_size);
    println!("   Output model: {}", args.output.as_ref().map(|p| p.display().to_string()).unwrap_or("(none)".to_string()));

    // Train BPE
    println!("\n🔄 Training BPE...");
    let c_text = std::ffi::CString::new(corpus_text.clone()).unwrap();
    unsafe {
        train_bpe_ffi(c_text.as_ptr(), args.vocab_size);
    }

    // Collect vocabulary and merges from C++
    let vocab_count = unsafe { get_vocab_count() };
    let merge_count = unsafe { get_merge_count() };
    
    let mut vocab = Vec::new();
    for i in 0..vocab_count {
        let vocab_ptr = unsafe { get_vocab_item(i) };
        if !vocab_ptr.is_null() {
            if let Ok(vocab_str) = unsafe { std::ffi::CStr::from_ptr(vocab_ptr).to_str() } {
                vocab.push(vocab_str.to_string());
            }
        }
    }
    
    let mut merges = Vec::new();
    for i in 0..merge_count {
        let mut first_ptr: *const std::ffi::c_char = std::ptr::null();
        let mut second_ptr: *const std::ffi::c_char = std::ptr::null();
        unsafe {
            get_merge_pair(i, &mut first_ptr, &mut second_ptr);
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
    
    println!("\n✓ BPE training complete!");
    println!("  Vocabulary size: {}", vocab.len());
    println!("  Total merges: {}", merges.len());

    // If output path specified, save model
    if let Some(output_path) = args.output {
        let model = BPEModel {
            vocab: vocab.clone(),
            merges: merges.clone(),
            vocab_size: args.vocab_size,
        };
        
        match serde_json::to_string_pretty(&model) {
            Ok(json) => {
                if let Err(e) = fs::write(&output_path, json) {
                    eprintln!("✗ Error writing model file: {}", e);
                } else {
                    println!("✓ Model saved to: {:?}", output_path);
                }
            }
            Err(e) => eprintln!("✗ Error serializing model: {}", e),
        }
    }

    // If tokenize string provided, apply learned merges
    if let Some(text) = args.tokenize.clone() {
        if merges.is_empty() {
            println!("\n⚠️  No merges learned yet. Cannot tokenize.");
        } else {
            let model_for_tokenize = BPEModel {
                vocab: vocab.clone(),
                merges: merges.clone(),
                vocab_size: args.vocab_size,
            };
            let tokens = model_for_tokenize.tokenize(&text);
            println!("\n📝 Tokenizing: \"{}\"", text);
            println!("   Tokens ({}): {:?}", tokens.len(), tokens);
        }
    }

    // If model path provided, load and use it
    if let Some(model_path) = args.model {
        match fs::read_to_string(&model_path) {
            Ok(json) => match serde_json::from_str::<BPEModel>(&json) {
                Ok(loaded_model) => {
                    println!("\n✓ Loaded model from: {:?}", model_path);
                    println!("   Vocab size: {}", loaded_model.vocab_size);
                    println!("   Merges learned: {}", loaded_model.merges.len());
                    
                    // If tokenize arg also provided, use loaded model
                    if let Some(text) = args.tokenize {
                        let tokens = loaded_model.tokenize(&text);
                        println!("\n📝 Tokenizing with loaded model: \"{}\"", text);
                        println!("   Tokens ({}): {:?}", tokens.len(), tokens);
                    }
                }
                Err(e) => eprintln!("✗ Error parsing model JSON: {}", e),
            },
            Err(e) => eprintln!("✗ Error reading model file: {}", e),
        }
    }

    println!("\n✅ Done!");
}
