use clap::Parser;
use std::fs;
use std::path::PathBuf;
use subword_tokenizer::{BPEModel, train};

mod tests;

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
    let model = match train(&corpus_text, args.vocab_size) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("✗ Error training BPE: {}", e);
            std::process::exit(1);
        }
    };

    println!("\n✓ BPE training complete!");
    println!("  Vocabulary size: {}", model.vocab.len());
    println!("  Total merges: {}", model.merges.len());

    // If output path specified, save model
    if let Some(output_path) = args.output {
        match model.save(&output_path) {
            Ok(_) => println!("✓ Model saved to: {:?}", output_path),
            Err(e) => eprintln!("✗ Error saving model: {}", e),
        }
    }

    // If tokenize string provided, apply learned merges
    if let Some(text) = args.tokenize.clone() {
        if model.merges.is_empty() {
            println!("\n⚠️  No merges learned yet. Cannot tokenize.");
        } else {
            let tokens = model.tokenize(&text);
            println!("\n📝 Tokenizing: \"{}\"", text);
            println!("   Tokens ({}): {:?}", tokens.len(), tokens);
        }
    }

    // If model path provided, load and use it
    if let Some(model_path) = args.model {
        match BPEModel::load(&model_path) {
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
            Err(e) => eprintln!("✗ Error loading model: {}", e),
        }
    }

    println!("\n✅ Done!");
}
