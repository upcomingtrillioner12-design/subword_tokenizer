use clap::Parser;
use std::fs;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "bpe-tokenizer")]
#[command(about = "Enterprise BPE Tokenizer")]
struct Args {
    #[arg(short, long, help = "Corpus file to train on")]
    corpus: Option<PathBuf>,

    #[arg(short, long, help = "Load existing model")]
    model: Option<PathBuf>,

    #[arg(short, long, default_value = "100")]
    vocab_size: i32,

    #[arg(short, long, help = "Text to tokenize")]
    text: Option<String>,

    #[arg(short, long, help = "Save model to file")]
    save: Option<PathBuf>,
}

fn main() {
    let args = Args::parse();

    let model = if let Some(model_path) = &args.model {
        println!("Loading model from: {}", model_path.display());
        subword_tokenizer::BPEModel::load(model_path).expect("Failed to load model")
    } else if let Some(corpus_path) = &args.corpus {
        let corpus = fs::read_to_string(corpus_path).expect("Failed to read corpus");
        println!("Training on: {} ({} chars)", corpus_path.display(), corpus.len());
        let m = subword_tokenizer::train(&corpus, args.vocab_size).expect("Training failed");
        println!("Trained! Vocab: {} | Merges: {}", m.vocab.len(), m.merges.len());
        m
    } else {
        eprintln!("Error: provide --corpus to train or --model to load");
        std::process::exit(1);
    };

    if let Some(save_path) = &args.save {
        model.save(save_path).expect("Failed to save model");
        println!("Model saved to: {}", save_path.display());
    }

    if let Some(text) = args.text {
        let ids = model.encode_with_special(&text);
        let tokens = subword_tokenizer::encode(&model, &text);
        println!("Text:    '{}'", text);
        println!("Tokens:  {:?}", tokens);
        println!("IDs:     {:?}", ids);
    }
}
