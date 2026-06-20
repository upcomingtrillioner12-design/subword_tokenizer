use std::env;
use subword_tokenizer::BPEModel;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Usage:");
        println!("  cargo run --release -- train <corpus.txt> <vocab_size>");
        println!("  cargo run --release -- expand <corpus.txt> <vocab_size>");
        println!("  cargo run --release -- tokenize <text>");
        println!("  cargo run --release -- count");
        return;
    }
    match args[1].as_str() {
        "train" => {
            let corpus = &args[2];
            let size: i32 = args[3].parse().unwrap();
            let mut bpe = BPEModel::new();
            bpe.train(corpus, size);
            bpe.save("model.json").unwrap();
        }
        "expand" => {
            let corpus = &args[2];
            let size: i32 = args[3].parse().unwrap();
            let mut bpe = BPEModel::new();
            let _ = bpe.load("model.json");
            bpe.train(corpus, size);
            bpe.save("model.json").unwrap();
        }
        "tokenize" => {
            let text = &args[2];
            let mut bpe = BPEModel::new();
            let _ = bpe.load("model.json");
            let ids = bpe.encode(text);
            println!("IDs: {:?}", ids);
            println!("Count: {}", ids.len());
        }
        "decode" => {
            let ids_str = &args[2];
            let ids: Vec<i32> = ids_str.split(',').map(|s| s.trim().parse().unwrap()).collect();
            let mut bpe = BPEModel::new();
            let _ = bpe.load("model.json");
            let text = bpe.decode(&ids);
            println!("Decoded: {}", text);
        }
        "count" => {
            let mut bpe = BPEModel::new();
            let _ = bpe.load("model.json");
            println!("Vocab: {} | Merges: {} | Total: {}", 
                bpe.vocab_size(), bpe.merge_count(), bpe.vocab_size() + 5);
        }
        _ => println!("Unknown: {}", args[1]),
    }
}
