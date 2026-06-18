fn main() {
    println!("ENTERPRISE SUBWORD TOKENIZER - BPE FROM SCRATCH");

    let corpus = "low low low low low lower lower lowest running running runner runner ran ran happily happy";
    let vocab_size = 100;

    println!("Training Corpus: {}", corpus);

    match subword_tokenizer::train(corpus, vocab_size) {
        Ok(model) => {
            println!("Training Complete!");
            println!("Vocabulary size: {}", model.vocab.len());
            println!("Merge rules: {}", model.merges.len());

            println!("\nFirst 5 merge rules:");
            for (i, (a, b)) in model.merges.iter().take(5).enumerate() {
                println!("  {}. '{}' + '{}' -> '{}{}'", i + 1, a, b, a, b);
            }

            println!("\nSample Vocabulary (first 15):");
            for (i, word) in model.vocab.iter().take(15).enumerate() {
                println!("  {}. '{}'", i + 1, word);
            }

            let test_words = vec!["lower", "running", "happy", "lowest"];
            println!("\nTokenizing test words:");
            for word in test_words {
                let tokens = subword_tokenizer::encode(&model, word);
                println!("  '{}' -> {:?}", word, tokens);
            }
        }
        Err(e) => {
            println!("Error: {}", e);
        }
    }

    println!("\nBPE Tokenizer Complete!");
}
