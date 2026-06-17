# Subword Tokenizer - BPE in Rust + C++

A high-performance Byte Pair Encoding (BPE) tokenizer built with Rust and C++ integration. Train, save, and use subword tokenizers with a simple CLI or import as a library.

[![Build Status](https://github.com/upcomingtrillioner12-design/subword_tokenizer/actions/workflows/ci.yml/badge.svg)](https://github.com/upcomingtrillioner12-design/subword_tokenizer/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Quick Start

### Prerequisites
- Rust 1.96+ ([install](https://rustup.rs/))
- C++ compiler (clang/gcc on Linux/macOS, MSVC on Windows)

### Installation

```bash
git clone https://github.com/upcomingtrillioner12-design/subword_tokenizer.git
cd subword_tokenizer
cargo build --release
```

### First Training Run

```bash
# Train on sample corpus
cargo run --release -- --vocab-size 512 --output my_model.json

# Tokenize with learned model
cargo run --release -- --model my_model.json --tokenize "natural language processing"
```

**Output:**
```
Input:  "natural language processing"
Output: ["n", "at", "u", "r", "al ", "l", "anguage ", "p", "roc", "ess", "ing"]
```

## Features

✅ **Core BPE Algorithm** - Pair counting with iterative merging
✅ **CLI Interface** - 5 flexible arguments for training and inference  
✅ **JSON Models** - Portable, human-readable vocabulary and merges
✅ **Deterministic Tokenization** - Same input → same output
✅ **Cross-Platform** - Windows, macOS, Linux support
✅ **Library Export** - Import as Rust crate for embedding
✅ **CI/CD Pipeline** - Automated testing on all platforms
✅ **Comprehensive Tests** - 17 unit tests (5 lib + 12 binary)

## Usage

### Command-Line Interface

#### Train on Corpus
```bash
cargo run --release -- \
  --corpus data.txt \
  --vocab-size 512 \
  --output model.json
```

**Parameters:**
- `--corpus <FILE>`: Input text file (uses default sample if omitted)
- `--vocab-size <N>`: Target vocabulary size (default: 350)
- `--output <FILE>`: Save trained model as JSON

#### Tokenize Text
```bash
# Using trained model
cargo run --release -- \
  --model model.json \
  --tokenize "your text here"

# Quick test (uses default sample)
cargo run --release -- --tokenize "hello world"
```

**Parameters:**
- `--model <FILE>`: Pre-trained model JSON file
- `--tokenize <TEXT>`: Text to tokenize

#### Combined Training + Tokenization
```bash
cargo run --release -- \
  --corpus data.txt \
  --vocab-size 1024 \
  --tokenize "test phrase"
```

### Library Usage

Add to `Cargo.toml`:
```toml
[dependencies]
subword-tokenizer = { path = "./subword_tokenizer" }
```

**Example Code:**
```rust
use subword_tokenizer::{BPEModel, train};

// Train on corpus
let corpus = "Your training text here...";
let model = train(corpus, 512)?;

// Tokenize new text
let tokens = model.tokenize("hello world");
println!("{:?}", tokens);

// Save and load models
model.save(&"my_model.json".into())?;
let loaded = BPEModel::load(&"my_model.json".into())?;
```

## Project Structure

```
├── src/
│   ├── main.rs              # CLI entry point
│   ├── lib.rs               # Public library API
│   ├── cpp/
│   │   └── bpe.cpp         # C++ BPE implementation
│   └── tests.rs             # Integration tests
├── .github/
│   └── workflows/
│       └── ci.yml          # GitHub Actions pipeline
├── experiments/
│   └── run_experiments.py   # Evaluation framework
├── doc/
│   ├── architecture.md      # System diagram (Mermaid)
│   ├── architecture.png     # Architecture visualization
│   └── project_reference.pdf # Complete documentation
├── Cargo.toml              # Rust dependencies
├── build.rs                # C++ build configuration
└── README.md               # This file
```

## Performance Metrics

Results from evaluation on 3 diverse datasets with 4 vocab sizes:

| Vocab Size | Compression | Model Size | Inference Time |
|-----------|------------|-----------|-----------------|
| **256**   | 7.76x      | 1.3 KB    | 3.62 ms        |
| **512**   | 2.62x ⭐   | 25 KB     | 3.86 ms        |
| **1024**  | 1.82x      | 57 KB     | 3.99 ms        |
| **2048**  | 1.82x      | 57 KB     | 3.98 ms        |

**Recommendation:** Use **vocab_size=512** for best balance of compression, model size, and speed.

📊 Full experiment results: [EXPERIMENTS.md](EXPERIMENTS.md)

## Testing

### Run All Tests
```bash
cargo test
```

**Results:** 17 tests total
- 5 library tests (tokenization logic)
- 12 binary tests (CLI integration)

### Specific Test Categories
```bash
# Library tests only
cargo test --lib

# Binary tests only
cargo test --test '*'

# Verbose output
cargo test -- --nocapture
```

### Test Coverage
- ✓ Character tokenization
- ✓ Single and multiple merges
- ✓ Edge cases (empty strings, single chars)
- ✓ Deterministic behavior
- ✓ Model serialization/deserialization
- ✓ Merge ordering
- ✓ Whitespace handling

## Building & Deployment

### Development Build
```bash
cargo build          # Optimized for faster compilation
cargo run -- --help  # Run with CLI
```

### Release Build
```bash
cargo build --release  # Fully optimized binary
./target/release/subword-tokenizer --help
```

### Run Tests
```bash
cargo test --release   # Run all tests in optimized mode
```

### Cross-Platform Compilation

**macOS:**
```bash
cargo build --release --target x86_64-apple-darwin
```

**Linux:**
```bash
cargo build --release --target x86_64-unknown-linux-gnu
```

**Windows:**
```bash
cargo build --release --target x86_64-pc-windows-msvc
```

## CI/CD Pipeline

Automated testing via GitHub Actions on every push and PR:

- **Platforms:** Ubuntu (Linux), macOS, Windows
- **Rust Versions:** stable, nightly
- **Quality Gates:** rustfmt, clippy
- **Coverage:** Build + Test all combinations

View results: [Actions](https://github.com/upcomingtrillioner12-design/subword_tokenizer/actions)

## API Reference

### BPEModel (struct)

```rust
pub struct BPEModel {
    pub vocab: Vec<String>,
    pub merges: Vec<(String, String)>,
    pub vocab_size: i32,
}
```

**Methods:**
- `new(vocab, merges, vocab_size)` - Create model
- `tokenize(&text)` - Apply merges to text
- `save(&path)` - Write to JSON file
- `load(&path)` - Read from JSON file

### train() function

```rust
pub fn train(
    corpus_text: &str,
    vocab_size: i32
) -> Result<BPEModel, Box<dyn std::error::Error>>
```

Trains BPE on corpus and returns learned model.

## Examples

### Example 1: Train on Custom Corpus
```bash
echo "The quick brown fox jumps over the lazy dog" > corpus.txt
cargo run --release -- --corpus corpus.txt --vocab-size 300 --output fox_model.json
```

### Example 2: Load and Tokenize
```bash
cargo run --release -- --model fox_model.json --tokenize "quick brown"
```

### Example 3: Inspect Model JSON
```bash
cat fox_model.json | jq '.vocab | length'  # Count tokens
cat fox_model.json | jq '.merges | length' # Count merge operations
```

### Example 4: Library Integration
```rust
use subword_tokenizer::{BPEModel, train};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let corpus = "sample text for training";
    let model = train(corpus, 256)?;
    
    let tokens = model.tokenize("sample");
    println!("Tokens: {:?}", tokens);
    
    model.save(&"model.json".into())?;
    Ok(())
}
```

## Architecture

System components:

```
CLI Input
    ↓
[Rust main.rs] ← Clap argument parsing
    ↓
[src/lib.rs] ← BPEModel, train() function
    ↓
[FFI Boundary] ← C++ calls
    ↓
[C++ bpe.cpp] ← Core algorithm (pair counting, merging)
    ↓
Results: vocabulary + merges
    ↓
[serde_json] → JSON serialization
    ↓
Output: Model file or tokenized text
```

Detailed architecture: [doc/architecture.md](doc/architecture.md)

## Performance Optimization Tips

1. **Vocab Size Selection**
   - Mobile/Edge: use vocab=256 (1.3 KB, 3.6 ms)
   - General NLP: use vocab=512 (25 KB, 3.9 ms) ⭐
   - High compression: use vocab=1024 (57 KB, 4.0 ms)

2. **Corpus Size**
   - Larger corpora produce better vocabulary coverage
   - Recommended minimum: 10,000 characters

3. **Inference Speed**
   - Speed is limited by merge count, not vocab size
   - Typical: ~1-4 ms per inference
   - Batch processing available (see API)

## Troubleshooting

### Build Errors

**"C++ compiler not found"**
```bash
# macOS
brew install clang

# Linux (Ubuntu/Debian)
sudo apt install build-essential

# Windows
# Install Visual Studio Build Tools from Microsoft
```

**"library stdc++ not found"** (macOS)
- Fixed in build.rs with conditional linking
- Should work automatically on v1.0+

### Runtime Issues

**"File not found"**
```bash
# Use absolute or relative paths from project root
cargo run -- --corpus ./data/corpus.txt

# Or use relative path
cargo run -- --corpus data.txt
```

**Tokenization produces unexpected results**
- Verify model was trained on similar text
- Check merges were learned: `jq '.merges | length' model.json`
- Try different vocab_size

## Documentation

- **Quick Start:** This README
- **Comprehensive Guide:** [doc/project_reference.pdf](doc/project_reference.pdf)
- **Experiments & Metrics:** [EXPERIMENTS.md](EXPERIMENTS.md)
- **Features Demo:** [FEATURE_SHOWCASE.md](FEATURE_SHOWCASE.md)
- **Architecture:** [doc/architecture.md](doc/architecture.md)

## Contributing

Contributions welcome! Areas for enhancement:

- [ ] Pretokenization (word-level, punctuation handling)
- [ ] Text normalization (lowercase, diacritics)
- [ ] Python bindings (PyO3)
- [ ] Parallel merge computation
- [ ] tiktoken compatibility
- [ ] Web UI for training

## Roadmap

**Q3 2026:**
- [ ] Word-level pretokenization
- [ ] Unicode normalization
- [ ] Python bindings (PyO3)

**Q4 2026:**
- [ ] Publish to crates.io
- [ ] Performance optimization (SIMD)
- [ ] Extended language support

**2027:**
- [ ] Transformers integration
- [ ] Web UI dashboard
- [ ] Community tokenizer zoo

## Performance Comparisons

Comparison with other tokenizers on standard benchmarks coming soon.

## License

MIT License - see [LICENSE](LICENSE) file

## Citation

If you use this tokenizer in research, please cite:

```bibtex
@software{subword_tokenizer_2026,
  author = {Singh, Jaydip},
  title = {Subword Tokenizer: BPE in Rust + C++},
  year = {2026},
  url = {https://github.com/upcomingtrillioner12-design/subword_tokenizer}
}
```

## Support

- 📖 Read the [documentation](doc/project_reference.pdf)
- 🐛 Report bugs: [GitHub Issues](https://github.com/upcomingtrillioner12-design/subword_tokenizer/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/upcomingtrillioner12-design/subword_tokenizer/discussions)

---

**Last Updated:** June 17, 2026

**Status:** ✅ Production Ready | 17 Tests Passing | 3 Platforms Supported

Made with ❤️ by the Subword Tokenizer Team
