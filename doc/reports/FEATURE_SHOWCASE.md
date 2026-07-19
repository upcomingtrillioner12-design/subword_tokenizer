# Subword Tokenizer - Feature Showcase

## ✅ All 4 Features Implemented

### 1. CLI Argument Parsing ✓
Train on custom corpus files with configurable parameters:
```bash
cargo run -- --corpus data/corpora/raw/test_corpus.txt --vocab-size 300 --output model.json
cargo run -- --model model.json --tokenize "hello world"
```

**Options:**
- `--corpus FILE`: Load corpus from file
- `--vocab-size SIZE`: Target vocabulary size (default: 350)
- `--output FILE`: Save trained model to JSON
- `--tokenize TEXT`: Apply tokenization to text
- `--model FILE`: Load and use trained model

### 2. Vocabulary Serialization (JSON) ✓
Complete model persistence with vocabulary and merge operations:
```json
{
  "vocab": ["a", "b", ..., "hello", "world"],
  "merges": [["a", "b"], ["ab", "c"], ...],
  "vocab_size": 350
}
```

### 3. Tokenizer Inference Mode ✓
Load trained models and apply learned merges to new text:
```
Input: "natural language processing"
Output: ["n", "at", "u", "r", "al ", "l", "anguage ", "p", "roc", "ess", "ing"]
```

### 4. Comprehensive Unit Tests ✓
12 passing tests covering:
- Single character tokenization
- Empty string edge cases
- Merge application (single & multiple)
- Deterministic behavior verification
- Serialization/deserialization
- Partial merges and ordering

**Result:** `12 passed; 0 failed`

## Example Workflow

1. Train: `cargo run -- --corpus data/corpora/raw/test_corpus.txt --vocab-size 350 --output trained.json`
2. Infer: `cargo run -- --model trained.json --tokenize "text to tokenize"`
3. Test:  `cargo test`

## What's Been Delivered

✓ Core BPE algorithm (pair counting + iterative merging)
✓ Full CLI with argument parsing
✓ JSON serialization/deserialization
✓ Inference mode with deterministic results
✓ 12 comprehensive unit tests
✓ macOS build fixes
✓ Error handling and user feedback

**Status:** Ready for development branch deployment
