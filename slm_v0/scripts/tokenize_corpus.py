#!/usr/bin/env python3
"""
Tokenize physics corpus using the Rust subword tokenizer.
Creates train/val/test splits and saves as binary files.
"""
import json
import subprocess
import ast
import shutil
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_PROJECT_DIR = ROOT / "subword_tokenizer"
TOKENIZER_ACTIVE_MODEL = TOKENIZER_PROJECT_DIR / "model.json"
TOKENIZER_BIN = TOKENIZER_PROJECT_DIR / "target" / "release" / "bpe-tokenizer"


def activate_tokenizer_model(tokenizer_path):
    """Activate selected tokenizer model for Rust CLI."""
    src = Path(tokenizer_path)
    if not src.exists():
        raise FileNotFoundError(f"Tokenizer model not found: {src}")
    shutil.copy2(src, TOKENIZER_ACTIVE_MODEL)


def tokenize_with_rust_tokenizer(text):
    """Call Rust tokenizer via CLI."""
    try:
        if TOKENIZER_BIN.exists():
            cmd = [str(TOKENIZER_BIN), "tokenize", text]
            cwd = str(TOKENIZER_PROJECT_DIR)
        else:
            cmd = ["cargo", "run", "--release", "--manifest-path", str(TOKENIZER_PROJECT_DIR / "Cargo.toml"), "--", "tokenize", text]
            cwd = str(ROOT)

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
        if result.returncode != 0:
            print(f"Tokenizer process failed: {result.stderr.strip() or result.stdout.strip()}")
            return []
        
        # Parse output: "IDs: [1, 2, 3]"
        for line in result.stdout.splitlines():
            if line.startswith("IDs:"):
                ids_str = line.split("IDs:", 1)[1].strip()
                return ast.literal_eval(ids_str)
    except Exception as e:
        print(f"Tokenizer error: {e}")
    
    return []

def tokenize_corpus(corpus_file, tokenizer_model, output_dir, seq_len=512, split_ratios=(0.98, 0.01, 0.01)):
    """Tokenize corpus and create train/val/test splits."""
    corpus_path = Path(corpus_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Tokenizing corpus: {corpus_file}")
    print(f"Tokenizer: {tokenizer_model}")
    print(f"Seq length: {seq_len}")

    activate_tokenizer_model(tokenizer_model)
    
    all_tokens = []
    doc_count = 0
    
    # Read corpus and tokenize
    print("Reading and tokenizing documents...")
    with open(corpus_path, "r") as f:
        for line in tqdm(f, desc="Tokenizing"):
            try:
                doc = json.loads(line.strip())
                text = doc.get("text", "")
                
                if len(text.split()) < 10:
                    continue
                
                tokens = tokenize_with_rust_tokenizer(text)
                all_tokens.extend(tokens)
                doc_count += 1
                
            except json.JSONDecodeError:
                continue
    
    print(f"Tokenized {doc_count} documents → {len(all_tokens)} tokens")
    
    # Create sequences of fixed length
    print("Creating fixed-length sequences...")
    sequences = []
    for i in range(0, len(all_tokens) - seq_len, seq_len):
        seq = all_tokens[i:i + seq_len]
        sequences.append(seq)
    
    print(f"Created {len(sequences)} sequences of length {seq_len}")
    
    # Train/val/test split
    n_train = int(len(sequences) * split_ratios[0])
    n_val = int(len(sequences) * split_ratios[1])
    
    train_seqs = sequences[:n_train]
    val_seqs = sequences[n_train:n_train + n_val]
    test_seqs = sequences[n_train + n_val:]
    
    print(f"Split: train={len(train_seqs)}, val={len(val_seqs)}, test={len(test_seqs)}")
    
    # Save as binary files
    def save_sequences(seqs, filename):
        arr = np.array(seqs, dtype=np.int32)
        np.save(output_path / filename, arr)
        print(f"Saved: {output_path / filename} ({arr.nbytes / 1e9:.2f} GB)")
    
    save_sequences(train_seqs, "train.npy")
    save_sequences(val_seqs, "val.npy")
    save_sequences(test_seqs, "test.npy")
    
    # Save stats
    stats = {
        "total_tokens": len(all_tokens),
        "total_sequences": len(sequences),
        "seq_length": seq_len,
        "train_sequences": len(train_seqs),
        "val_sequences": len(val_seqs),
        "test_sequences": len(test_seqs),
    }
    with open(output_path / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nStats saved to {output_path / 'stats.json'}")
    print(f"Total tokens: {stats['total_tokens']:,}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize physics corpus")
    parser.add_argument("--corpus", required=True, help="Input corpus file (JSONL)")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer model JSON")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    args = parser.parse_args()
    
    tokenize_corpus(
        corpus_file=args.corpus,
        tokenizer_model=args.tokenizer,
        output_dir=args.output,
        seq_len=args.seq_len
    )
