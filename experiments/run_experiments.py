#!/usr/bin/env python3
"""
Experiment matrix for BPE tokenizer evaluation.
Runs 12 experiments: 3 datasets × 4 vocab sizes
Measures compression ratio, token count, model size, and inference speed.
"""

import json
import subprocess
import os
import sys
import time
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class Experiment:
    dataset_name: str
    vocab_size: int
    corpus_chars: int
    tokens: int
    merges: int
    model_size_bytes: int
    compression_ratio: float
    inference_time_ms: float

# Define 3 diverse datasets
DATASETS = {
    "wikipedia_snippet": """
        Byte Pair Encoding (BPE) is a simple data compression technique that iteratively 
        replaces the most frequent pair of bytes in a sequence with a single, unused byte. 
        This algorithm is commonly used in natural language processing applications, particularly 
        in transformer-based models like BERT, GPT-2, and GPT-3. The BPE algorithm was originally 
        proposed as a compression mechanism but has become the standard tokenization method for 
        many modern language models. The process begins by treating each character as a separate 
        token, then iteratively merges the most common pairs of adjacent tokens until reaching 
        a target vocabulary size. This approach balances the vocabulary size with the ability 
        to represent arbitrary text sequences. The effectiveness of BPE depends on the target 
        vocabulary size and the structure of the training corpus.
    """,
    "code_snippet": """
        def tokenize(text: str) -> List[str]:
            tokens = list(text)
            for merge_pair in self.merges:
                merged = merge_pair[0] + merge_pair[1]
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if i + 1 < len(tokens) and tokens[i] == merge_pair[0] and tokens[i + 1] == merge_pair[1]:
                        new_tokens.append(merged)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                tokens = new_tokens
            return tokens
        
        class BPEModel:
            def __init__(self, vocab_size: int):
                self.vocab_size = vocab_size
                self.merges = []
                self.vocab = set()
    """,
    "mixed_content": """
        The quick brown fox jumps over the lazy dog. Natural language processing (NLP) is a 
        subfield of linguistics, computer science, and artificial intelligence concerned with 
        the interactions between computers and human language. NLP is used to apply machine learning 
        algorithms to text and speech. Machine learning is a type of artificial intelligence (AI) 
        that provides computers with the ability to learn and improve from experience without being 
        explicitly programmed. Deep learning is part of a broader family of machine learning methods 
        based on artificial neural networks with representation learning. Transformer models have 
        revolutionized the field of NLP by introducing self-attention mechanisms that allow models 
        to process input sequences in parallel while maintaining long-range dependencies. The 
        BERT model pre-trained on masked language modeling has achieved state-of-the-art results 
        on many NLP benchmarks including question answering, sentiment analysis, and named entity 
        recognition. GPT-3 is a large-scale language model trained on diverse internet text that 
        can perform various NLP tasks without task-specific fine-tuning.
    """
}

VOCAB_SIZES = [256, 512, 1024, 2048]

def run_experiment(dataset_name: str, dataset_text: str, vocab_size: int, 
                   tokenizer_bin: Path) -> Experiment:
    """Run a single BPE training experiment."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        corpus_file = Path(tmpdir) / f"{dataset_name}_corpus.txt"
        model_file = Path(tmpdir) / f"{dataset_name}_{vocab_size}_model.json"
        
        # Write corpus
        corpus_file.write_text(dataset_text.strip())
        corpus_chars = len(dataset_text.strip())
        
        # Train BPE
        start_time = time.time()
        result = subprocess.run(
            [str(tokenizer_bin), "--corpus", str(corpus_file), 
             "--vocab-size", str(vocab_size), "--output", str(model_file)],
            capture_output=True,
            text=True,
            timeout=10
        )
        train_time = time.time() - start_time
        
        if result.returncode != 0:
            print(f"✗ Training failed for {dataset_name} (vocab={vocab_size})")
            print(result.stderr)
            sys.exit(1)
        
        # Load model and extract metrics
        model_data = json.loads(model_file.read_text())
        tokens = len(model_data['vocab'])
        merges = len(model_data['merges'])
        model_size = model_file.stat().st_size
        
        # Tokenize test string for inference speed
        test_string = "tokenization performance"
        start_time = time.time()
        for _ in range(100):  # Run 100 times for more accurate measurement
            inference_result = subprocess.run(
                [str(tokenizer_bin), "--model", str(model_file), 
                 "--tokenize", test_string],
                capture_output=True,
                text=True,
                timeout=5
            )
        inference_time_ms = (time.time() - start_time) / 100 * 1000
        
        # Calculate compression ratio
        compression_ratio = corpus_chars / tokens if tokens > 0 else 0
        
        return Experiment(
            dataset_name=dataset_name,
            vocab_size=vocab_size,
            corpus_chars=corpus_chars,
            tokens=tokens,
            merges=merges,
            model_size_bytes=model_size,
            compression_ratio=compression_ratio,
            inference_time_ms=inference_time_ms
        )

def main():
    # Find the tokenizer binary - look in parent directory's target folder
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    tokenizer_bin = project_root / "target" / "release" / "subword-tokenizer"
    
    if not tokenizer_bin.exists():
        print(f"✗ Tokenizer binary not found at {tokenizer_bin}")
        print("Please run: cargo build --release")
        sys.exit(1)
    
    print("╔════════════════════════════════════════════╗")
    print("║   BPE Tokenizer Experiment Matrix          ║")
    print("║   3 Datasets × 4 Vocab Sizes = 12 Exps    ║")
    print("╚════════════════════════════════════════════╝\n")
    
    results: List[Experiment] = []
    total_experiments = len(DATASETS) * len(VOCAB_SIZES)
    current = 0
    
    # Run all experiments
    for dataset_name, dataset_text in DATASETS.items():
        print(f"\n📊 Dataset: {dataset_name}")
        print(f"   Corpus size: {len(dataset_text.strip())} chars")
        
        for vocab_size in VOCAB_SIZES:
            current += 1
            print(f"   [{current}/{total_experiments}] Training with vocab_size={vocab_size}...", end=" ", flush=True)
            
            exp = run_experiment(dataset_name, dataset_text, vocab_size, tokenizer_bin)
            results.append(exp)
            
            print(f"✓ tokens={exp.tokens}, merges={exp.merges}, "
                  f"ratio={exp.compression_ratio:.2f}, model={exp.model_size_bytes}B")
    
    # Generate report
    print("\n" + "="*120)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*120)
    
    print("\n{:<20} {:<12} {:<12} {:<12} {:<15} {:<15} {:<15}".format(
        "Dataset", "Vocab Size", "Tokens", "Merges", "Compression", "Model Size", "Inference (ms)"
    ))
    print("-"*120)
    
    for exp in results:
        print("{:<20} {:<12} {:<12} {:<12} {:<15.2f} {:<15} {:<15.4f}".format(
            exp.dataset_name, exp.vocab_size, exp.tokens, exp.merges,
            exp.compression_ratio, f"{exp.model_size_bytes}B", exp.inference_time_ms
        ))
    
    # Generate analysis by vocab size
    print("\n" + "="*120)
    print("ANALYSIS BY VOCABULARY SIZE")
    print("="*120)
    
    for vocab_size in VOCAB_SIZES:
        vocab_exps = [e for e in results if e.vocab_size == vocab_size]
        avg_compression = sum(e.compression_ratio for e in vocab_exps) / len(vocab_exps)
        avg_model_size = sum(e.model_size_bytes for e in vocab_exps) / len(vocab_exps)
        avg_inference = sum(e.inference_time_ms for e in vocab_exps) / len(vocab_exps)
        
        print(f"\nVocab Size: {vocab_size}")
        print(f"  Avg Compression Ratio: {avg_compression:.2f}")
        print(f"  Avg Model Size: {avg_model_size:.0f} bytes")
        print(f"  Avg Inference Time: {avg_inference:.4f} ms")
    
    # Generate analysis by dataset
    print("\n" + "="*120)
    print("ANALYSIS BY DATASET")
    print("="*120)
    
    for dataset_name in DATASETS.keys():
        dataset_exps = [e for e in results if e.dataset_name == dataset_name]
        avg_compression = sum(e.compression_ratio for e in dataset_exps) / len(dataset_exps)
        avg_tokens = sum(e.tokens for e in dataset_exps) / len(dataset_exps)
        
        print(f"\n{dataset_name}:")
        print(f"  Avg Tokens: {avg_tokens:.0f}")
        print(f"  Avg Compression Ratio: {avg_compression:.2f}")
        print(f"  Vocab Size Range: {min(e.vocab_size for e in dataset_exps)} - {max(e.vocab_size for e in dataset_exps)}")
    
    # Save results to JSON
    results_file = project_root / "experiments_results.json"
    results_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_experiments": len(results),
        "datasets": list(DATASETS.keys()),
        "vocab_sizes": VOCAB_SIZES,
        "experiments": [
            {
                "dataset": e.dataset_name,
                "vocab_size": e.vocab_size,
                "corpus_chars": e.corpus_chars,
                "tokens": e.tokens,
                "merges": e.merges,
                "model_size_bytes": e.model_size_bytes,
                "compression_ratio": round(e.compression_ratio, 2),
                "inference_time_ms": round(e.inference_time_ms, 4),
            }
            for e in results
        ]
    }
    
    results_file.write_text(json.dumps(results_data, indent=2))
    print(f"\n✓ Results saved to: {results_file}")
    
    print("\n" + "="*120)
    print("RECOMMENDATIONS")
    print("="*120)
    
    # Find best vocab size for different use cases
    by_compression = sorted(results, key=lambda e: e.compression_ratio, reverse=True)
    by_model_size = sorted(results, key=lambda e: e.model_size_bytes)
    by_inference = sorted(results, key=lambda e: e.inference_time_ms)
    
    print(f"\n✓ Best for compression: {by_compression[0].dataset_name} with vocab={by_compression[0].vocab_size} "
          f"(ratio={by_compression[0].compression_ratio:.2f})")
    print(f"✓ Best for model size: {by_model_size[0].dataset_name} with vocab={by_model_size[0].vocab_size} "
          f"({by_model_size[0].model_size_bytes} bytes)")
    print(f"✓ Best for speed: {by_inference[0].dataset_name} with vocab={by_inference[0].vocab_size} "
          f"({by_inference[0].inference_time_ms:.4f} ms)")
    
    print("\n✅ Experiment matrix complete!\n")

if __name__ == "__main__":
    main()
