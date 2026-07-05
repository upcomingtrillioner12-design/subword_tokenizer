#!/usr/bin/env python3
"""
Preprocess arXiv physics papers:
- Extract title + abstract + key sections
- Remove noise, normalize whitespace
- Deduplicate
- Chunk long documents
- Output as JSONL
"""
import json
import re
from pathlib import Path
from collections import defaultdict
import argparse
from tqdm import tqdm

def clean_text(text):
    """Remove noise and normalize whitespace."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove control characters
    text = ''.join(c for c in text if ord(c) >= 32 or c in '\n\t')
    return text.strip()

def extract_content(paper_data):
    """Extract and combine key content from a paper."""
    title = paper_data.get("title", "").strip()
    abstract = paper_data.get("summary", "").strip()
    
    # Combine
    content = f"Title: {title}\n\nAbstract: {abstract}"
    return content

def chunk_document(content, chunk_size=4000, overlap=200):
    """Split long documents into chunks."""
    words = content.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.split()) > 50:  # Only keep meaningful chunks
            chunks.append(chunk)
    
    return chunks if chunks else [content]

def preprocess_papers(input_file, output_file, chunk_size=4000):
    """Process papers from JSONL input to cleaned JSONL output."""
    input_path = Path(input_file)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    seen_hashes = set()
    doc_count = 0
    chunk_count = 0
    
    print(f"Processing papers from: {input_file}")
    print(f"Output: {output_file}")
    
    with open(input_path, "r") as infile, open(output_path, "w") as outfile:
        for line in tqdm(infile, desc="Papers processed"):
            try:
                paper = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            
            content = extract_content(paper)
            content = clean_text(content)
            
            # Deduplication via hash
            content_hash = hash(content) % (10 ** 9)
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)
            
            # Chunk
            chunks = chunk_document(content, chunk_size)
            
            for chunk_idx, chunk in enumerate(chunks):
                doc = {
                    "arxiv_id": paper.get("arxiv_id", "unknown"),
                    "title": paper.get("title", ""),
                    "chunk": chunk_idx,
                    "text": chunk,
                    "length": len(chunk.split()),
                }
                outfile.write(json.dumps(doc) + "\n")
                chunk_count += 1
            
            doc_count += 1
    
    print(f"\nProcessed {doc_count} unique papers → {chunk_count} chunks")
    print(f"Output saved to: {output_file}")
    return str(output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess arXiv papers")
    parser.add_argument("--input-file", required=True, help="Input JSONL file from download_arxiv.py")
    parser.add_argument("--output-file", required=True, help="Output JSONL file")
    parser.add_argument("--chunk-size", type=int, default=4000, help="Words per chunk")
    args = parser.parse_args()
    
    preprocess_papers(
        input_file=args.input_file,
        output_file=args.output_file,
        chunk_size=args.chunk_size
    )
