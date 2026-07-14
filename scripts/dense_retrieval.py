#!/usr/bin/env python3
"""
Dense Retrieval Engine

Implements vector-based semantic search using sentence transformers and FAISS.
Supports building and querying dense indexes with configurable embedding models.

Features:
- Sentence-transformers for semantic embeddings
- FAISS for efficient vector similarity search
- Batch processing for fast indexing
- Multiple index types (FlatL2, IVFFLAT, HNSW)
- Metadata management for document tracking
"""

import json
import argparse
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np

# External dependencies
try:
    from sentence_transformers import SentenceTransformer
    import faiss
except ImportError as e:
    raise ImportError(
        f"Missing dependencies: {e}\n"
        "Install with: pip install sentence-transformers faiss-cpu"
    )


class DenseRetriever:
    """
    Dense retriever using sentence transformers and FAISS.
    
    Attributes:
        embedding_model: SentenceTransformer model for generating embeddings
        index: FAISS index for vector similarity search
        metadata: List of chunk metadata (doc_id, text, etc.)
        embedding_dim: Dimensionality of embeddings
    """
    
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        """
        Initialize dense retriever.
        
        Args:
            model_name: HuggingFace model identifier for embeddings.
                       Recommended: "all-mpnet-base-v2" (384-dim) or
                                   "all-minilm-l6-v2" (384-dim, lighter)
        """
        print(f"Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        # Use get_embedding_dimension (newer) with fallback to get_sentence_embedding_dimension
        if hasattr(self.embedding_model, "get_embedding_dimension"):
            self.embedding_dim = self.embedding_model.get_embedding_dimension()
        else:
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        self.index = None
        self.metadata = []
        print(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def build_index_from_corpus(
        self,
        corpus_jsonl_path: str,
        output_dir: str,
        chunk_size: int = 220,
        chunk_overlap: int = 40,
        index_type: str = "flat"
    ) -> None:
        """
        Build FAISS index from corpus JSONL file.
        
        Args:
            corpus_jsonl_path: Path to corpus.jsonl (one doc per line)
            output_dir: Directory to save index and metadata
            chunk_size: Target tokens per chunk (approximate)
            chunk_overlap: Token overlap between chunks
            index_type: "flat" (exact) or "ivf" (approximate)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nBuilding dense index from {corpus_jsonl_path}")
        print(f"  Output directory: {output_dir}")
        print(f"  Chunk size: {chunk_size} tokens, overlap: {chunk_overlap}")
        
        # 1. Load and chunk corpus
        chunks = self._load_and_chunk_corpus(corpus_jsonl_path, chunk_size, chunk_overlap)
        print(f"  Total chunks: {len(chunks)}")
        
        # 2. Generate embeddings (batch processing)
        embeddings = self._generate_embeddings([c["text"] for c in chunks])
        embeddings = np.array(embeddings).astype("float32")
        print(f"  Embeddings shape: {embeddings.shape}")
        
        # 3. Build FAISS index
        if index_type.lower() == "flat":
            index = faiss.IndexFlatL2(self.embedding_dim)
            index.add(embeddings)
            print(f"  Index type: FlatL2 (exact search)")
        elif index_type.lower() == "ivf":
            # IVFFlat with sqrt(n) clusters
            nlist = max(64, min(1024, int(np.sqrt(len(chunks)))))
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
            index.train(embeddings)
            index.add(embeddings)
            print(f"  Index type: IVFFlat (approximate, nlist={nlist})")
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        # 4. Save index and metadata
        self.index = index
        self.metadata = chunks
        
        index_path = output_path / "dense_index.faiss"
        faiss.write_index(index, str(index_path))
        print(f"  Saved index: {index_path}")
        
        metadata_path = output_path / "dense_index_metadata.jsonl"
        with open(metadata_path, "w") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk) + "\n")
        print(f"  Saved metadata: {metadata_path}")
        
        print("✓ Index build complete")
    
    def load_index(self, index_path: str, metadata_path: str) -> None:
        """
        Load prebuilt index and metadata from disk.
        
        Args:
            index_path: Path to .faiss file
            metadata_path: Path to metadata .jsonl file
        """
        print(f"\nLoading index from {index_path}")
        self.index = faiss.read_index(index_path)
        
        self.metadata = []
        with open(metadata_path, "r") as f:
            for line in f:
                self.metadata.append(json.loads(line))
        
        print(f"  Index loaded. Total vectors: {self.index.ntotal}")
        print(f"  Metadata loaded. Total chunks: {len(self.metadata)}")
    
    def query(self, query_text: str, k: int = 5) -> List[Dict]:
        """
        Retrieve top-k chunks for a query.
        
        Args:
            query_text: Query text to search for
            k: Number of results to return
        
        Returns:
            List of dicts with keys: doc_id, chunk_id, text, score, rank
        """
        if self.index is None:
            raise ValueError("Index not loaded. Call build_index or load_index first.")
        
        # Embed query
        query_embedding = self.embedding_model.encode([query_text])[0]
        query_embedding = np.array([query_embedding]).astype("float32")
        
        # Search index
        scores, indices = self.index.search(query_embedding, k)
        scores = scores[0]
        indices = indices[0]
        
        # Format results
        results = []
        for rank, (idx, score) in enumerate(zip(indices, scores), 1):
            if idx >= 0 and idx < len(self.metadata):
                chunk = self.metadata[idx]
                results.append({
                    "rank": rank,
                    "doc_id": chunk["doc_id"],
                    "chunk_id": chunk.get("chunk_id", idx),
                    "text": chunk["text"],
                    "score": float(score),
                    "source": chunk.get("source", "unknown")
                })
        
        return results
    
    def batch_query(self, queries: List[str], k: int = 5) -> List[List[Dict]]:
        """
        Retrieve top-k chunks for multiple queries efficiently.
        
        Args:
            queries: List of query texts
            k: Number of results per query
        
        Returns:
            List of result lists (one per query)
        """
        if self.index is None:
            raise ValueError("Index not loaded. Call build_index or load_index first.")
        
        # Embed all queries at once
        query_embeddings = self.embedding_model.encode(queries)
        query_embeddings = np.array(query_embeddings).astype("float32")
        
        # Search all at once
        scores, indices = self.index.search(query_embeddings, k)
        
        # Format results
        all_results = []
        for q_idx, (query_scores, query_indices) in enumerate(zip(scores, indices)):
            results = []
            for rank, (idx, score) in enumerate(zip(query_indices, query_scores), 1):
                if idx >= 0 and idx < len(self.metadata):
                    chunk = self.metadata[idx]
                    results.append({
                        "rank": rank,
                        "doc_id": chunk["doc_id"],
                        "chunk_id": chunk.get("chunk_id", idx),
                        "text": chunk["text"],
                        "score": float(score),
                        "source": chunk.get("source", "unknown")
                    })
            all_results.append(results)
        
        return all_results
    
    def save(self, output_dir: str) -> None:
        """Save current index and metadata to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.index is None:
            raise ValueError("No index to save. Build or load an index first.")
        
        index_path = output_path / "dense_index.faiss"
        faiss.write_index(self.index, str(index_path))
        
        metadata_path = output_path / "dense_index_metadata.jsonl"
        with open(metadata_path, "w") as f:
            for chunk in self.metadata:
                f.write(json.dumps(chunk) + "\n")
        
        print(f"Saved index to {output_path}")
    
    # ========== Helper Methods ==========
    
    def _load_and_chunk_corpus(
        self,
        corpus_path: str,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[Dict]:
        """
        Load corpus and create overlapping chunks.
        
        Args:
            corpus_path: Path to corpus.jsonl
            chunk_size: Target tokens per chunk (approximate)
            chunk_overlap: Token overlap between chunks
        
        Returns:
            List of chunks with metadata
        """
        chunks = []
        chunk_id_counter = 0
        
        with open(corpus_path, "r") as f:
            for line_idx, line in enumerate(f):
                doc = json.loads(line)
                
                # Extract document metadata
                doc_id = doc.get("doc_id", doc.get("id", f"doc_{line_idx:06d}"))
                source = doc.get("source", "unknown")
                
                # Try multiple field names for text content
                text = doc.get("text", "")
                if not text and "abstract" in doc:
                    # For physics papers: combine title + abstract
                    title = doc.get("title", "")
                    abstract = doc.get("abstract", "")
                    text = f"{title} {abstract}".strip()
                
                if not text:
                    continue
                
                # Split into tokens (simple split for approximate chunking)
                tokens = text.split()
                
                # Create overlapping chunks
                step = max(1, chunk_size - chunk_overlap)
                for i in range(0, len(tokens), step):
                    chunk_tokens = tokens[i:i + chunk_size]
                    if chunk_tokens:
                        chunk_text = " ".join(chunk_tokens)
                        chunks.append({
                            "doc_id": doc_id,
                            "chunk_id": chunk_id_counter,
                            "text": chunk_text,
                            "source": source,
                            "doc_line": line_idx,
                            "token_start": i,
                            "token_count": len(chunk_tokens),
                            "title": doc.get("title", "")
                        })
                        chunk_id_counter += 1
        
        return chunks
    
    def _generate_embeddings(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for texts with batching.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            show_progress: Whether to print progress
        
        Returns:
            Array of embeddings (n_texts, embedding_dim)
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(batch, show_progress_bar=False)
            embeddings.extend(batch_embeddings)
            
            if show_progress and (i + batch_size) % (batch_size * 10) == 0:
                print(f"  Embedded {i + batch_size}/{len(texts)} texts")
        
        return embeddings


def main():
    """CLI interface for building and querying dense indexes."""
    parser = argparse.ArgumentParser(description="Dense Retrieval Engine")
    subparsers = parser.add_subparsers(dest="command", help="Subcommand")
    
    # Build subcommand
    build_parser = subparsers.add_parser("build", help="Build dense index from corpus")
    build_parser.add_argument("--input", required=True, help="Path to corpus.jsonl")
    build_parser.add_argument("--output-dir", required=True, help="Output directory for index")
    build_parser.add_argument("--model", default="all-mpnet-base-v2", help="Embedding model")
    build_parser.add_argument("--chunk-size", type=int, default=220, help="Chunk size in tokens")
    build_parser.add_argument("--chunk-overlap", type=int, default=40, help="Chunk overlap in tokens")
    build_parser.add_argument("--index-type", default="flat", choices=["flat", "ivf"], 
                             help="Index type (flat=exact, ivf=approximate)")
    
    # Query subcommand
    query_parser = subparsers.add_parser("query", help="Query dense index")
    query_parser.add_argument("--index", required=True, help="Path to .faiss index")
    query_parser.add_argument("--metadata", required=True, help="Path to metadata.jsonl")
    query_parser.add_argument("--model", default="all-mpnet-base-v2", help="Embedding model")
    query_parser.add_argument("--q", required=True, help="Query text")
    query_parser.add_argument("--k", type=int, default=5, help="Number of results")
    
    args = parser.parse_args()
    
    if args.command == "build":
        retriever = DenseRetriever(model_name=args.model)
        retriever.build_index_from_corpus(
            corpus_jsonl_path=args.input,
            output_dir=args.output_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            index_type=args.index_type
        )
    
    elif args.command == "query":
        retriever = DenseRetriever(model_name=args.model)
        retriever.load_index(args.index, args.metadata)
        
        print(f"\nQuery: {args.q}\n")
        results = retriever.query(args.q, k=args.k)
        
        for result in results:
            print(f"[{result['rank']}] score={result['score']:.2f} | {result['doc_id']}")
            print(f"    {result['text'][:100]}...\n")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
