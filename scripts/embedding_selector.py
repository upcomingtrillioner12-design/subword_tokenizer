"""
Embedding Model Selector - Support for multiple embeddings with unified interface
Supports: all-mpnet-base-v2, instructor-embedding, SciBERT
Phase 4 Task 7: Alternative embeddings evaluation
"""

import torch
import numpy as np
from typing import List, Union, Optional, Dict, Any
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel


class EmbeddingSelector:
    """Unified interface for multiple embedding models"""
    
    def __init__(self, model_name: str, device: Optional[str] = None, **kwargs):
        """
        Initialize embedding model
        
        Args:
            model_name: One of ['all-mpnet-base-v2', 'instructor', 'sciBERT', 'scientific']
            device: torch device (defaults to cuda/mps if available, else cpu)
            **kwargs: Additional arguments for model initialization
        """
        self.model_name = model_name
        self.device = device or ('mps' if torch.backends.mps.is_available() else 
                                  'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.model_type = self._infer_model_type(model_name)
        
        self._load_model()
        
    def _infer_model_type(self, model_name: str) -> str:
        """Infer model type from name"""
        model_lower = model_name.lower()
        if 'instructor' in model_lower:
            return 'sentence_transformers_instruct'
        elif 'scibert' in model_lower or 'scientific' in model_lower:
            return 'huggingface_pooled'
        elif 'mpnet' in model_lower or 'sentence' in model_lower:
            return 'sentence_transformers'
        else:
            return 'sentence_transformers'
    
    def _load_model(self):
        """Load embedding model based on type"""
        try:
            if self.model_type == 'sentence_transformers':
                # Standard SentenceTransformers models
                self.model = SentenceTransformer(self.model_name, device=self.device)
                
            elif self.model_type == 'sentence_transformers_instruct':
                # Instructor-embedding with instruction capability
                # Note: Falls back to sentence_transformers if instructor not installed
                try:
                    from InstructorEmbedding import INSTRUCTOR
                    self.model = INSTRUCTOR(self.model_name)
                except ImportError:
                    print(f"Warning: InstructorEmbedding not installed, falling back to all-mpnet")
                    self.model = SentenceTransformer('all-mpnet-base-v2', device=self.device)
                    self.model_type = 'sentence_transformers'
                    
            elif self.model_type == 'huggingface_pooled':
                # HuggingFace models with mean pooling
                self.model = SentenceTransformer(self.model_name, device=self.device)
                
            print(f"✓ Loaded {self.model_name} ({self.model_type}) on {self.device}")
            
        except Exception as e:
            print(f"Error loading {self.model_name}: {e}")
            print("Falling back to all-mpnet-base-v2")
            self.model = SentenceTransformer('all-mpnet-base-v2', device=self.device)
            self.model_type = 'sentence_transformers'
    
    def encode(self, texts: Union[str, List[str]], 
               instruction: Optional[str] = None,
               batch_size: int = 32,
               normalize: bool = True) -> np.ndarray:
        """
        Encode texts to embeddings
        
        Args:
            texts: String or list of strings
            instruction: Optional instruction for instructor-embedding
            batch_size: Batch size for processing
            normalize: Whether to L2 normalize embeddings
            
        Returns:
            np.ndarray of shape (n_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            if self.model_type == 'sentence_transformers_instruct' and instruction:
                # Instructor-embedding with custom instruction
                instruction_inputs = [[instruction, text] for text in texts]
                embeddings = self.model.encode(instruction_inputs, batch_size=batch_size)
            else:
                # Standard encoding
                embeddings = self.model.encode(texts, batch_size=batch_size, 
                                               normalize_embeddings=normalize,
                                               show_progress_bar=False)
            
            return np.array(embeddings)
            
        except Exception as e:
            print(f"Error encoding with {self.model_name}: {e}")
            return np.zeros((len(texts), 768))
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension"""
        try:
            if hasattr(self.model, 'get_sentence_embedding_dimension'):
                return self.model.get_sentence_embedding_dimension()
            elif hasattr(self.model, 'word_embedding_dimension'):
                return self.model.word_embedding_dimension
            else:
                # Test with dummy input
                dummy = self.encode("test")[0]
                return len(dummy)
        except:
            return 768
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata"""
        return {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'device': self.device,
            'embedding_dim': self.get_embedding_dim(),
            'supports_instruction': self.model_type == 'sentence_transformers_instruct'
        }


class EmbeddingComparator:
    """Compare multiple embedding models on same dataset"""
    
    def __init__(self, models: List[str], device: Optional[str] = None):
        """
        Initialize multiple models for comparison
        
        Args:
            models: List of model names
            device: Device to use
        """
        self.models = {}
        self.device = device
        
        for model_name in models:
            try:
                self.models[model_name] = EmbeddingSelector(model_name, device=device)
            except Exception as e:
                print(f"Failed to load {model_name}: {e}")
    
    def encode_all(self, texts: List[str], 
                   instructions: Optional[Dict[str, str]] = None) -> Dict[str, np.ndarray]:
        """
        Encode texts with all loaded models
        
        Args:
            texts: Texts to encode
            instructions: Optional dict mapping model_name -> instruction
            
        Returns:
            Dict mapping model_name -> embeddings
        """
        results = {}
        for model_name, model in self.models.items():
            instruction = instructions.get(model_name) if instructions else None
            results[model_name] = model.encode(texts, instruction=instruction)
        return results
    
    def compute_similarities(self, query: str, corpus: List[str],
                            top_k: int = 5) -> Dict[str, List[tuple]]:
        """
        Compute top-K similarities for query across all models
        
        Args:
            query: Query text
            corpus: Corpus texts
            top_k: Number of top results per model
            
        Returns:
            Dict mapping model_name -> [(text, similarity), ...]
        """
        query_embeddings = self.encode_all([query])
        corpus_embeddings_all = self.encode_all(corpus)
        
        results = {}
        for model_name in self.models.keys():
            query_emb = query_embeddings[model_name][0]
            corpus_embs = corpus_embeddings_all[model_name]
            
            # Cosine similarity
            similarities = corpus_embs @ query_emb / (
                np.linalg.norm(corpus_embs, axis=1) * np.linalg.norm(query_emb) + 1e-8
            )
            
            top_indices = np.argsort(similarities)[::-1][:top_k]
            results[model_name] = [(corpus[i], float(similarities[i])) 
                                   for i in top_indices]
        
        return results
    
    def get_all_model_info(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata for all models"""
        return {name: model.get_model_info() 
                for name, model in self.models.items()}


if __name__ == '__main__':
    # Example: Compare embeddings
    print("=== Embedding Model Comparison ===\n")
    
    # Initialize multiple models
    models = ['all-mpnet-base-v2']  # Add 'instructor' and 'sciBERT' if available
    comparator = EmbeddingComparator(models)
    
    # Print model info
    print("Loaded Models:")
    for name, info in comparator.get_all_model_info().items():
        print(f"  {name}: dim={info['embedding_dim']}, device={info['device']}")
    
    # Test encoding
    test_texts = [
        "What is Planck's constant?",
        "Calculate the derivative of x^2",
        "Define photosynthesis"
    ]
    
    print("\n=== Encoding Test ===")
    embeddings = comparator.encode_all(test_texts)
    for model_name, embs in embeddings.items():
        print(f"{model_name}: shape={embs.shape}")
    
    # Test similarity
    print("\n=== Similarity Test ===")
    query = "What is the speed of light?"
    corpus = [
        "3 × 10^8 m/s is a constant",
        "The derivative of cos is -sin",
        "Light travels at 299,792,458 m/s"
    ]
    
    similarities = comparator.compute_similarities(query, corpus, top_k=2)
    for model_name, results in similarities.items():
        print(f"\n{model_name}:")
        for text, sim in results:
            print(f"  {sim:.4f}: {text[:50]}...")
