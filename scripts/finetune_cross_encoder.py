"""
Task 10.2b: Cross-Encoder Fine-Tuning on Domain Preferences

Fine-tunes MS Marco pre-trained cross-encoder on STEM+adversarial preference pairs.
Goal: Better ranking of context documents → improved retrieval precision → accuracy boost.

Architecture:
  1. Collect top-5 retrieved docs for each STEM question
  2. Label as: 1.0 if contains answer information, 0.0 otherwise
  3. Fine-tune last 3 layers of cross-encoder (freeze earlier layers)
  4. Validate on held-out STEM questions
  5. Integrate into evaluation pipeline

Expected Impact:
  - Retrieval precision: +5-10pp
  - Downstream accuracy: +2-3pp on adversarial set
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


class PreferencePairDataset(Dataset):
    """Dataset of (query, document, label) tuples for cross-encoder fine-tuning."""
    
    def __init__(
        self,
        pairs: List[Dict],
        tokenizer,
        max_length: int = 512
    ):
        """
        Args:
            pairs: List of dicts with keys: query, document, label (0.0 or 1.0)
            tokenizer: Cross-encoder tokenizer
            max_length: Max token length for tokenization
        """
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        logger.info(f"Dataset: {len(pairs)} pairs, {len([p for p in pairs if p['label'] > 0.5])} positive")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        query = pair['query']
        document = pair['document']
        label = float(pair['label'])
        
        # Tokenize: cross-encoder expects [CLS] query [SEP] document [SEP]
        inputs = self.tokenizer(
            query,
            document,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float32)
        }


class CrossEncoderFineTuner:
    """Fine-tunes MS Marco cross-encoder on domain preferences."""
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        learning_rate: float = 2e-5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.device = device
        
        logger.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
        self.model.to(device)
        
        # Freeze earlier layers, only fine-tune last 3
        self._freeze_early_layers()
    
    def _freeze_early_layers(self):
        """Freeze all but last 3 layers to prevent catastrophic forgetting."""
        total_layers = len(list(self.model.transformer.layer))
        freeze_until = max(0, total_layers - 3)
        
        for i, layer in enumerate(self.model.transformer.layer):
            if i < freeze_until:
                for param in layer.parameters():
                    param.requires_grad = False
        
        # Always keep classifier trainable
        for param in self.model.classifier.parameters():
            param.requires_grad = True
        
        logger.info(f"Frozen layers 0-{freeze_until-1}, fine-tuning {total_layers-freeze_until} layers + classifier")
    
    def fine_tune(
        self,
        train_pairs: List[Dict],
        val_pairs: Optional[List[Dict]] = None,
        epochs: int = 3,
        batch_size: int = 8,
        early_stopping_patience: int = 2
    ) -> Dict:
        """
        Fine-tune the cross-encoder on preference pairs.
        
        Args:
            train_pairs: List of (query, document, label) dicts
            val_pairs: Validation pairs (optional)
            epochs: Number of fine-tuning epochs
            batch_size: Batch size
            early_stopping_patience: Stop if val loss doesn't improve
        
        Returns:
            Dict with training metrics
        """
        # Create datasets
        train_dataset = PreferencePairDataset(train_pairs, self.tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_loader = None
        if val_pairs:
            val_dataset = PreferencePairDataset(val_pairs, self.tokenizer)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate
        )
        
        # Loss function: Binary cross-entropy (1.0 = relevant, 0.0 = irrelevant)
        loss_fn = nn.BCEWithLogitsLoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device).unsqueeze(-1)  # Shape: (batch, 1)
                
                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                ).logits
                
                loss = loss_fn(logits, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            logger.info(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}")
            
            # Validation
            if val_loader:
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        labels = batch['label'].to(self.device).unsqueeze(-1)
                        
                        logits = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask
                        ).logits
                        
                        loss = loss_fn(logits, labels)
                        val_loss += loss.item()
                        
                        # Accuracy: threshold at 0.5
                        predictions = (logits > 0.0).float()
                        val_correct += (predictions == labels).sum().item()
                        val_total += labels.size(0)
                
                val_loss /= len(val_loader)
                val_accuracy = val_correct / val_total
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_accuracy)
                
                logger.info(f"         val_loss={val_loss:.4f}, val_acc={val_accuracy:.4f}")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    logger.info("  ✓ New best validation loss, saving checkpoint")
                    self.save_checkpoint(f"best_cross_encoder_finetuned.pt")
                else:
                    patience_counter += 1
                    logger.info(f"  ! No improvement ({patience_counter}/{early_stopping_patience})")
                    
                    if patience_counter >= early_stopping_patience:
                        logger.info("Early stopping triggered")
                        break
        
        logger.info(f"Fine-tuning complete. Best val_loss: {best_val_loss:.4f}")
        return history
    
    def evaluate_retrieval_precision(
        self,
        questions_file: str,
        retrieval_results_file: str,
        context_file: str,
        top_k: int = 5
    ) -> Dict:
        """
        Evaluate retrieval precision on a set of questions.
        
        This compares fine-tuned ranking against baseline ranking.
        
        Args:
            questions_file: JSON file with questions
            retrieval_results_file: JSON file with top-k retrieved docs per question
            context_file: JSON file with document texts
            top_k: Top-k precision to compute
        
        Returns:
            Dict with precision metrics before/after fine-tuning
        """
        # Load data
        with open(questions_file) as f:
            questions = json.load(f)
        with open(retrieval_results_file) as f:
            retrieval_results = json.load(f)
        with open(context_file) as f:
            contexts = json.load(f)
        
        self.model.eval()
        
        # For each question, re-rank the retrieved documents
        hits_at_k = 0
        total_questions = 0
        
        for question in questions:
            q_id = question['id']
            query = question['query']
            expected_answer = question['expected_answer']
            
            if q_id not in retrieval_results:
                continue
            
            retrieved_docs = retrieval_results[q_id]['docs'][:top_k]
            
            # Score each doc with fine-tuned model
            scores = []
            for doc_id in retrieved_docs:
                if doc_id not in contexts:
                    scores.append(0.0)
                    continue
                
                document = contexts[doc_id]['text']
                
                # Tokenize
                inputs = self.tokenizer(
                    query,
                    document,
                    max_length=512,
                    truncation=True,
                    padding='max_length',
                    return_tensors='pt'
                )
                
                # Score
                with torch.no_grad():
                    logits = self.model(
                        input_ids=inputs['input_ids'].to(self.device),
                        attention_mask=inputs['attention_mask'].to(self.device)
                    ).logits
                    score = torch.sigmoid(logits).item()
                
                scores.append(score)
            
            # Sort by score and check if top doc contains expected answer
            sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            top_doc_id = retrieved_docs[sorted_indices[0]] if sorted_indices else None
            
            if top_doc_id and top_doc_id in contexts:
                if expected_answer.lower() in contexts[top_doc_id]['text'].lower():
                    hits_at_k += 1
            
            total_questions += 1
        
        precision_at_k = hits_at_k / total_questions if total_questions > 0 else 0.0
        logger.info(f"Precision@{top_k}: {precision_at_k:.4f} ({hits_at_k}/{total_questions})")
        
        return {
            'precision_at_k': precision_at_k,
            'hits': hits_at_k,
            'total': total_questions
        }
    
    def save_checkpoint(self, path: str):
        """Save fine-tuned model checkpoint."""
        logger.info(f"Saving checkpoint to {path}")
        torch.save(self.model.state_dict(), path)
    
    def load_checkpoint(self, path: str):
        """Load fine-tuned model checkpoint."""
        logger.info(f"Loading checkpoint from {path}")
        self.model.load_state_dict(torch.load(path, map_location=self.device))


def collect_preference_pairs_from_stem(
    questions_file: str,
    retrieval_index: str,
    output_file: str,
    k_docs: int = 5
) -> List[Dict]:
    """
    Collect preference pairs from STEM benchmark.
    
    For each question, retrieve top-k documents and label as:
    - 1.0: Document contains answer information
    - 0.0: Document does not contain answer information
    
    Args:
        questions_file: Path to STEM questions JSON
        retrieval_index: Path to retrieval results JSON
        output_file: Path to save preference pairs
        k_docs: Number of documents to label per question
    
    Returns:
        List of (query, document, label) dicts
    """
    import sys
    sys.path.insert(0, '/Users/jdsingh/slm_v0/subword_tokenizer')
    
    from hybrid_retrieval import HybridRetrieval
    
    logger.info(f"Collecting preference pairs from {questions_file}")
    
    # Load questions
    with open(questions_file) as f:
        questions = json.load(f)
    
    pairs = []
    
    for question in questions:
        query = question['query']
        expected_answer = question['expected_answer']
        
        # Retrieve top-k documents
        retrieval = HybridRetrieval()
        results = retrieval.retrieve(query, k=k_docs)
        
        for i, result in enumerate(results):
            doc_text = result['text']
            doc_id = result.get('doc_id', f"{query}_doc_{i}")
            
            # Label: 1.0 if expected_answer in document
            label = 1.0 if expected_answer.lower() in doc_text.lower() else 0.0
            
            pairs.append({
                'query': query,
                'document': doc_text[:500],  # Truncate for efficiency
                'label': label,
                'question_id': question['id'],
                'doc_id': doc_id
            })
    
    logger.info(f"Collected {len(pairs)} preference pairs")
    logger.info(f"Positive examples: {len([p for p in pairs if p['label'] > 0.5])}")
    
    # Save pairs
    with open(output_file, 'w') as f:
        json.dump(pairs, f, indent=2)
    
    return pairs


def main():
    """CLI for fine-tuning cross-encoder on preference pairs."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Fine-tune cross-encoder on domain preference pairs"
    )
    parser.add_argument(
        "--questions",
        default="data/phase4_stem_60qa.json",
        help="Path to STEM questions file"
    )
    parser.add_argument(
        "--output-pairs",
        default="data/cross_encoder_preference_pairs.json",
        help="Path to save preference pairs"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of fine-tuning epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for fine-tuning"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--output-model",
        default="checkpoints/cross_encoder_finetuned.pt",
        help="Path to save fine-tuned model"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # Step 1: Collect preference pairs
    logger.info("=" * 70)
    logger.info("STEP 1: Collect Preference Pairs from STEM Benchmark")
    logger.info("=" * 70)
    pairs = collect_preference_pairs_from_stem(
        args.questions,
        "data/retrieval/",
        args.output_pairs
    )
    
    # Step 2: Fine-tune cross-encoder
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: Fine-tune Cross-Encoder")
    logger.info("=" * 70)
    
    # Split into train/val
    np.random.seed(42)
    indices = np.random.permutation(len(pairs))
    train_idx = indices[:int(0.8 * len(pairs))]
    val_idx = indices[int(0.8 * len(pairs)):]
    
    train_pairs = [pairs[i] for i in train_idx]
    val_pairs = [pairs[i] for i in val_idx]
    
    tuner = CrossEncoderFineTuner(learning_rate=args.learning_rate)
    history = tuner.fine_tune(
        train_pairs,
        val_pairs,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Step 3: Save model
    tuner.save_checkpoint(args.output_model)
    
    logger.info("\n" + "=" * 70)
    logger.info("Fine-tuning Complete!")
    logger.info(f"Model saved to: {args.output_model}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
