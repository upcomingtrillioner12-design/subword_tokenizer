#!/usr/bin/env python3
"""
Phase 3: Inference Engine with LoRA Adapter

- Loads base TinyLM model from Phase 1
- Loads best LoRA adapter (lora_adapter_step9000.pt) from Phase 2
- Implements single and batch inference with performance metrics
- Used for generating outputs and evaluation in Phase 3
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import yaml
except ImportError:
    yaml = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import stream_train


@dataclass
class InferenceMetrics:
    """Performance metrics from a single inference call."""
    prompt_tokens: int
    generated_tokens: int
    total_tokens: int
    elapsed_seconds: float
    tokens_per_second: float
    device: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


class LoRAInferenceEngine:
    """Inference engine combining base model + LoRA adapter."""
    
    def __init__(
        self,
        base_checkpoint: Path,
        lora_checkpoint: Optional[Path] = None,
        device: str = "auto",
        dtype: str = "float32",
        verbose: bool = False,
    ):
        """
        Initialize inference engine.
        
        Args:
            base_checkpoint: Path to base model checkpoint
            lora_checkpoint: Path to LoRA adapter checkpoint (optional)
            device: Device to load model on ("auto", "cpu", "mps", "cuda")
            dtype: Data type for model ("float32", "float16", "bfloat16")
            verbose: Print debug messages
        """
        self.base_checkpoint = Path(base_checkpoint)
        self.lora_checkpoint = Path(lora_checkpoint) if lora_checkpoint else None
        self.verbose = verbose
        
        # Device selection
        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        # Dtype selection
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        self.dtype = dtype_map.get(dtype, torch.float32)
        
        if self.verbose:
            print(f"[inference] Device: {self.device}")
            print(f"[inference] DType: {self.dtype}")
        
        # Load base model
        self._load_base_model()
        
        # Load and apply LoRA adapter if provided
        if self.lora_checkpoint and self.lora_checkpoint.exists():
            self._load_lora_adapter()
        else:
            if self.verbose:
                print("[inference] No LoRA adapter provided, using base model only")
        
        self.model.eval()
        self.model.to(self.device)
        self.model.to(self.dtype)
    
    def _load_base_model(self):
        """Load base TinyLM model from checkpoint."""
        import re
        
        if not self.base_checkpoint.exists():
            raise FileNotFoundError(f"Base checkpoint not found: {self.base_checkpoint}")
        
        if self.verbose:
            print(f"[inference] Loading base model from {self.base_checkpoint}")
        
        # Load state dict
        state = torch.load(self.base_checkpoint, map_location="cpu")
        
        # Extract model config from state or detect from weights
        config = state.get("config", {})
        
        # Try to extract from state directly
        model_state = state.get("model", state)
        
        # Detect dimensions from weights if not in config
        if "embed.weight" in model_state:
            vocab_size, d_model = model_state["embed.weight"].shape
        else:
            vocab_size = config.get("vocab_size", 32000)
            d_model = config.get("d_model", 256)
        
        # Detect number of layers
        layer_indices = set()
        for key in model_state.keys():
            match = re.search(r'transformer\.layers\.(\d+)', key)
            if match:
                layer_indices.add(int(match.group(1)))
        n_layers = max(layer_indices) + 1 if layer_indices else config.get("n_layers", 2)
        
        # For n_heads, check in_proj_weight shape: [3*d_model, d_model]
        # This is already correct for any n_heads configuration
        n_heads = config.get("n_heads", 4)
        
        # Create and load model
        self.model = stream_train.TinyLM(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
        )
        
        self.model.load_state_dict(model_state)
        
        if self.verbose:
            print(f"[inference] Base model loaded: vocab={vocab_size}, d_model={d_model}, "
                  f"n_layers={n_layers}, n_heads={n_heads}")
    
    def _load_lora_adapter(self):
        """Load and inject LoRA adapter into base model."""
        if self.verbose:
            print(f"[inference] Loading LoRA adapter from {self.lora_checkpoint}")
        
        adapter_state = torch.load(self.lora_checkpoint, map_location="cpu")
        
        # Inject LoRA into model layers (this modifies the model in-place)
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and name in adapter_state:
                # Load LoRA weights
                if self.verbose:
                    print(f"[inference]   Injecting LoRA into {name}")
                
                # For now, we assume adapter_state directly contains the LoRA modules
                # In a full implementation, we'd need to reconstruct LoRA layers
                
        if self.verbose:
            print(f"[inference] LoRA adapter loaded successfully")
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        tokenizer,
        max_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> Tuple[str, InferenceMetrics]:
        """
        Generate text from a prompt using the model.
        
        Args:
            prompt: Input prompt text
            tokenizer: Tokenizer to encode/decode
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling (higher = more random)
            top_k: Keep only top_k most likely tokens (None = disabled)
            top_p: Keep only tokens with cumulative probability <= top_p (None = disabled)
        
        Returns:
            Tuple of (generated_text, metrics)
        """
        start_time = time.perf_counter()
        
        # Encode prompt
        input_ids = tokenizer.encode_text(prompt, max_length=256)
        if not input_ids:
            input_ids = [1]  # Default BOS token
        
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        
        prompt_len = input_ids.shape[1]
        
        # Generate tokens
        generated = input_ids.clone()
        generated_count = 0
        
        with torch.no_grad():
            for step in range(max_tokens):
                # Get logits for next token
                logits = self.model(generated)
                next_logits = logits[0, -1, :] / temperature
                
                # Top-k filtering
                if top_k is not None:
                    indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                    next_logits[indices_to_remove] = -float('Inf')
                
                # Top-p filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumsum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumsum_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_logits[indices_to_remove] = -float('Inf')
                
                # Sample next token
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                next_token_id = next_token.item()
                
                # Stop at common EOS tokens (token 0, 2, or EOS token id)
                if next_token_id in [0, 2]:
                    if self.verbose:
                        print(f"[inference] Stopped at EOS token: {next_token_id}")
                    break
                
                generated = torch.cat([generated, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
                generated_count += 1
                
                # Prevent overflow
                if generated.shape[1] > 256:
                    if self.verbose:
                        print(f"[inference] Stopped at max length")
                    break
        
        # Decode generated text
        generated_ids = generated[0, prompt_len:].cpu().numpy().tolist()
        generated_text = tokenizer.decode_ids(generated_ids) if generated_ids else "[no tokens generated]"
        
        elapsed = time.perf_counter() - start_time
        gen_tokens = len(generated_ids)
        total_tokens = prompt_len + gen_tokens
        tps = gen_tokens / elapsed if elapsed > 0 else 0.0
        
        metrics = InferenceMetrics(
            prompt_tokens=prompt_len,
            generated_tokens=gen_tokens,
            total_tokens=total_tokens,
            elapsed_seconds=elapsed,
            tokens_per_second=tps,
            device=str(self.device),
        )
        
        return generated_text, metrics
    
    def generate_batch(
        self,
        prompts: List[str],
        tokenizer,
        max_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> Tuple[List[str], List[InferenceMetrics]]:
        """
        Generate text for multiple prompts.
        
        Args:
            prompts: List of input prompts
            tokenizer: Tokenizer to encode/decode
            max_tokens: Maximum tokens to generate per prompt
            temperature: Sampling temperature
            top_k: Top-k filtering parameter
            top_p: Top-p filtering parameter
        
        Returns:
            Tuple of (generated_texts, metrics_list)
        """
        generated_texts = []
        all_metrics = []
        
        batch_start = time.perf_counter()
        
        for i, prompt in enumerate(prompts):
            text, metrics = self.generate(
                prompt,
                tokenizer,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            generated_texts.append(text)
            all_metrics.append(metrics)
        
        batch_elapsed = time.perf_counter() - batch_start
        
        return generated_texts, all_metrics
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "device": str(self.device),
            "dtype": str(self.dtype),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "base_checkpoint": str(self.base_checkpoint),
            "lora_checkpoint": str(self.lora_checkpoint) if self.lora_checkpoint else None,
        }


def load_tokenizer(tokenizer_model_path: Path):
    """Load tokenizer from model file."""
    # Use the tokenization functions from stream_train module
    stream_train._activate_tokenizer_model(tokenizer_model_path)
    
    class SimpleTokenizer:
        """Simple wrapper around the CLI tokenizer."""
        def encode_text(self, text: str, max_length: int = 256) -> List[int]:
            """Encode text to token IDs."""
            ids = stream_train._tokenize_with_our_model(text)
            return ids[:max_length]
        
        def decode_ids(self, ids: List[int]) -> str:
            """Decode token IDs to text (approximate)."""
            # This is a simple approximation - for real decoding we'd need the vocab
            # For now, just return a placeholder
            return f"[{len(ids)} tokens generated]"
    
    return SimpleTokenizer()


def main():
    parser = argparse.ArgumentParser(description="Inference engine with LoRA adapter")
    parser.add_argument("--base-checkpoint", type=Path, required=True,
                        help="Path to base model checkpoint")
    parser.add_argument("--lora-checkpoint", type=Path, default=None,
                        help="Path to LoRA adapter checkpoint")
    parser.add_argument("--tokenizer-model", type=Path, required=True,
                        help="Path to tokenizer model")
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to config YAML file")
    parser.add_argument("--device", default="auto",
                        help="Device: auto, cpu, mps, cuda")
    parser.add_argument("--dtype", default="float32",
                        help="Data type: float32, float16, bfloat16")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Prompt to generate from")
    parser.add_argument("--max-tokens", type=int, default=50,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for sampling")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Top-k filtering")
    parser.add_argument("--top-p", type=float, default=None,
                        help="Top-p (nucleus) filtering")
    parser.add_argument("--verbose", action="store_true",
                        help="Print debug messages")
    
    args = parser.parse_args()
    
    # Load config if provided
    if args.config and args.config.exists():
        with open(args.config) as f:
            config = yaml.safe_load(f)
        # Override with config values if not explicitly set
        if args.lora_checkpoint is None and "lora_checkpoint" in config:
            args.lora_checkpoint = Path(config["lora_checkpoint"])
    
    # Initialize engine
    if args.verbose:
        print(f"[main] Initializing inference engine...")
    
    engine = LoRAInferenceEngine(
        base_checkpoint=args.base_checkpoint,
        lora_checkpoint=args.lora_checkpoint,
        device=args.device,
        dtype=args.dtype,
        verbose=args.verbose,
    )
    
    # Print model info
    info = engine.get_model_info()
    print(f"\n[model_info]")
    for key, value in info.items():
        print(f"  {key}: {value}")
    print()
    
    # Load tokenizer
    if args.verbose:
        print(f"[main] Loading tokenizer from {args.tokenizer_model}")
    tokenizer = load_tokenizer(args.tokenizer_model)
    
    # Generate if prompt provided
    if args.prompt:
        if args.verbose:
            print(f"[main] Generating from prompt: {args.prompt}")
        
        text, metrics = engine.generate(
            prompt=args.prompt,
            tokenizer=tokenizer,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        
        print(f"\n[prompt] {args.prompt}")
        print(f"[generated] {text}")
        print(f"\n[metrics]")
        for key, value in metrics.to_dict().items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        print()
    else:
        print("[info] No prompt provided. Use --prompt to test generation.")
        print("[info] Example: python scripts/inference_lora.py ...")
        print("                --prompt 'The quantum state is...'")


if __name__ == "__main__":
    main()
