#!/usr/bin/env python3
"""
Phase 3: Benchmark Inference - Compare Phase 1 vs Phase 2

- Run all 50 evaluation prompts through both models
- Compare: Speed, output quality, physics terminology accuracy
- Save detailed results and aggregated statistics
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
from collections import defaultdict

import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

import stream_train
from sampling_profiles import SAMPLING_PROFILES, resolve_sampling_config


@dataclass
class BenchmarkResult:
    """Result from comparing two models on a single prompt."""
    prompt_id: str
    category: str
    difficulty: str
    prompt_text: str
    
    # Phase 1 (Baseline) metrics
    phase1_tokens: int
    phase1_time: float
    phase1_tps: float
    
    # Phase 2 (LoRA) metrics
    phase2_tokens: int
    phase2_time: float
    phase2_tps: float
    
    # Comparative metrics
    token_diff: int
    time_diff: float
    speedup_factor: float
    quality_score: Optional[float] = None
    physics_accuracy: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


class BenchmarkEngine:
    """Run inference benchmarks comparing Phase 1 and Phase 2 models."""
    
    def __init__(
        self,
        base_checkpoint: Path,
        lora_checkpoint: Path,
        tokenizer_model: Path,
        eval_prompts_path: Path,
        generation_config: Optional[Dict] = None,
        device: str = "auto",
        verbose: bool = False,
    ):
        """Initialize benchmark engine with both models."""
        self.base_checkpoint = Path(base_checkpoint)
        self.lora_checkpoint = Path(lora_checkpoint)
        self.tokenizer_model = Path(tokenizer_model)
        self.eval_prompts_path = Path(eval_prompts_path)
        self.verbose = verbose
        self.generation_config = generation_config or SAMPLING_PROFILES["production"].copy()
        
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
        
        if self.verbose:
            print(f"[benchmark] Device: {self.device}")
        
        # Load base model (Phase 1 baseline)
        self.phase1_engine = self._load_phase1_model()
        
        # Load LoRA-tuned model (Phase 2)
        self.phase2_engine = self._load_phase2_model()
        
        # Load evaluation prompts
        self.eval_prompts = self._load_eval_prompts()
        
        if self.verbose:
            print(f"[benchmark] Loaded {len(self.eval_prompts)} evaluation prompts")
    
    def _load_phase1_model(self):
        """Load Phase 1 baseline model (base only)."""
        if self.verbose:
            print("[benchmark] Loading Phase 1 baseline model")
        
        # Import inference engine
        from scripts.inference_lora import LoRAInferenceEngine
        
        engine = LoRAInferenceEngine(
            base_checkpoint=self.base_checkpoint,
            lora_checkpoint=None,  # No LoRA for Phase 1
            device=str(self.device),
            verbose=False,
        )
        return engine
    
    def _load_phase2_model(self):
        """Load Phase 2 LoRA-tuned model."""
        if self.verbose:
            print("[benchmark] Loading Phase 2 LoRA model")
        
        # Import inference engine
        from scripts.inference_lora import LoRAInferenceEngine
        
        engine = LoRAInferenceEngine(
            base_checkpoint=self.base_checkpoint,
            lora_checkpoint=self.lora_checkpoint,
            device=str(self.device),
            verbose=False,
        )
        return engine
    
    def _load_eval_prompts(self) -> Dict:
        """Load evaluation prompts from JSON file."""
        with open(self.eval_prompts_path) as f:
            data = json.load(f)
        
        prompts = {}
        for category_key in ['quantum_mechanics', 'relativity_cosmology', 
                             'thermodynamics_statistical', 'electromagnetism', 'particle_physics']:
            if category_key in data:
                for prompt_data in data[category_key]['prompts']:
                    prompt_id = prompt_data['id']
                    prompts[prompt_id] = {
                        'category': category_key.replace('_', ' ').title(),
                        'difficulty': prompt_data['difficulty'],
                        'text': prompt_data['prompt'],
                    }
        
        return prompts
    
    def _load_tokenizer(self):
        """Load tokenizer."""
        stream_train._activate_tokenizer_model(self.tokenizer_model)
        
        class SimpleTokenizer:
            def encode_text(self, text: str, max_length: int = 256) -> List[int]:
                ids = stream_train._tokenize_with_our_model(text)
                return ids[:max_length]
            
            def decode_ids(self, ids: List[int]) -> str:
                return f"[{len(ids)} tokens]"
        
        return SimpleTokenizer()
    
    def run_benchmark(self, num_samples: Optional[int] = None) -> List[BenchmarkResult]:
        """Run benchmark on all (or subset of) prompts."""
        tokenizer = self._load_tokenizer()
        results = []
        
        prompt_list = list(self.eval_prompts.items())
        if num_samples:
            prompt_list = prompt_list[:num_samples]
        
        print(f"\n[benchmark] Running {len(prompt_list)} prompts...")
        print(f"[benchmark] Device: {self.device}\n")
        
        for idx, (prompt_id, prompt_data) in enumerate(prompt_list):
            if self.verbose or (idx + 1) % 5 == 0:
                print(f"[{idx+1}/{len(prompt_list)}] {prompt_id}...", end=" ")
            
            prompt_text = prompt_data['text']
            category = prompt_data['category']
            difficulty = prompt_data['difficulty']
            
            # Run Phase 1
            try:
                phase1_output, phase1_metrics = self.phase1_engine.generate(
                    prompt=prompt_text,
                    tokenizer=tokenizer,
                    max_tokens=int(self.generation_config["max_tokens"]),
                    temperature=float(self.generation_config["temperature"]),
                    top_k=self.generation_config["top_k"],
                    top_p=self.generation_config["top_p"],
                )
                phase1_tokens = phase1_metrics.generated_tokens
                phase1_time = phase1_metrics.elapsed_seconds
                phase1_tps = phase1_metrics.tokens_per_second
            except Exception as e:
                if self.verbose:
                    print(f"Phase 1 error: {e}")
                    import traceback
                    traceback.print_exc()
                phase1_tokens, phase1_time, phase1_tps = 0, 0.0, 0.0
            
            # Run Phase 2
            try:
                phase2_output, phase2_metrics = self.phase2_engine.generate(
                    prompt=prompt_text,
                    tokenizer=tokenizer,
                    max_tokens=int(self.generation_config["max_tokens"]),
                    temperature=float(self.generation_config["temperature"]),
                    top_k=self.generation_config["top_k"],
                    top_p=self.generation_config["top_p"],
                )
                phase2_tokens = phase2_metrics.generated_tokens
                phase2_time = phase2_metrics.elapsed_seconds
                phase2_tps = phase2_metrics.tokens_per_second
            except Exception as e:
                if self.verbose:
                    print(f"Phase 2 error: {e}")
                phase2_tokens, phase2_time, phase2_tps = 0, 0.0, 0.0
            
            # Calculate comparative metrics
            token_diff = phase2_tokens - phase1_tokens
            time_diff = phase2_time - phase1_time
            speedup = phase1_time / phase2_time if phase2_time > 0 else 0.0
            
            result = BenchmarkResult(
                prompt_id=prompt_id,
                category=category,
                difficulty=difficulty,
                prompt_text=prompt_text,
                phase1_tokens=phase1_tokens,
                phase1_time=phase1_time,
                phase1_tps=phase1_tps,
                phase2_tokens=phase2_tokens,
                phase2_time=phase2_time,
                phase2_tps=phase2_tps,
                token_diff=token_diff,
                time_diff=time_diff,
                speedup_factor=speedup,
            )
            
            results.append(result)
            
            if self.verbose or (idx + 1) % 5 == 0:
                print(f"P1:{phase1_tokens} P2:{phase2_tokens} (ΔT:{time_diff:.3f}s)")
        
        print(f"\n[benchmark] Completed {len(results)} prompts")
        return results
    
    def compute_statistics(self, results: List[BenchmarkResult]) -> Dict:
        """Compute aggregate statistics from results."""
        if not results:
            return {}
        
        stats = {
            'total_prompts': len(results),
            'by_difficulty': defaultdict(lambda: {'count': 0, 'metrics': {}}),
            'by_category': defaultdict(lambda: {'count': 0, 'metrics': {}}),
            'aggregate': {
                'phase1_avg_tokens': 0,
                'phase2_avg_tokens': 0,
                'phase1_avg_time': 0,
                'phase2_avg_time': 0,
                'avg_token_diff': 0,
                'avg_time_diff': 0,
                'avg_speedup': 0,
            }
        }
        
        phase1_tokens_list = []
        phase2_tokens_list = []
        phase1_times = []
        phase2_times = []
        token_diffs = []
        time_diffs = []
        speedups = []
        
        for result in results:
            # Aggregate data
            phase1_tokens_list.append(result.phase1_tokens)
            phase2_tokens_list.append(result.phase2_tokens)
            phase1_times.append(result.phase1_time)
            phase2_times.append(result.phase2_time)
            token_diffs.append(result.token_diff)
            time_diffs.append(result.time_diff)
            speedups.append(result.speedup_factor)
            
            # By difficulty
            diff = result.difficulty
            stats['by_difficulty'][diff]['count'] += 1
            if 'tokens' not in stats['by_difficulty'][diff]['metrics']:
                stats['by_difficulty'][diff]['metrics']['phase1_tokens'] = []
                stats['by_difficulty'][diff]['metrics']['phase2_tokens'] = []
            stats['by_difficulty'][diff]['metrics']['phase1_tokens'].append(result.phase1_tokens)
            stats['by_difficulty'][diff]['metrics']['phase2_tokens'].append(result.phase2_tokens)
            
            # By category
            cat = result.category
            stats['by_category'][cat]['count'] += 1
            if 'tokens' not in stats['by_category'][cat]['metrics']:
                stats['by_category'][cat]['metrics']['phase1_tokens'] = []
                stats['by_category'][cat]['metrics']['phase2_tokens'] = []
            stats['by_category'][cat]['metrics']['phase1_tokens'].append(result.phase1_tokens)
            stats['by_category'][cat]['metrics']['phase2_tokens'].append(result.phase2_tokens)
        
        # Compute averages
        stats['aggregate']['phase1_avg_tokens'] = np.mean(phase1_tokens_list) if phase1_tokens_list else 0
        stats['aggregate']['phase2_avg_tokens'] = np.mean(phase2_tokens_list) if phase2_tokens_list else 0
        stats['aggregate']['phase1_avg_time'] = np.mean(phase1_times) if phase1_times else 0
        stats['aggregate']['phase2_avg_time'] = np.mean(phase2_times) if phase2_times else 0
        stats['aggregate']['avg_token_diff'] = np.mean(token_diffs) if token_diffs else 0
        stats['aggregate']['avg_time_diff'] = np.mean(time_diffs) if time_diffs else 0
        stats['aggregate']['avg_speedup'] = np.mean([s for s in speedups if s > 0]) if speedups else 0
        
        # Compute per-difficulty/category means
        for diff in stats['by_difficulty']:
            if stats['by_difficulty'][diff]['metrics']['phase1_tokens']:
                stats['by_difficulty'][diff]['metrics']['avg_phase1_tokens'] = \
                    np.mean(stats['by_difficulty'][diff]['metrics']['phase1_tokens'])
                stats['by_difficulty'][diff]['metrics']['avg_phase2_tokens'] = \
                    np.mean(stats['by_difficulty'][diff]['metrics']['phase2_tokens'])
        
        for cat in stats['by_category']:
            if stats['by_category'][cat]['metrics']['phase1_tokens']:
                stats['by_category'][cat]['metrics']['avg_phase1_tokens'] = \
                    np.mean(stats['by_category'][cat]['metrics']['phase1_tokens'])
                stats['by_category'][cat]['metrics']['avg_phase2_tokens'] = \
                    np.mean(stats['by_category'][cat]['metrics']['phase2_tokens'])
        
        return stats


def main():
    parser = argparse.ArgumentParser(description="Benchmark Phase 1 vs Phase 2 inference")
    parser.add_argument("--base-checkpoint", type=Path, required=True,
                        help="Path to base model checkpoint")
    parser.add_argument("--lora-checkpoint", type=Path, required=True,
                        help="Path to LoRA adapter checkpoint")
    parser.add_argument("--tokenizer-model", type=Path, required=True,
                        help="Path to tokenizer model")
    parser.add_argument("--eval-prompts", type=Path, default=Path("data/eval_prompts.json"),
                        help="Path to evaluation prompts file")
    parser.add_argument("--output-dir", type=Path, default=Path("results"),
                        help="Output directory for results")
    parser.add_argument("--device", default="auto",
                        help="Device: auto, cpu, mps, cuda")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Number of prompts to benchmark (default: all)")
    parser.add_argument("--sampling-profile", choices=["production", "canonical"], default="production",
                        help="Sampling profile for generation")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Override max generated tokens")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Override temperature")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Override top-k")
    parser.add_argument("--top-p", type=float, default=None,
                        help="Override top-p")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed progress")
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    generation_config = resolve_sampling_config(
        profile=args.sampling_profile,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    # Initialize benchmark engine
    engine = BenchmarkEngine(
        base_checkpoint=args.base_checkpoint,
        lora_checkpoint=args.lora_checkpoint,
        tokenizer_model=args.tokenizer_model,
        eval_prompts_path=args.eval_prompts,
        generation_config=generation_config,
        device=args.device,
        verbose=args.verbose,
    )
    
    # Run benchmark
    results = engine.run_benchmark(num_samples=args.num_samples)
    
    # Compute statistics
    stats = engine.compute_statistics(results)
    
    # Save results
    results_file = args.output_dir / "phase3_benchmark_results.json"
    output_data = {
        'metadata': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_prompts': len(results),
            'device': str(engine.device),
            'sampling_profile': args.sampling_profile,
            'generation_config': generation_config,
        },
        'results': [r.to_dict() for r in results],
        'statistics': stats,
    }
    
    with open(results_file, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print(f"\n[main] Results saved to {results_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(f"\nTotal Prompts: {stats['total_prompts']}")
    print(f"\nAggregate Metrics:")
    agg = stats['aggregate']
    print(f"  Phase 1 avg tokens:        {agg['phase1_avg_tokens']:.1f}")
    print(f"  Phase 2 avg tokens:        {agg['phase2_avg_tokens']:.1f}")
    print(f"  Token difference:          {agg['avg_token_diff']:.1f}")
    print(f"  Phase 1 avg time:          {agg['phase1_avg_time']:.4f}s")
    print(f"  Phase 2 avg time:          {agg['phase2_avg_time']:.4f}s")
    print(f"  Time difference:           {agg['avg_time_diff']:.4f}s")
    print(f"  Average speedup:           {agg['avg_speedup']:.2f}x")
    
    print(f"\nBy Difficulty:")
    for diff in sorted(stats['by_difficulty'].keys()):
        d = stats['by_difficulty'][diff]
        print(f"  {diff.upper()}: {d['count']} prompts, "
              f"P1:{d['metrics'].get('avg_phase1_tokens', 0):.1f} tokens, "
              f"P2:{d['metrics'].get('avg_phase2_tokens', 0):.1f} tokens")
    
    print(f"\nBy Category:")
    for cat in sorted(stats['by_category'].keys()):
        c = stats['by_category'][cat]
        print(f"  {cat}: {c['count']} prompts, "
              f"P1:{c['metrics'].get('avg_phase1_tokens', 0):.1f} tokens, "
              f"P2:{c['metrics'].get('avg_phase2_tokens', 0):.1f} tokens")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
