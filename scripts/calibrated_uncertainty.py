"""
Task 10: Improved Uncertainty Calibration

The baseline uncertainty_score from Task 9 was decoupled from correctness
(all 20 questions correct but uncertainty ranging 0.0-0.8). This module
implements a 4-component calibration framework to improve confidence signals:

1. **Logprob Spread**: Variance across top-5 option scores (higher variance = lower confidence)
2. **Context Relevance**: Penalty if semantic_similarity < threshold (bad retrieval = uncertainty)
3. **Entailment Consistency**: Penalty if entailment score conflicts with generation confidence
4. **Faithfulness Grounding**: Penalty if faithfulness < threshold (hallucination risk)

Target: Push high-confidence correct answers to 0.1-0.3, ambiguous to 0.7-0.9
"""

import json
import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import math

logger = logging.getLogger(__name__)


@dataclass
class CalibratedUncertaintyConfig:
    """Configuration for calibrated uncertainty scoring"""
    # Logprob spread component
    logprob_spread_weight: float = 0.30
    logprob_spread_threshold: float = 0.5  # Consider spread < 0.5 as high confidence
    
    # Context relevance component
    context_weight: float = 0.25
    semantic_sim_threshold: float = 0.85  # Threshold below which apply penalty
    
    # Entailment consistency component
    entailment_weight: float = 0.25
    entailment_threshold: float = 0.85  # Below this = lower confidence
    
    # Faithfulness component
    faithfulness_weight: float = 0.20
    faithfulness_threshold: float = 0.15  # Below this = apply penalty
    
    # Overall scaling
    calibration_slope: float = 0.9  # Steepness of confidence curve
    calibration_offset: float = 0.1  # Minimum uncertainty baseline


class CalibratedUncertaintyEvaluator:
    """
    4-component uncertainty calibration based on:
    - Model confidence (logprob distribution)
    - Retrieval quality (semantic similarity)
    - Context grounding (entailment)
    - Faithfulness (hallucination detection)
    """
    
    def __init__(self, config: Optional[CalibratedUncertaintyConfig] = None):
        self.config = config or CalibratedUncertaintyConfig()
        self.stats = {
            'processed': 0,
            'high_confidence': 0,
            'low_confidence': 0,
            'calibration_adjustments': []
        }
    
    def compute_logprob_spread_component(
        self,
        option_scores: List[Dict[str, float]]
    ) -> Tuple[float, Dict]:
        """
        Compute confidence from logprob distribution across top-5 options.
        
        High variance across options = model uncertain which to pick
        Low variance = model confident in top option
        
        Returns: (confidence_score 0-1, component_info)
        """
        if not option_scores or len(option_scores) < 2:
            return 0.5, {'msg': 'Insufficient options', 'variance': 0}
        
        # Extract log probabilities (lower/more negative = lower prob)
        logprobs = [opt.get('avg_logprob', -100) for opt in option_scores]
        
        # Compute normalized spread (range / max)
        spread = (max(logprobs) - min(logprobs)) / abs(min(logprobs)) if min(logprobs) != 0 else 0
        
        # Normalize to 0-1 (higher spread = lower confidence)
        # If spread < threshold, confidence is high (low uncertainty component)
        if spread < self.config.logprob_spread_threshold:
            confidence = 0.1  # Very confident in top option
        else:
            confidence = min(0.8, 0.1 + (spread / self.config.logprob_spread_threshold) * 0.7)
        
        component_info = {
            'spread': float(spread),
            'num_options': len(option_scores),
            'logprob_range': float(max(logprobs) - min(logprobs)),
            'confidence': float(confidence)
        }
        
        return confidence, component_info
    
    def compute_context_relevance_component(
        self,
        semantic_similarity: float
    ) -> Tuple[float, Dict]:
        """
        Penalty if retrieved context is not semantically similar to answer.
        
        High semantic_similarity = answer well-grounded in context
        Low semantic_similarity = potential hallucination = high uncertainty
        
        Returns: (uncertainty_component 0-1, component_info)
        """
        # If similarity is good, no uncertainty penalty
        if semantic_similarity >= self.config.semantic_sim_threshold:
            uncertainty = 0.0  # No penalty
        else:
            # Linear penalty for poor semantic match
            gap = self.config.semantic_sim_threshold - semantic_similarity
            uncertainty = min(0.8, gap * 2.0)  # Scale factor ~2x for meaningful range
        
        component_info = {
            'semantic_similarity': float(semantic_similarity),
            'threshold': self.config.semantic_sim_threshold,
            'uncertainty': float(uncertainty)
        }
        
        return uncertainty, component_info
    
    def compute_entailment_consistency_component(
        self,
        entailment_score: float,
        factual_consistency: float
    ) -> Tuple[float, Dict]:
        """
        Check if context entails the generated answer.
        
        High entailment = answer logically follows from context
        Low entailment = answer disconnected from context = higher uncertainty
        
        Returns: (uncertainty_component 0-1, component_info)
        """
        # Average entailment signals
        entailment_avg = (entailment_score + factual_consistency) / 2
        
        # If strong entailment, low uncertainty component
        if entailment_avg >= self.config.entailment_threshold:
            uncertainty = 0.0
        else:
            # Quadratic penalty for weak entailment (more aggressive)
            gap = self.config.entailment_threshold - entailment_avg
            uncertainty = min(0.8, (gap ** 1.5) * 3.0)
        
        component_info = {
            'entailment_score': float(entailment_score),
            'factual_consistency': float(factual_consistency),
            'entailment_avg': float(entailment_avg),
            'threshold': self.config.entailment_threshold,
            'uncertainty': float(uncertainty)
        }
        
        return uncertainty, component_info
    
    def compute_faithfulness_component(
        self,
        faithfulness: float
    ) -> Tuple[float, Dict]:
        """
        Penalty if faithfulness score is low (indicates hallucination risk).
        
        High faithfulness = answer grounded in context
        Low faithfulness = potential hallucination = high uncertainty
        
        Returns: (uncertainty_component 0-1, component_info)
        """
        # Threshold-based penalty
        if faithfulness >= self.config.faithfulness_threshold:
            uncertainty = 0.0
        else:
            # Steep penalty for low faithfulness (hallucination risk)
            gap = self.config.faithfulness_threshold - faithfulness
            uncertainty = min(1.0, gap * 5.0)  # Aggressive scaling
        
        component_info = {
            'faithfulness': float(faithfulness),
            'threshold': self.config.faithfulness_threshold,
            'uncertainty': float(uncertainty)
        }
        
        return uncertainty, component_info
    
    def calibrate(
        self,
        metrics: Dict,
        option_scores: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Compute calibrated uncertainty score combining all 4 components.
        
        Args:
            metrics: Dict with keys: semantic_similarity, entailment_score,
                    factual_consistency, faithfulness, uncertainty_score (baseline)
            option_scores: List of option dicts with 'avg_logprob' key
        
        Returns:
            Dict with calibrated_uncertainty, component_scores, adjustment_info
        """
        self.stats['processed'] += 1
        
        # Extract metrics with defaults
        semantic_sim = metrics.get('semantic_similarity', 1.0)
        entailment = metrics.get('entailment_score', 0.5)
        factual_consistency = metrics.get('factual_consistency', 0.5)
        faithfulness = metrics.get('faithfulness', 0.0)
        baseline_uncertainty = metrics.get('uncertainty_score', 0.5)
        
        # Compute 4 components
        logprob_conf, logprob_info = self.compute_logprob_spread_component(option_scores or [])
        context_unc, context_info = self.compute_context_relevance_component(semantic_sim)
        entailment_unc, entailment_info = self.compute_entailment_consistency_component(
            entailment, factual_consistency
        )
        faithfulness_unc, faithfulness_info = self.compute_faithfulness_component(faithfulness)
        
        # Combine components with weighted average
        # Note: logprob_conf is confidence, so convert to uncertainty
        logprob_unc = 1.0 - logprob_conf
        
        combined_uncertainty = (
            self.config.logprob_spread_weight * logprob_unc +
            self.config.context_weight * context_unc +
            self.config.entailment_weight * entailment_unc +
            self.config.faithfulness_weight * faithfulness_unc
        )
        
        # Apply calibration curve (sigmoid-like scaling)
        # Amplify differences: low stays low, high stays high
        calibrated = self.config.calibration_offset + (
            combined_uncertainty * self.config.calibration_slope
        )
        calibrated = max(0.0, min(1.0, calibrated))  # Clip to [0, 1]
        
        # Track adjustment
        adjustment = calibrated - baseline_uncertainty
        self.stats['calibration_adjustments'].append(adjustment)
        
        if calibrated < 0.4:
            self.stats['high_confidence'] += 1
        else:
            self.stats['low_confidence'] += 1
        
        result = {
            'calibrated_uncertainty': float(calibrated),
            'baseline_uncertainty': float(baseline_uncertainty),
            'adjustment': float(adjustment),
            'components': {
                'logprob_spread': {
                    'confidence': float(logprob_conf),
                    'uncertainty': float(logprob_unc),
                    **logprob_info
                },
                'context_relevance': context_info,
                'entailment_consistency': entailment_info,
                'faithfulness': faithfulness_info
            },
            'combined_uncertainty': float(combined_uncertainty)
        }
        
        return result
    
    def get_stats(self) -> Dict:
        """Return calibration statistics"""
        adjustments = self.stats['calibration_adjustments']
        return {
            'processed': self.stats['processed'],
            'high_confidence': self.stats['high_confidence'],
            'low_confidence': self.stats['low_confidence'],
            'avg_adjustment': float(np.mean(adjustments)) if adjustments else 0.0,
            'adjustment_std': float(np.std(adjustments)) if adjustments else 0.0,
            'adjustment_range': (
                float(min(adjustments)) if adjustments else 0.0,
                float(max(adjustments)) if adjustments else 0.0
            )
        }


def apply_calibrated_uncertainty_to_results(
    results_file: str,
    output_file: str,
    config: Optional[CalibratedUncertaintyConfig] = None
) -> Dict:
    """
    Load Task 9 results and apply calibrated uncertainty scoring.
    
    Args:
        results_file: Path to rag_generation_eval JSON file
        output_file: Path to save calibrated results
        config: Optional CalibratedUncertaintyConfig
    
    Returns:
        Summary statistics
    """
    logger.info(f"Loading results from {results_file}")
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    evaluator = CalibratedUncertaintyEvaluator(config)
    
    # Apply calibration to each result
    calibration_details = []
    for result in data['results']:
        metrics = result.get('metrics', {})
        option_scores = result.get('option_scores', [])
        
        calibration = evaluator.calibrate(metrics, option_scores)
        result['calibration'] = calibration
        result['metrics']['calibrated_uncertainty'] = calibration['calibrated_uncertainty']
        
        calibration_details.append({
            'id': result['id'],
            'baseline': calibration['baseline_uncertainty'],
            'calibrated': calibration['calibrated_uncertainty'],
            'adjustment': calibration['adjustment']
        })
    
    # Recompute summary statistics
    calibrated_scores = [c['calibrated'] for c in calibration_details]
    
    data['summary']['avg_calibrated_uncertainty'] = float(np.mean(calibrated_scores))
    data['summary']['median_calibrated_uncertainty'] = float(np.median(calibrated_scores))
    data['summary']['std_calibrated_uncertainty'] = float(np.std(calibrated_scores))
    data['summary']['min_calibrated_uncertainty'] = float(np.min(calibrated_scores))
    data['summary']['max_calibrated_uncertainty'] = float(np.max(calibrated_scores))
    
    data['summary']['calibration_improvements'] = {
        'high_confidence_count': evaluator.stats['high_confidence'],
        'low_confidence_count': evaluator.stats['low_confidence'],
        'avg_adjustment': float(np.mean(evaluator.stats['calibration_adjustments'])),
        'adjustment_std': float(np.std(evaluator.stats['calibration_adjustments']))
    }
    
    # Save calibrated results
    logger.info(f"Saving calibrated results to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Calibration complete:")
    logger.info(f"  Baseline avg uncertainty: {data['summary']['avg_uncertainty_score']:.4f}")
    logger.info(f"  Calibrated avg uncertainty: {data['summary']['avg_calibrated_uncertainty']:.4f}")
    logger.info(f"  Calibration range: {data['summary']['min_calibrated_uncertainty']:.4f} - {data['summary']['max_calibrated_uncertainty']:.4f}")
    logger.info(f"  High confidence cases: {evaluator.stats['high_confidence']}/{evaluator.stats['processed']}")
    
    return {
        'calibration_details': calibration_details,
        'summary': data['summary'],
        'stats': evaluator.get_stats()
    }


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python calibrated_uncertainty.py <input_json> <output_json>")
        print("\nExample:")
        print("  python calibrated_uncertainty.py \\")
        print("    results/rag_generation_eval/rag_generation_eval_20260715_195330.json \\")
        print("    results/rag_generation_eval/rag_generation_eval_20260715_195330_calibrated.json")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    logging.basicConfig(level=logging.INFO)
    result = apply_calibrated_uncertainty_to_results(input_file, output_file)
    
    print("\n" + "="*70)
    print("CALIBRATION RESULTS SUMMARY")
    print("="*70)
    print(f"Baseline avg uncertainty: {result['summary'].get('avg_uncertainty_score', 0):.4f}")
    print(f"Calibrated avg uncertainty: {result['summary'].get('avg_calibrated_uncertainty', 0):.4f}")
    print(f"Uncertainty std dev: {result['summary'].get('std_calibrated_uncertainty', 0):.4f}")
    print(f"Range: {result['summary'].get('min_calibrated_uncertainty', 0):.4f} - {result['summary'].get('max_calibrated_uncertainty', 0):.4f}")
    print(f"High confidence (< 0.4): {result['stats']['high_confidence']} questions")
    print(f"Low confidence (>= 0.4): {result['stats']['low_confidence']} questions")
    print(f"Avg adjustment: {result['stats']['avg_adjustment']:.4f} ± {result['stats']['adjustment_std']:.4f}")
    print("="*70)
