#!/usr/bin/env python3
"""
Analyze faithfulness failures in RAG evaluation results.
Identify which answers are hallucinating vs grounded in context.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict


def extract_words(text: str) -> set:
    """Extract words from text."""
    return set(re.findall(r"[a-zA-Z0-9_]+", text.lower()))


def analyze_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze a single result for faithfulness issues."""
    query = result["query"]
    expected = result["expected_answer"]
    generated = result["generated_answer"]
    context = result.get("context", "")
    
    # If context not in result, reconstruct from retrieved_topk
    if not context:
        context_parts = []
        for doc in result.get("retrieved_topk", []):
            doc_text = doc.get("text", "")
            if doc_text:
                context_parts.append(doc_text)
        context = " ".join(context_parts)
    
    # Extract word sets
    query_words = extract_words(query)
    expected_words = extract_words(expected)
    generated_words = extract_words(generated)
    context_words = extract_words(context)
    
    # Faithfulness calculations
    # 1) Generated-answer faithfulness (matches evaluation metric)
    generated_in_context = generated_words & context_words
    generated_not_in_context = generated_words - context_words
    faithfulness = len(generated_in_context) / len(generated_words) if generated_words else 0.0

    # 2) Expected-answer grounding (diagnostic)
    expected_in_context = expected_words & context_words
    expected_not_in_context = expected_words - context_words
    
    # Score breakdown
    num_expected_words = len(expected_words)
    num_generated_words = len(generated_words)
    num_grounded = len(generated_in_context)
    num_hallucinated = len(generated_not_in_context)
    
    return {
        "id": result["id"],
        "category": result.get("category", "unknown"),
        "query": query,
        "expected_answer": expected,
        "generated_answer": generated,
        "context_snippet": context[:200] if context else "[no context]",
        "faithfulness_score": faithfulness,
        "num_expected_words": num_expected_words,
        "num_generated_words": num_generated_words,
        "num_grounded_words": num_grounded,
        "num_hallucinated_words": num_hallucinated,
        "hallucinated_words": sorted(list(generated_not_in_context)),
        "grounded_words": sorted(list(generated_in_context)),
        "expected_grounding": {
            "ratio": (len(expected_in_context) / len(expected_words)) if expected_words else 0.0,
            "grounded_words": sorted(list(expected_in_context)),
            "missing_words": sorted(list(expected_not_in_context)),
        },
        "token_f1": result["metrics"]["token_f1"],
        "contains_expected": result["metrics"]["contains_expected"],
        "reported_faithfulness": result["metrics"]["faithfulness"],
    }


def categorize_hallucinations(analyses: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Categorize results by hallucination severity."""
    categories = {
        "perfect_faith": [],    # faithfulness == 1.0
        "high_faith": [],       # 0.7 <= faithfulness < 1.0
        "medium_faith": [],     # 0.3 <= faithfulness < 0.7
        "low_faith": [],        # 0.0 < faithfulness < 0.3
        "no_faith": [],         # faithfulness == 0.0
    }
    
    for analysis in analyses:
        faith = analysis["faithfulness_score"]
        if faith == 1.0:
            categories["perfect_faith"].append(analysis)
        elif faith >= 0.7:
            categories["high_faith"].append(analysis)
        elif faith >= 0.3:
            categories["medium_faith"].append(analysis)
        elif faith > 0.0:
            categories["low_faith"].append(analysis)
        else:
            categories["no_faith"].append(analysis)
    
    return categories


def print_analysis(eval_file: Path, output_file: Path = None):
    """Print detailed faithfulness analysis."""
    with open(eval_file) as f:
        report = json.load(f)
    
    results = report["results"]
    analyses = [analyze_result(r) for r in results]
    categories = categorize_hallucinations(analyses)
    
    # Summary statistics
    total = len(analyses)
    avg_faith = sum(a["faithfulness_score"] for a in analyses) / total
    
    output_lines = []
    
    output_lines.append("=" * 100)
    output_lines.append("FAITHFULNESS ANALYSIS REPORT")
    output_lines.append("=" * 100)
    output_lines.append("")
    output_lines.append(f"Total questions analyzed: {total}")
    output_lines.append(f"Average faithfulness: {avg_faith:.4f}")
    output_lines.append("")
    output_lines.append("Distribution by faithfulness level:")
    output_lines.append(f"  Perfect (1.0):       {len(categories['perfect_faith']):3d} ({len(categories['perfect_faith'])/total*100:5.1f}%)")
    output_lines.append(f"  High (0.7-1.0):      {len(categories['high_faith']):3d} ({len(categories['high_faith'])/total*100:5.1f}%)")
    output_lines.append(f"  Medium (0.3-0.7):    {len(categories['medium_faith']):3d} ({len(categories['medium_faith'])/total*100:5.1f}%)")
    output_lines.append(f"  Low (0.0-0.3):       {len(categories['low_faith']):3d} ({len(categories['low_faith'])/total*100:5.1f}%)")
    output_lines.append(f"  None (0.0):          {len(categories['no_faith']):3d} ({len(categories['no_faith'])/total*100:5.1f}%)")
    output_lines.append("")
    output_lines.append("-" * 100)
    output_lines.append("")
    
    # Detailed breakdowns
    for level_name, level_analyses in [
        ("NO FAITH (0.0) - Complete Hallucination", categories["no_faith"]),
        ("LOW FAITH (0.0-0.3) - Mostly Hallucination", categories["low_faith"]),
        ("MEDIUM FAITH (0.3-0.7) - Mixed", categories["medium_faith"]),
        ("HIGH FAITH (0.7-1.0) - Mostly Grounded", categories["high_faith"]),
        ("PERFECT FAITH (1.0) - Fully Grounded", categories["perfect_faith"]),
    ]:
        if not level_analyses:
            continue
        
        output_lines.append("")
        output_lines.append(f"{level_name} ({len(level_analyses)} questions)")
        output_lines.append("-" * 100)
        
        for analysis in level_analyses:
            output_lines.append("")
            output_lines.append(f"ID: {analysis['id']} ({analysis['category']})")
            output_lines.append(f"Query: {analysis['query']}")
            output_lines.append(f"Expected: {analysis['expected_answer']}")
            output_lines.append(f"Generated: {analysis['generated_answer']}")
            output_lines.append(f"Faithfulness: {analysis['faithfulness_score']:.4f} (reported: {analysis['reported_faithfulness']:.4f})")
            output_lines.append(f"Word breakdown: {analysis['num_grounded_words']}/{analysis['num_expected_words']} grounded, "
                              f"{analysis['num_hallucinated_words']} hallucinated")
            if analysis["hallucinated_words"]:
                output_lines.append(f"Hallucinated words: {', '.join(analysis['hallucinated_words'])}")
            if analysis["grounded_words"]:
                output_lines.append(f"Grounded words: {', '.join(analysis['grounded_words'])}")
    
    output_lines.append("")
    output_lines.append("-" * 100)
    output_lines.append("")
    output_lines.append("KEY INSIGHTS:")
    output_lines.append("")
    
    # Calculate insights
    total_grounded = sum(a["num_grounded_words"] for a in analyses)
    total_hallucinated = sum(a["num_hallucinated_words"] for a in analyses)
    total_words = total_grounded + total_hallucinated
    
    output_lines.append(f"1. Global word-level faithfulness: {total_grounded}/{total_words} = {total_grounded/total_words*100:.1f}% grounded")
    output_lines.append(f"   - Grounded words: {total_grounded}")
    output_lines.append(f"   - Hallucinated words: {total_hallucinated}")
    output_lines.append("")
    
    # Most hallucinated words
    hallucinated_freq = defaultdict(int)
    for analysis in analyses:
        for word in analysis["hallucinated_words"]:
            hallucinated_freq[word] += 1
    
    if hallucinated_freq:
        top_hallucinated = sorted(hallucinated_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        output_lines.append(f"2. Most commonly hallucinated words:")
        for word, count in top_hallucinated:
            output_lines.append(f"   - '{word}' (hallucinated {count} time(s))")
        output_lines.append("")
    
    # Accuracy vs faithfulness correlation
    perfect_accuracy_low_faith = [a for a in analyses if a["token_f1"] == 1.0 and a["faithfulness_score"] < 0.5]
    if perfect_accuracy_low_faith:
        output_lines.append(f"3. Perfect accuracy but low faithfulness: {len(perfect_accuracy_low_faith)} questions")
        output_lines.append("   (Model gets right answer despite not using context)")
        for a in perfect_accuracy_low_faith:
            output_lines.append(f"   - {a['id']}: {a['expected_answer']} (faith={a['faithfulness_score']:.2f})")
        output_lines.append("")
    
    # Print and optionally save
    text = "\n".join(output_lines)
    print(text)
    
    if output_file:
        output_file.write_text(text)
        print(f"\nAnalysis saved to: {output_file}")


if __name__ == "__main__":
    import sys
    
    # Use latest evaluation file if no argument provided
    results_dir = Path("/Users/jdsingh/slm_v0/subword_tokenizer/results/rag_generation_eval")
    
    if len(sys.argv) > 1:
        eval_file = Path(sys.argv[1])
    else:
        # Get latest file
        json_files = list(results_dir.glob("rag_generation_eval_*.json"))
        if not json_files:
            print("No evaluation files found!")
            sys.exit(1)
        eval_file = max(json_files)
    
    output_file = results_dir / f"faithfulness_analysis_{eval_file.stem.split('_', 3)[-1]}.md"
    print_analysis(eval_file, output_file)
