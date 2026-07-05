#!/usr/bin/env python3
"""
Download arXiv physics papers and save to structured format.
"""
import arxiv
import json
from pathlib import Path
from tqdm import tqdm
import argparse
from datetime import datetime

def download_arxiv_papers(category="physics", limit=10000, output_dir="data/physics_papers/raw"):
    """Stream arXiv papers and save as JSONL."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_file = output_path / f"arxiv_{category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    
    print(f"Downloading {limit} papers from category: {category}")
    print(f"Output: {output_file}")
    
    client = arxiv.Client(
        page_size=1000,
        delay_seconds=3,
        num_retries=5
    )
    
    search = arxiv.Search(
        query=f"cat:{category}",
        sort_by=arxiv.SortCriterion.SubmittedDate,
        max_results=limit
    )
    
    count = 0
    with open(output_file, "w") as f:
        for result in tqdm(client.results(search), total=limit, desc="Papers downloaded"):
            paper_data = {
                "arxiv_id": result.entry_id.split("/abs/")[-1],
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "published": result.published.isoformat() if result.published else None,
                "summary": result.summary,
                "categories": result.categories,
                "pdf_url": result.pdf_url,
            }
            f.write(json.dumps(paper_data) + "\n")
            count += 1
            if count >= limit:
                break
    
    print(f"\nDownloaded {count} papers to {output_file}")
    return str(output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download arXiv physics papers")
    parser.add_argument("--category", default="physics", help="arXiv category (default: physics)")
    parser.add_argument("--limit", type=int, default=10000, help="Max papers to download (default: 10000)")
    parser.add_argument("--output-dir", default="data/physics_papers/raw", help="Output directory")
    args = parser.parse_args()
    
    download_arxiv_papers(
        category=args.category,
        limit=args.limit,
        output_dir=args.output_dir
    )
