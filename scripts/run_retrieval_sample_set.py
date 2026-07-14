#!/usr/bin/env python3
import json
from pathlib import Path
import importlib.util


def main() -> None:
    root = Path('/Users/jdsingh/slm_v0')
    sub = root / 'subword_tokenizer'
    meta_path = root / 'data' / 'offline_physics' / 'corpus_metadata.json'
    work_jsonl = sub / 'data' / 'retrieval' / 'offline_physics_source_texts.jsonl'
    index_path = sub / 'data' / 'retrieval' / 'bm25_index.json'
    res_dir = sub / 'results' / 'retrieval_baseline'
    res_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads(meta_path.read_text())
    sources = [Path(p) for p in meta.get('sources', [])]

    work_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with work_jsonl.open('w', encoding='utf-8') as out:
        for i, p in enumerate(sources):
            if not p.exists():
                continue
            txt = p.read_text(encoding='utf-8', errors='ignore')
            out.write(json.dumps({'doc_id': i, 'title': p.name, 'text': txt}, ensure_ascii=False) + '\n')

    spec = importlib.util.spec_from_file_location('retrieval_baseline', sub / 'scripts' / 'retrieval_baseline.py')
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    mod.build_index(work_jsonl, index_path, chunk_size=220, chunk_overlap=40)

    index = json.loads(index_path.read_text())
    queries = [
        'What causes black holes to evaporate?',
        'Explain quantum entanglement and nonlocality.',
        'What is the Higgs mechanism in particle physics?',
        'How does general relativity describe gravity?',
        'What evidence supports dark matter in galaxies?',
        'What is wave-particle duality?',
        'How are gravitational waves detected?',
        'What is superconductivity at low temperatures?',
    ]

    json_out = []
    md_lines = [
        '# Retrieval Quality Sample Set (Initial Baseline)',
        '',
        'Corpus source list: /Users/jdsingh/slm_v0/data/offline_physics/corpus_metadata.json',
        f'Index path: {index_path}',
        'Top-k: 5',
        '',
    ]

    for qi, q in enumerate(queries, start=1):
        scores = mod.bm25_score(mod.tokenize(q), index)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
        md_lines.append(f'## Q{qi}: {q}')
        q_item = {'query': q, 'top_k': []}
        if not ranked:
            md_lines.append('- No results')
            md_lines.append('')
            json_out.append(q_item)
            continue

        for rank, (doc_id, score) in enumerate(ranked, start=1):
            d = index['docs'][doc_id]
            snippet = d['text'][:220].replace('\n', ' ')
            item = {
                'rank': rank,
                'score': round(float(score), 4),
                'doc_id': int(doc_id),
                'title': d.get('title', ''),
                'source_line': d.get('source_line'),
                'snippet': snippet,
            }
            q_item['top_k'].append(item)
            md_lines.append(f"- [{rank}] score={score:.4f} | title={item['title']} | doc_id={doc_id}")
            md_lines.append(f"  - snippet: {snippet}...")

        md_lines.append('')
        json_out.append(q_item)

    json_path = res_dir / 'retrieval_quality_sample_set.json'
    md_path = res_dir / 'retrieval_quality_sample_set.md'

    json_path.write_text(
        json.dumps({'index_meta': index['meta'], 'queries': json_out}, indent=2),
        encoding='utf-8',
    )
    md_path.write_text('\n'.join(md_lines), encoding='utf-8')

    print(f'[OK] Sample JSON: {json_path}')
    print(f'[OK] Sample Markdown: {md_path}')
    print(f"[OK] Indexed chunks: {index['meta']['num_docs']}")


if __name__ == '__main__':
    main()
