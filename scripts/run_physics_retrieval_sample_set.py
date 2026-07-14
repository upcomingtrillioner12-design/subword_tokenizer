#!/usr/bin/env python3
"""
Generate synthetic physics paper JSONL for true physics retrieval baseline.

Creates realistic physics paper records with title + abstract fields
for evaluation of retrieval quality on actual domain-relevant content.
"""

import json
from pathlib import Path


def generate_physics_papers() -> list[dict]:
    """Generate synthetic physics papers covering key topic areas."""
    papers = [
        {
            "title": "Hawking Radiation and Black Hole Thermodynamics",
            "abstract": "We investigate the mechanism by which black holes emit radiation through quantum effects near the event horizon. The relationship between black hole entropy and surface area is established through semiclassical gravitational calculations. This work provides the foundation for understanding black hole evaporation rates and the information paradox.",
            "category": "General Relativity",
        },
        {
            "title": "Quantum Entanglement and Bell Nonlocality in Multipartite Systems",
            "abstract": "This paper examines the phenomenon of quantum entanglement in systems with three or more particles, demonstrating violation of Bell inequalities in various configurations. We establish upper bounds on classical correlation strengths and provide experimental protocols for testing genuine multipartite nonlocality.",
            "category": "Quantum Mechanics",
        },
        {
            "title": "The Higgs Mechanism and Electroweak Symmetry Breaking",
            "abstract": "We provide a comprehensive review of the Higgs mechanism in the standard model. The generation of gauge boson masses through spontaneous symmetry breaking is explored both theoretically and phenomenologically. Recent experimental evidence from collider experiments is discussed in detail.",
            "category": "Particle Physics",
        },
        {
            "title": "General Relativity: Spacetime Curvature and Gravitational Fields",
            "abstract": "This foundational work describes how matter and energy curve spacetime geometry, leading to gravitational interactions. The Einstein field equations and their solutions for various mass distributions are presented, including the Schwarzschild and Kerr metrics.",
            "category": "General Relativity",
        },
        {
            "title": "Dark Matter Detection and Galactic Rotation Curves",
            "abstract": "We analyze the observational evidence for dark matter in galactic systems through rotation curve measurements and gravitational lensing. Various dark matter candidates are evaluated, including WIMPs, axions, and primordial black holes. Detection strategies are compared across multiple experiments.",
            "category": "Cosmology",
        },
        {
            "title": "Wave-Particle Duality in Quantum Mechanics",
            "abstract": "An exposition of the complementarity principle describing how quantum objects exhibit both wave and particle properties depending on measurement context. Double-slit experiments and their interpretations are analyzed through the framework of quantum field theory.",
            "category": "Quantum Mechanics",
        },
        {
            "title": "LIGO and Gravitational Wave Detection: Observing the Cosmos",
            "abstract": "This paper describes the detection of gravitational waves through laser interferometry. The sensitivity improvements in modern detectors enable observation of black hole mergers and neutron star coalescence. We discuss how gravitational wave astronomy opens new windows on astrophysical phenomena.",
            "category": "General Relativity",
        },
        {
            "title": "Superconductivity and the BCS Theory of Cooper Pairs",
            "abstract": "We examine the microscopic theory of superconductivity through electron-phonon interactions. The formation of Cooper pairs below the critical temperature and the resulting energy gap are analyzed. Experimental signatures and applications in quantum devices are discussed.",
            "category": "Condensed Matter",
        },
        {
            "title": "Quantum Field Theory and the Standard Model",
            "abstract": "Comprehensive treatment of quantum field theory as applied to the standard model of particle physics. Path integral formulation, renormalization, and asymptotic freedom are covered. Running coupling constants and their role in high-energy physics are emphasized.",
            "category": "Particle Physics",
        },
        {
            "title": "Thermodynamics and Statistical Mechanics of Phase Transitions",
            "abstract": "This work develops the theory of phase transitions using statistical mechanics principles. Critical phenomena, scaling laws, and universality classes are examined. Applications to magnetic systems, fluids, and quantum gases are provided.",
            "category": "Thermodynamics",
        },
        {
            "title": "String Theory and Extra Dimensions",
            "abstract": "An introduction to string theory as a candidate for quantum gravity. The compactification of extra dimensions and the emergence of gauge theories from geometry are discussed. Recent developments in string phenomenology and dualities are reviewed.",
            "category": "Quantum Gravity",
        },
        {
            "title": "Spin-Orbit Coupling and Topological Insulators",
            "abstract": "We investigate how spin-orbit interactions modify electronic band structures in condensed matter systems. The emergence of topologically protected surface states and their robustness to disorder is analyzed. Potential applications in quantum computing are explored.",
            "category": "Condensed Matter",
        },
        {
            "title": "Quantum Cryptography and Key Distribution",
            "abstract": "This paper reviews quantum mechanical principles applied to secure communication. The BB84 protocol for quantum key distribution and its security proofs are presented. Practical implementations and security against eavesdropping are discussed.",
            "category": "Quantum Information",
        },
        {
            "title": "Neutron Stars and Extreme Matter States",
            "abstract": "We study the equation of state for nuclear matter at extreme densities found in neutron star cores. Quark-gluon plasma formation and potential color-flavor locked phases are examined. Observational constraints from mass-radius relationships are analyzed.",
            "category": "Astrophysics",
        },
        {
            "title": "Quantum Computing and Entanglement-Based Algorithms",
            "abstract": "This work explores how quantum entanglement enables computational speedup beyond classical algorithms. Shor's algorithm for factorization and Grover's search algorithm are detailed. Practical requirements for fault-tolerant quantum computing are discussed.",
            "category": "Quantum Information",
        },
        {
            "title": "Inflation and the Cosmic Microwave Background",
            "abstract": "We present the inflationary paradigm for early universe cosmology. Primordial perturbations and their evolution into large-scale structure are traced. The prediction of CMB anisotropies and polarization patterns is covered in detail.",
            "category": "Cosmology",
        },
        {
            "title": "Laser Physics and Quantum Optics",
            "abstract": "Fundamental principles of laser operation including population inversion and stimulated emission are reviewed. Squeezed light, entangled photons, and quantum measurement in optical systems are examined for fundamental and applied purposes.",
            "category": "Optics",
        },
        {
            "title": "Symmetries and Conservation Laws in Physics",
            "abstract": "Emmy Noether's theorem connecting continuous symmetries to conserved quantities forms the basis of this investigation. Applications in quantum mechanics, field theory, and general relativity demonstrate the central role of symmetry principles.",
            "category": "Theoretical Physics",
        },
        {
            "title": "Plasma Physics and Fusion Energy",
            "abstract": "Behavior of ionized gases under extreme conditions relevant to fusion reactions is analyzed. Plasma confinement techniques in tokamaks and stellarators are discussed. Progress toward controlled nuclear fusion is evaluated.",
            "category": "Plasma Physics",
        },
        {
            "title": "Molecular Dynamics and Computational Materials Science",
            "abstract": "Numerical simulation techniques for studying many-body systems at atomic scale are presented. Force fields and interatomic potentials are reviewed. Applications to materials discovery and phase transitions are highlighted.",
            "category": "Computational Physics",
        },
        {
            "title": "Quantum Mechanics Interpretation: Copenhagen vs Many-Worlds",
            "abstract": "We examine foundational interpretations of quantum mechanics and their experimental implications. The measurement problem, wavefunction collapse, and alternative interpretations are analyzed. Recent experimental tests constraining interpretation possibilities are discussed.",
            "category": "Quantum Foundations",
        },
        {
            "title": "Black Hole Thermodynamics and the Information Paradox",
            "abstract": "The connection between black hole physics and thermodynamics, including Hawking radiation and entropy, is explored. The apparent loss of quantum information in black hole evaporation is analyzed. Recent proposals for resolution including the holographic principle are reviewed.",
            "category": "Quantum Gravity",
        },
        {
            "title": "Bose-Einstein Condensation in Trapped Atomic Gases",
            "abstract": "Phenomena emerging when ultracold atoms condense into a single quantum state are investigated. The transition to Bose-Einstein condensate phase and collective excitations are characterized. Experimental techniques and applications are described.",
            "category": "Atomic Physics",
        },
        {
            "title": "Exotic Hadrons and Quark Model Extensions",
            "abstract": "Theoretical predictions and experimental evidence for hadronic states beyond traditional quark model classifications are examined. Tetraquarks, pentaquarks, and other multiquark states are analyzed. Implications for understanding the strong nuclear force are discussed.",
            "category": "Particle Physics",
        },
        {
            "title": "Relativistic Quantum Mechanics and the Dirac Equation",
            "abstract": "The development of Dirac's equation merging special relativity with quantum mechanics is presented. Prediction of antimatter and the treatment of electrons in electromagnetic fields are covered. Connections to quantum field theory are established.",
            "category": "Quantum Mechanics",
        },
    ]
    return papers


def main() -> None:
    root = Path('/Users/jdsingh/slm_v0')
    sub = root / 'subword_tokenizer'
    output_dir = sub / 'data' / 'retrieval'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate synthetic physics papers
    papers = generate_physics_papers()

    # Write JSONL
    jsonl_path = output_dir / 'synthetic_physics_papers.jsonl'
    with jsonl_path.open('w', encoding='utf-8') as f:
        for i, paper in enumerate(papers):
            record = {
                'doc_id': i,
                'title': paper['title'],
                'abstract': paper['abstract'],
                'category': paper['category'],
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f'[OK] Generated {len(papers)} physics papers')
    print(f'[OK] Written to: {jsonl_path}')

    # Now build retrieval index and run sample set on this corpus
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        'retrieval_baseline', sub / 'scripts' / 'retrieval_baseline.py'
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    # Build index on physics papers
    index_path = output_dir / 'bm25_physics_papers_index.json'
    print(f'\n[*] Building BM25 index on {len(papers)} physics papers...')
    mod.build_index(jsonl_path, index_path, chunk_size=220, chunk_overlap=40)

    index = json.loads(index_path.read_text())

    # Evaluate queries
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
        '# Retrieval Quality Sample Set (True Physics Papers)',
        '',
        'Corpus: Synthetic physics papers with titles and abstracts',
        f'Index path: {index_path}',
        'Top-k: 5',
        '',
        '## Evaluation Summary',
        '',
        f'- Total indexed chunks: {index["meta"]["num_docs"]}',
        f'- Queries evaluated: {len(queries)}',
        f'- Vocabulary size: {len(index["meta"].get("doc_freq", {}))}',
        '',
        '## Per-Query Results',
        '',
    ]

    for qi, q in enumerate(queries, start=1):
        scores = mod.bm25_score(mod.tokenize(q), index)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]

        md_lines.append(f'### Q{qi}: {q}')
        md_lines.append('')
        q_item = {'query': q, 'top_k': []}

        if not ranked:
            md_lines.append('*No matching results found.*')
            md_lines.append('')
            json_out.append(q_item)
            continue

        for rank, (doc_id, score) in enumerate(ranked, start=1):
            d = index['docs'][doc_id]
            snippet = d['text'][:220].replace('\n', ' ')
            title = d.get('title', 'N/A')
            item = {
                'rank': rank,
                'score': round(float(score), 4),
                'doc_id': int(doc_id),
                'title': title,
                'snippet': snippet,
            }
            q_item['top_k'].append(item)
            md_lines.append(f'**[{rank}]** Score: {score:.4f}')
            md_lines.append(f'- Title: *{title}*')
            md_lines.append(f'- Snippet: {snippet}...')
            md_lines.append('')

        json_out.append(q_item)

    res_dir = sub / 'results' / 'retrieval_baseline'
    res_dir.mkdir(parents=True, exist_ok=True)

    json_path = res_dir / 'retrieval_quality_sample_set_physics.json'
    md_path = res_dir / 'retrieval_quality_sample_set_physics.md'

    json_path.write_text(
        json.dumps({'index_meta': index['meta'], 'queries': json_out}, indent=2),
        encoding='utf-8',
    )
    md_path.write_text('\n'.join(md_lines), encoding='utf-8')

    print(f'[OK] Sample JSON: {json_path}')
    print(f'[OK] Sample Markdown: {md_path}')
    print(f'[OK] Indexed chunks: {index["meta"]["num_docs"]}')


if __name__ == '__main__':
    main()
