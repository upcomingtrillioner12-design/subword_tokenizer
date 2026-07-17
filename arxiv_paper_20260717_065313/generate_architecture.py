#!/usr/bin/env python3
"""
Generate simple architecture diagram
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')

# Define components with positions
components = [
    ("arXiv Stream", (1, 4.5), (1.5, 0.8), 'lightblue'),
    ("Rust BPE\nTokenizer", (3.5, 4.5), (1.5, 0.8), 'lightgreen'),
    ("TinyLM\nPre-Train", (6, 4.5), (1.5, 0.8), 'lightcoral'),
    ("LoRA\nFine-Tune", (8.5, 4.5), (1.5, 0.8), 'gold'),
    ("Checkpoint\nRanking", (8.5, 2.5), (1.5, 0.8), 'lightgray'),
    ("Best Model\nDeployment", (8.5, 0.8), (1.5, 0.8), 'lightpink'),
]

# Draw boxes
for label, (x, y), (w, h), color in components:
    box = FancyBboxPatch((x-w/2, y-h/2), w, h, 
                         boxstyle="round,pad=0.1",
                         facecolor=color, edgecolor='black',
                         linewidth=2)
    ax.add_patch(box)
    ax.text(x, y, label, ha='center', va='center', fontsize=10, fontweight='bold')

# Draw arrows
arrows = [
    ((1, 4.5), (2.75, 4.5)),
    ((3.5, 4.5), (5.25, 4.5)),
    ((6, 4.5), (7.75, 4.5)),
    ((8.5, 4.5), (8.5, 3.3)),
    ((8.5, 2.5), (8.5, 1.6)),
]

for start, end in arrows:
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

# Title
ax.text(5, 5.7, 'Coplay-Sync: System Architecture', 
        ha='center', va='center', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('architecture.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('architecture.pdf', bbox_inches='tight', facecolor='white')
print("✅ Architecture diagram generated")
