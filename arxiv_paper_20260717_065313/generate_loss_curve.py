#!/usr/bin/env python3
"""
Generate professional loss curve figure from checkpoint data
"""

import matplotlib.pyplot as plt
import numpy as np

# Checkpoint data from actual experiments
steps = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
losses = [0.0107, 0.0081944, 0.0082776, 0.0079121, 0.0065578, 
          0.0064018, 0.0066084, 0.0064529, 0.0062082, 0.0060005, 0.0061344]

# Create figure with publication-quality styling
plt.figure(figsize=(8, 5))
plt.style.use('seaborn-v0_8-whitegrid')

# Plot with markers
plt.plot(steps, losses, 'b-', linewidth=2.5, label='Validation Loss', marker='o', 
         markersize=8, markerfacecolor='white', markeredgewidth=2)

# Highlight best point
best_idx = np.argmin(losses)
plt.plot(steps[best_idx], losses[best_idx], 'r*', markersize=15, 
         label=f'Best: Step {steps[best_idx]}, Loss = {losses[best_idx]:.4f}')

# Formatting
plt.xlabel('Training Steps', fontsize=12, fontweight='bold')
plt.ylabel('Evaluation Loss', fontsize=12, fontweight='bold')
plt.title('LoRA Fine-Tuning: Evaluation Loss Progression', 
          fontsize=14, fontweight='bold')
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save
plt.savefig('loss_curve.png', dpi=300, bbox_inches='tight')
plt.savefig('loss_curve.pdf', bbox_inches='tight')
print("✅ Loss curve generated: loss_curve.png, loss_curve.pdf")
