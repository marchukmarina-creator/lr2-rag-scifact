"""Plot the original and post-hoc scores from preserved experiment records."""
import json
import sys
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt

archive, summary_path, destination = map(Path, sys.argv[1:])
with zipfile.ZipFile(archive) as source:
    original = json.loads(source.read('reports/generated/summary.json'))
corrected = json.loads(summary_path.read_text(encoding='utf-8'))
groups = ['exp1_llm_only', 'exp2_rag_k1', 'exp1_rag_k3', 'exp2_rag_k5']
labels = ['LLM-only', 'RAG k=1', 'RAG k=3', 'RAG k=5']
first = {row['condition']: row['accuracy'] for row in original['experiment1']}
first.update({f'exp2_rag_k{row["k"]}': row['accuracy'] for row in original['k_comparison'] if row['k'] != 3})
before = [first[g] for g in groups]
after = [corrected['groups'][g]['accuracy'] for g in groups]
destination.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 4.5))
x = range(len(groups))
ax.bar([i-.19 for i in x], before, width=.38, label='Original parsing')
ax.bar([i+.19 for i in x], after, width=.38, label='Post-hoc DOC normalization')
ax.set_xticks(list(x), labels); ax.set_ylim(0, 1.1); ax.set_ylabel('Accuracy (90 claims)')
ax.legend(); ax.grid(axis='y', alpha=.2)
for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', padding=3, fontsize=8)
fig.tight_layout(); fig.savefig(destination / 'accuracy_comparison.png', dpi=180); plt.close(fig)
