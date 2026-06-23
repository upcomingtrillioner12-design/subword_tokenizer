#!/usr/bin/env python3
import sys, re
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

def clean(text):
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"\S+@\S+\.\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    text = "".join(ch for ch in text if ch == "\n" or 32 <= ord(ch) < 127)
    return text.strip()

max_docs = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] else None
print(f"Loading OpenWebText (max_docs={max_docs})...")

ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
out = Path("openwebtext.txt")
buffer, count = [], 0

with open(out, "w", encoding="utf-8") as f:
    for ex in tqdm(ds, desc="Docs", unit="doc"):
        if max_docs and count >= max_docs: break
        t = clean(ex["text"])
        if len(t) < 20: continue
        buffer.extend([t, ""])
        count += 1
        if len(buffer) >= 10000:
            f.write("\n".join(buffer) + "\n")
            buffer.clear()
    if buffer:
        f.write("\n".join(buffer) + "\n")

print(f"Done: {count:,} docs → {out} ({out.stat().st_size/1024/1024:.1f} MB)")
